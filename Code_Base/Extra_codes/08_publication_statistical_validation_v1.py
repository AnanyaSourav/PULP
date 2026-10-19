#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 08 — PUBLICATION-GRADE STATISTICAL VALIDATION v1
==============================================================================

PURPOSE
-------
Perform leakage-safe, deterministic, publication-oriented statistical
characterization of the feature representations intended for modeling:

    1. Final molecular descriptors
    2. Final retained 3D-QSAR descriptors
    3. MoLFormer 768-dimensional embeddings

This script builds on Script 07, but fixes its key methodological limitation:
metadata/target-derived variables such as Activity_Status are NEVER considered
predictors.

IMPORTANT PRINCIPLES
--------------------
1. TRAIN drives screening and shortlist selection.
2. TEST is used only for independent replication.
3. TEST never determines:
       - feature selection
       - direction/orientation
       - thresholds
       - shortlist membership
4. No features are removed.
5. No predictive model is trained.
6. All random procedures use fixed seeds.
7. Statistical significance and practical effect are reported separately.

STATISTICAL ANALYSES
--------------------
For every feature:

    TRAIN:
        - Mann-Whitney U
        - Benjamini-Hochberg FDR
        - Kolmogorov-Smirnov test
        - ROC-AUC
        - orientation-corrected ROC-AUC
        - Average Precision / PR-AUC
        - Cliff's delta
        - mean / median / IQR
        - prevalence / zero fraction

    TEST:
        - same statistics independently
        - predictor orientation is FIXED from TRAIN
        - independent FDR
        - effect-direction replication

Train-derived shortlist:
        - FDR < 0.05
        AND
        (
          |Cliff's delta| >= 0.20
          OR oriented ROC-AUC >= 0.60
        )

For shortlisted features only:
        - stratified percentile bootstrap confidence intervals
        - permutation-based empirical significance
        - independent TEST replication

Representation-level stability:
        - Pearson correlation of TRAIN vs TEST effect sizes
        - Spearman correlation
        - direction-retention rate
        - FDR replication rate
        - practical-effect replication rate

INPUTS
------
feature_engineering/split/train_80.tsv
feature_engineering/split/test_20.tsv

feature_engineering/final_228_features/
    train_80_final_228_features.tsv
    test_20_final_228_features.tsv

feature_engineering/molformer_embeddings/
    train_80_molformer_embeddings.npy
    test_20_molformer_embeddings.npy

OUTPUT
------
feature_engineering/publication_statistics/

    excluded_columns.tsv

    train_molecular_statistics.tsv
    test_molecular_replication.tsv

    train_3d_statistics.tsv
    test_3d_replication.tsv

    train_molformer_statistics.tsv
    test_molformer_replication.tsv

    all_train_statistics.tsv
    all_test_replication.tsv

    replicated_feature_statistics.tsv

    shortlisted_features.tsv
    shortlisted_bootstrap_permutation.tsv

    representation_summary.tsv
    effect_stability_summary.tsv

    publication_statistical_summary.json
    publication_statistical_report.txt

NO MODEL IS TRAINED.
NO FEATURE IS REMOVED.
==============================================================================

"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import json
import math
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    ks_2samp,
    pearsonr,
    spearmanr,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MASTER_TRAIN = "feature_engineering/split/train_80.tsv"
MASTER_TEST = "feature_engineering/split/test_20.tsv"

FINAL_TRAIN = (
    "feature_engineering/final_228_features/"
    "train_80_final_228_features.tsv"
)

FINAL_TEST = (
    "feature_engineering/final_228_features/"
    "test_20_final_228_features.tsv"
)

MOLFORMER_TRAIN = (
    "feature_engineering/molformer_embeddings/"
    "train_80_molformer_embeddings.npy"
)

MOLFORMER_TEST = (
    "feature_engineering/molformer_embeddings/"
    "test_20_molformer_embeddings.npy"
)

OUTPUT_DIR = "feature_engineering/publication_statistics"

ID_COLUMN = "np_id"
LABEL_COLUMN = "Activity_Label"

ACTIVE_LABEL = "Active"
INACTIVE_LABEL = "Inactive"

RANDOM_SEED = 42

FDR_ALPHA = 0.05

# Practical relevance thresholds
MIN_ABS_CLIFF_DELTA = 0.20
MIN_ORIENTED_AUC = 0.60

# Number of strongest FDR-significant features retained per representation
# in addition to all practically meaningful features.
TOP_N_PER_REPRESENTATION = 10

# Publication inference settings.
# Increase to 2000-5000 for the final manuscript if desired.
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000

BOOTSTRAP_CI = 0.95

MIN_CLASS_N = 20


# =============================================================================
# TRUE 3D DESCRIPTORS GENERATED BY SCRIPT 04
# =============================================================================

TRUE_3D_DESCRIPTORS = [
    "PMI1",
    "PMI2",
    "PMI3",
    "NPR1",
    "NPR2",
    "RadiusOfGyration",
    "InertialShapeFactor",
    "Eccentricity",
    "Asphericity",
    "SpherocityIndex",
    "PBF",
]


# =============================================================================
# ALWAYS-EXCLUDED TARGET / METADATA / QC COLUMNS
# =============================================================================

FORBIDDEN_PREDICTORS = {
    "np_id",
    "Activity_Label",
    "Activity_Status",

    "SMILES",
    "Canonical_SMILES",

    "pref_name",
    "iupac_name",

    "chembl_id",
    "pubchem_cid",

    "InChI",
    "InChIKey",

    # Script-04 QC fields
    "3D_Status",
    "Optimization_Method",
    "Best_Conformer_Energy",
    "Conformers_Generated",
    "Metal_Containing",
}


# =============================================================================
# SETUP
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

START_TIME = datetime.now()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def subsection(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def sha256_file(path, block_size=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as handle:

        while True:

            chunk = handle.read(block_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def bh_fdr(pvalues):
    """
    Deterministic Benjamini-Hochberg correction.
    """

    pvalues = np.asarray(pvalues, dtype=float)

    adjusted = np.full(
        len(pvalues),
        np.nan,
        dtype=float
    )

    valid_mask = np.isfinite(pvalues)

    if valid_mask.sum() == 0:
        return adjusted

    valid_p = pvalues[valid_mask]

    order = np.argsort(valid_p)

    ranked = valid_p[order]

    n = len(ranked)

    ranks = np.arange(
        1,
        n + 1,
        dtype=float
    )

    qvalues = ranked * n / ranks

    qvalues = np.minimum.accumulate(
        qvalues[::-1]
    )[::-1]

    qvalues = np.clip(
        qvalues,
        0,
        1
    )

    restored = np.empty_like(qvalues)

    restored[order] = qvalues

    adjusted[valid_mask] = restored

    return adjusted


def cliffs_delta_from_auc(auc):
    """
    For continuous X with Active treated as positive:

        delta = 2*AUC - 1
    """

    if not np.isfinite(auc):
        return np.nan

    return (2.0 * auc) - 1.0


def effect_category(delta):

    if not np.isfinite(delta):
        return "not_testable"

    value = abs(delta)

    if value < 0.147:
        return "negligible"

    elif value < 0.330:
        return "small"

    elif value < 0.474:
        return "medium"

    else:
        return "large"


def safe_auc(y, scores):

    try:

        if len(np.unique(scores)) < 2:
            return np.nan

        return float(
            roc_auc_score(
                y,
                scores
            )
        )

    except Exception:
        return np.nan


def safe_ap(y, scores):

    try:

        if len(np.unique(scores)) < 2:
            return np.nan

        return float(
            average_precision_score(
                y,
                scores
            )
        )

    except Exception:
        return np.nan


def percentile_ci(values, confidence=0.95):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    alpha = 1.0 - confidence

    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)

    return (
        float(np.percentile(values, lower)),
        float(np.percentile(values, upper))
    )


# =============================================================================
# CLASS LABEL CONVERSION
# =============================================================================

def binary_labels(labels):

    labels = np.asarray(labels)

    valid = np.isin(
        labels,
        [
            ACTIVE_LABEL,
            INACTIVE_LABEL
        ]
    )

    if not valid.all():

        bad = np.unique(
            labels[~valid]
        )

        raise RuntimeError(
            f"Unexpected activity labels: {bad}"
        )

    return (
        labels == ACTIVE_LABEL
    ).astype(int)


# =============================================================================
# SINGLE FEATURE STATISTICS
# =============================================================================

def feature_statistics(
    values,
    labels,
    feature,
    representation,
    fixed_direction=None
):

    values = np.asarray(
        values,
        dtype=float
    )

    labels = np.asarray(labels)

    finite = np.isfinite(values)

    x = values[finite]
    lab = labels[finite]

    y = binary_labels(lab)

    active = x[
        lab == ACTIVE_LABEL
    ]

    inactive = x[
        lab == INACTIVE_LABEL
    ]

    result = {

        "representation":
            representation,

        "feature":
            feature,

        "n_total":
            int(len(values)),

        "n_finite":
            int(len(x)),

        "n_nonfinite":
            int((~finite).sum()),

        "n_active":
            int(len(active)),

        "n_inactive":
            int(len(inactive)),

        "active_mean":
            np.nan,

        "inactive_mean":
            np.nan,

        "active_median":
            np.nan,

        "inactive_median":
            np.nan,

        "active_q25":
            np.nan,

        "active_q75":
            np.nan,

        "inactive_q25":
            np.nan,

        "inactive_q75":
            np.nan,

        "median_difference":
            np.nan,

        "zero_fraction":
            np.nan,

        "variance":
            np.nan,

        "raw_auc":
            np.nan,

        "train_orientation":
            np.nan,

        "oriented_auc":
            np.nan,

        "average_precision":
            np.nan,

        "pr_baseline":
            np.nan,

        "ap_over_baseline":
            np.nan,

        "cliffs_delta":
            np.nan,

        "abs_cliffs_delta":
            np.nan,

        "effect_category":
            "not_testable",

        "mannwhitney_u":
            np.nan,

        "mannwhitney_p":
            np.nan,

        "ks_statistic":
            np.nan,

        "ks_p":
            np.nan,

        "effect_direction":
            "not_testable",
    }

    if (
        len(active) < MIN_CLASS_N
        or len(inactive) < MIN_CLASS_N
    ):
        return result

    # -------------------------------------------------------------------------
    # Descriptive statistics
    # -------------------------------------------------------------------------

    result["active_mean"] = float(
        np.mean(active)
    )

    result["inactive_mean"] = float(
        np.mean(inactive)
    )

    result["active_median"] = float(
        np.median(active)
    )

    result["inactive_median"] = float(
        np.median(inactive)
    )

    result["active_q25"] = float(
        np.quantile(active, 0.25)
    )

    result["active_q75"] = float(
        np.quantile(active, 0.75)
    )

    result["inactive_q25"] = float(
        np.quantile(inactive, 0.25)
    )

    result["inactive_q75"] = float(
        np.quantile(inactive, 0.75)
    )

    result["median_difference"] = (
        result["active_median"]
        -
        result["inactive_median"]
    )

    result["zero_fraction"] = float(
        np.mean(x == 0)
    )

    result["variance"] = float(
        np.var(x)
    )

    result["pr_baseline"] = float(
        np.mean(y)
    )

    # -------------------------------------------------------------------------
    # Mann-Whitney U
    # -------------------------------------------------------------------------

    try:

        mw = mannwhitneyu(
            active,
            inactive,
            alternative="two-sided",
            method="asymptotic"
        )

        result["mannwhitney_u"] = float(
            mw.statistic
        )

        result["mannwhitney_p"] = float(
            mw.pvalue
        )

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # KS
    # -------------------------------------------------------------------------

    try:

        ks = ks_2samp(
            active,
            inactive,
            alternative="two-sided",
            method="auto"
        )

        result["ks_statistic"] = float(
            ks.statistic
        )

        result["ks_p"] = float(
            ks.pvalue
        )

    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Raw AUC
    # -------------------------------------------------------------------------

    raw_auc = safe_auc(
        y,
        x
    )

    result["raw_auc"] = raw_auc

    # -------------------------------------------------------------------------
    # Determine direction from TRAIN only.
    # TEST receives fixed_direction from TRAIN.
    # -------------------------------------------------------------------------

    if fixed_direction is None:

        if not np.isfinite(raw_auc):

            direction = 1

        elif raw_auc >= 0.5:

            direction = 1

        else:

            direction = -1

    else:

        direction = int(
            fixed_direction
        )

    result["train_orientation"] = direction

    oriented_scores = (
        direction * x
    )

    oriented_auc = safe_auc(
        y,
        oriented_scores
    )

    ap = safe_ap(
        y,
        oriented_scores
    )

    result["oriented_auc"] = oriented_auc
    result["average_precision"] = ap

    if (
        np.isfinite(ap)
        and result["pr_baseline"] > 0
    ):

        result["ap_over_baseline"] = (
            ap /
            result["pr_baseline"]
        )

    # -------------------------------------------------------------------------
    # Cliff delta uses RAW direction
    # -------------------------------------------------------------------------

    delta = cliffs_delta_from_auc(
        raw_auc
    )

    result["cliffs_delta"] = delta

    if np.isfinite(delta):

        result["abs_cliffs_delta"] = abs(
            delta
        )

        result["effect_category"] = (
            effect_category(delta)
        )

        if delta > 0:
            result["effect_direction"] = (
                "higher_in_active"
            )

        elif delta < 0:
            result["effect_direction"] = (
                "lower_in_active"
            )

        else:
            result["effect_direction"] = (
                "no_direction"
            )

    return result


# =============================================================================
# MATRIX ANALYSIS
# =============================================================================

def analyze_matrix(
    matrix,
    features,
    labels,
    representation,
    fixed_directions=None
):

    rows = []

    total = len(features)

    for index, feature in enumerate(
        features
    ):

        if (
            index == 0
            or (index + 1) % 50 == 0
            or index == total - 1
        ):

            print(
                f"  {representation}: "
                f"{index + 1:,}/{total:,} "
                f"{feature}"
            )

        direction = None

        if fixed_directions is not None:

            direction = fixed_directions.get(
                feature,
                1
            )

        result = feature_statistics(
            matrix[:, index],
            labels,
            feature,
            representation,
            fixed_direction=direction
        )

        rows.append(result)

    result_df = pd.DataFrame(
        rows
    )

    result_df["fdr_q"] = bh_fdr(
        result_df[
            "mannwhitney_p"
        ].to_numpy()
    )

    result_df[
        "fdr_significant"
    ] = (
        result_df["fdr_q"]
        <= FDR_ALPHA
    )

    result_df[
        "practically_meaningful"
    ] = (
        (
            result_df[
                "abs_cliffs_delta"
            ] >= MIN_ABS_CLIFF_DELTA
        )
        |
        (
            result_df[
                "oriented_auc"
            ] >= MIN_ORIENTED_AUC
        )
    )

    result_df[
        "statistically_and_practically_relevant"
    ] = (
        result_df[
            "fdr_significant"
        ]
        &
        result_df[
            "practically_meaningful"
        ]
    )

    return result_df


# =============================================================================
# BOOTSTRAP
# =============================================================================

def bootstrap_feature(
    values,
    labels,
    orientation,
    n_bootstrap,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    values = np.asarray(
        values,
        dtype=float
    )

    labels = np.asarray(labels)

    finite = np.isfinite(values)

    x = values[finite]
    lab = labels[finite]

    active = x[
        lab == ACTIVE_LABEL
    ]

    inactive = x[
        lab == INACTIVE_LABEL
    ]

    if (
        len(active) < MIN_CLASS_N
        or len(inactive) < MIN_CLASS_N
    ):

        return {
            "auc_lower": np.nan,
            "auc_upper": np.nan,
            "delta_lower": np.nan,
            "delta_upper": np.nan,
            "ap_lower": np.nan,
            "ap_upper": np.nan,
        }

    auc_values = []
    delta_values = []
    ap_values = []

    for _ in range(n_bootstrap):

        active_sample = active[
            rng.integers(
                0,
                len(active),
                len(active)
            )
        ]

        inactive_sample = inactive[
            rng.integers(
                0,
                len(inactive),
                len(inactive)
            )
        ]

        x_boot = np.concatenate(
            [
                active_sample,
                inactive_sample
            ]
        )

        y_boot = np.concatenate(
            [
                np.ones(
                    len(active_sample),
                    dtype=int
                ),
                np.zeros(
                    len(inactive_sample),
                    dtype=int
                )
            ]
        )

        raw_auc = safe_auc(
            y_boot,
            x_boot
        )

        oriented_auc = safe_auc(
            y_boot,
            orientation * x_boot
        )

        ap = safe_ap(
            y_boot,
            orientation * x_boot
        )

        if np.isfinite(oriented_auc):
            auc_values.append(
                oriented_auc
            )

        if np.isfinite(raw_auc):
            delta_values.append(
                cliffs_delta_from_auc(
                    raw_auc
                )
            )

        if np.isfinite(ap):
            ap_values.append(ap)

    auc_lower, auc_upper = percentile_ci(
        auc_values,
        BOOTSTRAP_CI
    )

    delta_lower, delta_upper = percentile_ci(
        delta_values,
        BOOTSTRAP_CI
    )

    ap_lower, ap_upper = percentile_ci(
        ap_values,
        BOOTSTRAP_CI
    )

    return {

        "auc_lower":
            auc_lower,

        "auc_upper":
            auc_upper,

        "delta_lower":
            delta_lower,

        "delta_upper":
            delta_upper,

        "ap_lower":
            ap_lower,

        "ap_upper":
            ap_upper,
    }


# =============================================================================
# PERMUTATION TEST
# =============================================================================

def permutation_feature(
    values,
    labels,
    orientation,
    n_permutations,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    values = np.asarray(
        values,
        dtype=float
    )

    labels = np.asarray(labels)

    finite = np.isfinite(values)

    x = values[finite]
    lab = labels[finite]

    y = binary_labels(lab)

    observed = safe_auc(
        y,
        orientation * x
    )

    if not np.isfinite(observed):

        return {
            "observed_oriented_auc":
                np.nan,

            "permutation_p":
                np.nan,

            "null_auc_mean":
                np.nan,

            "null_auc_std":
                np.nan,
        }

    null_aucs = np.empty(
        n_permutations,
        dtype=float
    )

    for i in range(
        n_permutations
    ):

        y_perm = rng.permutation(y)

        null_aucs[i] = safe_auc(
            y_perm,
            orientation * x
        )

    valid = np.isfinite(
        null_aucs
    )

    null_valid = null_aucs[
        valid
    ]

    empirical_p = (
        1
        +
        np.sum(
            null_valid >= observed
        )
    ) / (
        1
        +
        len(null_valid)
    )

    return {

        "observed_oriented_auc":
            float(observed),

        "permutation_p":
            float(empirical_p),

        "null_auc_mean":
            float(
                np.mean(null_valid)
            ),

        "null_auc_std":
            float(
                np.std(null_valid)
            ),
    }


# =============================================================================
# HEADER
# =============================================================================

section(
    "SCRIPT 08 — PUBLICATION-GRADE STATISTICAL VALIDATION"
)

print(
    "Started:",
    START_TIME.isoformat()
)

print(
    "Random seed:",
    RANDOM_SEED
)

print(
    "FDR alpha:",
    FDR_ALPHA
)

print(
    "Bootstrap replicates:",
    N_BOOTSTRAP
)

print(
    "Permutation replicates:",
    N_PERMUTATIONS
)


# =============================================================================
# INPUT FILE CHECK
# =============================================================================

subsection(
    "1. INPUT FILE CHECK"
)

INPUT_FILES = [
    MASTER_TRAIN,
    MASTER_TEST,
    FINAL_TRAIN,
    FINAL_TEST,
    MOLFORMER_TRAIN,
    MOLFORMER_TEST,
]

input_hashes = {}

for path in INPUT_FILES:

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Required input missing:\n{path}"
        )

    print(
        "FOUND:",
        path
    )

    input_hashes[path] = sha256_file(
        path
    )


# =============================================================================
# LOAD MASTER DATA
# =============================================================================

subsection(
    "2. LOADING MASTER TRAIN / TEST"
)

master_train = pd.read_csv(
    MASTER_TRAIN,
    sep="\t",
    dtype={
        ID_COLUMN: str
    },
    low_memory=False
)

master_test = pd.read_csv(
    MASTER_TEST,
    sep="\t",
    dtype={
        ID_COLUMN: str
    },
    low_memory=False
)

print(
    "TRAIN:",
    master_train.shape
)

print(
    "TEST :",
    master_test.shape
)

for name, df in [
    ("TRAIN", master_train),
    ("TEST", master_test),
]:

    for required in [
        ID_COLUMN,
        LABEL_COLUMN
    ]:

        if required not in df.columns:

            raise RuntimeError(
                f"{required} missing from {name}"
            )

    if df[ID_COLUMN].duplicated().any():

        raise RuntimeError(
            f"Duplicate IDs in {name}"
        )

overlap = set(
    master_train[ID_COLUMN]
) & set(
    master_test[ID_COLUMN]
)

if overlap:

    raise RuntimeError(
        f"TRAIN/TEST ID overlap: "
        f"{len(overlap)}"
    )

train_labels = master_train[
    LABEL_COLUMN
].to_numpy()

test_labels = master_test[
    LABEL_COLUMN
].to_numpy()

print()
print(
    "TRAIN class distribution:"
)

print(
    master_train[
        LABEL_COLUMN
    ].value_counts()
)

print()
print(
    "TEST class distribution:"
)

print(
    master_test[
        LABEL_COLUMN
    ].value_counts()
)


# =============================================================================
# LOAD FINAL HANDCRAFTED FEATURES
# =============================================================================

subsection(
    "3. LOADING FINAL 182-FEATURE HANDCRAFTED MATRICES"
)

final_train = pd.read_csv(
    FINAL_TRAIN,
    sep="\t",
    dtype={
        ID_COLUMN: str
    },
    low_memory=False
)

final_test = pd.read_csv(
    FINAL_TEST,
    sep="\t",
    dtype={
        ID_COLUMN: str
    },
    low_memory=False
)

print(
    "Final TRAIN:",
    final_train.shape
)

print(
    "Final TEST :",
    final_test.shape
)


# =============================================================================
# ALIGN FINAL FEATURES TO MASTER
# =============================================================================

def align_final_features(
    final_df,
    master_df,
    name
):

    if ID_COLUMN not in final_df.columns:

        raise RuntimeError(
            f"{ID_COLUMN} missing from {name}"
        )

    if LABEL_COLUMN not in final_df.columns:

        raise RuntimeError(
            f"{LABEL_COLUMN} missing from {name}"
        )

    if final_df[
        ID_COLUMN
    ].duplicated().any():

        raise RuntimeError(
            f"Duplicate IDs in {name}"
        )

    aligned = (
        master_df[
            [
                ID_COLUMN,
                LABEL_COLUMN
            ]
        ]
        .merge(
            final_df,
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
            suffixes=(
                "_master",
                ""
            )
        )
    )

    if len(aligned) != len(
        master_df
    ):

        raise RuntimeError(
            f"Row alignment failed: {name}"
        )

    # Confirm target consistency
    if not np.array_equal(
        aligned[
            f"{LABEL_COLUMN}_master"
        ].astype(str).values,
        aligned[
            LABEL_COLUMN
        ].astype(str).values
    ):

        raise RuntimeError(
            f"Activity labels differ: {name}"
        )

    aligned = aligned.drop(
        columns=[
            f"{LABEL_COLUMN}_master"
        ]
    )

    return aligned


final_train = align_final_features(
    final_train,
    master_train,
    "FINAL TRAIN"
)

final_test = align_final_features(
    final_test,
    master_test,
    "FINAL TEST"
)

print(
    "Aligned final TRAIN:",
    final_train.shape
)

print(
    "Aligned final TEST :",
    final_test.shape
)


# =============================================================================
# IDENTIFY TRUE PREDICTORS
# =============================================================================

subsection(
    "4. PREDICTOR INVENTORY / LEAKAGE EXCLUSION"
)

excluded_records = []

candidate_features = []

for column in final_train.columns:

    if column in {
        ID_COLUMN,
        LABEL_COLUMN
    }:

        excluded_records.append({
            "column": column,
            "reason": "identifier_or_target"
        })

        continue

    if column in FORBIDDEN_PREDICTORS:

        excluded_records.append({
            "column": column,
            "reason": "forbidden_metadata_or_target_derived"
        })

        continue

    if not pd.api.types.is_numeric_dtype(
        final_train[column]
    ):

        excluded_records.append({
            "column": column,
            "reason": "non_numeric"
        })

        continue

    candidate_features.append(
        column
    )


# =============================================================================
# HARD LEAKAGE CHECK
# =============================================================================

target_binary = binary_labels(
    train_labels
)

safe_candidates = []

for feature in candidate_features:

    values = pd.to_numeric(
        final_train[feature],
        errors="coerce"
    ).to_numpy()

    finite = np.isfinite(values)

    if finite.sum() == 0:

        excluded_records.append({
            "column": feature,
            "reason": "no_finite_values"
        })

        continue

    x = values[finite]
    y = target_binary[finite]

    # Direct equality check
    if (
        len(x) == len(y)
        and np.array_equal(x, y)
    ):

        excluded_records.append({
            "column": feature,
            "reason": "exact_target_leakage"
        })

        continue

    # Perfect inverse target
    if (
        len(x) == len(y)
        and np.array_equal(
            x,
            1 - y
        )
    ):

        excluded_records.append({
            "column": feature,
            "reason": "exact_inverse_target_leakage"
        })

        continue

    safe_candidates.append(
        feature
    )


# =============================================================================
# SPLIT MOLECULAR VS 3D
# =============================================================================

three_d_features = [
    feature
    for feature in TRUE_3D_DESCRIPTORS
    if feature in safe_candidates
]

molecular_features = [
    feature
    for feature in safe_candidates
    if feature not in three_d_features
]

print(
    "Final candidate predictors:",
    len(safe_candidates)
)

print(
    "Molecular descriptors:",
    len(molecular_features)
)

print(
    "Retained 3D descriptors:",
    len(three_d_features)
)

print()
print(
    "Retained 3D feature names:"
)

for feature in three_d_features:
    print(" ", feature)


excluded_df = pd.DataFrame(
    excluded_records
)

excluded_path = os.path.join(
    OUTPUT_DIR,
    "excluded_columns.tsv"
)

excluded_df.to_csv(
    excluded_path,
    sep="\t",
    index=False
)


# =============================================================================
# CREATE HANDCRAFTED MATRICES
# =============================================================================

molecular_train_matrix = (
    final_train[
        molecular_features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)

molecular_test_matrix = (
    final_test[
        molecular_features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)

three_d_train_matrix = (
    final_train[
        three_d_features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)

three_d_test_matrix = (
    final_test[
        three_d_features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


# =============================================================================
# LOAD MOLFORMER
# =============================================================================

subsection(
    "5. LOADING MOLFORMER REPRESENTATIONS"
)

molformer_train = np.load(
    MOLFORMER_TRAIN,
    mmap_mode="r"
)

molformer_test = np.load(
    MOLFORMER_TEST,
    mmap_mode="r"
)

print(
    "MoLFormer TRAIN:",
    molformer_train.shape
)

print(
    "MoLFormer TEST :",
    molformer_test.shape
)

if molformer_train.shape[0] != len(
    master_train
):

    raise RuntimeError(
        "MoLFormer TRAIN row mismatch"
    )

if molformer_test.shape[0] != len(
    master_test
):

    raise RuntimeError(
        "MoLFormer TEST row mismatch"
    )

molformer_features = [
    f"molformer_{i:04d}"
    for i in range(
        1,
        molformer_train.shape[1] + 1
    )
]


# =============================================================================
# TRAIN STATISTICS
# =============================================================================

section(
    "6. TRAIN-ONLY FEATURE SCREENING"
)

molecular_train_stats = analyze_matrix(
    molecular_train_matrix,
    molecular_features,
    train_labels,
    "molecular"
)

three_d_train_stats = analyze_matrix(
    three_d_train_matrix,
    three_d_features,
    train_labels,
    "3d"
)

molformer_train_stats = analyze_matrix(
    molformer_train,
    molformer_features,
    train_labels,
    "molformer"
)


# =============================================================================
# TRAIN-DERIVED ORIENTATION
# =============================================================================

def orientation_dict(df):

    return {
        row["feature"]:
            int(row["train_orientation"])
        for _, row in df.iterrows()
    }


molecular_directions = orientation_dict(
    molecular_train_stats
)

three_d_directions = orientation_dict(
    three_d_train_stats
)

molformer_directions = orientation_dict(
    molformer_train_stats
)


# =============================================================================
# TEST REPLICATION
# =============================================================================

section(
    "7. INDEPENDENT TEST REPLICATION"
)

molecular_test_stats = analyze_matrix(
    molecular_test_matrix,
    molecular_features,
    test_labels,
    "molecular",
    fixed_directions=molecular_directions
)

three_d_test_stats = analyze_matrix(
    three_d_test_matrix,
    three_d_features,
    test_labels,
    "3d",
    fixed_directions=three_d_directions
)

molformer_test_stats = analyze_matrix(
    molformer_test,
    molformer_features,
    test_labels,
    "molformer",
    fixed_directions=molformer_directions
)


# =============================================================================
# SAVE FAMILY STATISTICS
# =============================================================================

def save_table(df, filename):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    df.to_csv(
        path,
        sep="\t",
        index=False
    )

    print(
        "Written:",
        path
    )

    return path


subsection(
    "8. WRITING FULL STATISTICAL TABLES"
)

save_table(
    molecular_train_stats,
    "train_molecular_statistics.tsv"
)

save_table(
    molecular_test_stats,
    "test_molecular_replication.tsv"
)

save_table(
    three_d_train_stats,
    "train_3d_statistics.tsv"
)

save_table(
    three_d_test_stats,
    "test_3d_replication.tsv"
)

save_table(
    molformer_train_stats,
    "train_molformer_statistics.tsv"
)

save_table(
    molformer_test_stats,
    "test_molformer_replication.tsv"
)


# =============================================================================
# COMBINE TRAIN / TEST TABLES
# =============================================================================

all_train = pd.concat(
    [
        molecular_train_stats,
        three_d_train_stats,
        molformer_train_stats
    ],
    ignore_index=True
)

all_test = pd.concat(
    [
        molecular_test_stats,
        three_d_test_stats,
        molformer_test_stats
    ],
    ignore_index=True
)

save_table(
    all_train,
    "all_train_statistics.tsv"
)

save_table(
    all_test,
    "all_test_replication.tsv"
)


# =============================================================================
# MERGE TRAIN / TEST RESULTS
# =============================================================================

subsection(
    "9. TRAIN / TEST EFFECT REPLICATION"
)

replicated = all_train.merge(
    all_test,
    on=[
        "representation",
        "feature"
    ],
    suffixes=(
        "_train",
        "_test"
    ),
    validate="one_to_one"
)

replicated[
    "same_effect_direction"
] = (
    np.sign(
        replicated[
            "cliffs_delta_train"
        ]
    )
    ==
    np.sign(
        replicated[
            "cliffs_delta_test"
        ]
    )
)

replicated[
    "replicated_fdr"
] = (
    replicated[
        "fdr_significant_train"
    ]
    &
    replicated[
        "fdr_significant_test"
    ]
)

replicated[
    "replicated_practical_effect"
] = (
    replicated[
        "practically_meaningful_train"
    ]
    &
    replicated[
        "practically_meaningful_test"
    ]
)

replicated[
    "fully_replicated"
] = (
    replicated[
        "replicated_fdr"
    ]
    &
    replicated[
        "replicated_practical_effect"
    ]
    &
    replicated[
        "same_effect_direction"
    ]
)

replicated[
    "abs_delta_difference"
] = abs(
    replicated[
        "abs_cliffs_delta_train"
    ]
    -
    replicated[
        "abs_cliffs_delta_test"
    ]
)

save_table(
    replicated,
    "replicated_feature_statistics.tsv"
)


# =============================================================================
# TRAIN-ONLY SHORTLIST
# =============================================================================

section(
    "10. TRAIN-DERIVED SHORTLIST"
)

shortlisted_frames = []

for representation in [
    "molecular",
    "3d",
    "molformer"
]:

    subset = all_train[
        all_train[
            "representation"
        ] == representation
    ].copy()

    meaningful = subset[
        subset[
            "statistically_and_practically_relevant"
        ]
    ]

    significant_ranked = (
        subset[
            subset[
                "fdr_significant"
            ]
        ]
        .sort_values(
            [
                "abs_cliffs_delta",
                "oriented_auc"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(
            TOP_N_PER_REPRESENTATION
        )
    )

    combined = pd.concat(
        [
            meaningful,
            significant_ranked
        ]
    ).drop_duplicates(
        subset=[
            "feature"
        ]
    )

    shortlisted_frames.append(
        combined
    )

shortlist = pd.concat(
    shortlisted_frames,
    ignore_index=True
)

shortlist = shortlist.sort_values(
    [
        "representation",
        "abs_cliffs_delta"
    ],
    ascending=[
        True,
        False
    ]
)

print(
    "TRAIN-derived shortlist:",
    len(shortlist)
)

print()

print(
    shortlist[
        [
            "representation",
            "feature",
            "oriented_auc",
            "average_precision",
            "cliffs_delta",
            "fdr_q"
        ]
    ].to_string(
        index=False
    )
)

save_table(
    shortlist,
    "shortlisted_features.tsv"
)


# =============================================================================
# FEATURE LOOKUP
# =============================================================================

matrix_lookup = {

    "molecular": {
        "train":
            molecular_train_matrix,

        "test":
            molecular_test_matrix,

        "features":
            molecular_features,
    },

    "3d": {
        "train":
            three_d_train_matrix,

        "test":
            three_d_test_matrix,

        "features":
            three_d_features,
    },

    "molformer": {
        "train":
            molformer_train,

        "test":
            molformer_test,

        "features":
            molformer_features,
    }
}


# =============================================================================
# BOOTSTRAP + PERMUTATION SHORTLIST
# =============================================================================

section(
    "11. BOOTSTRAP CONFIDENCE INTERVALS + PERMUTATION VALIDATION"
)

bootstrap_rows = []

for row_number, row in shortlist.reset_index(
    drop=True
).iterrows():

    representation = row[
        "representation"
    ]

    feature = row[
        "feature"
    ]

    orientation = int(
        row[
            "train_orientation"
        ]
    )

    family = matrix_lookup[
        representation
    ]

    feature_index = family[
        "features"
    ].index(feature)

    train_values = (
        family[
            "train"
        ][:, feature_index]
    )

    test_values = (
        family[
            "test"
        ][:, feature_index]
    )

    print()
    print(
        f"[{row_number + 1}/{len(shortlist)}] "
        f"{representation} :: {feature}"
    )

    train_boot = bootstrap_feature(
        train_values,
        train_labels,
        orientation,
        N_BOOTSTRAP,
        RANDOM_SEED
        +
        (row_number * 10)
        +
        1
    )

    test_boot = bootstrap_feature(
        test_values,
        test_labels,
        orientation,
        N_BOOTSTRAP,
        RANDOM_SEED
        +
        (row_number * 10)
        +
        2
    )

    permutation = permutation_feature(
        train_values,
        train_labels,
        orientation,
        N_PERMUTATIONS,
        RANDOM_SEED
        +
        (row_number * 10)
        +
        3
    )

    test_row = replicated[
        (
            replicated[
                "representation"
            ] == representation
        )
        &
        (
            replicated[
                "feature"
            ] == feature
        )
    ].iloc[0]

    bootstrap_rows.append({

        "representation":
            representation,

        "feature":
            feature,

        "train_orientation":
            orientation,

        # TRAIN observed
        "train_oriented_auc":
            row[
                "oriented_auc"
            ],

        "train_ap":
            row[
                "average_precision"
            ],

        "train_cliffs_delta":
            row[
                "cliffs_delta"
            ],

        "train_fdr_q":
            row[
                "fdr_q"
            ],

        # TRAIN CI
        "train_auc_ci_lower":
            train_boot[
                "auc_lower"
            ],

        "train_auc_ci_upper":
            train_boot[
                "auc_upper"
            ],

        "train_delta_ci_lower":
            train_boot[
                "delta_lower"
            ],

        "train_delta_ci_upper":
            train_boot[
                "delta_upper"
            ],

        "train_ap_ci_lower":
            train_boot[
                "ap_lower"
            ],

        "train_ap_ci_upper":
            train_boot[
                "ap_upper"
            ],

        # Permutation
        "permutation_p":
            permutation[
                "permutation_p"
            ],

        "permutation_null_auc_mean":
            permutation[
                "null_auc_mean"
            ],

        "permutation_null_auc_sd":
            permutation[
                "null_auc_std"
            ],

        # TEST observed
        "test_oriented_auc":
            test_row[
                "oriented_auc_test"
            ],

        "test_ap":
            test_row[
                "average_precision_test"
            ],

        "test_cliffs_delta":
            test_row[
                "cliffs_delta_test"
            ],

        "test_fdr_q":
            test_row[
                "fdr_q_test"
            ],

        # TEST CI
        "test_auc_ci_lower":
            test_boot[
                "auc_lower"
            ],

        "test_auc_ci_upper":
            test_boot[
                "auc_upper"
            ],

        "test_delta_ci_lower":
            test_boot[
                "delta_lower"
            ],

        "test_delta_ci_upper":
            test_boot[
                "delta_upper"
            ],

        "test_ap_ci_lower":
            test_boot[
                "ap_lower"
            ],

        "test_ap_ci_upper":
            test_boot[
                "ap_upper"
            ],

        # replication
        "same_effect_direction":
            bool(
                test_row[
                    "same_effect_direction"
                ]
            ),

        "replicated_fdr":
            bool(
                test_row[
                    "replicated_fdr"
                ]
            ),

        "replicated_practical_effect":
            bool(
                test_row[
                    "replicated_practical_effect"
                ]
            ),

        "fully_replicated":
            bool(
                test_row[
                    "fully_replicated"
                ]
            ),
    })


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_df[
    "permutation_fdr_q"
] = bh_fdr(
    bootstrap_df[
        "permutation_p"
    ].to_numpy()
)

save_table(
    bootstrap_df,
    "shortlisted_bootstrap_permutation.tsv"
)


# =============================================================================
# EFFECT STABILITY
# =============================================================================

section(
    "12. REPRESENTATION EFFECT-STABILITY ANALYSIS"
)

stability_rows = []

for representation in [
    "molecular",
    "3d",
    "molformer"
]:

    subset = replicated[
        replicated[
            "representation"
        ] == representation
    ].copy()

    valid = subset[
        np.isfinite(
            subset[
                "cliffs_delta_train"
            ]
        )
        &
        np.isfinite(
            subset[
                "cliffs_delta_test"
            ]
        )
    ]

    if len(valid) >= 3:

        pearson_r, pearson_p = pearsonr(
            valid[
                "cliffs_delta_train"
            ],
            valid[
                "cliffs_delta_test"
            ]
        )

        spearman_r, spearman_p = spearmanr(
            valid[
                "cliffs_delta_train"
            ],
            valid[
                "cliffs_delta_test"
            ]
        )

    else:

        pearson_r = np.nan
        pearson_p = np.nan

        spearman_r = np.nan
        spearman_p = np.nan

    stability_rows.append({

        "representation":
            representation,

        "features":
            int(len(subset)),

        "testable_effect_pairs":
            int(len(valid)),

        "train_fdr_significant":
            int(
                subset[
                    "fdr_significant_train"
                ].sum()
            ),

        "test_fdr_significant":
            int(
                subset[
                    "fdr_significant_test"
                ].sum()
            ),

        "fdr_replicated":
            int(
                subset[
                    "replicated_fdr"
                ].sum()
            ),

        "practical_train":
            int(
                subset[
                    "practically_meaningful_train"
                ].sum()
            ),

        "practical_test":
            int(
                subset[
                    "practically_meaningful_test"
                ].sum()
            ),

        "practical_replicated":
            int(
                subset[
                    "replicated_practical_effect"
                ].sum()
            ),

        "fully_replicated":
            int(
                subset[
                    "fully_replicated"
                ].sum()
            ),

        "same_direction_fraction":
            float(
                subset[
                    "same_effect_direction"
                ].mean()
            ),

        "median_abs_delta_train":
            float(
                subset[
                    "abs_cliffs_delta_train"
                ].median()
            ),

        "median_abs_delta_test":
            float(
                subset[
                    "abs_cliffs_delta_test"
                ].median()
            ),

        "median_abs_train_test_delta_difference":
            float(
                subset[
                    "abs_delta_difference"
                ].median()
            ),

        "pearson_effect_r":
            float(pearson_r),

        "pearson_effect_p":
            float(pearson_p),

        "spearman_effect_rho":
            float(spearman_r),

        "spearman_effect_p":
            float(spearman_p),
    })


stability_df = pd.DataFrame(
    stability_rows
)

save_table(
    stability_df,
    "effect_stability_summary.tsv"
)


# =============================================================================
# REPRESENTATION SUMMARY
# =============================================================================

section(
    "13. REPRESENTATION-LEVEL SUMMARY"
)

representation_rows = []

for representation in [
    "molecular",
    "3d",
    "molformer"
]:

    train_sub = all_train[
        all_train[
            "representation"
        ] == representation
    ]

    test_sub = all_test[
        all_test[
            "representation"
        ] == representation
    ]

    rep_sub = replicated[
        replicated[
            "representation"
        ] == representation
    ]

    valid_train = train_sub[
        np.isfinite(
            train_sub[
                "oriented_auc"
            ]
        )
    ]

    valid_test = test_sub[
        np.isfinite(
            test_sub[
                "oriented_auc"
            ]
        )
    ]

    best_train = (
        valid_train.sort_values(
            "abs_cliffs_delta",
            ascending=False
        ).iloc[0]
        if len(valid_train)
        else None
    )

    representation_rows.append({

        "representation":
            representation,

        "features":
            int(len(train_sub)),

        "testable_train":
            int(len(valid_train)),

        "train_fdr_significant":
            int(
                train_sub[
                    "fdr_significant"
                ].sum()
            ),

        "train_practically_meaningful":
            int(
                train_sub[
                    "practically_meaningful"
                ].sum()
            ),

        "train_statistically_and_practically_relevant":
            int(
                train_sub[
                    "statistically_and_practically_relevant"
                ].sum()
            ),

        "test_fdr_significant":
            int(
                test_sub[
                    "fdr_significant"
                ].sum()
            ),

        "test_practically_meaningful":
            int(
                test_sub[
                    "practically_meaningful"
                ].sum()
            ),

        "fully_replicated":
            int(
                rep_sub[
                    "fully_replicated"
                ].sum()
            ),

        "median_train_oriented_auc":
            float(
                valid_train[
                    "oriented_auc"
                ].median()
            )
            if len(valid_train)
            else np.nan,

        "median_test_oriented_auc":
            float(
                valid_test[
                    "oriented_auc"
                ].median()
            )
            if len(valid_test)
            else np.nan,

        "median_train_ap":
            float(
                valid_train[
                    "average_precision"
                ].median()
            )
            if len(valid_train)
            else np.nan,

        "median_test_ap":
            float(
                valid_test[
                    "average_precision"
                ].median()
            )
            if len(valid_test)
            else np.nan,

        "max_train_abs_cliffs_delta":
            float(
                valid_train[
                    "abs_cliffs_delta"
                ].max()
            )
            if len(valid_train)
            else np.nan,

        "max_test_abs_cliffs_delta":
            float(
                valid_test[
                    "abs_cliffs_delta"
                ].max()
            )
            if len(valid_test)
            else np.nan,

        "best_train_feature":
            (
                str(
                    best_train[
                        "feature"
                    ]
                )
                if best_train is not None
                else None
            ),

        "best_train_feature_auc":
            (
                float(
                    best_train[
                        "oriented_auc"
                    ]
                )
                if best_train is not None
                else np.nan
            ),

        "best_train_feature_ap":
            (
                float(
                    best_train[
                        "average_precision"
                    ]
                )
                if best_train is not None
                else np.nan
            ),

        "best_train_feature_delta":
            (
                float(
                    best_train[
                        "cliffs_delta"
                    ]
                )
                if best_train is not None
                else np.nan
            ),
    })


representation_df = pd.DataFrame(
    representation_rows
)

save_table(
    representation_df,
    "representation_summary.tsv"
)

print()
print(
    representation_df.to_string(
        index=False
    )
)


# =============================================================================
# MACHINE-READABLE JSON REPORT
# =============================================================================

section(
    "14. MACHINE-READABLE REPORT"
)

END_TIME = datetime.now()

report = {

    "script":
        "08_publication_statistical_validation_v1.py",

    "status":
        "PASS",

    "started":
        START_TIME.isoformat(),

    "finished":
        END_TIME.isoformat(),

    "methodological_policy": {

        "train_drives_screening":
            True,

        "test_used_for_feature_selection":
            False,

        "test_orientation_fixed_from_train":
            True,

        "models_trained":
            False,

        "features_removed":
            False,

        "target_leakage_exclusion":
            True,
    },

    "randomness": {

        "seed":
            RANDOM_SEED,

        "bootstrap_replicates":
            N_BOOTSTRAP,

        "permutation_replicates":
            N_PERMUTATIONS,

        "bootstrap_confidence":
            BOOTSTRAP_CI,
    },

    "statistical_framework": {

        "group_test":
            "Mann-Whitney U, two-sided",

        "multiple_testing":
            "Benjamini-Hochberg FDR",

        "fdr_alpha":
            FDR_ALPHA,

        "distribution_test":
            "Kolmogorov-Smirnov",

        "effect_size":
            "Cliff's delta derived from raw ROC-AUC",

        "discrimination":
            "ROC-AUC",

        "imbalance_sensitive_metric":
            "Average Precision / PR-AUC",

        "train_shortlist_condition": (
            f"FDR <= {FDR_ALPHA} AND "
            f"(|delta| >= {MIN_ABS_CLIFF_DELTA} "
            f"OR oriented AUC >= {MIN_ORIENTED_AUC})"
        ),

        "confidence_intervals":
            "stratified percentile bootstrap",

        "empirical_validation":
            "label permutation on TRAIN shortlist",
    },

    "class_distribution": {

        "train": {

            "active":
                int(
                    (
                        train_labels
                        ==
                        ACTIVE_LABEL
                    ).sum()
                ),

            "inactive":
                int(
                    (
                        train_labels
                        ==
                        INACTIVE_LABEL
                    ).sum()
                ),

            "active_fraction":
                float(
                    (
                        train_labels
                        ==
                        ACTIVE_LABEL
                    ).mean()
                ),
        },

        "test": {

            "active":
                int(
                    (
                        test_labels
                        ==
                        ACTIVE_LABEL
                    ).sum()
                ),

            "inactive":
                int(
                    (
                        test_labels
                        ==
                        INACTIVE_LABEL
                    ).sum()
                ),

            "active_fraction":
                float(
                    (
                        test_labels
                        ==
                        ACTIVE_LABEL
                    ).mean()
                ),
        },
    },

    "predictor_inventory": {

        "final_handcrafted_total":
            int(
                len(
                    safe_candidates
                )
            ),

        "molecular":
            int(
                len(
                    molecular_features
                )
            ),

        "3d":
            int(
                len(
                    three_d_features
                )
            ),

        "molformer":
            int(
                len(
                    molformer_features
                )
            ),

        "excluded_columns":
            int(
                len(
                    excluded_df
                )
            ),
    },

    "input_sha256":
        input_hashes,

    "outputs": {

        "excluded_columns":
            "excluded_columns.tsv",

        "all_train":
            "all_train_statistics.tsv",

        "all_test":
            "all_test_replication.tsv",

        "replicated":
            "replicated_feature_statistics.tsv",

        "shortlist":
            "shortlisted_features.tsv",

        "bootstrap_permutation":
            "shortlisted_bootstrap_permutation.tsv",

        "representation_summary":
            "representation_summary.tsv",

        "effect_stability":
            "effect_stability_summary.tsv",
    }
}


json_path = os.path.join(
    OUTPUT_DIR,
    "publication_statistical_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        report,
        handle,
        indent=2,
        default=str
    )

print(
    "Written:",
    json_path
)


# =============================================================================
# HUMAN-READABLE REPORT
# =============================================================================

section(
    "15. HUMAN-READABLE PUBLICATION REPORT"
)

text_path = os.path.join(
    OUTPUT_DIR,
    "publication_statistical_report.txt"
)

with open(
    text_path,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "=" * 78 + "\n"
    )

    handle.write(
        "SCRIPT 08 — PUBLICATION-GRADE STATISTICAL VALIDATION REPORT\n"
    )

    handle.write(
        "=" * 78 + "\n\n"
    )

    handle.write(
        f"Generated: {END_TIME.isoformat()}\n\n"
    )

    handle.write(
        "ANALYSIS PRINCIPLES\n"
    )

    handle.write(
        "-" * 78 + "\n"
    )

    handle.write(
        "TRAIN used for screening and direction determination: YES\n"
    )

    handle.write(
        "TEST used for feature selection: NO\n"
    )

    handle.write(
        "TEST used for independent replication: YES\n"
    )

    handle.write(
        "Target-derived metadata excluded: YES\n"
    )

    handle.write(
        "Features removed by this script: NO\n"
    )

    handle.write(
        "Predictive model trained: NO\n\n"
    )

    handle.write(
        "CLASS DISTRIBUTION\n"
    )

    handle.write(
        "-" * 78 + "\n"
    )

    handle.write(
        f"TRAIN Active: "
        f"{(train_labels == ACTIVE_LABEL).sum():,}\n"
    )

    handle.write(
        f"TRAIN Inactive: "
        f"{(train_labels == INACTIVE_LABEL).sum():,}\n"
    )

    handle.write(
        f"TRAIN Active fraction: "
        f"{(train_labels == ACTIVE_LABEL).mean():.6f}\n\n"
    )

    handle.write(
        f"TEST Active: "
        f"{(test_labels == ACTIVE_LABEL).sum():,}\n"
    )

    handle.write(
        f"TEST Inactive: "
        f"{(test_labels == INACTIVE_LABEL).sum():,}\n"
    )

    handle.write(
        f"TEST Active fraction: "
        f"{(test_labels == ACTIVE_LABEL).mean():.6f}\n\n"
    )

    handle.write(
        "PREDICTOR INVENTORY\n"
    )

    handle.write(
        "-" * 78 + "\n"
    )

    handle.write(
        f"Final handcrafted predictors: "
        f"{len(safe_candidates)}\n"
    )

    handle.write(
        f"Molecular descriptors: "
        f"{len(molecular_features)}\n"
    )

    handle.write(
        f"3D descriptors: "
        f"{len(three_d_features)}\n"
    )

    handle.write(
        f"MoLFormer dimensions: "
        f"{len(molformer_features)}\n\n"
    )

    handle.write(
        "STATISTICAL FRAMEWORK\n"
    )

    handle.write(
        "-" * 78 + "\n"
    )

    handle.write(
        "Primary class comparison: Mann-Whitney U\n"
    )

    handle.write(
        "Multiple testing: Benjamini-Hochberg FDR\n"
    )

    handle.write(
        f"FDR alpha: {FDR_ALPHA}\n"
    )

    handle.write(
        "Effect size: Cliff's delta\n"
    )

    handle.write(
        "Discrimination: ROC-AUC\n"
    )

    handle.write(
        "Class-imbalance-sensitive metric: Average Precision / PR-AUC\n"
    )

    handle.write(
        "Distributional difference: Kolmogorov-Smirnov\n"
    )

    handle.write(
        f"Bootstrap replicates: {N_BOOTSTRAP}\n"
    )

    handle.write(
        f"Permutation replicates: {N_PERMUTATIONS}\n\n"
    )

    handle.write(
        "REPRESENTATION SUMMARY\n"
    )

    handle.write(
        "-" * 78 + "\n"
    )

    handle.write(
        representation_df.to_string(
            index=False
        )
    )

    handle.write(
        "\n\n"
    )

    handle.write(
        "EFFECT STABILITY\n"
    )

    handle.write(
        "-" * 78 + "\n"
    )

    handle.write(
        stability_df.to_string(
            index=False
        )
    )

    handle.write(
        "\n\n"
    )

    handle.write(
        "TRAIN-DERIVED SHORTLIST\n"
    )

    handle.write(
        "-" * 78 + "\n"
    )

    handle.write(
        shortlist[
            [
                "representation",
                "feature",
                "oriented_auc",
                "average_precision",
                "cliffs_delta",
                "fdr_q"
            ]
        ].to_string(
            index=False
        )
    )

    handle.write(
        "\n\n"
    )

    handle.write(
        "INTERPRETATION RULE\n"
    )

    handle.write(
        "-" * 78 + "\n"
    )

    handle.write(
        "Statistical significance alone is not considered sufficient evidence\n"
    )

    handle.write(
        "of predictive relevance because the training sample contains >48,000\n"
    )

    handle.write(
        "molecules. Practical effect size, train/test directional stability,\n"
    )

    handle.write(
        "bootstrap uncertainty, permutation validation, and later multivariate\n"
    )

    handle.write(
        "predictive performance must be considered jointly.\n\n"
    )

    handle.write(
        "No features were removed and no predictive model was trained.\n"
    )


print(
    "Written:",
    text_path
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section(
    "SCRIPT 08 COMPLETE"
)

print(
    "FINAL STATUS: PASS"
)

print()

print(
    "Final handcrafted predictors:",
    len(safe_candidates)
)

print(
    "Molecular descriptors:",
    len(molecular_features)
)

print(
    "3D descriptors:",
    len(three_d_features)
)

print(
    "MoLFormer dimensions:",
    len(molformer_features)
)

print(
    "TRAIN-derived shortlisted features:",
    len(shortlist)
)

print()

print(
    "No feature removed."
)

print(
    "No predictive model trained."
)

print(
    "TEST used only for independent replication."
)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Finished:",
    END_TIME.isoformat()
)

print(
    "=" * 78
)
