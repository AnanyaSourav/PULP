#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 07 — STATISTICAL FEATURE ANALYSIS v1
==============================================================================

Purpose
-------
Statistically characterize:

1. Molecular descriptors
2. 3D-QSAR descriptors
3. MoLFormer embeddings

between Active and Inactive molecules.

IMPORTANT
---------
- TRAIN is used for statistical feature screening.
- TEST is NEVER used for feature selection.
- TEST is used only for independent replication/validation.
- No features are removed by this script.
- No model is trained by this script.
- All procedures are deterministic.
- Multiple-testing correction uses Benjamini-Hochberg FDR.

Primary statistical tests
-------------------------
1. Mann-Whitney U test
2. ROC-AUC
3. Kolmogorov-Smirnov statistic
4. Cliff's delta
5. Benjamini-Hochberg adjusted p-value

Additional quantities
---------------------
- Active median
- Inactive median
- Active mean
- Inactive mean
- Median difference
- IQR
- missingness
- zero fraction
- effect direction

Outputs
-------
feature_engineering/statistical_analysis/

    train_molecular_statistics.tsv
    train_3d_statistics.tsv
    train_molformer_statistics.tsv

    test_molecular_replication.tsv
    test_3d_replication.tsv
    test_molformer_replication.tsv

    statistical_feature_summary.json
    statistical_feature_analysis_report.txt

No features are removed.
==============================================================================


==============================================================================
"""

import os
import sys
import json
import hashlib
import platform
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.metrics import roc_auc_score


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_SPLIT = "feature_engineering/split/train_80.tsv"
TEST_SPLIT  = "feature_engineering/split/test_20.tsv"

MOLECULAR_TRAIN = (
    "feature_engineering/molecular_descriptors/"
    "train_80_molecular_descriptors.tsv"
)

MOLECULAR_TEST = (
    "feature_engineering/molecular_descriptors/"
    "test_20_molecular_descriptors.tsv"
)

THREE_D_TRAIN = (
    "feature_engineering/3d_qsar_descriptors/"
    "train_80_3d_qsar_descriptors.tsv"
)

THREE_D_TEST = (
    "feature_engineering/3d_qsar_descriptors/"
    "test_20_3d_qsar_descriptors.tsv"
)

MOLFORMER_TRAIN = (
    "feature_engineering/molformer_embeddings/"
    "train_80_molformer_embeddings.npy"
)

MOLFORMER_TEST = (
    "feature_engineering/molformer_embeddings/"
    "test_20_molformer_embeddings.npy"
)

OUTPUT_DIR = "feature_engineering/statistical_analysis"

ID_COLUMN = "np_id"
ACTIVITY_COLUMN = "Activity_Label"

ACTIVE_LABEL = "Active"
INACTIVE_LABEL = "Inactive"

FDR_ALPHA = 0.05

# Minimum number of observations required per class
MIN_CLASS_N = 10


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sha256_file(path, block_size=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def bh_fdr(pvalues):

    """
    Benjamini-Hochberg FDR correction.

    Deterministic implementation.
    """

    pvalues = np.asarray(pvalues, dtype=float)

    n = len(pvalues)

    adjusted = np.full(n, np.nan, dtype=float)

    valid = np.isfinite(pvalues)

    if valid.sum() == 0:
        return adjusted

    pv = pvalues[valid]

    order = np.argsort(pv)

    ranked = pv[order]

    ranks = np.arange(1, len(ranked) + 1)

    q = ranked * len(ranked) / ranks

    q = np.minimum.accumulate(q[::-1])[::-1]

    q = np.clip(q, 0.0, 1.0)

    restored = np.empty_like(q)

    restored[order] = q

    adjusted[valid] = restored

    return adjusted


def cliffs_delta_from_auc(auc):

    """
    For continuous feature X and binary outcome:

        Cliff's delta = 2*AUC - 1

    Direction:
        positive -> feature tends to be larger in Active
        negative -> feature tends to be smaller in Active
    """

    return (2.0 * auc) - 1.0


def effect_interpretation(delta):

    """
    Conventional approximate interpretation.

    This is descriptive, not a biological threshold.
    """

    a = abs(delta)

    if a < 0.147:
        return "negligible"

    if a < 0.33:
        return "small"

    if a < 0.474:
        return "medium"

    return "large"


def safe_auc(x_active, x_inactive):

    """

    Computes AUC where larger feature values indicate Active.

    """

    x = np.concatenate([
        np.asarray(x_active),
        np.asarray(x_inactive)
    ])

    y = np.concatenate([
        np.ones(len(x_active)),
        np.zeros(len(x_inactive))
    ])

    if len(np.unique(x)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, x))
    except Exception:
        return np.nan


def analyze_feature(
    values,
    labels,
    feature_name
):

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)

    finite = np.isfinite(values)

    values = values[finite]
    labels = labels[finite]

    active = values[labels == ACTIVE_LABEL]
    inactive = values[labels == INACTIVE_LABEL]

    result = {
        "feature": feature_name,
        "n_total": int(len(values)),
        "n_active": int(len(active)),
        "n_inactive": int(len(inactive)),
        "missing_or_nonfinite": int((~finite).sum()),

        "active_mean": np.nan,
        "inactive_mean": np.nan,

        "active_median": np.nan,
        "inactive_median": np.nan,

        "active_q25": np.nan,
        "active_q75": np.nan,

        "inactive_q25": np.nan,
        "inactive_q75": np.nan,

        "median_difference_active_minus_inactive": np.nan,

        "auc": np.nan,
        "auc_abs_effect": np.nan,

        "cliffs_delta": np.nan,
        "effect_size_abs": np.nan,
        "effect_interpretation": "not_testable",

        "mannwhitney_u": np.nan,
        "mannwhitney_p": np.nan,

        "ks_statistic": np.nan,
        "ks_p": np.nan,

        "zero_fraction": np.nan,
    }

    if len(active) < MIN_CLASS_N or len(inactive) < MIN_CLASS_N:
        return result

    result["active_mean"] = float(np.mean(active))
    result["inactive_mean"] = float(np.mean(inactive))

    result["active_median"] = float(np.median(active))
    result["inactive_median"] = float(np.median(inactive))

    result["active_q25"] = float(np.quantile(active, 0.25))
    result["active_q75"] = float(np.quantile(active, 0.75))

    result["inactive_q25"] = float(np.quantile(inactive, 0.25))
    result["inactive_q75"] = float(np.quantile(inactive, 0.75))

    result["median_difference_active_minus_inactive"] = (
        result["active_median"] -
        result["inactive_median"]
    )

    result["zero_fraction"] = float(
        np.mean(values == 0)
    )

    # -------------------------------------------------------------------------
    # Mann-Whitney U
    # -------------------------------------------------------------------------

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            mw = mannwhitneyu(
                active,
                inactive,
                alternative="two-sided",
                method="asymptotic"
            )

            result["mannwhitney_u"] = float(mw.statistic)
            result["mannwhitney_p"] = float(mw.pvalue)

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # ROC AUC
    # -------------------------------------------------------------------------

    auc = safe_auc(active, inactive)

    result["auc"] = auc

    if np.isfinite(auc):

        result["auc_abs_effect"] = abs(auc - 0.5)

        delta = cliffs_delta_from_auc(auc)

        result["cliffs_delta"] = delta
        result["effect_size_abs"] = abs(delta)

        result["effect_interpretation"] = effect_interpretation(delta)

    # -------------------------------------------------------------------------
    # KS test
    # -------------------------------------------------------------------------

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:

            ks = ks_2samp(
                active,
                inactive,
                alternative="two-sided",
                method="auto"
            )

            result["ks_statistic"] = float(ks.statistic)
            result["ks_p"] = float(ks.pvalue)

        except Exception:
            pass

    return result


def analyze_dataframe(df, labels, dataset_name):

    feature_columns = []

    for col in df.columns:

        if col == ID_COLUMN:
            continue

        if col == ACTIVITY_COLUMN:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    print(f"\n{dataset_name}: {len(feature_columns)} numeric features")

    rows = []

    for i, feature in enumerate(feature_columns, start=1):

        if i == 1 or i % 50 == 0 or i == len(feature_columns):

            print(
                f"  Analyzing {i:,}/{len(feature_columns):,}: "
                f"{feature}"
            )

        result = analyze_feature(
            df[feature].to_numpy(),
            labels,
            feature
        )

        rows.append(result)

    result_df = pd.DataFrame(rows)

    result_df["fdr_q"] = bh_fdr(
        result_df["mannwhitney_p"].to_numpy()
    )

    result_df["significant_fdr_0_05"] = (
        result_df["fdr_q"] <= FDR_ALPHA
    )

    # Stronger practical criterion
    result_df["meaningful_auc"] = (
        result_df["auc_abs_effect"] >= 0.10
    )

    result_df["meaningful_effect"] = (
        result_df["effect_size_abs"] >= 0.20
    )

    result_df["statistically_and_practically_relevant"] = (
        result_df["significant_fdr_0_05"] &
        (
            result_df["meaningful_auc"] |
            result_df["meaningful_effect"]
        )
    )

    result_df = result_df.sort_values(
        by=[
            "statistically_and_practically_relevant",
            "effect_size_abs",
            "auc_abs_effect",
            "fdr_q"
        ],
        ascending=[
            False,
            False,
            False,
            True
        ],
        na_position="last"
    )

    result_df.insert(
        0,
        "dataset",
        dataset_name
    )

    return result_df


# =============================================================================
# LOAD SPLITS
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

START_TIME = datetime.now()

print("=" * 78)
print("SCRIPT 07 — STATISTICAL FEATURE ANALYSIS v1")
print("=" * 78)

print(f"Started: {START_TIME.isoformat()}")

print("\nPython:")
print(sys.version)

print("\nNumPy:", np.__version__)
print("Pandas:", pd.__version__)

print("\nOutput directory:")
print(OUTPUT_DIR)


# =============================================================================
# INPUT CHECK
# =============================================================================

INPUT_FILES = [
    TRAIN_SPLIT,
    TEST_SPLIT,
    MOLECULAR_TRAIN,
    MOLECULAR_TEST,
    THREE_D_TRAIN,
    THREE_D_TEST,
    MOLFORMER_TRAIN,
    MOLFORMER_TEST,
]

print("\n" + "-" * 78)
print("1. INPUT FILE CHECK")
print("-" * 78)

for path in INPUT_FILES:

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )

    print("FOUND:", path)


# =============================================================================
# HASH INPUTS
# =============================================================================

print("\n" + "-" * 78)
print("2. INPUT SHA256")
print("-" * 78)

input_hashes = {}

for path in INPUT_FILES:

    print(f"\n{path}")

    digest = sha256_file(path)

    input_hashes[path] = digest

    print(digest)


# =============================================================================
# LOAD MASTER TRAIN / TEST SPLITS
# =============================================================================

print("\n" + "-" * 78)
print("3. LOADING MASTER SPLITS")
print("-" * 78)

train_master = pd.read_csv(
    TRAIN_SPLIT,
    sep="\t"
)

test_master = pd.read_csv(
    TEST_SPLIT,
    sep="\t"
)

print("TRAIN shape:", train_master.shape)
print("TEST shape :", test_master.shape)


# =============================================================================
# VALIDATE MASTER SPLITS
# =============================================================================

for name, df in [
    ("TRAIN", train_master),
    ("TEST", test_master)
]:

    if ID_COLUMN not in df.columns:

        raise RuntimeError(
            f"{ID_COLUMN} missing from {name}"
        )

    if ACTIVITY_COLUMN not in df.columns:

        raise RuntimeError(
            f"{ACTIVITY_COLUMN} missing from {name}"
        )

    duplicates = int(
        df[ID_COLUMN].duplicated().sum()
    )

    if duplicates != 0:

        raise RuntimeError(
            f"{name} contains {duplicates} duplicate IDs"
        )

train_overlap = len(
    set(train_master[ID_COLUMN]) &
    set(test_master[ID_COLUMN])
)

if train_overlap != 0:

    raise RuntimeError(
        f"TRAIN/TEST ID overlap detected: {train_overlap}"
    )


# =============================================================================
# ACTIVITY QC
# =============================================================================

print("\n" + "-" * 78)
print("4. ACTIVITY LABEL QC")
print("-" * 78)

for name, df in [
    ("TRAIN", train_master),
    ("TEST", test_master)
]:

    counts = df[ACTIVITY_COLUMN].value_counts()

    print(f"\n{name}")

    print(counts)

    if ACTIVE_LABEL not in counts:
        raise RuntimeError(
            f"{ACTIVE_LABEL} missing from {name}"
        )

    if INACTIVE_LABEL not in counts:
        raise RuntimeError(
            f"{INACTIVE_LABEL} missing from {name}"
        )


train_labels = train_master[ACTIVITY_COLUMN].to_numpy()
test_labels = test_master[ACTIVITY_COLUMN].to_numpy()


# =============================================================================
# HELPER TO ALIGN TABULAR FEATURES
# =============================================================================

def load_and_align_table(path, master, dataset_name):

    print(f"\nLoading {path}")

    df = pd.read_csv(
        path,
        sep="\t"
    )

    print("Raw shape:", df.shape)

    if ID_COLUMN not in df.columns:

        raise RuntimeError(
            f"{ID_COLUMN} missing from {path}"
        )

    if df[ID_COLUMN].duplicated().any():

        raise RuntimeError(
            f"Duplicate IDs found in {path}"
        )

    master_ids = master[ID_COLUMN].astype(str)
    feature_ids = df[ID_COLUMN].astype(str)

    master_set = set(master_ids)
    feature_set = set(feature_ids)

    missing = master_set - feature_set
    extra = feature_set - master_set

    if missing:

        raise RuntimeError(
            f"{path}: {len(missing)} master IDs missing"
        )

    if extra:

        print(
            f"WARNING: {len(extra)} extra IDs in {path}; "
            "they will be discarded."
        )

    df[ID_COLUMN] = df[ID_COLUMN].astype(str)

    aligned = (
        master[
            [ID_COLUMN]
        ]
        .astype({ID_COLUMN: str})
        .merge(
            df,
            on=ID_COLUMN,
            how="left",
            sort=False,
            validate="one_to_one"
        )
    )

    if len(aligned) != len(master):

        raise RuntimeError(
            f"Alignment failed for {path}"
        )

    print(
        f"{dataset_name} aligned shape:",
        aligned.shape
    )

    return aligned


# =============================================================================
# MOLECULAR DESCRIPTORS
# =============================================================================

print("\n" + "=" * 78)
print("5. MOLECULAR DESCRIPTOR STATISTICS")
print("=" * 78)

molecular_train = load_and_align_table(
    MOLECULAR_TRAIN,
    train_master,
    "MOLECULAR TRAIN"
)

molecular_test = load_and_align_table(
    MOLECULAR_TEST,
    test_master,
    "MOLECULAR TEST"
)

molecular_train_stats = analyze_dataframe(
    molecular_train,
    train_labels,
    "molecular_train"
)

molecular_test_stats = analyze_dataframe(
    molecular_test,
    test_labels,
    "molecular_test"
)

molecular_train_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train_molecular_statistics.tsv"
    ),
    sep="\t",
    index=False
)

molecular_test_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_molecular_replication.tsv"
    ),
    sep="\t",
    index=False
)


# =============================================================================
# 3D DESCRIPTORS
# =============================================================================

print("\n" + "=" * 78)
print("6. 3D-QSAR DESCRIPTOR STATISTICS")
print("=" * 78)

three_d_train = load_and_align_table(
    THREE_D_TRAIN,
    train_master,
    "3D TRAIN"
)

three_d_test = load_and_align_table(
    THREE_D_TEST,
    test_master,
    "3D TEST"
)

three_d_train_stats = analyze_dataframe(
    three_d_train,
    train_labels,
    "3d_train"
)

three_d_test_stats = analyze_dataframe(
    three_d_test,
    test_labels,
    "3d_test"
)

three_d_train_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train_3d_statistics.tsv"
    ),
    sep="\t",
    index=False
)

three_d_test_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_3d_replication.tsv"
    ),
    sep="\t",
    index=False
)


# =============================================================================
# MOLFORMER
# =============================================================================

print("\n" + "=" * 78)
print("7. MOLFORMER EMBEDDING STATISTICS")
print("=" * 78)

print("Loading TRAIN embeddings...")

mf_train = np.load(
    MOLFORMER_TRAIN,
    mmap_mode="r"
)

print("TRAIN shape:", mf_train.shape)

print("Loading TEST embeddings...")

mf_test = np.load(
    MOLFORMER_TEST,
    mmap_mode="r"
)

print("TEST shape :", mf_test.shape)


if mf_train.shape[0] != len(train_master):

    raise RuntimeError(
        "MoLFormer TRAIN row count does not match master TRAIN"
    )

if mf_test.shape[0] != len(test_master):

    raise RuntimeError(
        "MoLFormer TEST row count does not match master TEST"
    )


def analyze_embeddings(
    matrix,
    labels,
    dataset_name
):

    n_rows, n_features = matrix.shape

    print(
        f"\n{dataset_name}: "
        f"{n_rows:,} rows × {n_features:,} dimensions"
    )

    rows = []

    for i in range(n_features):

        if (
            i == 0
            or (i + 1) % 50 == 0
            or i == n_features - 1
        ):

            print(
                f"  Analyzing dimension "
                f"{i + 1:,}/{n_features:,}"
            )

        result = analyze_feature(
            matrix[:, i],
            labels,
            f"molformer_{i + 1:04d}"
        )

        rows.append(result)

    result_df = pd.DataFrame(rows)

    result_df["fdr_q"] = bh_fdr(
        result_df["mannwhitney_p"].to_numpy()
    )

    result_df["significant_fdr_0_05"] = (
        result_df["fdr_q"] <= FDR_ALPHA
    )

    result_df["meaningful_auc"] = (
        result_df["auc_abs_effect"] >= 0.10
    )

    result_df["meaningful_effect"] = (
        result_df["effect_size_abs"] >= 0.20
    )

    result_df["statistically_and_practically_relevant"] = (
        result_df["significant_fdr_0_05"] &
        (
            result_df["meaningful_auc"] |
            result_df["meaningful_effect"]
        )
    )

    result_df.insert(
        0,
        "dataset",
        dataset_name
    )

    result_df = result_df.sort_values(
        by=[
            "statistically_and_practically_relevant",
            "effect_size_abs",
            "auc_abs_effect",
            "fdr_q"
        ],
        ascending=[
            False,
            False,
            False,
            True
        ],
        na_position="last"
    )

    return result_df


molformer_train_stats = analyze_embeddings(
    mf_train,
    train_labels,
    "molformer_train"
)

molformer_test_stats = analyze_embeddings(
    mf_test,
    test_labels,
    "molformer_test"
)

molformer_train_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train_molformer_statistics.tsv"
    ),
    sep="\t",
    index=False
)

molformer_test_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_molformer_replication.tsv"
    ),
    sep="\t",
    index=False
)


# =============================================================================
# SUMMARY FUNCTION
# =============================================================================

def summarize_statistics(df):

    valid = df[
        np.isfinite(df["auc"])
    ].copy()

    return {
        "total_features": int(len(df)),

        "testable_features": int(len(valid)),

        "fdr_significant": int(
            valid["significant_fdr_0_05"].sum()
        ),

        "meaningful_auc": int(
            valid["meaningful_auc"].sum()
        ),

        "meaningful_effect": int(
            valid["meaningful_effect"].sum()
        ),

        "statistically_and_practically_relevant": int(
            valid[
                "statistically_and_practically_relevant"
            ].sum()
        ),

        "median_auc": float(
            valid["auc"].median()
        ) if len(valid) else None,

        "median_abs_auc_effect": float(
            valid["auc_abs_effect"].median()
        ) if len(valid) else None,

        "maximum_abs_auc_effect": float(
            valid["auc_abs_effect"].max()
        ) if len(valid) else None,

        "maximum_abs_cliffs_delta": float(
            valid["effect_size_abs"].max()
        ) if len(valid) else None,

        "best_feature": (
            str(
                valid.sort_values(
                    "effect_size_abs",
                    ascending=False
                ).iloc[0]["feature"]
            )
            if len(valid)
            else None
        )
    }


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("8. STATISTICAL SUMMARY")
print("=" * 78)

summaries = {

    "molecular_train":
        summarize_statistics(molecular_train_stats),

    "molecular_test":
        summarize_statistics(molecular_test_stats),

    "3d_train":
        summarize_statistics(three_d_train_stats),

    "3d_test":
        summarize_statistics(three_d_test_stats),

    "molformer_train":
        summarize_statistics(molformer_train_stats),

    "molformer_test":
        summarize_statistics(molformer_test_stats),
}


for name, summary in summaries.items():

    print("\n" + name)

    for key, value in summary.items():

        print(
            f"  {key}: {value}"
        )


# =============================================================================
# TOP TRAIN FEATURES
# =============================================================================

print("\n" + "=" * 78)
print("9. TOP TRAIN FEATURES")
print("=" * 78)


def print_top(df, name, n=20):

    print(f"\n{name}")

    cols = [
        "feature",
        "auc",
        "auc_abs_effect",
        "cliffs_delta",
        "effect_interpretation",
        "mannwhitney_p",
        "fdr_q",
        "ks_statistic",
        "statistically_and_practically_relevant"
    ]

    available = [
        c for c in cols
        if c in df.columns
    ]

    print(
        df[
            available
        ]
        .head(n)
        .to_string(index=False)
    )


print_top(
    molecular_train_stats,
    "MOLECULAR"
)

print_top(
    three_d_train_stats,
    "3D-QSAR"
)

print_top(
    molformer_train_stats,
    "MOLFORMER"
)


# =============================================================================
# CROSS-REPRESENTATION SUMMARY
# =============================================================================

representation_summary = pd.DataFrame([

    {
        "representation": "molecular",
        **summaries["molecular_train"]
    },

    {
        "representation": "3d",
        **summaries["3d_train"]
    },

    {
        "representation": "molformer",
        **summaries["molformer_train"]
    }

])

representation_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "representation_statistical_summary.tsv"
    ),
    sep="\t",
    index=False
)


# =============================================================================
# MACHINE-READABLE JSON
# =============================================================================

END_TIME = datetime.now()

report = {

    "script":
        "07_statistical_feature_analysis_v1.py",

    "status":
        "PASS",

    "generated":
        END_TIME.isoformat(),

    "statistical_framework": {

        "primary_test":
            "Mann-Whitney U",

        "secondary_discrimination_metric":
            "ROC-AUC",

        "distributional_test":
            "Kolmogorov-Smirnov",

        "effect_size":
            "Cliffs delta = 2*AUC - 1",

        "multiple_testing":
            "Benjamini-Hochberg FDR",

        "fdr_alpha":
            FDR_ALPHA,

        "train_used_for_feature_selection":
            True,

        "test_used_for_feature_selection":
            False,

        "deterministic":
            True
    },

    "inputs": {

        "train_split": {
            "file": TRAIN_SPLIT,
            "sha256": input_hashes[TRAIN_SPLIT],
            "rows": len(train_master)
        },

        "test_split": {
            "file": TEST_SPLIT,
            "sha256": input_hashes[TEST_SPLIT],
            "rows": len(test_master)
        },

        "molecular_train": {
            "file": MOLECULAR_TRAIN,
            "sha256": input_hashes[MOLECULAR_TRAIN]
        },

        "molecular_test": {
            "file": MOLECULAR_TEST,
            "sha256": input_hashes[MOLECULAR_TEST]
        },

        "3d_train": {
            "file": THREE_D_TRAIN,
            "sha256": input_hashes[THREE_D_TRAIN]
        },

        "3d_test": {
            "file": THREE_D_TEST,
            "sha256": input_hashes[THREE_D_TEST]
        },

        "molformer_train": {
            "file": MOLFORMER_TRAIN,
            "sha256": input_hashes[MOLFORMER_TRAIN]
        },

        "molformer_test": {
            "file": MOLFORMER_TEST,
            "sha256": input_hashes[MOLFORMER_TEST]
        }
    },

    "class_distribution": {

        "train": {
            "active": int(
                (train_labels == ACTIVE_LABEL).sum()
            ),
            "inactive": int(
                (train_labels == INACTIVE_LABEL).sum()
            )
        },

        "test": {
            "active": int(
                (test_labels == ACTIVE_LABEL).sum()
            ),
            "inactive": int(
                (test_labels == INACTIVE_LABEL).sum()
            )
        }
    },

    "representations": summaries,

    "outputs": {

        "train_molecular":
            "train_molecular_statistics.tsv",

        "test_molecular":
            "test_molecular_replication.tsv",

        "train_3d":
            "train_3d_statistics.tsv",

        "test_3d":
            "test_3d_replication.tsv",

        "train_molformer":
            "train_molformer_statistics.tsv",

        "test_molformer":
            "test_molformer_replication.tsv",

        "representation_summary":
            "representation_statistical_summary.tsv",

        "json":
            "statistical_feature_summary.json",

        "report":
            "statistical_feature_analysis_report.txt"
    }
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "statistical_feature_summary.json"
    ),
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# =============================================================================
# HUMAN-READABLE REPORT
# =============================================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "statistical_feature_analysis_report.txt"
)

with open(report_path, "w") as f:

    f.write(
        "==============================================================================\n"
    )

    f.write(
        "SCRIPT 07 — STATISTICAL FEATURE ANALYSIS REPORT\n"
    )

    f.write(
        "==============================================================================\n\n"
    )

    f.write(
        f"Generated: {END_TIME.isoformat()}\n\n"
    )

    f.write(
        "STATISTICAL FRAMEWORK\n"
    )

    f.write(
        "------------------------------------------------------------------------------\n"
    )

    f.write(
        "Primary test: Mann-Whitney U\n"
    )

    f.write(
        "Effect metric: ROC-AUC\n"
    )

    f.write(
        "Effect size: Cliff's delta = 2*AUC - 1\n"
    )

    f.write(
        "Distribution test: Kolmogorov-Smirnov\n"
    )

    f.write(
        "Multiple testing: Benjamini-Hochberg FDR\n"
    )

    f.write(
        f"FDR alpha: {FDR_ALPHA}\n"
    )

    f.write(
        "TRAIN used for feature screening: YES\n"
    )

    f.write(
        "TEST used for feature selection: NO\n\n"
    )

    f.write(
        "REPRESENTATION SUMMARY\n"
    )

    f.write(
        "------------------------------------------------------------------------------\n"
    )

    for representation in [
        "molecular_train",
        "3d_train",
        "molformer_train"
    ]:

        f.write(
            f"\n{representation}\n"
        )

        for key, value in summaries[representation].items():

            f.write(
                f"  {key}: {value}\n"
            )

    f.write(
        "\n\nIMPORTANT INTERPRETATION\n"
    )

    f.write(
        "------------------------------------------------------------------------------\n"
    )

    f.write(
        "Statistical significance alone is NOT sufficient for feature selection.\n"
    )

    f.write(
        "With ~48,000 molecules, extremely small effects can produce very small\n"
    )

    f.write(
        "p-values. Therefore AUC and Cliff's delta should be considered together\n"
    )

    f.write(
        "with FDR-adjusted significance and independent TEST replication.\n"
    )

    f.write(
        "\nNo features were removed by this script.\n"
    )

    f.write(
        "No predictive model was trained by this script.\n"
    )


print("\n" + "=" * 78)
print("10. OUTPUTS")
print("=" * 78)

print(
    "Written:",
    os.path.join(
        OUTPUT_DIR,
        "train_molecular_statistics.tsv"
    )
)

print(
    "Written:",
    os.path.join(
        OUTPUT_DIR,
        "test_molecular_replication.tsv"
    )
)

print(
    "Written:",
    os.path.join(
        OUTPUT_DIR,
        "train_3d_statistics.tsv"
    )
)

print(
    "Written:",
    os.path.join(
        OUTPUT_DIR,
        "test_3d_replication.tsv"
    )
)

print(
    "Written:",
    os.path.join(
        OUTPUT_DIR,
        "train_molformer_statistics.tsv"
    )
)

print(
    "Written:",
    os.path.join(
        OUTPUT_DIR,
        "test_molformer_replication.tsv"
    )
)

print(
    "Written:",
    os.path.join(
        OUTPUT_DIR,
        "representation_statistical_summary.tsv"
    )
)

print(
    "Written:",
    os.path.join(
        OUTPUT_DIR,
        "statistical_feature_summary.json"
    )
)

print(
    "Written:",
    report_path
)


# =============================================================================
# FINAL
# =============================================================================

print("\n" + "=" * 78)
print("SCRIPT 07 STATISTICAL FEATURE ANALYSIS COMPLETE")
print("=" * 78)

print("\nFINAL STATUS: PASS")

print(
    "\nIMPORTANT:"
    "\n  No features were removed."
    "\n  No model was trained."
    "\n  TRAIN statistics are for screening."
    "\n  TEST statistics are for independent replication."
)

print(
    f"\nFinished: {END_TIME.isoformat()}"
)

print("=" * 78)
