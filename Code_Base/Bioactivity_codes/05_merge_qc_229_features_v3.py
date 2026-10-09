#!/usr/bin/env python3

"""
======================================================================
SCRIPT 05 v3
RDKit + 3D FEATURE INTEGRATION WITH STRICT TRAIN-ONLY QC
======================================================================

Purpose
-------
Merge:
    217 RDKit molecular descriptors
    +
    11 3D structure-derived descriptors

Initial feature space:
    228 descriptors

QC decisions are derived ONLY from TRAIN.

TEST is NEVER used to decide:
    - missingness removal
    - variance removal
    - identical-feature removal
    - correlation removal

TEST is transformed using the decisions learned from TRAIN.

No scaling is performed here.

Expected input:
    feature_engineering/molecular_descriptors/train_80_molecular_descriptors.tsv
    feature_engineering/molecular_descriptors/test_20_molecular_descriptors.tsv

    feature_engineering/3d_qsar_descriptors/train_80_3d_qsar_descriptors.tsv
    feature_engineering/3d_qsar_descriptors/test_20_3d_qsar_descriptors.tsv

Output:
    feature_engineering/final_228_features_v3/
        train_80_final_features.tsv
        test_20_final_features.tsv
        feature_list.txt
        removed_features.txt
        feature_removal_log.tsv
        preprocessing_statistics.tsv
        qc_summary.txt

======================================================================
"""

import os
import sys
import time
import math
import traceback

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

RDKit_TRAIN = (
    "feature_engineering/molecular_descriptors/"
    "train_80_molecular_descriptors.tsv"
)

RDKit_TEST = (
    "feature_engineering/molecular_descriptors/"
    "test_20_molecular_descriptors.tsv"
)

D3_TRAIN = (
    "feature_engineering/3d_qsar_descriptors/"
    "train_80_3d_qsar_descriptors.tsv"
)

D3_TEST = (
    "feature_engineering/3d_qsar_descriptors/"
    "test_20_3d_qsar_descriptors.tsv"
)

DESCRIPTOR_LIST = (
    "feature_engineering/molecular_descriptors/"
    "descriptor_list.txt"
)

OUTPUT_DIR = "feature_engineering/final_228_features_v3"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train_80_final_features.tsv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "test_20_final_features.tsv"
)

FEATURE_LIST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "feature_list.txt"
)

REMOVED_FEATURES_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "removed_features.txt"
)

REMOVAL_LOG_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "feature_removal_log.tsv"
)

STATS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "preprocessing_statistics.tsv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "qc_summary.txt"
)


# Expected descriptors
EXPECTED_RDKIT = 217
EXPECTED_3D = 11
EXPECTED_TOTAL = EXPECTED_RDKIT + EXPECTED_3D


# QC thresholds
MISSINGNESS_THRESHOLD = 0.20
VARIANCE_THRESHOLD = 1e-12
CORRELATION_THRESHOLD = 0.95


# Metadata columns in the original RDKit/3D files
METADATA_COLUMNS = [
    "np_id",
    "pref_name",
    "iupac_name",
    "SMILES",
    "Canonical_SMILES",
    "Activity_Status",
    "Activity_Label",
    "chembl_id",
    "pubchem_cid",
    "MW",
    "LogS",
    "LogD",
    "LogP",
    "nHA",
    "nHD",
    "TPSA",
    "nRot",
    "nRing",
    "InChI",
    "InChIKey",
]


# 3D QC columns are NOT ML features
D3_QC_COLUMNS = [
    "3D_Status",
    "Optimization_Method",
    "Best_Conformer_Energy",
    "Conformers_Generated",
    "Metal_Containing",
]


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def die(message):
    print("\nERROR:", message)
    sys.exit(1)


def check_file(path):
    if not os.path.isfile(path):
        die(f"Required file not found:\n{path}")

    print(f"FOUND: {path}")


def load_tsv(path):
    print(f"\nLoading:\n{path}")
    df = pd.read_csv(
        path,
        sep="\t",
        low_memory=False
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    return df


def duplicate_ids(df):
    if "np_id" not in df.columns:
        die("np_id column missing.")

    return int(df["np_id"].duplicated().sum())


def validate_ids(df, name):
    dup = duplicate_ids(df)

    print(
        f"{name} duplicate np_id: {dup:,}"
    )

    if dup != 0:
        die(
            f"{name} contains duplicate np_id values."
        )


def validate_train_test_separation(train_df, test_df):

    overlap = set(train_df["np_id"]) & set(test_df["np_id"])

    print(
        f"Train/test overlapping IDs: {len(overlap):,}"
    )

    if overlap:
        die(
            "Train/test ID leakage detected."
        )

    print("Train/test separation: PASS")


def validate_row_counts(
    rd_train,
    rd_test,
    d3_train,
    d3_test
):

    n_train = len(rd_train)
    n_test = len(rd_test)

    if len(d3_train) != n_train:
        die("RDKit TRAIN and 3D TRAIN row counts differ.")

    if len(d3_test) != n_test:
        die("RDKit TEST and 3D TEST row counts differ.")

    print(
        f"RDKit TRAIN: PASS ({n_train:,})"
    )
    print(
        f"RDKit TEST: PASS ({n_test:,})"
    )
    print(
        f"3D TRAIN: PASS ({len(d3_train):,})"
    )
    print(
        f"3D TEST: PASS ({len(d3_test):,})"
    )


def validate_descriptor_list():

    with open(
        DESCRIPTOR_LIST,
        "r",
        encoding="utf-8"
    ) as f:

        descriptors = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # Preserve order while checking duplicates
    if len(descriptors) != len(set(descriptors)):
        die(
            "descriptor_list.txt contains duplicate descriptor names."
        )

    if len(descriptors) != EXPECTED_RDKIT:
        die(
            f"Expected {EXPECTED_RDKIT} RDKit descriptors "
            f"but descriptor_list.txt contains "
            f"{len(descriptors)}."
        )

    print(
        f"RDKit descriptor list: {len(descriptors)}"
    )

    return descriptors


def validate_rdkit_columns(
    train_df,
    test_df,
    descriptors
):

    train_missing = [
        x for x in descriptors
        if x not in train_df.columns
    ]

    test_missing = [
        x for x in descriptors
        if x not in test_df.columns
    ]

    if train_missing:
        die(
            "RDKit TRAIN is missing descriptors:\n"
            + "\n".join(train_missing)
        )

    if test_missing:
        die(
            "RDKit TEST is missing descriptors:\n"
            + "\n".join(test_missing)
        )

    print(
        "RDKit TRAIN descriptor validation: PASS"
    )
    print(
        "RDKit TEST descriptor validation : PASS"
    )


def identify_3d_features(train_df, test_df):

    excluded = (
        set(METADATA_COLUMNS)
        |
        set(D3_QC_COLUMNS)
    )

    train_features = [
        c for c in train_df.columns
        if c not in excluded
    ]

    test_features = [
        c for c in test_df.columns
        if c not in excluded
    ]

    if set(train_features) != set(test_features):
        train_only = sorted(
            set(train_features) - set(test_features)
        )

        test_only = sorted(
            set(test_features) - set(train_features)
        )

        die(
            "TRAIN/TEST 3D descriptor mismatch.\n"
            f"TRAIN only: {train_only}\n"
            f"TEST only : {test_only}"
        )

    # Preserve TRAIN order
    features = train_features

    if len(features) != EXPECTED_3D:
        die(
            f"Expected {EXPECTED_3D} 3D descriptors "
            f"but found {len(features)}."
        )

    print(
        f"3D descriptors: {len(features)}"
    )

    return features


def validate_id_matching(
    rd_train,
    d3_train,
    rd_test,
    d3_test
):

    for name, a, b in [
        (
            "Training",
            rd_train,
            d3_train
        ),
        (
            "Testing",
            rd_test,
            d3_test
        )
    ]:

        a_ids = set(a["np_id"])
        b_ids = set(b["np_id"])

        missing_a = a_ids - b_ids
        missing_b = b_ids - a_ids

        print(
            f"{name} RDKit IDs missing from 3D: "
            f"{len(missing_a):,}"
        )

        print(
            f"{name} 3D IDs missing from RDKit: "
            f"{len(missing_b):,}"
        )

        if missing_a or missing_b:
            die(
                f"{name} RDKit/3D ID mismatch."
            )

    print("RDKit <-> 3D ID matching: PASS")


# =====================================================================
# MERGING
# =====================================================================

def merge_features(
    rd_train,
    d3_train,
    rd_test,
    d3_test,
    rdkit_features,
    d3_features
):

    # Always merge by np_id.
    # This prevents accidental row-order mismatch.

    rd_train_indexed = rd_train.set_index(
        "np_id",
        drop=False
    )

    d3_train_indexed = d3_train.set_index(
        "np_id",
        drop=False
    )

    rd_test_indexed = rd_test.set_index(
        "np_id",
        drop=False
    )

    d3_test_indexed = d3_test.set_index(
        "np_id",
        drop=False
    )

    train_ids = rd_train["np_id"].tolist()
    test_ids = rd_test["np_id"].tolist()

    train = pd.DataFrame({
        "np_id": train_ids,
        "Activity_Label": [
            rd_train_indexed.loc[x, "Activity_Label"]
            for x in train_ids
        ]
    })

    test = pd.DataFrame({
        "np_id": test_ids,
        "Activity_Label": [
            rd_test_indexed.loc[x, "Activity_Label"]
            for x in test_ids
        ]
    })

    for feature in rdkit_features:
        train[feature] = pd.to_numeric(
            rd_train_indexed.loc[
                train_ids,
                feature
            ],
            errors="coerce"
        ).to_numpy()

        test[feature] = pd.to_numeric(
            rd_test_indexed.loc[
                test_ids,
                feature
            ],
            errors="coerce"
        ).to_numpy()

    for feature in d3_features:
        train[feature] = pd.to_numeric(
            d3_train_indexed.loc[
                train_ids,
                feature
            ],
            errors="coerce"
        ).to_numpy()

        test[feature] = pd.to_numeric(
            d3_test_indexed.loc[
                test_ids,
                feature
            ],
            errors="coerce"
        ).to_numpy()

    print(
        f"\nMerged TRAIN rows: {len(train):,}"
    )

    print(
        f"Merged TEST rows : {len(test):,}"
    )

    expected_columns = (
        2 + EXPECTED_TOTAL
    )

    if train.shape[1] != expected_columns:
        die(
            f"Unexpected TRAIN merged column count: "
            f"{train.shape[1]}; expected {expected_columns}."
        )

    if test.shape[1] != expected_columns:
        die(
            f"Unexpected TEST merged column count: "
            f"{test.shape[1]}; expected {expected_columns}."
        )

    return train, test


# =====================================================================
# TARGET VALIDATION
# =====================================================================

def validate_target(train, test):

    valid_labels = {
        "Active",
        "Inactive"
    }

    train_labels = set(
        train["Activity_Label"].dropna().unique()
    )

    test_labels = set(
        test["Activity_Label"].dropna().unique()
    )

    if not train_labels.issubset(valid_labels):
        die(
            f"Unexpected TRAIN labels: {train_labels}"
        )

    if not test_labels.issubset(valid_labels):
        die(
            f"Unexpected TEST labels: {test_labels}"
        )

    if train["Activity_Label"].isna().any():
        die("TRAIN contains missing Activity_Label.")

    if test["Activity_Label"].isna().any():
        die("TEST contains missing Activity_Label.")

    print("\nTarget validation: PASS")

    print("\nTraining target distribution:")
    print(
        train["Activity_Label"].value_counts()
    )

    print("\nTesting target distribution:")
    print(
        test["Activity_Label"].value_counts()
    )


# =====================================================================
# FEATURE CONVERSION
# =====================================================================

def convert_numeric(
    train,
    test,
    features
):

    train_missing_before = (
        train[features].isna().sum().sum()
    )

    test_missing_before = (
        test[features].isna().sum().sum()
    )

    # Convert everything to numeric.
    # Important: this is done identically for TRAIN and TEST.
    for feature in features:

        train[feature] = pd.to_numeric(
            train[feature],
            errors="coerce"
        )

        test[feature] = pd.to_numeric(
            test[feature],
            errors="coerce"
        )

    train_missing_after_conversion = (
        train[features].isna().sum().sum()
    )

    test_missing_after_conversion = (
        test[features].isna().sum().sum()
    )

    print(
        "\nTraining missing values before conversion:",
        train_missing_before
    )

    print(
        "Testing missing values before conversion :",
        test_missing_before
    )

    print(
        "Training missing values after conversion :",
        train_missing_after_conversion
    )

    print(
        "Testing missing values after conversion  :",
        test_missing_after_conversion
    )

    return train, test


# =====================================================================
# TRAIN-ONLY MISSINGNESS
# =====================================================================

def train_only_missingness_qc(
    train,
    test,
    features,
    removal_log
):

    print(
        "\n============================================================"
    )
    print("TRAIN-ONLY MISSINGNESS QC")
    print(
        "============================================================"
    )

    n = len(train)

    train_missing_fraction = (
        train[features]
        .isna()
        .sum()
        / n
    )

    remove = [
        f for f in features
        if train_missing_fraction[f]
        > MISSINGNESS_THRESHOLD
    ]

    print(
        f"Missingness threshold: "
        f"{MISSINGNESS_THRESHOLD * 100:.1f}%"
    )

    print(
        f"Features removed by TRAIN missingness: "
        f"{len(remove)}"
    )

    for feature in remove:

        removal_log.append({
            "Feature": feature,
            "Stage": "Missingness",
            "Reason": "TRAIN missingness above threshold",
            "Statistic": float(
                train_missing_fraction[feature]
            ),
            "Threshold": MISSINGNESS_THRESHOLD
        })

    retained = [
        f for f in features
        if f not in remove
    ]

    # Important:
    # Test missingness is reported but NEVER used for removal.
    test_missing_fraction = (
        test[retained]
        .isna()
        .sum()
        / len(test)
    )

    if len(retained) > 0:
        max_test_missing = (
            float(test_missing_fraction.max())
        )
    else:
        max_test_missing = 0.0

    print(
        f"Maximum TEST missingness among retained "
        f"features: {max_test_missing * 100:.4f}%"
    )

    return retained


# =====================================================================
# TRAIN-ONLY IMPUTATION
# =====================================================================

def train_only_imputation(
    train,
    test,
    features,
    removal_log
):

    print(
        "\n============================================================"
    )
    print("TRAIN-ONLY MEDIAN IMPUTATION")
    print(
        "============================================================"
    )

    statistics = []

    for feature in features:

        train_values = train[feature]

        median = train_values.median()

        if pd.isna(median):
            removal_log.append({
                "Feature": feature,
                "Stage": "Imputation",
                "Reason": "TRAIN median unavailable",
                "Statistic": np.nan,
                "Threshold": np.nan
            })

            continue

        train_missing = int(
            train[feature].isna().sum()
        )

        test_missing = int(
            test[feature].isna().sum()
        )

        if train_missing > 0:
            train[feature] = train[feature].fillna(
                median
            )

        if test_missing > 0:
            test[feature] = test[feature].fillna(
                median
            )

        statistics.append({
            "Feature": feature,
            "TRAIN_Median": median,
            "TRAIN_Missing_Before": train_missing,
            "TEST_Missing_Before": test_missing
        })

    stats_df = pd.DataFrame(statistics)

    return train, test, stats_df


# =====================================================================
# TRAIN-ONLY VARIANCE QC
# =====================================================================

def train_only_variance_qc(
    train,
    test,
    features,
    removal_log
):

    print(
        "\n============================================================"
    )
    print("TRAIN-ONLY VARIANCE QC")
    print(
        "============================================================"
    )

    remove = []

    for feature in features:

        values = train[feature]

        variance = values.var(
            ddof=0
        )

        if not np.isfinite(variance):

            remove.append(feature)

            removal_log.append({
                "Feature": feature,
                "Stage": "Variance",
                "Reason": "TRAIN variance is non-finite",
                "Statistic": variance,
                "Threshold": VARIANCE_THRESHOLD
            })

        elif variance <= VARIANCE_THRESHOLD:

            remove.append(feature)

            removal_log.append({
                "Feature": feature,
                "Stage": "Variance",
                "Reason": "TRAIN zero/near-zero variance",
                "Statistic": variance,
                "Threshold": VARIANCE_THRESHOLD
            })

    print(
        f"Variance threshold: {VARIANCE_THRESHOLD}"
    )

    print(
        f"Features removed: {len(remove)}"
    )

    if remove:
        print("\nRemoved:")
        for feature in remove:
            print(feature)

    retained = [
        f for f in features
        if f not in remove
    ]

    return retained


# =====================================================================
# TRAIN-ONLY IDENTICAL FEATURE QC
# =====================================================================

def train_only_identical_feature_qc(
    train,
    test,
    features,
    removal_log
):

    print(
        "\n============================================================"
    )
    print("TRAIN-ONLY IDENTICAL FEATURE QC")
    print(
        "============================================================"
    )

    """
    IMPORTANT:

    Identical-feature detection is performed ONLY on TRAIN.

    TEST is never examined to decide whether a feature should
    be removed.

    If:
        A == B in TRAIN

    then B is removed from BOTH TRAIN and TEST.

    If:
        A == B only in TEST

    nothing is removed.

    This specifically fixes the v2 issue involving:

        fr_alkyl_carbamate
        fr_benzodiazepine
        fr_tetrazole
    """

    retained = list(features)

    removed = set()

    identical_pairs = []

    # Efficient grouping by exact TRAIN values.
    # We use pandas hashing instead of an O(n^2) comparison.

    signatures = {}

    for feature in features:

        series = train[feature]

        # Include dtype and values.
        # Since all features have already been converted to numeric,
        # this is deterministic.

        signature = pd.util.hash_pandas_object(
            series,
            index=False
        ).sum()

        # Hash collision protection:
        # when signatures match, perform exact comparison.

        if signature in signatures:

            for previous_feature in signatures[signature]:

                if train[feature].equals(
                    train[previous_feature]
                ):

                    identical_pairs.append(
                        (
                            previous_feature,
                            feature
                        )
                    )

                    # Keep the first feature.
                    # Remove the later feature.
                    if feature not in removed:
                        removed.add(feature)

                    removal_log.append({
                        "Feature": feature,
                        "Stage": "IdenticalFeature",
                        "Reason": (
                            "Identical to another feature "
                            "in TRAIN"
                        ),
                        "Statistic": (
                            f"identical_to={previous_feature}"
                        ),
                        "Threshold": 0
                    })

                    break

            signatures[signature].append(
                feature
            )

        else:
            signatures[signature] = [
                feature
            ]

    print(
        f"TRAIN identical feature pairs: "
        f"{len(identical_pairs)}"
    )

    if identical_pairs:

        print("\nTRAIN identical pairs:")

        for a, b in identical_pairs:
            print(
                f"{a} <--> {b}"
            )

    else:
        print(
            "No identical feature-value columns in TRAIN."
        )

    print(
        f"Features removed by TRAIN identical-feature QC: "
        f"{len(removed)}"
    )

    retained = [
        f for f in features
        if f not in removed
    ]

    return retained, identical_pairs


# =====================================================================
# TRAIN-ONLY CORRELATION QC
# =====================================================================

def train_only_correlation_qc(
    train,
    test,
    features,
    removal_log
):

    print(
        "\n============================================================"
    )
    print("TRAIN-ONLY CORRELATION QC")
    print(
        "============================================================"
    )

    if len(features) <= 1:
        return features, []

    print(
        f"Calculating TRAIN correlation matrix for "
        f"{len(features)} features..."
    )

    corr = train[features].corr(
        method="pearson"
    ).abs()

    upper = np.triu(
        np.ones(
            corr.shape,
            dtype=bool
        ),
        k=1
    )

    upper_df = corr.where(
        upper
    )

    removal = set()
    pairs = []

    for feature in upper_df.columns:

        correlated = upper_df.index[
            upper_df[feature]
            > CORRELATION_THRESHOLD
        ].tolist()

        for other in correlated:

            r = float(
                upper_df.loc[
                    other,
                    feature
                ]
            )

            pairs.append(
                (
                    other,
                    feature,
                    r
                )
            )

            # Deterministic policy:
            # remove the later feature in the current feature order.
            if feature not in removal:
                removal.add(feature)

                removal_log.append({
                    "Feature": feature,
                    "Stage": "Correlation",
                    "Reason": (
                        "TRAIN correlation above threshold"
                    ),
                    "Statistic": r,
                    "Threshold": CORRELATION_THRESHOLD
                })

    print(
        f"Correlation threshold: "
        f"{CORRELATION_THRESHOLD}"
    )

    print(
        f"Features removed: {len(removal)}"
    )

    if pairs:

        print(
            "\nHighly correlated TRAIN pairs:"
        )

        for a, b, r in pairs:
            print(
                f"{a} <-> {b}: r={r:.5f}"
            )

    retained = [
        f for f in features
        if f not in removal
    ]

    return retained, pairs


# =====================================================================
# FINAL NUMERICAL QC
# =====================================================================

def final_numerical_qc(
    train,
    test,
    features
):

    print(
        "\n============================================================"
    )
    print("FINAL NUMERICAL QC")
    print(
        "============================================================"
    )

    train_nan = int(
        train[features]
        .isna()
        .sum()
        .sum()
    )

    test_nan = int(
        test[features]
        .isna()
        .sum()
        .sum()
    )

    train_inf = int(
        np.isinf(
            train[features].to_numpy()
        ).sum()
    )

    test_inf = int(
        np.isinf(
            test[features].to_numpy()
        ).sum()
    )

    print(
        f"Training NaN: {train_nan}"
    )

    print(
        f"Testing NaN : {test_nan}"
    )

    print(
        f"Training Inf: {train_inf}"
    )

    print(
        f"Testing Inf : {test_inf}"
    )

    if train_nan != 0:
        die(
            "TRAIN still contains NaN."
        )

    if test_nan != 0:
        die(
            "TEST still contains NaN."
        )

    if train_inf != 0:
        die(
            "TRAIN still contains Inf."
        )

    if test_inf != 0:
        die(
            "TEST still contains Inf."
        )

    # Column-name duplication check
    train_duplicates = [
        c for c in train.columns[
            train.columns.duplicated()
        ]
    ]

    test_duplicates = [
        c for c in test.columns[
            test.columns.duplicated()
        ]
    ]

    print(
        f"TRAIN duplicate column names: "
        f"{len(train_duplicates)}"
    )

    print(
        f"TEST duplicate column names : "
        f"{len(test_duplicates)}"
    )

    if train_duplicates:
        die(
            f"Duplicate TRAIN column names: "
            f"{train_duplicates}"
        )

    if test_duplicates:
        die(
            f"Duplicate TEST column names: "
            f"{test_duplicates}"
        )

    print(
        "Final numerical QC: PASS"
    )


# =====================================================================
# IMPORTANT TEST-ONLY DIAGNOSTIC
# =====================================================================

def test_only_identical_diagnostic(
    train,
    test,
    features
):

    print(
        "\n============================================================"
    )
    print("TEST-ONLY IDENTICAL FEATURE DIAGNOSTIC")
    print(
        "============================================================"
    )

    """
    This section is DIAGNOSTIC ONLY.

    It NEVER removes anything.

    It exists specifically to identify cases such as:

        fr_alkyl_carbamate
        fr_benzodiazepine
        fr_tetrazole

    being identical in TEST but not TRAIN.
    """

    signatures = {}
    identical_pairs = []

    for feature in features:

        series = test[feature]

        signature = pd.util.hash_pandas_object(
            series,
            index=False
        ).sum()

        if signature in signatures:

            for previous_feature in signatures[signature]:

                if test[feature].equals(
                    test[previous_feature]
                ):

                    identical_pairs.append(
                        (
                            previous_feature,
                            feature
                        )
                    )

            signatures[signature].append(
                feature
            )

        else:

            signatures[signature] = [
                feature
            ]

    print(
        f"TEST identical feature pairs: "
        f"{len(identical_pairs)}"
    )

    if identical_pairs:

        print(
            "\nNOTE:"
        )

        print(
            "These are TEST-only diagnostics."
        )

        print(
            "NO features will be removed because of them."
        )

        for a, b in identical_pairs:
            print(
                f"TEST IDENTICAL: {a} <--> {b}"
            )

    else:

        print(
            "No identical feature-value columns in TEST."
        )

    return identical_pairs


# =====================================================================
# SAVE FILES
# =====================================================================

def save_outputs(
    train,
    test,
    features,
    removal_log,
    stats_df,
    summary_lines
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Final datasets
    # ---------------------------------------------------------------

    train.to_csv(
        TRAIN_OUTPUT,
        sep="\t",
        index=False
    )

    test.to_csv(
        TEST_OUTPUT,
        sep="\t",
        index=False
    )

    # ---------------------------------------------------------------
    # Feature list
    # ---------------------------------------------------------------

    with open(
        FEATURE_LIST_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        for feature in features:
            f.write(
                feature + "\n"
            )

    # ---------------------------------------------------------------
    # Removed features
    # ---------------------------------------------------------------

    removed_features = [
        x["Feature"]
        for x in removal_log
    ]

    # Preserve order
    seen = set()
    unique_removed = []

    for feature in removed_features:

        if feature not in seen:

            seen.add(feature)
            unique_removed.append(
                feature
            )

    with open(
        REMOVED_FEATURES_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        for feature in unique_removed:
            f.write(
                feature + "\n"
            )

    # ---------------------------------------------------------------
    # Removal log
    # ---------------------------------------------------------------

    removal_df = pd.DataFrame(
        removal_log,
        columns=[
            "Feature",
            "Stage",
            "Reason",
            "Statistic",
            "Threshold"
        ]
    )

    removal_df.to_csv(
        REMOVAL_LOG_OUTPUT,
        sep="\t",
        index=False
    )

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    if stats_df is None:
        stats_df = pd.DataFrame()

    stats_df.to_csv(
        STATS_OUTPUT,
        sep="\t",
        index=False
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        for line in summary_lines:
            f.write(
                line + "\n"
            )

    print(
        "\nSaved:"
    )

    print(TRAIN_OUTPUT)
    print(TEST_OUTPUT)
    print(FEATURE_LIST_OUTPUT)
    print(REMOVED_FEATURES_OUTPUT)
    print(REMOVAL_LOG_OUTPUT)
    print(STATS_OUTPUT)
    print(SUMMARY_OUTPUT)


# =====================================================================
# MAIN
# =====================================================================

def main():

    start_time = time.time()

    print(
        "=================================================================="
    )

    print(
        "SCRIPT 05 v3 — RDKit + 3D FEATURE INTEGRATION"
    )

    print(
        "=================================================================="
    )

    print(
        "\nExpected feature composition:"
    )

    print(
        f"  RDKit descriptors: {EXPECTED_RDKIT}"
    )

    print(
        f"  3D descriptors:    {EXPECTED_3D}"
    )

    print(
        f"  TOTAL:             {EXPECTED_TOTAL}"
    )

    print(
        "\nQC POLICY:"
    )

    print(
        "  All feature-removal decisions are TRAIN-only."
    )

    print(
        "  TEST is transformed using TRAIN-derived decisions."
    )

    print(
        "  TEST-only duplicate/identical features are diagnostic only."
    )

    print(
        "  No scaling is performed."
    )

    # -----------------------------------------------------------------
    # Check inputs
    # -----------------------------------------------------------------

    print(
        "\n=================================================================="
    )

    print(
        "CHECKING INPUT FILES"
    )

    print(
        "=================================================================="
    )

    for path in [
        RDKit_TRAIN,
        RDKit_TEST,
        D3_TRAIN,
        D3_TEST,
        DESCRIPTOR_LIST
    ]:
        check_file(path)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    rd_train = load_tsv(
        RDKit_TRAIN
    )

    rd_test = load_tsv(
        RDKit_TEST
    )

    d3_train = load_tsv(
        D3_TRAIN
    )

    d3_test = load_tsv(
        D3_TEST
    )

    # -----------------------------------------------------------------
    # Basic validation
    # -----------------------------------------------------------------

    print(
        "\n=================================================================="
    )

    print(
        "ROW COUNT VALIDATION"
    )

    print(
        "=================================================================="
    )

    validate_row_counts(
        rd_train,
        rd_test,
        d3_train,
        d3_test
    )

    print(
        "\n=================================================================="
    )

    print(
        "ID VALIDATION"
    )

    print(
        "=================================================================="
    )

    validate_ids(
        rd_train,
        "RDKit TRAIN"
    )

    validate_ids(
        rd_test,
        "RDKit TEST"
    )

    validate_ids(
        d3_train,
        "3D TRAIN"
    )

    validate_ids(
        d3_test,
        "3D TEST"
    )

    validate_train_test_separation(
        rd_train,
        rd_test
    )

    validate_train_test_separation(
        d3_train,
        d3_test
    )

    # -----------------------------------------------------------------
    # Descriptor validation
    # -----------------------------------------------------------------

    print(
        "\n=================================================================="
    )

    print(
        "DESCRIPTOR VALIDATION"
    )

    print(
        "=================================================================="
    )

    rdkit_features = validate_descriptor_list()

    validate_rdkit_columns(
        rd_train,
        rd_test,
        rdkit_features
    )

    d3_features = identify_3d_features(
        d3_train,
        d3_test
    )

    print(
        f"\nRDKit features: {len(rdkit_features)}"
    )

    print(
        f"3D features:    {len(d3_features)}"
    )

    print(
        f"TOTAL features: "
        f"{len(rdkit_features) + len(d3_features)}"
    )

    if (
        len(rdkit_features)
        +
        len(d3_features)
        != EXPECTED_TOTAL
    ):
        die(
            "Initial feature count does not match expected total."
        )

    # -----------------------------------------------------------------
    # ID matching
    # -----------------------------------------------------------------

    validate_id_matching(
        rd_train,
        d3_train,
        rd_test,
        d3_test
    )

    # -----------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------

    print(
        "\n=================================================================="
    )

    print(
        "MERGING RDKit + 3D FEATURES"
    )

    print(
        "=================================================================="
    )

    train, test = merge_features(
        rd_train,
        d3_train,
        rd_test,
        d3_test,
        rdkit_features,
        d3_features
    )

    # -----------------------------------------------------------------
    # Target validation
    # -----------------------------------------------------------------

    validate_target(
        train,
        test
    )

    # -----------------------------------------------------------------
    # Feature list
    # -----------------------------------------------------------------

    all_features = (
        rdkit_features
        +
        d3_features
    )

    removal_log = []

    # -----------------------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------------------

    train, test = convert_numeric(
        train,
        test,
        all_features
    )

    # -----------------------------------------------------------------
    # Missingness
    # -----------------------------------------------------------------

    features = train_only_missingness_qc(
        train,
        test,
        all_features,
        removal_log
    )

    # -----------------------------------------------------------------
    # Imputation
    # -----------------------------------------------------------------

    train, test, stats_df = train_only_imputation(
        train,
        test,
        features,
        removal_log
    )

    # -----------------------------------------------------------------
    # Variance
    # -----------------------------------------------------------------

    features = train_only_variance_qc(
        train,
        test,
        features,
        removal_log
    )

    # -----------------------------------------------------------------
    # IDENTICAL FEATURE QC
    # -----------------------------------------------------------------

    features, train_identical_pairs = (
        train_only_identical_feature_qc(
            train,
            test,
            features,
            removal_log
        )
    )

    # -----------------------------------------------------------------
    # Correlation
    # -----------------------------------------------------------------

    features, correlation_pairs = (
        train_only_correlation_qc(
            train,
            test,
            features,
            removal_log
        )
    )

    # -----------------------------------------------------------------
    # Final test-only diagnostic
    # -----------------------------------------------------------------

    test_identical_pairs = (
        test_only_identical_diagnostic(
            train,
            test,
            features
        )
    )

    # -----------------------------------------------------------------
    # Final numerical QC
    # -----------------------------------------------------------------

    final_numerical_qc(
        train,
        test,
        features
    )

    # -----------------------------------------------------------------
    # Enforce identical feature ordering
    # -----------------------------------------------------------------

    train = train[
        ["np_id", "Activity_Label"]
        +
        features
    ]

    test = test[
        ["np_id", "Activity_Label"]
        +
        features
    ]

    if list(train.columns) != list(test.columns):
        die(
            "TRAIN/TEST feature ordering mismatch."
        )

    print(
        "\nTRAIN/TEST feature ordering: PASS"
    )

    # -----------------------------------------------------------------
    # Final dimensions
    # -----------------------------------------------------------------

    print(
        f"\nInitial features: {EXPECTED_TOTAL}"
    )

    print(
        f"Final features:   {len(features)}"
    )

    print(
        f"Training final shape: {train.shape}"
    )

    print(
        f"Testing final shape : {test.shape}"
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    runtime = (
        time.time()
        -
        start_time
    )

    summary_lines = [

        "SCRIPT 05 v3 — FINAL QC SUMMARY",
        "=" * 70,
        "",
        "FEATURE COMPOSITION",
        f"RDKit descriptors: {EXPECTED_RDKIT}",
        f"3D descriptors:    {EXPECTED_3D}",
        f"Initial features:  {EXPECTED_TOTAL}",
        f"Final features:    {len(features)}",
        "",
        "DATASET SIZE",
        f"Training molecules: {len(train):,}",
        f"Testing molecules:  {len(test):,}",
        "",
        "TRAIN-ONLY QC",
        f"Missingness threshold: {MISSINGNESS_THRESHOLD}",
        f"Variance threshold: {VARIANCE_THRESHOLD}",
        f"Correlation threshold: {CORRELATION_THRESHOLD}",
        f"TRAIN identical pairs: {len(train_identical_pairs)}",
        f"Correlation pairs above threshold: {len(correlation_pairs)}",
        "",
        "TEST DIAGNOSTICS",
        f"TEST identical pairs: {len(test_identical_pairs)}",
        "TEST-only identical features were NOT removed.",
        "",
        "FINAL NUMERICAL QC",
        f"Training NaN: {train[features].isna().sum().sum()}",
        f"Testing NaN:  {test[features].isna().sum().sum()}",
        f"Training Inf: {np.isinf(train[features].to_numpy()).sum()}",
        f"Testing Inf:  {np.isinf(test[features].to_numpy()).sum()}",
        "",
        "FINAL SHAPE",
        f"Training: {train.shape}",
        f"Testing:  {test.shape}",
        "",
        f"Runtime: {runtime / 60:.2f} minutes",
        "",
        "QC STATUS: PASS"
    ]

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    save_outputs(
        train,
        test,
        features,
        removal_log,
        stats_df,
        summary_lines
    )

    print(
        "\n=================================================================="
    )

    print(
        "FINAL QC: PASS"
    )

    print(
        "=================================================================="
    )


# =====================================================================
# RUN
# =====================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nInterrupted by user."
        )

        sys.exit(130)

    except Exception as e:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FATAL ERROR"
        )

        print(
            "=" * 70
        )

        print(
            str(e)
        )

        traceback.print_exc()

        sys.exit(1)
