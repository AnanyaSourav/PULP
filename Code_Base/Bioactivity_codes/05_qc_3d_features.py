#!/usr/bin/env python3

"""
SCRIPT 05
=========
QC and cleaning of molecular + 3D QSAR descriptors.

INPUT
-----
feature_engineering/3d_qsar_descriptors/train_80_3d_qsar_descriptors.tsv
feature_engineering/3d_qsar_descriptors/test_20_3d_qsar_descriptors.tsv

FEATURES
--------
217 RDKit molecular descriptors
+
12 3D QSAR descriptors

QC / PROVENANCE COLUMNS
-----------------------
3D_Status
Optimization_Method
Best_Conformer_Energy
Conformers_Generated
Metal_Containing

These QC/provenance columns are NOT used as ML features.

IMPORTANT
---------
All feature-cleaning decisions are learned from TRAINING data only.

TEST data is NEVER used to determine:
- imputation values
- variance thresholds
- correlation filtering
- feature selection

Outputs
-------
feature_engineering/final_features/
    train_80_features_qc.tsv
    test_20_features_qc.tsv

    feature_list.txt
    removed_features.txt
    qc_summary.txt

    preprocessing_statistics.tsv
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = (
    "feature_engineering/3d_qsar_descriptors/"
    "train_80_3d_qsar_descriptors.tsv"
)

TEST_FILE = (
    "feature_engineering/3d_qsar_descriptors/"
    "test_20_3d_qsar_descriptors.tsv"
)

OUTPUT_DIR = "feature_engineering/final_features"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train_80_features_qc.tsv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "test_20_features_qc.tsv"
)

FEATURE_LIST = os.path.join(
    OUTPUT_DIR,
    "feature_list.txt"
)

REMOVED_FEATURES = os.path.join(
    OUTPUT_DIR,
    "removed_features.txt"
)

QC_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "qc_summary.txt"
)

PREPROCESSING_STATS = os.path.join(
    OUTPUT_DIR,
    "preprocessing_statistics.tsv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Minimum variance threshold.
# Constant features have variance = 0.
VARIANCE_THRESHOLD = 0.0

# Highly correlated feature threshold.
CORRELATION_THRESHOLD = 0.95

# Maximum fraction of missing values allowed in TRAINING
# before a feature is removed.
MAX_MISSING_FRACTION = 0.20

# Numerical precision
FLOAT_PRECISION = 8


# ============================================================
# COLUMNS THAT ARE NOT ML FEATURES
# ============================================================

IDENTIFIER_COLUMNS = [
    "np_id",
    "pref_name",
    "iupac_name",
    "SMILES",
    "Canonical_SMILES",
    "chembl_id",
    "pubchem_cid",
    "InChI",
    "InChIKey",
]

TARGET_COLUMNS = [
    "Activity_Status",
    "Activity_Label",
]

QC_COLUMNS = [
    "3D_Status",
    "Optimization_Method",
    "Best_Conformer_Energy",
    "Conformers_Generated",
    "Metal_Containing",
]


# ============================================================
# UTILITY
# ============================================================

def log(message):
    print(message, flush=True)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    log("")
    log("=" * 70)
    log("LOADING 3D FEATURE DATASETS")
    log("=" * 70)

    log(f"Training: {TRAIN_FILE}")
    train = pd.read_csv(
        TRAIN_FILE,
        sep="\t",
        low_memory=False
    )

    log(f"Testing : {TEST_FILE}")
    test = pd.read_csv(
        TEST_FILE,
        sep="\t",
        low_memory=False
    )

    log("")
    log(f"Training molecules: {len(train):,}")
    log(f"Testing molecules : {len(test):,}")

    return train, test


# ============================================================
# BASIC STRUCTURE CHECK
# ============================================================

def basic_checks(train, test):

    log("")
    log("=" * 70)
    log("BASIC DATASET QC")
    log("=" * 70)

    expected_train = 48152
    expected_test = 12038

    if len(train) != expected_train:
        log(
            f"WARNING: Expected {expected_train:,} training molecules "
            f"but found {len(train):,}"
        )
    else:
        log("Training row count: PASS")

    if len(test) != expected_test:
        log(
            f"WARNING: Expected {expected_test:,} testing molecules "
            f"but found {len(test):,}"
        )
    else:
        log("Testing row count: PASS")

    # Duplicate IDs
    train_dup = train["np_id"].duplicated().sum()
    test_dup = test["np_id"].duplicated().sum()

    log(f"Training duplicate np_id: {train_dup}")
    log(f"Testing duplicate np_id : {test_dup}")

    # Train/test overlap
    train_ids = set(train["np_id"].astype(str))
    test_ids = set(test["np_id"].astype(str))

    overlap = train_ids.intersection(test_ids)

    log(f"Train/test ID overlap: {len(overlap)}")

    if len(overlap) > 0:
        raise RuntimeError(
            "DATA LEAKAGE: molecules occur in both train and test."
        )

    log("Train/test separation: PASS")


# ============================================================
# DETERMINE CANDIDATE FEATURES
# ============================================================

def identify_candidate_features(train, test):

    log("")
    log("=" * 70)
    log("IDENTIFYING FEATURE COLUMNS")
    log("=" * 70)

    excluded = set(
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
        + QC_COLUMNS
    )

    common_columns = [
        c for c in train.columns
        if c in test.columns
        and c not in excluded
    ]

    log(f"Total columns in train: {len(train.columns)}")
    log(f"Total columns in test : {len(test.columns)}")

    log("")
    log(f"Candidate ML features: {len(common_columns)}")

    return common_columns


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

def convert_numeric(train, test, features):

    log("")
    log("=" * 70)
    log("CONVERTING FEATURES TO NUMERIC")
    log("=" * 70)

    train_features = train[features].copy()
    test_features = test[features].copy()

    for col in features:

        train_features[col] = pd.to_numeric(
            train_features[col],
            errors="coerce"
        )

        test_features[col] = pd.to_numeric(
            test_features[col],
            errors="coerce"
        )

    return train_features, test_features


# ============================================================
# NaN / INF QC
# ============================================================

def calculate_missing_statistics(train_features, test_features):

    log("")
    log("=" * 70)
    log("MISSING / INVALID VALUE QC")
    log("=" * 70)

    train_nan = train_features.isna().sum()
    test_nan = test_features.isna().sum()

    train_inf = np.isinf(
        train_features.to_numpy(dtype=float)
    ).sum(axis=0)

    test_inf = np.isinf(
        test_features.to_numpy(dtype=float)
    ).sum(axis=0)

    train_inf = pd.Series(
        train_inf,
        index=train_features.columns
    )

    test_inf = pd.Series(
        test_inf,
        index=test_features.columns
    )

    # Convert inf to NaN
    train_features = train_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    test_features = test_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    stats = pd.DataFrame({
        "Feature": train_features.columns,
        "Train_NaN": train_nan.values,
        "Test_NaN": test_nan.values,
        "Train_Inf": train_inf.values,
        "Test_Inf": test_inf.values,
    })

    stats["Train_Missing_Fraction"] = (
        stats["Train_NaN"] / len(train_features)
    )

    stats["Test_Missing_Fraction"] = (
        stats["Test_NaN"] / len(test_features)
    )

    return train_features, test_features, stats


# ============================================================
# REMOVE FEATURES WITH TOO MANY MISSING VALUES
# ============================================================

def remove_high_missing_features(
    train_features,
    test_features,
    missing_stats
):

    log("")
    log("=" * 70)
    log("MISSING-VALUE FEATURE FILTER")
    log("=" * 70)

    remove = missing_stats.loc[
        missing_stats["Train_Missing_Fraction"]
        > MAX_MISSING_FRACTION,
        "Feature"
    ].tolist()

    if remove:

        log(
            f"Removing {len(remove)} features with "
            f"> {MAX_MISSING_FRACTION * 100:.1f}% "
            f"training missing values"
        )

        for feature in remove:
            log(
                f"  REMOVE {feature} "
                f"({missing_stats.loc[missing_stats['Feature'] == feature, 'Train_Missing_Fraction'].iloc[0] * 100:.2f}%)"
            )

        train_features = train_features.drop(
            columns=remove
        )

        test_features = test_features.drop(
            columns=remove
        )

    else:
        log("No high-missing features removed.")

    return train_features, test_features, remove


# ============================================================
# TRAIN-ONLY MEDIAN IMPUTATION
# ============================================================

def median_imputation(train_features, test_features):

    log("")
    log("=" * 70)
    log("TRAIN-ONLY MEDIAN IMPUTATION")
    log("=" * 70)

    medians = train_features.median()

    train_missing_before = train_features.isna().sum().sum()
    test_missing_before = test_features.isna().sum().sum()

    train_features = train_features.fillna(medians)
    test_features = test_features.fillna(medians)

    train_missing_after = train_features.isna().sum().sum()
    test_missing_after = test_features.isna().sum().sum()

    log(
        f"Training missing values: "
        f"{train_missing_before:,} -> "
        f"{train_missing_after:,}"
    )

    log(
        f"Testing missing values : "
        f"{test_missing_before:,} -> "
        f"{test_missing_after:,}"
    )

    return train_features, test_features, medians


# ============================================================
# CONSTANT / ZERO-VARIANCE FILTER
# ============================================================

def remove_low_variance_features(
    train_features,
    test_features
):

    log("")
    log("=" * 70)
    log("ZERO-VARIANCE FEATURE FILTER")
    log("=" * 70)

    variances = train_features.var(
        axis=0,
        ddof=0
    )

    remove = variances[
        variances <= VARIANCE_THRESHOLD
    ].index.tolist()

    log(f"Zero/near-zero variance features: {len(remove)}")

    if remove:

        for feature in remove:
            log(f"  REMOVE {feature}")

        train_features = train_features.drop(
            columns=remove
        )

        test_features = test_features.drop(
            columns=remove
        )

    return train_features, test_features, remove, variances


# ============================================================
# HIGH CORRELATION FILTER
# ============================================================

def remove_highly_correlated_features(
    train_features,
    test_features
):

    log("")
    log("=" * 70)
    log("HIGH-CORRELATION FEATURE FILTER")
    log("=" * 70)

    log(
        f"Calculating training correlation matrix "
        f"for {train_features.shape[1]} features..."
    )

    corr = train_features.corr(
        method="pearson"
    ).abs()

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    remove = []

    for column in upper.columns:

        correlated = upper[column][
            upper[column] > CORRELATION_THRESHOLD
        ]

        if len(correlated) > 0:
            remove.append(column)

    remove = list(dict.fromkeys(remove))

    log(
        f"Features above correlation threshold "
        f"({CORRELATION_THRESHOLD}): {len(remove)}"
    )

    if remove:

        train_features = train_features.drop(
            columns=remove
        )

        test_features = test_features.drop(
            columns=remove
        )

    return train_features, test_features, remove


# ============================================================
# CREATE PREPROCESSING STATISTICS
# ============================================================

def create_statistics(
    train_features,
    medians
):

    stats = pd.DataFrame({
        "Feature": train_features.columns,
        "Training_Median": [
            medians.get(c, np.nan)
            for c in train_features.columns
        ],
        "Training_Mean": train_features.mean().values,
        "Training_Std": train_features.std(
            ddof=0
        ).values,
        "Training_Min": train_features.min().values,
        "Training_Max": train_features.max().values,
    })

    return stats


# ============================================================
# SAVE FINAL DATASETS
# ============================================================

def save_outputs(
    train,
    test,
    train_features,
    test_features
):

    log("")
    log("=" * 70)
    log("CREATING FINAL FEATURE MATRICES")
    log("=" * 70)

    # Preserve identifiers and target
    metadata_columns = (
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
    )

    # Only columns that actually exist
    metadata_columns = [
        c for c in metadata_columns
        if c in train.columns
    ]

    train_final = pd.concat(
        [
            train[metadata_columns].reset_index(drop=True),
            train_features.reset_index(drop=True)
        ],
        axis=1
    )

    test_final = pd.concat(
        [
            test[metadata_columns].reset_index(drop=True),
            test_features.reset_index(drop=True)
        ],
        axis=1
    )

    train_final.to_csv(
        TRAIN_OUTPUT,
        sep="\t",
        index=False,
        float_format=f"%.{FLOAT_PRECISION}g"
    )

    test_final.to_csv(
        TEST_OUTPUT,
        sep="\t",
        index=False,
        float_format=f"%.{FLOAT_PRECISION}g"
    )

    log(f"Saved: {TRAIN_OUTPUT}")
    log(f"Saved: {TEST_OUTPUT}")

    return train_final, test_final


# ============================================================
# SAVE FEATURE LIST
# ============================================================

def save_feature_list(features):

    with open(
        FEATURE_LIST,
        "w"
    ) as f:

        for feature in features:
            f.write(feature + "\n")

    log(f"Feature list saved: {FEATURE_LIST}")


# ============================================================
# SAVE REMOVED FEATURE LIST
# ============================================================

def save_removed_features(
    missing_removed,
    variance_removed,
    correlation_removed
):

    with open(
        REMOVED_FEATURES,
        "w"
    ) as f:

        f.write(
            "# Features removed during QC\n"
        )

        f.write(
            "\n# HIGH MISSINGNESS\n"
        )

        for feature in missing_removed:
            f.write(feature + "\n")

        f.write(
            "\n# ZERO VARIANCE\n"
        )

        for feature in variance_removed:
            f.write(feature + "\n")

        f.write(
            "\n# HIGH CORRELATION\n"
        )

        for feature in correlation_removed:
            f.write(feature + "\n")

    log(
        f"Removed feature list saved: "
        f"{REMOVED_FEATURES}"
    )


# ============================================================
# QC SUMMARY
# ============================================================

def save_qc_summary(
    train,
    test,
    original_feature_count,
    final_feature_count,
    missing_removed,
    variance_removed,
    correlation_removed,
    train_features,
    test_features
):

    with open(
        QC_SUMMARY,
        "w"
    ) as f:

        f.write(
            "3D QSAR FEATURE ENGINEERING QC SUMMARY\n"
        )
        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Training molecules: {len(train):,}\n"
        )

        f.write(
            f"Testing molecules : {len(test):,}\n\n"
        )

        f.write(
            f"Initial ML features: "
            f"{original_feature_count}\n"
        )

        f.write(
            f"Final ML features: "
            f"{final_feature_count}\n\n"
        )

        f.write(
            f"High-missing features removed: "
            f"{len(missing_removed)}\n"
        )

        f.write(
            f"Zero-variance features removed: "
            f"{len(variance_removed)}\n"
        )

        f.write(
            f"High-correlation features removed: "
            f"{len(correlation_removed)}\n\n"
        )

        f.write(
            f"Correlation threshold: "
            f"{CORRELATION_THRESHOLD}\n"
        )

        f.write(
            f"Maximum missing fraction: "
            f"{MAX_MISSING_FRACTION}\n\n"
        )

        f.write(
            f"Final training shape: "
            f"{train_features.shape}\n"
        )

        f.write(
            f"Final testing shape: "
            f"{test_features.shape}\n\n"
        )

        f.write(
            "QC PRINCIPLE\n"
        )
        f.write(
            "All preprocessing decisions were learned "
            "from TRAINING data only.\n"
        )
        f.write(
            "The testing dataset was not used for "
            "feature filtering or imputation.\n"
        )

    log(f"QC summary saved: {QC_SUMMARY}")


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    train_final,
    test_final,
    features
):

    log("")
    log("=" * 70)
    log("FINAL VALIDATION")
    log("=" * 70)

    train_feature_matrix = train_final[
        features
    ]

    test_feature_matrix = test_final[
        features
    ]

    # NaN
    train_nan = train_feature_matrix.isna().sum().sum()
    test_nan = test_feature_matrix.isna().sum().sum()

    # Inf
    train_inf = np.isinf(
        train_feature_matrix.to_numpy(
            dtype=float
        )
    ).sum()

    test_inf = np.isinf(
        test_feature_matrix.to_numpy(
            dtype=float
        )
    ).sum()

    log(f"Training NaN: {train_nan}")
    log(f"Testing NaN : {test_nan}")

    log(f"Training Inf: {train_inf}")
    log(f"Testing Inf : {test_inf}")

    # Duplicate features
    duplicate_features = (
        train_feature_matrix.columns[
            train_feature_matrix.columns.duplicated()
        ].tolist()
    )

    log(
        f"Duplicate feature columns: "
        f"{len(duplicate_features)}"
    )

    # Shape
    log(
        f"Training final shape: "
        f"{train_final.shape}"
    )

    log(
        f"Testing final shape : "
        f"{test_final.shape}"
    )

    if train_nan > 0 or test_nan > 0:
        raise RuntimeError(
            "Final datasets still contain NaN values."
        )

    if train_inf > 0 or test_inf > 0:
        raise RuntimeError(
            "Final datasets still contain Inf values."
        )

    if duplicate_features:
        raise RuntimeError(
            "Duplicate feature columns detected."
        )

    log("")
    log("FINAL QC: PASS")


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    ensure_output_dir()

    log("")
    log("=" * 70)
    log("SCRIPT 05 — 3D QSAR FEATURE QC")
    log("=" * 70)

    log(f"RDKit/3D input train: {TRAIN_FILE}")
    log(f"RDKit/3D input test : {TEST_FILE}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train, test = load_data()

    # --------------------------------------------------------
    # Basic QC
    # --------------------------------------------------------

    basic_checks(
        train,
        test
    )

    # --------------------------------------------------------
    # Candidate features
    # --------------------------------------------------------

    features = identify_candidate_features(
        train,
        test
    )

    original_feature_count = len(features)

    # --------------------------------------------------------
    # Convert numeric
    # --------------------------------------------------------

    train_features, test_features = convert_numeric(
        train,
        test,
        features
    )

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    (
        train_features,
        test_features,
        missing_stats
    ) = calculate_missing_statistics(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # Missing feature removal
    # --------------------------------------------------------

    (
        train_features,
        test_features,
        missing_removed
    ) = remove_high_missing_features(
        train_features,
        test_features,
        missing_stats
    )

    # --------------------------------------------------------
    # TRAIN-ONLY MEDIAN IMPUTATION
    # --------------------------------------------------------

    (
        train_features,
        test_features,
        medians
    ) = median_imputation(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # Zero variance
    # --------------------------------------------------------

    (
        train_features,
        test_features,
        variance_removed,
        variances
    ) = remove_low_variance_features(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # High correlation
    # --------------------------------------------------------

    (
        train_features,
        test_features,
        correlation_removed
    ) = remove_highly_correlated_features(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    stats = create_statistics(
        train_features,
        medians
    )

    stats.to_csv(
        PREPROCESSING_STATS,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Save final feature matrices
    # --------------------------------------------------------

    (
        train_final,
        test_final
    ) = save_outputs(
        train,
        test,
        train_features,
        test_features
    )

    final_features = list(
        train_features.columns
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    save_feature_list(
        final_features
    )

    # --------------------------------------------------------
    # Save removed features
    # --------------------------------------------------------

    save_removed_features(
        missing_removed,
        variance_removed,
        correlation_removed
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("FEATURE QC SUMMARY")
    log("=" * 70)

    log(
        f"Initial features: "
        f"{original_feature_count}"
    )

    log(
        f"Removed by missingness: "
        f"{len(missing_removed)}"
    )

    log(
        f"Removed by zero variance: "
        f"{len(variance_removed)}"
    )

    log(
        f"Removed by correlation: "
        f"{len(correlation_removed)}"
    )

    log(
        f"Final features: "
        f"{len(final_features)}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_validation(
        train_final,
        test_final,
        final_features
    )

    elapsed = time.time() - start

    log("")
    log("=" * 70)
    log("SCRIPT 05 COMPLETED")
    log("=" * 70)

    log(
        f"Runtime: {elapsed / 60:.2f} minutes"
    )

    log("")
    log("OUTPUT FILES:")
    log(TRAIN_OUTPUT)
    log(TEST_OUTPUT)
    log(FEATURE_LIST)
    log(REMOVED_FEATURES)
    log(QC_SUMMARY)
    log(PREPROCESSING_STATS)


if __name__ == "__main__":
    main()
