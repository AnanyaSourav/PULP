#!/usr/bin/env python3

"""
SCRIPT 05 v2
============

Merge:
    217 RDKit molecular descriptors
    +
     12 3D conformation-dependent descriptors
    =
    229 candidate ML features

Then perform TRAIN-ONLY QC.

IMPORTANT
---------
The test dataset is NEVER used to determine:
    - missingness filtering
    - imputation values
    - variance filtering
    - any other preprocessing decision

The 229 original features are preserved unless a feature is:
    1. >20% missing in TRAINING data
    2. zero variance in TRAINING data

No correlation filtering is performed here.

Correlation/feature selection should be handled separately during
model development so that the complete 229-feature representation
remains available.

INPUTS
------
RDKit descriptors:
    feature_engineering/molecular_descriptors/
        train_80_molecular_descriptors.tsv
        test_20_molecular_descriptors.tsv

3D descriptors:
    feature_engineering/3d_qsar_descriptors/
        train_80_3d_qsar_descriptors.tsv
        test_20_3d_qsar_descriptors.tsv

OUTPUTS
-------
feature_engineering/final_features/
    train_80_features_229.tsv
    test_20_features_229.tsv

    feature_list_229.txt
    removed_features_229.txt
    preprocessing_statistics_229.tsv
    qc_summary_229.txt
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

RDKit_TRAIN = (
    "feature_engineering/molecular_descriptors/"
    "train_80_molecular_descriptors.tsv"
)

RDKit_TEST = (
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

OUTPUT_DIR = (
    "feature_engineering/final_features"
)

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train_80_features_229.tsv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "test_20_features_229.tsv"
)

FEATURE_LIST = os.path.join(
    OUTPUT_DIR,
    "feature_list_229.txt"
)

REMOVED_FEATURES = os.path.join(
    OUTPUT_DIR,
    "removed_features_229.txt"
)

PREPROCESSING_STATS = os.path.join(
    OUTPUT_DIR,
    "preprocessing_statistics_229.tsv"
)

QC_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "qc_summary_229.txt"
)


# ============================================================
# EXPECTED DATASET SIZE
# ============================================================

EXPECTED_TRAIN = 48152
EXPECTED_TEST = 12038

EXPECTED_RDKIT_FEATURES = 217
EXPECTED_3D_FEATURES = 12
EXPECTED_TOTAL_FEATURES = 229


# ============================================================
# QC PARAMETERS
# ============================================================

MAX_MISSING_FRACTION = 0.20

VARIANCE_THRESHOLD = 0.0

FLOAT_PRECISION = 8


# ============================================================
# FEATURE / METADATA DEFINITIONS
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

THREE_D_QC_COLUMNS = [
    "3D_Status",
    "Optimization_Method",
    "Best_Conformer_Energy",
    "Conformers_Generated",
    "Metal_Containing",
]


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message, flush=True)


# ============================================================
# DIRECTORY
# ============================================================

def prepare_output_directory():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_all_data():

    log("")
    log("=" * 75)
    log("LOADING INPUT DATASETS")
    log("=" * 75)

    log("")
    log("RDKit TRAIN:")
    log(RDKit_TRAIN)

    rdkit_train = pd.read_csv(
        RDKit_TRAIN,
        sep="\t",
        low_memory=False
    )

    log(
        f"Rows: {len(rdkit_train):,}"
    )

    log("")
    log("RDKit TEST:")
    log(RDKit_TEST)

    rdkit_test = pd.read_csv(
        RDKit_TEST,
        sep="\t",
        low_memory=False
    )

    log(
        f"Rows: {len(rdkit_test):,}"
    )

    log("")
    log("3D TRAIN:")
    log(THREE_D_TRAIN)

    three_d_train = pd.read_csv(
        THREE_D_TRAIN,
        sep="\t",
        low_memory=False
    )

    log(
        f"Rows: {len(three_d_train):,}"
    )

    log("")
    log("3D TEST:")
    log(THREE_D_TEST)

    three_d_test = pd.read_csv(
        THREE_D_TEST,
        sep="\t",
        low_memory=False
    )

    log(
        f"Rows: {len(three_d_test):,}"
    )

    return (
        rdkit_train,
        rdkit_test,
        three_d_train,
        three_d_test
    )


# ============================================================
# BASIC ROW COUNT CHECK
# ============================================================

def check_row_counts(
    rdkit_train,
    rdkit_test,
    three_d_train,
    three_d_test
):

    log("")
    log("=" * 75)
    log("ROW COUNT VALIDATION")
    log("=" * 75)

    datasets = {
        "RDKit TRAIN": (
            rdkit_train,
            EXPECTED_TRAIN
        ),
        "RDKit TEST": (
            rdkit_test,
            EXPECTED_TEST
        ),
        "3D TRAIN": (
            three_d_train,
            EXPECTED_TRAIN
        ),
        "3D TEST": (
            three_d_test,
            EXPECTED_TEST
        ),
    }

    for name, (df, expected) in datasets.items():

        actual = len(df)

        if actual == expected:
            log(
                f"{name}: PASS ({actual:,})"
            )
        else:
            raise RuntimeError(
                f"{name}: expected {expected:,} "
                f"rows but found {actual:,}"
            )


# ============================================================
# CHECK NP_ID
# ============================================================

def check_ids(
    rdkit_train,
    rdkit_test,
    three_d_train,
    three_d_test
):

    log("")
    log("=" * 75)
    log("ID VALIDATION")
    log("=" * 75)

    datasets = {
        "RDKit TRAIN": rdkit_train,
        "RDKit TEST": rdkit_test,
        "3D TRAIN": three_d_train,
        "3D TEST": three_d_test,
    }

    for name, df in datasets.items():

        if "np_id" not in df.columns:
            raise RuntimeError(
                f"{name}: np_id column missing."
            )

        duplicates = (
            df["np_id"]
            .astype(str)
            .duplicated()
            .sum()
        )

        log(
            f"{name} duplicate np_id: "
            f"{duplicates}"
        )

        if duplicates > 0:
            raise RuntimeError(
                f"{name} contains duplicate np_id."
            )


# ============================================================
# CHECK TRAIN / TEST ID OVERLAP
# ============================================================

def check_train_test_separation(
    rdkit_train,
    rdkit_test
):

    log("")
    log("=" * 75)
    log("TRAIN / TEST LEAKAGE CHECK")
    log("=" * 75)

    train_ids = set(
        rdkit_train["np_id"]
        .astype(str)
    )

    test_ids = set(
        rdkit_test["np_id"]
        .astype(str)
    )

    overlap = train_ids.intersection(
        test_ids
    )

    log(
        f"Train/test overlapping IDs: "
        f"{len(overlap)}"
    )

    if len(overlap) > 0:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            "same np_id occurs in train and test."
        )

    log("Train/test separation: PASS")


# ============================================================
# CHECK RDKit / 3D ID MATCHING
# ============================================================

def check_id_matching(
    rdkit_train,
    rdkit_test,
    three_d_train,
    three_d_test
):

    log("")
    log("=" * 75)
    log("RDKit ↔ 3D ID MATCHING")
    log("=" * 75)

    rdkit_train_ids = set(
        rdkit_train["np_id"].astype(str)
    )

    three_d_train_ids = set(
        three_d_train["np_id"].astype(str)
    )

    rdkit_test_ids = set(
        rdkit_test["np_id"].astype(str)
    )

    three_d_test_ids = set(
        three_d_test["np_id"].astype(str)
    )

    train_missing_in_3d = (
        rdkit_train_ids - three_d_train_ids
    )

    train_extra_3d = (
        three_d_train_ids - rdkit_train_ids
    )

    test_missing_in_3d = (
        rdkit_test_ids - three_d_test_ids
    )

    test_extra_3d = (
        three_d_test_ids - rdkit_test_ids
    )

    log(
        "Training RDKit IDs missing from 3D: "
        f"{len(train_missing_in_3d)}"
    )

    log(
        "Training 3D IDs missing from RDKit: "
        f"{len(train_extra_3d)}"
    )

    log(
        "Testing RDKit IDs missing from 3D: "
        f"{len(test_missing_in_3d)}"
    )

    log(
        "Testing 3D IDs missing from RDKit: "
        f"{len(test_extra_3d)}"
    )

    if (
        train_missing_in_3d
        or train_extra_3d
        or test_missing_in_3d
        or test_extra_3d
    ):

        raise RuntimeError(
            "RDKit and 3D datasets do not contain "
            "identical np_id sets."
        )

    log("RDKit ↔ 3D ID matching: PASS")


# ============================================================
# DETERMINE RDKit FEATURES
# ============================================================

def identify_rdkit_features(
    rdkit_train,
    rdkit_test
):

    excluded = set(
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
        + THREE_D_QC_COLUMNS
    )

    features = [
        c for c in rdkit_train.columns
        if c not in excluded
        and c in rdkit_test.columns
    ]

    log("")
    log(
        f"RDKit candidate features: "
        f"{len(features)}"
    )

    if len(features) != EXPECTED_RDKIT_FEATURES:

        raise RuntimeError(
            f"Expected {EXPECTED_RDKIT_FEATURES} "
            f"RDKit descriptors but found "
            f"{len(features)}."
        )

    return features


# ============================================================
# DETERMINE 3D FEATURES
# ============================================================

def identify_3d_features(
    three_d_train,
    three_d_test
):

    excluded = set(
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
        + THREE_D_QC_COLUMNS
    )

    features = [
        c for c in three_d_train.columns
        if c not in excluded
        and c in three_d_test.columns
    ]

    log(
        f"3D candidate features: "
        f"{len(features)}"
    )

    if len(features) != EXPECTED_3D_FEATURES:

        raise RuntimeError(
            f"Expected {EXPECTED_3D_FEATURES} "
            f"3D descriptors but found "
            f"{len(features)}."
        )

    log("")
    log("3D descriptors:")

    for feature in features:
        log(
            f"  {feature}"
        )

    return features


# ============================================================
# CHECK FEATURE NAME COLLISIONS
# ============================================================

def check_feature_collisions(
    rdkit_features,
    three_d_features
):

    overlap = set(
        rdkit_features
    ).intersection(
        three_d_features
    )

    log("")
    log("=" * 75)
    log("FEATURE NAME COLLISION CHECK")
    log("=" * 75)

    log(
        f"RDKit features: {len(rdkit_features)}"
    )

    log(
        f"3D features: {len(three_d_features)}"
    )

    log(
        f"Feature name overlap: {len(overlap)}"
    )

    if overlap:

        log("Colliding feature names:")

        for feature in sorted(overlap):
            log(
                f"  {feature}"
            )

        raise RuntimeError(
            "RDKit and 3D descriptor names overlap."
        )

    log("Feature name collision check: PASS")


# ============================================================
# MERGE RDKit + 3D
# ============================================================

def merge_datasets(
    rdkit_train,
    rdkit_test,
    three_d_train,
    three_d_test,
    rdkit_features,
    three_d_features
):

    log("")
    log("=" * 75)
    log("MERGING RDKit + 3D FEATURES")
    log("=" * 75)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = [
        c for c in (
            IDENTIFIER_COLUMNS
            + TARGET_COLUMNS
        )
        if c in rdkit_train.columns
    ]

    # --------------------------------------------------------
    # RDKit subset
    # --------------------------------------------------------

    rdkit_train_subset = rdkit_train[
        metadata + rdkit_features
    ].copy()

    rdkit_test_subset = rdkit_test[
        metadata + rdkit_features
    ].copy()

    # --------------------------------------------------------
    # 3D subset
    # --------------------------------------------------------

    three_d_train_subset = three_d_train[
        ["np_id"] + three_d_features
    ].copy()

    three_d_test_subset = three_d_test[
        ["np_id"] + three_d_features
    ].copy()

    # --------------------------------------------------------
    # Merge by np_id
    # --------------------------------------------------------

    train = rdkit_train_subset.merge(
        three_d_train_subset,
        on="np_id",
        how="inner",
        validate="one_to_one"
    )

    test = rdkit_test_subset.merge(
        three_d_test_subset,
        on="np_id",
        how="inner",
        validate="one_to_one"
    )

    log(
        f"Merged training molecules: "
        f"{len(train):,}"
    )

    log(
        f"Merged testing molecules: "
        f"{len(test):,}"
    )

    if len(train) != EXPECTED_TRAIN:

        raise RuntimeError(
            "Training merge lost molecules."
        )

    if len(test) != EXPECTED_TEST:

        raise RuntimeError(
            "Testing merge lost molecules."
        )

    expected_features = (
        len(rdkit_features)
        + len(three_d_features)
    )

    actual_features = len(
        set(train.columns)
        - set(metadata)
    )

    log(
        f"Expected ML features: "
        f"{expected_features}"
    )

    log(
        f"Actual ML features: "
        f"{actual_features}"
    )

    if actual_features != EXPECTED_TOTAL_FEATURES:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_FEATURES} "
            f"features but found "
            f"{actual_features}."
        )

    log(
        "RDKit + 3D merge: PASS"
    )

    return train, test


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

def convert_features_numeric(
    train,
    test,
    features
):

    log("")
    log("=" * 75)
    log("NUMERIC FEATURE CONVERSION")
    log("=" * 75)

    train_features = train[
        features
    ].copy()

    test_features = test[
        features
    ].copy()

    for feature in features:

        train_features[feature] = pd.to_numeric(
            train_features[feature],
            errors="coerce"
        )

        test_features[feature] = pd.to_numeric(
            test_features[feature],
            errors="coerce"
        )

    return (
        train_features,
        test_features
    )


# ============================================================
# INF → NAN
# ============================================================

def replace_infinity(
    train_features,
    test_features
):

    train_array = train_features.to_numpy(
        dtype=float
    )

    test_array = test_features.to_numpy(
        dtype=float
    )

    train_inf = np.isinf(
        train_array
    ).sum()

    test_inf = np.isinf(
        test_array
    ).sum()

    train_features = train_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    test_features = test_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    log(
        f"Training Inf values: "
        f"{train_inf:,}"
    )

    log(
        f"Testing Inf values : "
        f"{test_inf:,}"
    )

    return (
        train_features,
        test_features,
        train_inf,
        test_inf
    )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def missing_value_qc(
    train_features,
    test_features
):

    log("")
    log("=" * 75)
    log("MISSING VALUE QC")
    log("=" * 75)

    train_nan = (
        train_features.isna().sum()
    )

    test_nan = (
        test_features.isna().sum()
    )

    stats = pd.DataFrame({
        "Feature": train_features.columns,
        "Train_NaN": train_nan.values,
        "Test_NaN": test_nan.values,
    })

    stats[
        "Train_Missing_Fraction"
    ] = (
        stats["Train_NaN"]
        / len(train_features)
    )

    stats[
        "Test_Missing_Fraction"
    ] = (
        stats["Test_NaN"]
        / len(test_features)
    )

    total_train_nan = (
        train_nan.sum()
    )

    total_test_nan = (
        test_nan.sum()
    )

    log(
        f"Training missing values: "
        f"{total_train_nan:,}"
    )

    log(
        f"Testing missing values : "
        f"{total_test_nan:,}"
    )

    high_missing = stats.loc[
        stats["Train_Missing_Fraction"]
        > MAX_MISSING_FRACTION,
        "Feature"
    ].tolist()

    log(
        f"Features with > "
        f"{MAX_MISSING_FRACTION * 100:.1f}% "
        f"training missingness: "
        f"{len(high_missing)}"
    )

    return stats, high_missing


# ============================================================
# REMOVE HIGH-MISSING FEATURES
# ============================================================

def remove_high_missing(
    train_features,
    test_features,
    high_missing
):

    if not high_missing:

        log(
            "No high-missing features removed."
        )

        return (
            train_features,
            test_features
        )

    log("")
    log(
        "Removing high-missing features:"
    )

    for feature in high_missing:
        log(
            f"  {feature}"
        )

    train_features = train_features.drop(
        columns=high_missing
    )

    test_features = test_features.drop(
        columns=high_missing
    )

    return (
        train_features,
        test_features
    )


# ============================================================
# TRAIN-ONLY MEDIAN IMPUTATION
# ============================================================

def train_only_median_imputation(
    train_features,
    test_features
):

    log("")
    log("=" * 75)
    log("TRAIN-ONLY MEDIAN IMPUTATION")
    log("=" * 75)

    # IMPORTANT:
    # Medians are calculated ONLY from TRAINING.

    medians = train_features.median(
        numeric_only=True
    )

    train_before = (
        train_features.isna().sum().sum()
    )

    test_before = (
        test_features.isna().sum().sum()
    )

    train_features = train_features.fillna(
        medians
    )

    test_features = test_features.fillna(
        medians
    )

    train_after = (
        train_features.isna().sum().sum()
    )

    test_after = (
        test_features.isna().sum().sum()
    )

    log(
        f"Training NaN: "
        f"{train_before:,} -> {train_after:,}"
    )

    log(
        f"Testing NaN : "
        f"{test_before:,} -> {test_after:,}"
    )

    if train_after > 0:

        raise RuntimeError(
            "Training still contains NaN "
            "after median imputation."
        )

    if test_after > 0:

        raise RuntimeError(
            "Testing still contains NaN "
            "after median imputation."
        )

    return (
        train_features,
        test_features,
        medians
    )


# ============================================================
# ZERO VARIANCE
# ============================================================

def zero_variance_qc(
    train_features,
    test_features
):

    log("")
    log("=" * 75)
    log("ZERO-VARIANCE FEATURE QC")
    log("=" * 75)

    # TRAIN ONLY
    variances = train_features.var(
        axis=0,
        ddof=0
    )

    remove = variances[
        variances <= VARIANCE_THRESHOLD
    ].index.tolist()

    log(
        f"Zero-variance features: "
        f"{len(remove)}"
    )

    for feature in remove:
        log(
            f"  REMOVE {feature}"
        )

    if remove:

        train_features = train_features.drop(
            columns=remove
        )

        test_features = test_features.drop(
            columns=remove
        )

    return (
        train_features,
        test_features,
        remove,
        variances
    )


# ============================================================
# PREPROCESSING STATISTICS
# ============================================================

def generate_statistics(
    train_features,
    medians
):

    stats = pd.DataFrame({
        "Feature": train_features.columns,

        "Training_Median": [
            medians.get(
                feature,
                np.nan
            )
            for feature in train_features.columns
        ],

        "Training_Mean":
            train_features.mean().values,

        "Training_Std":
            train_features.std(
                ddof=0
            ).values,

        "Training_Min":
            train_features.min().values,

        "Training_Max":
            train_features.max().values,

        "Training_Variance":
            train_features.var(
                ddof=0
            ).values,
    })

    return stats


# ============================================================
# SAVE FEATURE LIST
# ============================================================

def save_feature_list(features):

    with open(
        FEATURE_LIST,
        "w"
    ) as f:

        for feature in features:
            f.write(
                feature + "\n"
            )

    log(
        f"Feature list saved: "
        f"{FEATURE_LIST}"
    )


# ============================================================
# SAVE REMOVED FEATURES
# ============================================================

def save_removed_features(
    high_missing,
    zero_variance
):

    with open(
        REMOVED_FEATURES,
        "w"
    ) as f:

        f.write(
            "# SCRIPT 05 v2 REMOVED FEATURES\n"
        )

        f.write(
            "\n# HIGH TRAINING MISSINGNESS\n"
        )

        for feature in high_missing:
            f.write(
                feature + "\n"
            )

        f.write(
            "\n# ZERO TRAINING VARIANCE\n"
        )

        for feature in zero_variance:
            f.write(
                feature + "\n"
            )

    log(
        f"Removed feature list saved: "
        f"{REMOVED_FEATURES}"
    )


# ============================================================
# SAVE FINAL DATASETS
# ============================================================

def save_final_datasets(
    train,
    test,
    train_features,
    test_features
):

    metadata = [
        c for c in (
            IDENTIFIER_COLUMNS
            + TARGET_COLUMNS
        )
        if c in train.columns
    ]

    train_final = pd.concat(
        [
            train[
                metadata
            ].reset_index(drop=True),

            train_features.reset_index(
                drop=True
            )
        ],
        axis=1
    )

    test_final = pd.concat(
        [
            test[
                metadata
            ].reset_index(drop=True),

            test_features.reset_index(
                drop=True
            )
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

    log("")
    log(
        f"Saved: {TRAIN_OUTPUT}"
    )

    log(
        f"Saved: {TEST_OUTPUT}"
    )

    return (
        train_final,
        test_final
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    train_final,
    test_final,
    features
):

    log("")
    log("=" * 75)
    log("FINAL VALIDATION")
    log("=" * 75)

    # --------------------------------------------------------
    # Row counts
    # --------------------------------------------------------

    log(
        f"Training molecules: "
        f"{len(train_final):,}"
    )

    log(
        f"Testing molecules : "
        f"{len(test_final):,}"
    )

    if len(train_final) != EXPECTED_TRAIN:
        raise RuntimeError(
            "Final training row count incorrect."
        )

    if len(test_final) != EXPECTED_TEST:
        raise RuntimeError(
            "Final testing row count incorrect."
        )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    log(
        f"Final ML features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    train_nan = (
        train_final[
            features
        ].isna().sum().sum()
    )

    test_nan = (
        test_final[
            features
        ].isna().sum().sum()
    )

    log(
        f"Training NaN: {train_nan}"
    )

    log(
        f"Testing NaN : {test_nan}"
    )

    # --------------------------------------------------------
    # Inf
    # --------------------------------------------------------

    train_inf = np.isinf(
        train_final[
            features
        ].to_numpy(
            dtype=float
        )
    ).sum()

    test_inf = np.isinf(
        test_final[
            features
        ].to_numpy(
            dtype=float
        )
    ).sum()

    log(
        f"Training Inf: {train_inf}"
    )

    log(
        f"Testing Inf : {test_inf}"
    )

    # --------------------------------------------------------
    # Duplicate columns
    # --------------------------------------------------------

    duplicate_columns = (
        train_final.columns[
            train_final.columns.duplicated()
        ].tolist()
    )

    log(
        f"Duplicate columns: "
        f"{len(duplicate_columns)}"
    )

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    train_duplicate_ids = (
        train_final["np_id"]
        .duplicated()
        .sum()
    )

    test_duplicate_ids = (
        test_final["np_id"]
        .duplicated()
        .sum()
    )

    log(
        f"Training duplicate IDs: "
        f"{train_duplicate_ids}"
    )

    log(
        f"Testing duplicate IDs: "
        f"{test_duplicate_ids}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    if train_nan != 0:
        raise RuntimeError(
            "Training contains NaN."
        )

    if test_nan != 0:
        raise RuntimeError(
            "Testing contains NaN."
        )

    if train_inf != 0:
        raise RuntimeError(
            "Training contains Inf."
        )

    if test_inf != 0:
        raise RuntimeError(
            "Testing contains Inf."
        )

    if duplicate_columns:
        raise RuntimeError(
            "Duplicate columns detected."
        )

    if train_duplicate_ids != 0:
        raise RuntimeError(
            "Duplicate training np_id detected."
        )

    if test_duplicate_ids != 0:
        raise RuntimeError(
            "Duplicate testing np_id detected."
        )

    log("")
    log("FINAL QC: PASS")


# ============================================================
# WRITE QC SUMMARY
# ============================================================

def write_summary(
    initial_features,
    final_features,
    high_missing,
    zero_variance,
    train_final,
    test_final
):

    with open(
        QC_SUMMARY,
        "w"
    ) as f:

        f.write(
            "SCRIPT 05 v2\n"
        )

        f.write(
            "RDKit + 3D QSAR FEATURE QC SUMMARY\n"
        )

        f.write(
            "=" * 75 + "\n\n"
        )

        f.write(
            "DATASETS\n"
        )

        f.write(
            f"Training molecules: "
            f"{len(train_final):,}\n"
        )

        f.write(
            f"Testing molecules: "
            f"{len(test_final):,}\n\n"
        )

        f.write(
            "FEATURES\n"
        )

        f.write(
            f"RDKit descriptors: "
            f"{EXPECTED_RDKIT_FEATURES}\n"
        )

        f.write(
            f"3D descriptors: "
            f"{EXPECTED_3D_FEATURES}\n"
        )

        f.write(
            f"Initial total: "
            f"{initial_features}\n"
        )

        f.write(
            f"Final total: "
            f"{final_features}\n\n"
        )

        f.write(
            "REMOVED FEATURES\n"
        )

        f.write(
            f"High missingness: "
            f"{len(high_missing)}\n"
        )

        f.write(
            f"Zero variance: "
            f"{len(zero_variance)}\n\n"
        )

        f.write(
            "QC PARAMETERS\n"
        )

        f.write(
            f"Maximum training missingness: "
            f"{MAX_MISSING_FRACTION * 100:.1f}%\n"
        )

        f.write(
            f"Variance threshold: "
            f"{VARIANCE_THRESHOLD}\n"
        )

        f.write(
            "Correlation filtering: NOT PERFORMED\n\n"
        )

        f.write(
            "DATA LEAKAGE CONTROL\n"
        )

        f.write(
            "All missing-value filtering, "
            "imputation and variance decisions "
            "were calculated using TRAINING data only.\n"
        )

        f.write(
            "No test-set statistics were used.\n"
        )

        f.write(
            "\n3D QC/provenance columns excluded "
            "from ML features:\n"
        )

        for column in THREE_D_QC_COLUMNS:
            f.write(
                f"  {column}\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    prepare_output_directory()

    log("")
    log("=" * 75)
    log("SCRIPT 05 v2 — RDKit + 3D FEATURE INTEGRATION")
    log("=" * 75)

    log("")
    log(
        "Expected feature composition:"
    )

    log(
        f"  RDKit descriptors: "
        f"{EXPECTED_RDKIT_FEATURES}"
    )

    log(
        f"  3D descriptors: "
        f"{EXPECTED_3D_FEATURES}"
    )

    log(
        f"  TOTAL: "
        f"{EXPECTED_TOTAL_FEATURES}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        rdkit_train,
        rdkit_test,
        three_d_train,
        three_d_test
    ) = load_all_data()

    # --------------------------------------------------------
    # Row checks
    # --------------------------------------------------------

    check_row_counts(
        rdkit_train,
        rdkit_test,
        three_d_train,
        three_d_test
    )

    # --------------------------------------------------------
    # ID checks
    # --------------------------------------------------------

    check_ids(
        rdkit_train,
        rdkit_test,
        three_d_train,
        three_d_test
    )

    check_train_test_separation(
        rdkit_train,
        rdkit_test
    )

    check_id_matching(
        rdkit_train,
        rdkit_test,
        three_d_train,
        three_d_test
    )

    # --------------------------------------------------------
    # Identify features
    # --------------------------------------------------------

    rdkit_features = identify_rdkit_features(
        rdkit_train,
        rdkit_test
    )

    three_d_features = identify_3d_features(
        three_d_train,
        three_d_test
    )

    check_feature_collisions(
        rdkit_features,
        three_d_features
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    train, test = merge_datasets(
        rdkit_train,
        rdkit_test,
        three_d_train,
        three_d_test,
        rdkit_features,
        three_d_features
    )

    all_features = (
        rdkit_features
        + three_d_features
    )

    initial_feature_count = len(
        all_features
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    (
        train_features,
        test_features
    ) = convert_features_numeric(
        train,
        test,
        all_features
    )

    # --------------------------------------------------------
    # Inf handling
    # --------------------------------------------------------

    (
        train_features,
        test_features,
        train_inf,
        test_inf
    ) = replace_infinity(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # Missing QC
    # --------------------------------------------------------

    (
        missing_stats,
        high_missing
    ) = missing_value_qc(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # Remove high-missing
    # --------------------------------------------------------

    (
        train_features,
        test_features
    ) = remove_high_missing(
        train_features,
        test_features,
        high_missing
    )

    # --------------------------------------------------------
    # TRAIN ONLY median imputation
    # --------------------------------------------------------

    (
        train_features,
        test_features,
        medians
    ) = train_only_median_imputation(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # TRAIN ONLY variance filter
    # --------------------------------------------------------

    (
        train_features,
        test_features,
        zero_variance,
        variances
    ) = zero_variance_qc(
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # Final feature list
    # --------------------------------------------------------

    final_features = list(
        train_features.columns
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    statistics = generate_statistics(
        train_features,
        medians
    )

    statistics.to_csv(
        PREPROCESSING_STATS,
        sep="\t",
        index=False
    )

    log(
        f"Preprocessing statistics saved: "
        f"{PREPROCESSING_STATS}"
    )

    # --------------------------------------------------------
    # Save final datasets
    # --------------------------------------------------------

    (
        train_final,
        test_final
    ) = save_final_datasets(
        train,
        test,
        train_features,
        test_features
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    save_feature_list(
        final_features
    )

    # --------------------------------------------------------
    # Save removed feature list
    # --------------------------------------------------------

    save_removed_features(
        high_missing,
        zero_variance
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    write_summary(
        initial_feature_count,
        len(final_features),
        high_missing,
        zero_variance,
        train_final,
        test_final
    )

    log(
        f"QC summary saved: "
        f"{QC_SUMMARY}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_validation(
        train_final,
        test_final,
        final_features
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    log("")
    log("=" * 75)
    log("SCRIPT 05 v2 COMPLETED")
    log("=" * 75)

    log(
        f"Runtime: "
        f"{elapsed:.2f} seconds"
    )

    log("")
    log("FINAL FEATURE COMPOSITION:")
    log(
        f"  RDKit: "
        f"{len(rdkit_features)}"
    )
    log(
        f"  3D: "
        f"{len(three_d_features)}"
    )
    log(
        f"  Final: "
        f"{len(final_features)}"
    )

    log("")
    log("OUTPUTS:")
    log(
        TRAIN_OUTPUT
    )
    log(
        TEST_OUTPUT
    )
    log(
        FEATURE_LIST
    )
    log(
        REMOVED_FEATURES
    )
    log(
        PREPROCESSING_STATS
    )
    log(
        QC_SUMMARY
    )


if __name__ == "__main__":
    main()
