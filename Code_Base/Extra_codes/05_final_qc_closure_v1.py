#!/usr/bin/env python3

"""
======================================================================
SCRIPT 05 — FINAL QC CLOSURE
======================================================================

Purpose
-------
Final targeted QC before freezing the phytochemical feature matrix
for downstream machine-learning modeling.

This is NOT another full feature-engineering QC cycle.

Primary objective:
    Investigate the remaining identical feature-value pairs detected
    in TEST and determine whether they represent a real feature problem
    or merely identical values within the finite TEST dataset.

Reference principle:
    Feature-removal decisions are based primarily on TRAIN.

The script does NOT automatically remove features.

Checks
------
1. Input file existence
2. Train/test shape
3. Column-name consistency
4. Duplicate column names
5. Feature-order consistency
6. Exact duplicate feature-value columns
7. Known TEST duplicate pairs
8. TRAIN-vs-TEST comparison of suspicious features
9. Unique-value counts
10. Constant features
11. Near-zero variance features
12. NaN values
13. Infinite values
14. Duplicate np_id
15. Train/test np_id overlap
16. Activity-label distribution
17. Numeric feature integrity
18. Exact duplicate rows
19. Machine-readable JSON report
20. Human-readable TXT report
21. Suspicious-pair CSV

Output
------
05_final_qc_closure_output/
    final_qc_closure_report.json
    final_qc_closure_report.txt
    suspicious_feature_pairs.csv

======================================================================
"""

import os
import sys
import json
import math
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

TRAIN_FILE = (
    "feature_engineering/final_228_features/"
    "train_80_final_228_features.tsv"
)

TEST_FILE = (
    "feature_engineering/final_228_features/"
    "test_20_final_228_features.tsv"
)

OUTPUT_DIR = "05_final_qc_closure_output"

ID_COLUMN = "np_id"
LABEL_COLUMN = "Activity_Label"

# Current known suspicious pairs from previous QC
KNOWN_TEST_PAIRS = [
    ("fr_alkyl_carbamate", "fr_benzodiazepine"),
    ("fr_alkyl_carbamate", "fr_tetrazole"),
    ("fr_benzodiazepine", "fr_tetrazole"),
]

# Near-zero variance threshold
# A feature is flagged if one value represents >= 99.9% of observations
NZV_DOMINANCE_THRESHOLD = 0.999

# Minimum acceptable number of unique values for a non-constant
# continuous/count feature.
MIN_UNIQUE_WARNING = 2


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def safe_json_value(value):
    """Convert NumPy/Pandas values into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "INF" if value > 0 else "-INF"
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "INF" if value > 0 else "-INF"

    return value


def json_clean(obj):
    """Recursively make object JSON serializable."""

    if isinstance(obj, dict):
        return {
            str(k): json_clean(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [json_clean(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_clean(v) for v in obj]

    return safe_json_value(obj)


def sha256_file(path):
    """Calculate SHA256 hash of a file."""

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def feature_columns(df):
    """Return feature columns excluding ID and label."""

    return [
        c for c in df.columns
        if c not in {ID_COLUMN, LABEL_COLUMN}
    ]


def find_duplicate_column_names(columns):
    """Find duplicated column names."""

    seen = {}
    duplicates = {}

    for i, col in enumerate(columns):

        if col not in seen:
            seen[col] = [i]

        else:
            seen[col].append(i)

    for col, positions in seen.items():

        if len(positions) > 1:
            duplicates[col] = positions

    return duplicates


def find_identical_feature_pairs(df, features):
    """
    Find pairs of features having exactly identical values
    across all rows.
    """

    identical_pairs = []

    # Convert only feature matrix
    data = df[features]

    # Hash each column for efficient comparison
    hashes = {}

    for col in features:

        series = data[col]

        # Include values and missingness in hash
        h = pd.util.hash_pandas_object(
            series,
            index=False
        ).values

        digest = hashlib.sha256(h.tobytes()).hexdigest()

        hashes.setdefault(digest, []).append(col)

    # Confirm actual equality within hash groups
    for cols in hashes.values():

        if len(cols) < 2:
            continue

        for i in range(len(cols)):

            for j in range(i + 1, len(cols)):

                c1 = cols[i]
                c2 = cols[j]

                equal = data[c1].equals(data[c2])

                if equal:
                    identical_pairs.append(
                        (c1, c2)
                    )

    return identical_pairs


def column_profile(series):
    """Return detailed statistics for one feature."""

    n = len(series)

    missing = int(series.isna().sum())

    finite_nonmissing = None
    inf_count = 0

    if pd.api.types.is_numeric_dtype(series):

        arr = pd.to_numeric(
            series,
            errors="coerce"
        ).to_numpy(dtype=float)

        inf_count = int(np.isinf(arr).sum())

        finite_values = arr[
            np.isfinite(arr)
        ]

        if len(finite_values) > 0:
            finite_nonmissing = {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values)),
                "median": float(np.median(finite_values)),
            }

    value_counts = series.value_counts(
        dropna=False
    )

    unique_count = int(series.nunique(
        dropna=False
    ))

    dominant_count = 0
    dominant_fraction = 0.0

    if len(value_counts) > 0:

        dominant_count = int(value_counts.iloc[0])

        dominant_fraction = (
            dominant_count / n
            if n > 0
            else 0.0
        )

    return {
        "dtype": str(series.dtype),
        "n": int(n),
        "missing_count": missing,
        "missing_fraction": (
            missing / n if n > 0 else 0.0
        ),
        "infinite_count": inf_count,
        "unique_count": unique_count,
        "dominant_value_count": dominant_count,
        "dominant_fraction": dominant_fraction,
        "constant": unique_count <= 1,
        "near_zero_variance": (
            unique_count > 1
            and dominant_fraction >= NZV_DOMINANCE_THRESHOLD
        ),
        "numeric": bool(
            pd.api.types.is_numeric_dtype(series)
        ),
        "numeric_summary": finite_nonmissing,
    }


def class_distribution(df):

    if LABEL_COLUMN not in df.columns:
        return {
            "available": False
        }

    counts = df[LABEL_COLUMN].value_counts(
        dropna=False
    )

    total = len(df)

    result = {
        "available": True,
        "n": int(total),
        "counts": {},
        "fractions": {},
    }

    for key, value in counts.items():

        key_string = str(key)

        result["counts"][key_string] = int(value)

        result["fractions"][key_string] = (
            float(value / total)
            if total > 0
            else 0.0
        )

    return result


# ======================================================================
# LOAD DATA
# ======================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print_section("SCRIPT 05 — FINAL QC CLOSURE")

print("Starting final targeted QC...")
print()

if not os.path.exists(TRAIN_FILE):

    print("ERROR: TRAIN file not found:")
    print(TRAIN_FILE)
    sys.exit(1)

if not os.path.exists(TEST_FILE):

    print("ERROR: TEST file not found:")
    print(TEST_FILE)
    sys.exit(1)


# ======================================================================
# FILE METADATA
# ======================================================================

print_section("1. FILE INFORMATION")

train_hash = sha256_file(TRAIN_FILE)
test_hash = sha256_file(TEST_FILE)

print("TRAIN:")
print(" ", TRAIN_FILE)
print(" SHA256:", train_hash)

print()
print("TEST:")
print(" ", TEST_FILE)
print(" SHA256:", test_hash)


# ======================================================================
# READ DATA
# ======================================================================

print_section("2. LOADING DATA")

try:

    train = pd.read_csv(
        TRAIN_FILE,
        sep="\t",
        low_memory=False
    )

    test = pd.read_csv(
        TEST_FILE,
        sep="\t",
        low_memory=False
    )

except Exception as e:

    print("ERROR while reading files:")
    print(e)

    sys.exit(1)


print("TRAIN shape:", train.shape)
print("TEST shape :", test.shape)


# ======================================================================
# INITIAL REPORT STRUCTURE
# ======================================================================

report = {

    "metadata": {
        "script": "05_final_qc_closure_v1.py",
        "timestamp": datetime.now().isoformat(),
        "train_file": TRAIN_FILE,
        "test_file": TEST_FILE,
        "train_sha256": train_hash,
        "test_sha256": test_hash,
        "id_column": ID_COLUMN,
        "label_column": LABEL_COLUMN,
    },

    "dataset": {},

    "column_qc": {},

    "feature_qc": {},

    "duplicate_feature_analysis": {},

    "known_suspicious_pairs": [],

    "identifier_qc": {},

    "label_qc": {},

    "numeric_qc": {},

    "final_decision": {},

}


# ======================================================================
# DATASET SHAPE
# ======================================================================

print_section("3. DATASET SHAPE")

train_features = feature_columns(train)
test_features = feature_columns(test)

print("TRAIN rows:", len(train))
print("TEST rows :", len(test))

print("TRAIN features:", len(train_features))
print("TEST features :", len(test_features))

report["dataset"] = {

    "train_rows": int(len(train)),
    "test_rows": int(len(test)),

    "train_columns": int(len(train.columns)),
    "test_columns": int(len(test.columns)),

    "train_features": int(len(train_features)),
    "test_features": int(len(test_features)),

}


# ======================================================================
# COLUMN QC
# ======================================================================

print_section("4. COLUMN STRUCTURE QC")

train_duplicate_names = find_duplicate_column_names(
    list(train.columns)
)

test_duplicate_names = find_duplicate_column_names(
    list(test.columns)
)

print("TRAIN duplicate column names:")

if train_duplicate_names:
    print(train_duplicate_names)
else:
    print("NONE")

print()
print("TEST duplicate column names:")

if test_duplicate_names:
    print(test_duplicate_names)
else:
    print("NONE")


same_feature_set = (
    set(train_features) == set(test_features)
)

same_feature_order = (
    train_features == test_features
)

train_only = sorted(
    set(train_features) - set(test_features)
)

test_only = sorted(
    set(test_features) - set(train_features)
)


print()
print("Same feature set:", same_feature_set)
print("Same feature order:", same_feature_order)

if train_only:
    print("TRAIN-only features:", train_only)

if test_only:
    print("TEST-only features:", test_only)


report["column_qc"] = {

    "train_duplicate_column_names": train_duplicate_names,
    "test_duplicate_column_names": test_duplicate_names,

    "same_feature_set": same_feature_set,
    "same_feature_order": same_feature_order,

    "train_only_features": train_only,
    "test_only_features": test_only,

}


# ======================================================================
# IDENTICAL FEATURE-VALUE ANALYSIS
# ======================================================================

print_section("5. IDENTICAL FEATURE-VALUE ANALYSIS")

train_identical_pairs = find_identical_feature_pairs(
    train,
    train_features
)

test_identical_pairs = find_identical_feature_pairs(
    test,
    test_features
)

print("TRAIN identical pairs:")

if train_identical_pairs:
    for a, b in train_identical_pairs:
        print(" ", a, "<-->", b)
else:
    print(" NONE")

print()
print("TEST identical pairs:")

if test_identical_pairs:
    for a, b in test_identical_pairs:
        print(" ", a, "<-->", b)
else:
    print(" NONE")


report["duplicate_feature_analysis"] = {

    "train_identical_pairs": [
        list(x)
        for x in train_identical_pairs
    ],

    "test_identical_pairs": [
        list(x)
        for x in test_identical_pairs
    ],

    "train_identical_pair_count": len(
        train_identical_pairs
    ),

    "test_identical_pair_count": len(
        test_identical_pairs
    ),

}


# ======================================================================
# ANALYSE KNOWN SUSPICIOUS PAIRS
# ======================================================================

print_section("6. DETAILED ANALYSIS OF KNOWN TEST PAIRS")

pair_rows = []

for feature_a, feature_b in KNOWN_TEST_PAIRS:

    print()
    print("-" * 70)

    print(
        f"PAIR: {feature_a} <--> {feature_b}"
    )

    row = {

        "feature_a": feature_a,
        "feature_b": feature_b,

        "present_train_a": feature_a in train.columns,
        "present_train_b": feature_b in train.columns,

        "present_test_a": feature_a in test.columns,
        "present_test_b": feature_b in test.columns,

    }

    if (
        feature_a not in train.columns
        or feature_b not in train.columns
        or feature_a not in test.columns
        or feature_b not in test.columns
    ):

        print("ERROR: One or more features missing.")

        row["status"] = "MISSING_FEATURE"

        pair_rows.append(row)

        report["known_suspicious_pairs"].append(
            row
        )

        continue


    train_a = train[feature_a]
    train_b = train[feature_b]

    test_a = test[feature_a]
    test_b = test[feature_b]


    # --------------------------------------------------------------
    # TRAIN equality
    # --------------------------------------------------------------

    train_identical = train_a.equals(
        train_b
    )

    test_identical = test_a.equals(
        test_b
    )


    # --------------------------------------------------------------
    # Unique values
    # --------------------------------------------------------------

    train_a_unique = int(
        train_a.nunique(dropna=False)
    )

    train_b_unique = int(
        train_b.nunique(dropna=False)
    )

    test_a_unique = int(
        test_a.nunique(dropna=False)
    )

    test_b_unique = int(
        test_b.nunique(dropna=False)
    )


    # --------------------------------------------------------------
    # Correlation
    # --------------------------------------------------------------

    train_corr = None
    test_corr = None

    if (
        pd.api.types.is_numeric_dtype(train_a)
        and pd.api.types.is_numeric_dtype(train_b)
    ):

        train_corr = train_a.corr(
            train_b
        )

    if (
        pd.api.types.is_numeric_dtype(test_a)
        and pd.api.types.is_numeric_dtype(test_b)
    ):

        test_corr = test_a.corr(
            test_b
        )


    # --------------------------------------------------------------
    # Difference analysis
    # --------------------------------------------------------------

    train_difference_count = None
    test_difference_count = None

    if (
        pd.api.types.is_numeric_dtype(train_a)
        and pd.api.types.is_numeric_dtype(train_b)
    ):

        train_difference_count = int(
            (train_a != train_b).sum()
        )

    if (
        pd.api.types.is_numeric_dtype(test_a)
        and pd.api.types.is_numeric_dtype(test_b)
    ):

        test_difference_count = int(
            (test_a != test_b).sum()
        )


    # --------------------------------------------------------------
    # Profiles
    # --------------------------------------------------------------

    train_profile_a = column_profile(
        train_a
    )

    train_profile_b = column_profile(
        train_b
    )

    test_profile_a = column_profile(
        test_a
    )

    test_profile_b = column_profile(
        test_b
    )


    print()
    print("TRAIN")
    print(
        f" {feature_a}: "
        f"unique={train_a_unique}, "
        f"constant={train_profile_a['constant']}, "
        f"NZV={train_profile_a['near_zero_variance']}"
    )

    print(
        f" {feature_b}: "
        f"unique={train_b_unique}, "
        f"constant={train_profile_b['constant']}, "
        f"NZV={train_profile_b['near_zero_variance']}"
    )

    print(
        " Identical:",
        train_identical
    )

    print(
        " Correlation:",
        train_corr
    )


    print()
    print("TEST")
    print(
        f" {feature_a}: "
        f"unique={test_a_unique}, "
        f"constant={test_profile_a['constant']}, "
        f"NZV={test_profile_a['near_zero_variance']}"
    )

    print(
        f" {feature_b}: "
        f"unique={test_b_unique}, "
        f"constant={test_profile_b['constant']}, "
        f"NZV={test_profile_b['near_zero_variance']}"
    )

    print(
        " Identical:",
        test_identical
    )

    print(
        " Correlation:",
        test_corr
    )


    # --------------------------------------------------------------
    # Determine interpretation
    # --------------------------------------------------------------

    if train_identical:

        interpretation = (
            "TRUE_TRAIN_REDUNDANCY"
        )

    elif test_identical:

        if (
            train_profile_a["constant"]
            or train_profile_b["constant"]
        ):

            interpretation = (
                "TEST_IDENTITY_WITH_TRAIN_CONSTANT_FEATURE"
            )

        elif (
            train_profile_a["near_zero_variance"]
            or train_profile_b["near_zero_variance"]
        ):

            interpretation = (
                "TEST_IDENTITY_WITH_TRAIN_NZV_FEATURE"
            )

        else:

            interpretation = (
                "TEST_ONLY_VALUE_IDENTITY"
            )

    else:

        interpretation = (
            "NO_CURRENT_IDENTITY"
        )


    row.update({

        "train_identical": train_identical,
        "test_identical": test_identical,

        "train_unique_a": train_a_unique,
        "train_unique_b": train_b_unique,

        "test_unique_a": test_a_unique,
        "test_unique_b": test_b_unique,

        "train_difference_count": train_difference_count,
        "test_difference_count": test_difference_count,

        "train_correlation": train_corr,
        "test_correlation": test_corr,

        "train_a_constant": train_profile_a["constant"],
        "train_b_constant": train_profile_b["constant"],

        "train_a_nzv": train_profile_a[
            "near_zero_variance"
        ],

        "train_b_nzv": train_profile_b[
            "near_zero_variance"
        ],

        "test_a_constant": test_profile_a["constant"],
        "test_b_constant": test_profile_b["constant"],

        "test_a_nzv": test_profile_a[
            "near_zero_variance"
        ],

        "test_b_nzv": test_profile_b[
            "near_zero_variance"
        ],

        "interpretation": interpretation,

    })


    pair_rows.append(row)

    report["known_suspicious_pairs"].append(
        row
    )


# ======================================================================
# FEATURE-WIDE QC
# ======================================================================

print_section("7. FEATURE-WIDE QUALITY CONTROL")


def feature_qc(df, features, dataset_name):

    results = {

        "dataset": dataset_name,

        "feature_count": len(features),

        "constant_features": [],

        "near_zero_variance_features": [],

        "features_with_missing": [],

        "features_with_infinite": [],

        "non_numeric_features": [],

    }


    for col in features:

        s = df[col]

        profile = column_profile(s)


        if profile["constant"]:
            results["constant_features"].append(
                col
            )

        if profile["near_zero_variance"]:
            results[
                "near_zero_variance_features"
            ].append(col)

        if profile["missing_count"] > 0:
            results["features_with_missing"].append(
                col
            )

        if profile["infinite_count"] > 0:
            results["features_with_infinite"].append(
                col
            )

        if not profile["numeric"]:
            results["non_numeric_features"].append(
                col
            )


    return results


train_feature_qc = feature_qc(
    train,
    train_features,
    "TRAIN"
)

test_feature_qc = feature_qc(
    test,
    test_features,
    "TEST"
)


print()
print("TRAIN")
print(
    " Constant:",
    len(train_feature_qc["constant_features"])
)

print(
    " Near-zero variance:",
    len(
        train_feature_qc[
            "near_zero_variance_features"
        ]
    )
)

print(
    " Missing:",
    len(
        train_feature_qc[
            "features_with_missing"
        ]
    )
)

print(
    " Infinite:",
    len(
        train_feature_qc[
            "features_with_infinite"
        ]
    )
)

print(
    " Non-numeric:",
    len(
        train_feature_qc[
            "non_numeric_features"
        ]
    )
)


print()
print("TEST")
print(
    " Constant:",
    len(test_feature_qc["constant_features"])
)

print(
    " Near-zero variance:",
    len(
        test_feature_qc[
            "near_zero_variance_features"
        ]
    )
)

print(
    " Missing:",
    len(
        test_feature_qc[
            "features_with_missing"
        ]
    )
)

print(
    " Infinite:",
    len(
        test_feature_qc[
            "features_with_infinite"
        ]
    )
)

print(
    " Non-numeric:",
    len(
        test_feature_qc[
            "non_numeric_features"
        ]
    )
)


report["feature_qc"] = {

    "train": train_feature_qc,
    "test": test_feature_qc,

}


# ======================================================================
# IDENTIFIER QC
# ======================================================================

print_section("8. IDENTIFIER / DATA-LEAKAGE QC")

identifier_report = {

    "id_column_present_train": (
        ID_COLUMN in train.columns
    ),

    "id_column_present_test": (
        ID_COLUMN in test.columns
    ),

    "train_duplicate_ids": 0,

    "test_duplicate_ids": 0,

    "train_test_id_overlap_count": 0,

}


if (
    ID_COLUMN in train.columns
    and ID_COLUMN in test.columns
):

    train_duplicate_ids = int(
        train[ID_COLUMN].duplicated().sum()
    )

    test_duplicate_ids = int(
        test[ID_COLUMN].duplicated().sum()
    )

    train_ids = set(
        train[ID_COLUMN].astype(str)
    )

    test_ids = set(
        test[ID_COLUMN].astype(str)
    )

    overlap = train_ids.intersection(
        test_ids
    )

    identifier_report[
        "train_duplicate_ids"
    ] = train_duplicate_ids

    identifier_report[
        "test_duplicate_ids"
    ] = test_duplicate_ids

    identifier_report[
        "train_test_id_overlap_count"
    ] = len(overlap)

    print(
        "TRAIN duplicate np_id:",
        train_duplicate_ids
    )

    print(
        "TEST duplicate np_id:",
        test_duplicate_ids
    )

    print(
        "TRAIN/TEST np_id overlap:",
        len(overlap)
    )

    if overlap:

        print()
        print("WARNING: overlapping IDs detected.")

        identifier_report[
            "overlap_examples"
        ] = sorted(
            list(overlap)
        )[:20]


report["identifier_qc"] = identifier_report


# ======================================================================
# LABEL QC
# ======================================================================

print_section("9. ACTIVITY LABEL QC")

train_label_qc = class_distribution(
    train
)

test_label_qc = class_distribution(
    test
)

print("TRAIN:")
print(
    json.dumps(
        train_label_qc,
        indent=2
    )
)

print()
print("TEST:")
print(
    json.dumps(
        test_label_qc,
        indent=2
    )
)

report["label_qc"] = {

    "train": train_label_qc,
    "test": test_label_qc,

}


# ======================================================================
# EXACT DUPLICATE ROWS
# ======================================================================

print_section("10. EXACT DUPLICATE ROW QC")

train_duplicate_rows = int(
    train.duplicated().sum()
)

test_duplicate_rows = int(
    test.duplicated().sum()
)

train_duplicate_feature_rows = int(
    train[train_features].duplicated().sum()
)

test_duplicate_feature_rows = int(
    test[test_features].duplicated().sum()
)

print(
    "TRAIN exact duplicate rows:",
    train_duplicate_rows
)

print(
    "TEST exact duplicate rows:",
    test_duplicate_rows
)

print(
    "TRAIN duplicate feature vectors:",
    train_duplicate_feature_rows
)

print(
    "TEST duplicate feature vectors:",
    test_duplicate_feature_rows
)

report["feature_qc"]["duplicate_rows"] = {

    "train_full_row_duplicates":
        train_duplicate_rows,

    "test_full_row_duplicates":
        test_duplicate_rows,

    "train_feature_vector_duplicates":
        train_duplicate_feature_rows,

    "test_feature_vector_duplicates":
        test_duplicate_feature_rows,

}


# ======================================================================
# NUMERIC QC
# ======================================================================

print_section("11. NUMERIC FEATURE MATRIX QC")

train_numeric_problems = []

test_numeric_problems = []

for col in train_features:

    if not pd.api.types.is_numeric_dtype(
        train[col]
    ):

        train_numeric_problems.append(
            col
        )

for col in test_features:

    if not pd.api.types.is_numeric_dtype(
        test[col]
    ):

        test_numeric_problems.append(
            col
        )

print(
    "TRAIN non-numeric feature count:",
    len(train_numeric_problems)
)

print(
    "TEST non-numeric feature count:",
    len(test_numeric_problems)
)

report["numeric_qc"] = {

    "train_non_numeric_features":
        train_numeric_problems,

    "test_non_numeric_features":
        test_numeric_problems,

}


# ======================================================================
# FINAL DECISION LOGIC
# ======================================================================

print_section("12. FINAL QC DECISION")

critical_failures = []

warnings = []


# --------------------------------------------------------------
# Duplicate column names
# --------------------------------------------------------------

if train_duplicate_names:
    critical_failures.append(
        "TRAIN contains duplicate column names."
    )

if test_duplicate_names:
    critical_failures.append(
        "TEST contains duplicate column names."
    )


# --------------------------------------------------------------
# Feature mismatch
# --------------------------------------------------------------

if not same_feature_set:

    critical_failures.append(
        "TRAIN and TEST feature sets differ."
    )


# --------------------------------------------------------------
# Order mismatch
# --------------------------------------------------------------

if not same_feature_order:

    critical_failures.append(
        "TRAIN and TEST feature order differs."
    )


# --------------------------------------------------------------
# Missing / Inf
# --------------------------------------------------------------

if (
    train_feature_qc[
        "features_with_missing"
    ]
):

    critical_failures.append(
        "TRAIN contains missing feature values."
    )

if (
    test_feature_qc[
        "features_with_missing"
    ]
):

    critical_failures.append(
        "TEST contains missing feature values."
    )

if (
    train_feature_qc[
        "features_with_infinite"
    ]
):

    critical_failures.append(
        "TRAIN contains infinite feature values."
    )

if (
    test_feature_qc[
        "features_with_infinite"
    ]
):

    critical_failures.append(
        "TEST contains infinite feature values."
    )


# --------------------------------------------------------------
# Non-numeric features
# --------------------------------------------------------------

if train_numeric_problems:

    critical_failures.append(
        "TRAIN contains non-numeric feature columns."
    )

if test_numeric_problems:

    critical_failures.append(
        "TEST contains non-numeric feature columns."
    )


# --------------------------------------------------------------
# ID overlap
# --------------------------------------------------------------

if (
    identifier_report[
        "train_test_id_overlap_count"
    ] > 0
):

    critical_failures.append(
        "TRAIN/TEST compound ID overlap detected."
    )


# --------------------------------------------------------------
# TRAIN identical feature pairs
# --------------------------------------------------------------

if train_identical_pairs:

    critical_failures.append(
        "TRAIN contains identical feature-value pairs."
    )


# --------------------------------------------------------------
# TRAIN constant features
# --------------------------------------------------------------

if train_feature_qc[
    "constant_features"
]:

    critical_failures.append(
        "TRAIN contains constant features."
    )


# --------------------------------------------------------------
# TRAIN NZV
# --------------------------------------------------------------

if train_feature_qc[
    "near_zero_variance_features"
]:

    warnings.append(
        "TRAIN contains near-zero-variance features; "
        "review before final model fitting."
    )


# --------------------------------------------------------------
# TEST-only identical pairs
# --------------------------------------------------------------

if test_identical_pairs and not train_identical_pairs:

    warnings.append(
        "TEST contains feature pairs that are identical "
        "within TEST but not within TRAIN. These should "
        "NOT automatically be removed."
    )


# --------------------------------------------------------------
# Final state
# --------------------------------------------------------------

if critical_failures:

    final_status = "FAIL"

elif warnings:

    final_status = "PASS_WITH_REVIEW"

else:

    final_status = "PASS"


print()
print("FINAL STATUS:", final_status)

print()

if critical_failures:

    print("CRITICAL FAILURES:")

    for item in critical_failures:
        print(" -", item)

else:

    print("CRITICAL FAILURES: NONE")


print()

if warnings:

    print("WARNINGS:")

    for item in warnings:
        print(" -", item)

else:

    print("WARNINGS: NONE")


# ======================================================================
# FEATURE REMOVAL RECOMMENDATION
# ======================================================================

print_section("13. FEATURE REMOVAL RECOMMENDATION")

if train_identical_pairs:

    removal_recommendation = (
        "REVIEW_REQUIRED: TRAIN contains identical "
        "feature-value pairs. Investigate before modeling."
    )

elif train_feature_qc[
    "constant_features"
]:

    removal_recommendation = (
        "REVIEW_REQUIRED: TRAIN contains constant features. "
        "Remove constant features before modeling."
    )

elif train_feature_qc[
    "near_zero_variance_features"
]:

    removal_recommendation = (
        "REVIEW_REQUIRED: TRAIN contains near-zero-variance "
        "features. Review whether these should be removed."
    )

elif test_identical_pairs:

    removal_recommendation = (
        "DO_NOT_REMOVE_FOR_TEST_IDENTITY: The identical "
        "TEST pairs are not identical in TRAIN. Their "
        "TEST-only identity does not by itself justify "
        "feature removal."
    )

else:

    removal_recommendation = (
        "NO_FEATURE_REMOVAL_REQUIRED_FROM_IDENTICALITY_QC."
    )


print(removal_recommendation)


report["final_decision"] = {

    "status": final_status,

    "critical_failures": critical_failures,

    "warnings": warnings,

    "feature_removal_recommendation":
        removal_recommendation,

    "ready_to_freeze": (
        len(critical_failures) == 0
        and not train_feature_qc[
            "constant_features"
        ]
    ),

}


# ======================================================================
# SAVE SUSPICIOUS PAIR CSV
# ======================================================================

print_section("14. WRITING MACHINE-READABLE OUTPUT")

pair_csv = os.path.join(
    OUTPUT_DIR,
    "suspicious_feature_pairs.csv"
)

pair_df = pd.DataFrame(
    pair_rows
)

pair_df.to_csv(
    pair_csv,
    index=False
)

print(
    "Written:",
    pair_csv
)


# ======================================================================
# SAVE JSON REPORT
# ======================================================================

json_file = os.path.join(
    OUTPUT_DIR,
    "final_qc_closure_report.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_clean(report),
        f,
        indent=2,
        allow_nan=False
    )

print(
    "Written:",
    json_file
)


# ======================================================================
# HUMAN-READABLE REPORT
# ======================================================================

txt_file = os.path.join(
    OUTPUT_DIR,
    "final_qc_closure_report.txt"
)

with open(
    txt_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 78 + "\n"
    )

    f.write(
        "SCRIPT 05 — FINAL QC CLOSURE REPORT\n"
    )

    f.write(
        "=" * 78 + "\n\n"
    )

    f.write(
        f"Generated: "
        f"{report['metadata']['timestamp']}\n\n"
    )

    f.write(
        "INPUT FILES\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"TRAIN: {TRAIN_FILE}\n"
    )

    f.write(
        f"TEST : {TEST_FILE}\n\n"
    )

    f.write(
        "DATASET\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"TRAIN rows: "
        f"{len(train)}\n"
    )

    f.write(
        f"TEST rows : "
        f"{len(test)}\n"
    )

    f.write(
        f"TRAIN features: "
        f"{len(train_features)}\n"
    )

    f.write(
        f"TEST features : "
        f"{len(test_features)}\n\n"
    )


    f.write(
        "COLUMN QC\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"TRAIN duplicate column names: "
        f"{len(train_duplicate_names)}\n"
    )

    f.write(
        f"TEST duplicate column names: "
        f"{len(test_duplicate_names)}\n"
    )

    f.write(
        f"Same feature set: "
        f"{same_feature_set}\n"
    )

    f.write(
        f"Same feature order: "
        f"{same_feature_order}\n\n"
    )


    f.write(
        "IDENTICAL FEATURE-VALUE COLUMNS\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"TRAIN pairs: "
        f"{len(train_identical_pairs)}\n"
    )

    f.write(
        f"TEST pairs: "
        f"{len(test_identical_pairs)}\n\n"
    )

    if train_identical_pairs:

        for a, b in train_identical_pairs:

            f.write(
                f"TRAIN: {a} <--> {b}\n"
            )

    if test_identical_pairs:

        for a, b in test_identical_pairs:

            f.write(
                f"TEST: {a} <--> {b}\n"
            )

    f.write("\n")


    f.write(
        "TRAIN FEATURE QC\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"Constant features: "
        f"{len(train_feature_qc['constant_features'])}\n"
    )

    f.write(
        f"Near-zero variance: "
        f"{len(train_feature_qc['near_zero_variance_features'])}\n"
    )

    f.write(
        f"Features with missing values: "
        f"{len(train_feature_qc['features_with_missing'])}\n"
    )

    f.write(
        f"Features with infinite values: "
        f"{len(train_feature_qc['features_with_infinite'])}\n"
    )

    f.write(
        f"Non-numeric features: "
        f"{len(train_feature_qc['non_numeric_features'])}\n\n"
    )


    f.write(
        "TEST FEATURE QC\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"Constant features: "
        f"{len(test_feature_qc['constant_features'])}\n"
    )

    f.write(
        f"Near-zero variance: "
        f"{len(test_feature_qc['near_zero_variance_features'])}\n"
    )

    f.write(
        f"Features with missing values: "
        f"{len(test_feature_qc['features_with_missing'])}\n"
    )

    f.write(
        f"Features with infinite values: "
        f"{len(test_feature_qc['features_with_infinite'])}\n"
    )

    f.write(
        f"Non-numeric features: "
        f"{len(test_feature_qc['non_numeric_features'])}\n\n"
    )


    f.write(
        "IDENTIFIER QC\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"TRAIN duplicate IDs: "
        f"{identifier_report['train_duplicate_ids']}\n"
    )

    f.write(
        f"TEST duplicate IDs: "
        f"{identifier_report['test_duplicate_ids']}\n"
    )

    f.write(
        f"TRAIN/TEST ID overlap: "
        f"{identifier_report['train_test_id_overlap_count']}\n\n"
    )


    f.write(
        "DUPLICATE ROW QC\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"TRAIN full-row duplicates: "
        f"{train_duplicate_rows}\n"
    )

    f.write(
        f"TEST full-row duplicates: "
        f"{test_duplicate_rows}\n"
    )

    f.write(
        f"TRAIN duplicate feature vectors: "
        f"{train_duplicate_feature_rows}\n"
    )

    f.write(
        f"TEST duplicate feature vectors: "
        f"{test_duplicate_feature_rows}\n\n"
    )


    f.write(
        "FINAL DECISION\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"STATUS: {final_status}\n\n"
    )

    if critical_failures:

        f.write(
            "CRITICAL FAILURES:\n"
        )

        for item in critical_failures:

            f.write(
                f" - {item}\n"
            )

        f.write("\n")


    if warnings:

        f.write(
            "WARNINGS:\n"
        )

        for item in warnings:

            f.write(
                f" - {item}\n"
            )

        f.write("\n")


    f.write(
        "FEATURE REMOVAL RECOMMENDATION:\n"
    )

    f.write(
        removal_recommendation
        + "\n\n"
    )


    f.write(
        "FREEZE RECOMMENDATION:\n"
    )

    if report["final_decision"][
        "ready_to_freeze"
    ]:

        f.write(
            "YES — feature matrix can proceed "
            "to downstream modeling, subject to "
            "review of warnings above.\n"
        )

    else:

        f.write(
            "NO — resolve critical QC failures "
            "before downstream modeling.\n"
        )


# ======================================================================
# FINAL CONSOLE SUMMARY
# ======================================================================

print_section("FINAL SUMMARY")

print("STATUS:", final_status)

print()
print(
    "TRAIN identical feature pairs:",
    len(train_identical_pairs)
)

print(
    "TEST identical feature pairs:",
    len(test_identical_pairs)
)

print(
    "TRAIN constant features:",
    len(
        train_feature_qc[
            "constant_features"
        ]
    )
)

print(
    "TRAIN near-zero-variance features:",
    len(
        train_feature_qc[
            "near_zero_variance_features"
        ]
    )
)

print(
    "TRAIN missing-value features:",
    len(
        train_feature_qc[
            "features_with_missing"
        ]
    )
)

print(
    "TEST missing-value features:",
    len(
        test_feature_qc[
            "features_with_missing"
        ]
    )
)

print(
    "TRAIN/TEST ID overlap:",
    identifier_report[
        "train_test_id_overlap_count"
    ]
)

print()
print(
    "FEATURE REMOVAL:"
)

print(
    removal_recommendation
)

print()
print(
    "Reports:"
)

print(
    " ",
    json_file
)

print(
    " ",
    txt_file
)

print(
    " ",
    pair_csv
)

print()
print("=" * 78)
print("SCRIPT 05 FINAL QC CLOSURE COMPLETE")
print("=" * 78)
