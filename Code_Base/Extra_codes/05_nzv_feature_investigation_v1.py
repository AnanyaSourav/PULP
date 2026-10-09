#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 05 — NZV FEATURE INVESTIGATION v1
==============================================================================

Purpose
-------
Investigate the 22 near-zero-variance (NZV) features identified by
05_final_qc_closure.py.

IMPORTANT:
    This script DOES NOT remove features.

It evaluates:
    1. Feature identity
    2. Unique values
    3. Value frequencies
    4. Dominant-value prevalence
    5. Variance
    6. Train/test behavior
    7. Constant status
    8. NZV status
    9. Activity association
   10. Feature redundancy
   11. Test-set stability
   12. Duplicate-value behavior
   13. Final KEEP / REVIEW / REMOVE recommendation

INPUT
-----
feature_engineering/final_228_features/train_80_final_228_features.tsv
feature_engineering/final_228_features/test_20_final_228_features.tsv

OUTPUT
------
05_nzv_feature_investigation_output/
    nzv_feature_investigation_report.txt
    nzv_feature_investigation_report.json
    nzv_feature_analysis.tsv
    nzv_value_frequencies.tsv
    nzv_activity_association.tsv
    nzv_redundancy.tsv
    nzv_test_behavior.tsv

Decision philosophy
-------------------
NZV != automatic removal.

A feature will only receive REMOVE when there is strong evidence that
it contains no useful information.

Features that are sparse but potentially biologically meaningful are
flagged for REVIEW rather than automatically removed.

==============================================================================
"""

import os
import json
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_FILE = (
    "feature_engineering/final_228_features/"
    "train_80_final_228_features.tsv"
)

TEST_FILE = (
    "feature_engineering/final_228_features/"
    "test_20_final_228_features.tsv"
)

OUTPUT_DIR = "05_nzv_feature_investigation_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ANALYSIS_FILE = os.path.join(
    OUTPUT_DIR,
    "nzv_feature_analysis.tsv"
)

VALUE_FREQ_FILE = os.path.join(
    OUTPUT_DIR,
    "nzv_value_frequencies.tsv"
)

ACTIVITY_FILE = os.path.join(
    OUTPUT_DIR,
    "nzv_activity_association.tsv"
)

REDUNDANCY_FILE = os.path.join(
    OUTPUT_DIR,
    "nzv_redundancy.tsv"
)

TEST_BEHAVIOR_FILE = os.path.join(
    OUTPUT_DIR,
    "nzv_test_behavior.tsv"
)

REPORT_TXT = os.path.join(
    OUTPUT_DIR,
    "nzv_feature_investigation_report.txt"
)

REPORT_JSON = os.path.join(
    OUTPUT_DIR,
    "nzv_feature_investigation_report.json"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def is_numeric_series(series):
    converted = pd.to_numeric(series, errors="coerce")

    return converted.notna().all()


def calculate_nzv(series):
    """
    Practical NZV definition used here:

    - exactly one unique value -> CONSTANT
    - two or more unique values where the most frequent value dominates
      strongly -> NZV

    The original QC already identified the NZV features.
    This function is used to characterize them.
    """

    s = series.dropna()

    if len(s) == 0:
        return {
            "unique": 0,
            "constant": False,
            "nzv": False,
            "dominant_fraction": np.nan,
            "second_fraction": np.nan,
            "frequency_ratio": np.nan,
        }

    counts = s.value_counts(dropna=False)

    unique = len(counts)

    dominant_fraction = counts.iloc[0] / len(s)

    if unique == 1:
        return {
            "unique": 1,
            "constant": True,
            "nzv": False,
            "dominant_fraction": dominant_fraction,
            "second_fraction": 0.0,
            "frequency_ratio": np.inf,
        }

    second_fraction = counts.iloc[1] / len(s)

    if second_fraction == 0:
        ratio = np.inf
    else:
        ratio = counts.iloc[0] / counts.iloc[1]

    # Conservative characterization threshold
    nzv = (
        dominant_fraction >= 0.95
        and ratio >= 19
    )

    return {
        "unique": unique,
        "constant": False,
        "nzv": nzv,
        "dominant_fraction": dominant_fraction,
        "second_fraction": second_fraction,
        "frequency_ratio": ratio,
    }


def point_biserial_like(feature, labels):
    """
    For binary activity labels:

    Active = 1
    Inactive = 0

    Pearson correlation between numeric feature and binary label
    is used as a simple association measure.

    This is descriptive only.
    """

    x = safe_numeric(feature)
    y = labels.map({
        "Active": 1,
        "Inactive": 0,
        1: 1,
        0: 0
    })

    valid = x.notna() & y.notna()

    if valid.sum() < 3:
        return np.nan

    if x.loc[valid].nunique() < 2:
        return np.nan

    if y.loc[valid].nunique() < 2:
        return np.nan

    return x.loc[valid].corr(y.loc[valid])


def activity_value_table(feature, labels):
    """
    Calculate active/inactive counts and rates for each distinct value.
    """

    temp = pd.DataFrame({
        "value": feature,
        "activity": labels
    })

    rows = []

    for value, group in temp.groupby(
        "value",
        dropna=False
    ):

        n = len(group)

        active = (
            group["activity"]
            .astype(str)
            .str.lower()
            .eq("active")
            .sum()
        )

        inactive = (
            group["activity"]
            .astype(str)
            .str.lower()
            .eq("inactive")
            .sum()
        )

        rows.append({
            "value": value,
            "n": n,
            "fraction": n / len(temp),
            "active": int(active),
            "inactive": int(inactive),
            "active_fraction": active / n if n else np.nan,
        })

    return pd.DataFrame(rows)


def classify_feature(
    train_unique,
    train_dominant,
    train_second,
    train_variance,
    test_unique,
    test_constant,
    activity_corr,
    duplicate_with_test=False
):

    # ------------------------------------------------------------------
    # REMOVE CONDITIONS
    # ------------------------------------------------------------------

    # Truly constant in both train and test
    if train_unique == 1 and test_constant:
        return "REMOVE"

    # No variation in train
    if train_unique == 1:
        return "REMOVE"

    # ------------------------------------------------------------------
    # REVIEW CONDITIONS
    # ------------------------------------------------------------------

    # Feature becomes constant in test
    if test_constant:
        return "REVIEW"

    # Extremely dominated but not constant
    if train_dominant >= 0.995:
        return "REVIEW"

    # Activity association exists
    if (
        pd.notna(activity_corr)
        and abs(activity_corr) >= 0.02
    ):
        return "KEEP"

    # Very sparse but potentially informative
    if train_dominant >= 0.95:
        return "REVIEW"

    # Default for non-constant NZV
    return "REVIEW"


# ============================================================================
# HEADER
# ============================================================================

print("=" * 78)
print("SCRIPT 05 — NZV FEATURE INVESTIGATION v1")
print("=" * 78)

print()
print("Purpose:")
print("  Investigate the 22 TRAIN NZV features.")
print("  NO FEATURES WILL BE REMOVED.")
print()

start_time = datetime.now()


# ============================================================================
# 1. FILE INFORMATION
# ============================================================================

print("=" * 78)
print("1. FILE INFORMATION")
print("=" * 78)

print("TRAIN:")
print(f"  {TRAIN_FILE}")

print("TEST:")
print(f"  {TEST_FILE}")

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"TRAIN file not found:\n{TRAIN_FILE}"
    )

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"TEST file not found:\n{TEST_FILE}"
    )

train_sha = sha256_file(TRAIN_FILE)
test_sha = sha256_file(TEST_FILE)

print()
print("TRAIN SHA256:")
print(f"  {train_sha}")

print("TEST SHA256:")
print(f"  {test_sha}")


# ============================================================================
# 2. LOAD DATA
# ============================================================================

print()
print("=" * 78)
print("2. LOADING DATA")
print("=" * 78)

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

print(f"TRAIN shape: {train.shape}")
print(f"TEST shape : {test.shape}")


# ============================================================================
# 3. IDENTIFY LABEL AND FEATURE COLUMNS
# ============================================================================

print()
print("=" * 78)
print("3. IDENTIFYING COLUMNS")
print("=" * 78)

possible_id = [
    "np_id",
    "id",
    "ID",
    "compound_id"
]

possible_activity = [
    "Activity_Label",
    "activity",
    "Activity",
    "bioactivity",
    "Bioactivity",
    "label",
    "Label",
    "class",
    "Class"
]

id_col = None

for col in possible_id:
    if col in train.columns:
        id_col = col
        break

activity_col = None

for col in possible_activity:
    if col in train.columns:
        activity_col = col
        break

print(f"ID column       : {id_col}")
print(f"Activity column : {activity_col}")

if activity_col is None:
    raise RuntimeError(
        "Could not identify activity column."
    )

metadata_columns = set()

if id_col is not None:
    metadata_columns.add(id_col)

metadata_columns.add(activity_col)

feature_columns = [
    c for c in train.columns
    if c not in metadata_columns
]

print(f"Total feature columns: {len(feature_columns)}")


# ============================================================================
# 4. IDENTIFY NZV FEATURES
# ============================================================================

print()
print("=" * 78)
print("4. IDENTIFYING TRAIN NZV FEATURES")
print("=" * 78)

nzv_features = []

for feature in feature_columns:

    info = calculate_nzv(train[feature])

    if info["nzv"]:
        nzv_features.append(feature)

print(f"NZV features detected: {len(nzv_features)}")

for i, feature in enumerate(nzv_features, 1):
    print(f"  {i:02d}. {feature}")


# ============================================================================
# 5. DETAILED FEATURE ANALYSIS
# ============================================================================

print()
print("=" * 78)
print("5. DETAILED NZV FEATURE ANALYSIS")
print("=" * 78)

analysis_rows = []
value_frequency_rows = []
activity_rows = []
test_rows = []


for idx, feature in enumerate(nzv_features, 1):

    print()
    print("-" * 70)
    print(f"[{idx}/{len(nzv_features)}] {feature}")
    print("-" * 70)

    train_s = train[feature]
    test_s = test[feature]

    train_numeric = safe_numeric(train_s)
    test_numeric = safe_numeric(test_s)

    train_info = calculate_nzv(train_s)
    test_info = calculate_nzv(test_s)

    train_unique = train_s.nunique(dropna=True)
    test_unique = test_s.nunique(dropna=True)

    train_missing = int(train_s.isna().sum())
    test_missing = int(test_s.isna().sum())

    train_inf = int(
        np.isinf(train_numeric).sum()
    )

    test_inf = int(
        np.isinf(test_numeric).sum()
    )

    if is_numeric_series(train_s):

        train_variance = float(
            train_numeric.var()
        )

        train_mean = float(
            train_numeric.mean()
        )

        train_min = float(
            train_numeric.min()
        )

        train_max = float(
            train_numeric.max()
        )

        train_std = float(
            train_numeric.std()
        )

        test_variance = float(
            test_numeric.var()
        )

    else:

        train_variance = np.nan
        train_mean = np.nan
        train_min = np.nan
        train_max = np.nan
        train_std = np.nan
        test_variance = np.nan

    # ------------------------------------------------------------------
    # Activity association
    # ------------------------------------------------------------------

    activity_corr = point_biserial_like(
        train_s,
        train[activity_col]
    )

    # ------------------------------------------------------------------
    # Value frequencies
    # ------------------------------------------------------------------

    counts = train_s.value_counts(
        dropna=False
    )

    for rank, (value, count) in enumerate(
        counts.items(),
        start=1
    ):

        value_frequency_rows.append({
            "feature": feature,
            "rank": rank,
            "value": value,
            "count": int(count),
            "fraction": float(count / len(train_s)),
        })

    # ------------------------------------------------------------------
    # Activity-specific value analysis
    # ------------------------------------------------------------------

    act_table = activity_value_table(
        train_s,
        train[activity_col]
    )

    for _, row in act_table.iterrows():

        activity_rows.append({
            "feature": feature,
            "value": row["value"],
            "n": int(row["n"]),
            "fraction": float(row["fraction"]),
            "active": int(row["active"]),
            "inactive": int(row["inactive"]),
            "active_fraction": float(
                row["active_fraction"]
            ),
        })

    # ------------------------------------------------------------------
    # Test behavior
    # ------------------------------------------------------------------

    test_constant = test_unique == 1

    train_test_same_values = set(
        train_s.dropna().unique()
    ) == set(
        test_s.dropna().unique()
    )

    # If all test values equal one train value
    test_value_in_train = all(
        value in set(train_s.dropna().unique())
        for value in test_s.dropna().unique()
    )

    test_rows.append({
        "feature": feature,
        "train_unique": int(train_unique),
        "test_unique": int(test_unique),
        "train_constant": bool(
            train_unique == 1
        ),
        "test_constant": bool(test_constant),
        "train_dominant_fraction": float(
            train_info["dominant_fraction"]
        ),
        "test_dominant_fraction": float(
            test_info["dominant_fraction"]
        ),
        "train_test_value_sets_identical": bool(
            train_test_same_values
        ),
        "all_test_values_seen_in_train": bool(
            test_value_in_train
        ),
        "test_missing": test_missing,
        "test_infinite": test_inf,
    })

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    recommendation = classify_feature(
        train_unique=train_unique,
        train_dominant=train_info[
            "dominant_fraction"
        ],
        train_second=train_info[
            "second_fraction"
        ],
        train_variance=train_variance,
        test_unique=test_unique,
        test_constant=test_constant,
        activity_corr=activity_corr
    )

    # ------------------------------------------------------------------
    # Analysis row
    # ------------------------------------------------------------------

    analysis_rows.append({
        "feature": feature,

        "train_unique": int(train_unique),
        "train_constant": bool(
            train_unique == 1
        ),

        "train_nzv": True,

        "train_dominant_fraction": float(
            train_info["dominant_fraction"]
        ),

        "train_second_fraction": float(
            train_info["second_fraction"]
        ),

        "train_frequency_ratio": float(
            train_info["frequency_ratio"]
        ),

        "train_mean": train_mean,
        "train_std": train_std,
        "train_variance": train_variance,
        "train_min": train_min,
        "train_max": train_max,

        "train_missing": train_missing,
        "train_infinite": train_inf,

        "test_unique": int(test_unique),
        "test_constant": bool(test_constant),

        "test_dominant_fraction": float(
            test_info["dominant_fraction"]
        ),

        "test_variance": test_variance,

        "test_missing": test_missing,
        "test_infinite": test_inf,

        "activity_correlation": (
            float(activity_corr)
            if pd.notna(activity_corr)
            else np.nan
        ),

        "recommendation": recommendation,
    })

    print(
        f"Unique train values : {train_unique}"
    )

    print(
        f"Dominant fraction   : "
        f"{train_info['dominant_fraction']:.6f}"
    )

    print(
        f"Second fraction     : "
        f"{train_info['second_fraction']:.6f}"
    )

    print(
        f"Frequency ratio     : "
        f"{train_info['frequency_ratio']:.3f}"
    )

    print(
        f"Variance            : "
        f"{train_variance}"
    )

    print(
        f"Test unique values  : {test_unique}"
    )

    print(
        f"Test constant       : {test_constant}"
    )

    print(
        f"Activity correlation: "
        f"{activity_corr}"
    )

    print(
        f"Recommendation      : {recommendation}"
    )


# ============================================================================
# 6. REDUNDANCY ANALYSIS
# ============================================================================

print()
print("=" * 78)
print("6. NZV-TO-NZV REDUNDANCY ANALYSIS")
print("=" * 78)

redundancy_rows = []

for i in range(len(nzv_features)):

    feature_a = nzv_features[i]

    a = safe_numeric(train[feature_a])

    for j in range(i + 1, len(nzv_features)):

        feature_b = nzv_features[j]

        b = safe_numeric(train[feature_b])

        valid = a.notna() & b.notna()

        if valid.sum() < 3:
            corr = np.nan
        elif a.loc[valid].nunique() < 2:
            corr = np.nan
        elif b.loc[valid].nunique() < 2:
            corr = np.nan
        else:
            corr = a.loc[valid].corr(
                b.loc[valid]
            )

        identical = train[
            feature_a
        ].equals(
            train[feature_b]
        )

        redundancy_rows.append({
            "feature_1": feature_a,
            "feature_2": feature_b,
            "pearson_correlation": corr,
            "identical_train_values": bool(
                identical
            )
        })

redundancy_df = pd.DataFrame(
    redundancy_rows
)

if len(redundancy_df) > 0:

    high_corr = redundancy_df[
        redundancy_df[
            "pearson_correlation"
        ].abs() >= 0.90
    ]

    identical_pairs = redundancy_df[
        redundancy_df[
            "identical_train_values"
        ]
    ]

    print(
        f"NZV pairs examined: "
        f"{len(redundancy_df)}"
    )

    print(
        f"Highly correlated pairs "
        f"(|r| >= 0.90): {len(high_corr)}"
    )

    print(
        f"Identical pairs: "
        f"{len(identical_pairs)}"
    )

else:

    high_corr = pd.DataFrame()
    identical_pairs = pd.DataFrame()

    print("No NZV pairs available.")


# ============================================================================
# 7. SAVE TABULAR REPORTS
# ============================================================================

print()
print("=" * 78)
print("7. WRITING MACHINE-READABLE TABLES")
print("=" * 78)

analysis_df = pd.DataFrame(
    analysis_rows
)

value_frequency_df = pd.DataFrame(
    value_frequency_rows
)

activity_df = pd.DataFrame(
    activity_rows
)

test_behavior_df = pd.DataFrame(
    test_rows
)

analysis_df.to_csv(
    ANALYSIS_FILE,
    sep="\t",
    index=False
)

value_frequency_df.to_csv(
    VALUE_FREQ_FILE,
    sep="\t",
    index=False
)

activity_df.to_csv(
    ACTIVITY_FILE,
    sep="\t",
    index=False
)

redundancy_df.to_csv(
    REDUNDANCY_FILE,
    sep="\t",
    index=False
)

test_behavior_df.to_csv(
    TEST_BEHAVIOR_FILE,
    sep="\t",
    index=False
)

print(f"Written: {ANALYSIS_FILE}")
print(f"Written: {VALUE_FREQ_FILE}")
print(f"Written: {ACTIVITY_FILE}")
print(f"Written: {REDUNDANCY_FILE}")
print(f"Written: {TEST_BEHAVIOR_FILE}")


# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================

recommendation_counts = (
    analysis_df[
        "recommendation"
    ]
    .value_counts()
    .to_dict()
)

constant_train = int(
    analysis_df[
        "train_constant"
    ].sum()
)

constant_test = int(
    analysis_df[
        "test_constant"
    ].sum()
)

test_collapsed = int(
    (
        analysis_df[
            "test_unique"
        ]
        <
        analysis_df[
            "train_unique"
        ]
    ).sum()
)

activity_associated = int(
    (
        analysis_df[
            "activity_correlation"
        ].abs()
        >= 0.02
    ).sum()
)


# ============================================================================
# 9. FINAL DECISION LOGIC
# ============================================================================

print()
print("=" * 78)
print("8. FINAL NZV INVESTIGATION SUMMARY")
print("=" * 78)

print(
    f"Total NZV features investigated: "
    f"{len(nzv_features)}"
)

print(
    f"Constant in TRAIN: "
    f"{constant_train}"
)

print(
    f"Constant in TEST: "
    f"{constant_test}"
)

print(
    f"Features collapsing in TEST: "
    f"{test_collapsed}"
)

print(
    f"NZV features with |activity correlation| >= 0.02: "
    f"{activity_associated}"
)

print()
print("Recommendations:")

for key in [
    "KEEP",
    "REVIEW",
    "REMOVE"
]:

    print(
        f"  {key}: "
        f"{recommendation_counts.get(key, 0)}"
    )


# ============================================================================
# 10. PRINT FEATURE-BY-FEATURE DECISIONS
# ============================================================================

print()
print("=" * 78)
print("9. FEATURE-BY-FEATURE DECISIONS")
print("=" * 78)

for _, row in analysis_df.iterrows():

    print(
        f"{row['feature']:<40} "
        f"{row['recommendation']}"
    )


# ============================================================================
# 11. OVERALL STATUS
# ============================================================================

if recommendation_counts.get("REMOVE", 0) > 0:

    overall_status = "REVIEW_REQUIRED"

elif recommendation_counts.get("REVIEW", 0) > 0:

    overall_status = "REVIEW_REQUIRED"

else:

    overall_status = "NZV_INVESTIGATION_PASS"


# ============================================================================
# 12. MACHINE-READABLE JSON
# ============================================================================

report = {

    "script": "05_nzv_feature_investigation_v1",

    "generated": datetime.now().isoformat(),

    "input_files": {

        "train": TRAIN_FILE,
        "test": TEST_FILE,

        "train_sha256": train_sha,
        "test_sha256": test_sha
    },

    "dataset": {

        "train_rows": int(len(train)),
        "test_rows": int(len(test)),

        "train_columns": int(len(train.columns)),
        "test_columns": int(len(test.columns)),

        "train_features": int(len(feature_columns)),
        "test_features": int(len(feature_columns))
    },

    "activity": {

        "column": activity_col,

        "train_counts": (
            train[activity_col]
            .value_counts()
            .to_dict()
        ),

        "test_counts": (
            test[activity_col]
            .value_counts()
            .to_dict()
        )
    },

    "nzv": {

        "count": int(len(nzv_features)),

        "features": nzv_features,

        "recommendation_counts": {
            k: int(v)
            for k, v in recommendation_counts.items()
        },

        "constant_train": constant_train,

        "constant_test": constant_test,

        "test_collapsed_features": test_collapsed,

        "activity_associated_features": activity_associated
    },

    "redundancy": {

        "pairs_examined": int(
            len(redundancy_df)
        ),

        "high_correlation_pairs": int(
            len(high_corr)
        ),

        "identical_pairs": int(
            len(identical_pairs)
        )
    },

    "overall_status": overall_status,

    "important_note": (
        "This script investigates NZV features only. "
        "It does not remove or modify any feature."
    ),

    "output_files": {

        "analysis": ANALYSIS_FILE,
        "value_frequencies": VALUE_FREQ_FILE,
        "activity_association": ACTIVITY_FILE,
        "redundancy": REDUNDANCY_FILE,
        "test_behavior": TEST_BEHAVIOR_FILE,
        "report_text": REPORT_TXT,
        "report_json": REPORT_JSON
    }
}


with open(
    REPORT_JSON,
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        default=str
    )

print()
print(f"Written: {REPORT_JSON}")


# ============================================================================
# 13. HUMAN-READABLE REPORT
# ============================================================================

end_time = datetime.now()

with open(
    REPORT_TXT,
    "w"
) as f:

    f.write(
        "=" * 78 + "\n"
    )

    f.write(
        "SCRIPT 05 — NZV FEATURE INVESTIGATION REPORT\n"
    )

    f.write(
        "=" * 78 + "\n\n"
    )

    f.write(
        f"Generated: {datetime.now().isoformat()}\n\n"
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
        f"TRAIN rows: {len(train)}\n"
    )

    f.write(
        f"TEST rows : {len(test)}\n"
    )

    f.write(
        f"TRAIN features: {len(feature_columns)}\n"
    )

    f.write(
        f"TEST features : {len(feature_columns)}\n\n"
    )

    f.write(
        "NZV FEATURES\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"Total NZV features: {len(nzv_features)}\n\n"
    )

    for i, feature in enumerate(
        nzv_features,
        1
    ):

        row = analysis_df[
            analysis_df["feature"] == feature
        ].iloc[0]

        f.write(
            f"{i:02d}. {feature}\n"
        )

        f.write(
            f"    Train unique: "
            f"{row['train_unique']}\n"
        )

        f.write(
            f"    Dominant fraction: "
            f"{row['train_dominant_fraction']:.8f}\n"
        )

        f.write(
            f"    Second fraction: "
            f"{row['train_second_fraction']:.8f}\n"
        )

        f.write(
            f"    Frequency ratio: "
            f"{row['train_frequency_ratio']}\n"
        )

        f.write(
            f"    Variance: "
            f"{row['train_variance']}\n"
        )

        f.write(
            f"    Test unique: "
            f"{row['test_unique']}\n"
        )

        f.write(
            f"    Test constant: "
            f"{row['test_constant']}\n"
        )

        f.write(
            f"    Activity correlation: "
            f"{row['activity_correlation']}\n"
        )

        f.write(
            f"    Recommendation: "
            f"{row['recommendation']}\n\n"
        )

    f.write(
        "REDUNDANCY\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"NZV pairs examined: "
        f"{len(redundancy_df)}\n"
    )

    f.write(
        f"Highly correlated pairs "
        f"(|r| >= 0.90): "
        f"{len(high_corr)}\n"
    )

    f.write(
        f"Identical pairs: "
        f"{len(identical_pairs)}\n\n"
    )

    f.write(
        "RECOMMENDATION SUMMARY\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    for key in [
        "KEEP",
        "REVIEW",
        "REMOVE"
    ]:

        f.write(
            f"{key}: "
            f"{recommendation_counts.get(key, 0)}\n"
        )

    f.write("\n")

    f.write(
        "OVERALL STATUS\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        f"{overall_status}\n\n"
    )

    f.write(
        "IMPORTANT\n"
    )

    f.write(
        "-" * 78 + "\n"
    )

    f.write(
        "No features were removed or modified by this script.\n"
    )

    f.write(
        "The purpose is evidence-based investigation before any "
        "feature-freezing decision.\n"
    )


print(f"Written: {REPORT_TXT}")


# ============================================================================
# 14. FINAL CONSOLE SUMMARY
# ============================================================================

print()
print("=" * 78)
print("FINAL SUMMARY")
print("=" * 78)

print(
    f"NZV features investigated: {len(nzv_features)}"
)

print(
    f"KEEP    : "
    f"{recommendation_counts.get('KEEP', 0)}"
)

print(
    f"REVIEW  : "
    f"{recommendation_counts.get('REVIEW', 0)}"
)

print(
    f"REMOVE  : "
    f"{recommendation_counts.get('REMOVE', 0)}"
)

print()
print(
    f"OVERALL STATUS: {overall_status}"
)

print()
print("IMPORTANT:")
print(
    "No features were removed."
)

print(
    "Use the generated reports to make the final NZV decision."
)

print()
print("Reports:")
print(
    f"  {ANALYSIS_FILE}"
)

print(
    f"  {VALUE_FREQ_FILE}"
)

print(
    f"  {ACTIVITY_FILE}"
)

print(
    f"  {REDUNDANCY_FILE}"
)

print(
    f"  {TEST_BEHAVIOR_FILE}"
)

print(
    f"  {REPORT_JSON}"
)

print(
    f"  {REPORT_TXT}"
)

print()
print("=" * 78)
print("SCRIPT 05 NZV INVESTIGATION COMPLETE")
print("=" * 78)

print(
    f"End time: {end_time.strftime('%a %b %d %H:%M:%S IST %Y')}"
)
