#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# INPUT
# ============================================================

INPUT_FILE = (
    "feature_engineering/"
    "cleaned_phytochemicals_activity.tsv"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("feature_engineering/split")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

TRAIN_FILE = (
    OUTPUT_DIR /
    "train_80.tsv"
)

TEST_FILE = (
    OUTPUT_DIR /
    "test_20.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

print("=" * 70)
print("80:20 STRATIFIED TRAIN-TEST SPLIT")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)

print(
    f"\nTotal molecules: {len(df):,}"
)


# ============================================================
# CHECK ACTIVITY COLUMN
# ============================================================

if "Activity_Status" not in df.columns:
    raise ValueError(
        "Activity_Status column not found."
    )

# Convert activity status to integer
df["Activity_Status"] = (
    pd.to_numeric(
        df["Activity_Status"],
        errors="raise"
    )
    .astype(int)
)


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\nOriginal class distribution:")
print(
    df["Activity_Status"]
    .value_counts()
    .sort_index()
)

print("\nOriginal class percentages:")
print(
    (
        df["Activity_Status"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(3)
)


# ============================================================
# RANDOM STRATIFIED 80:20 SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["Activity_Status"],
    shuffle=True
)


# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# SAVE TRAINING SET
# ============================================================

train_df.to_csv(
    TRAIN_FILE,
    sep="\t",
    index=False
)


# ============================================================
# SAVE TESTING SET
# ============================================================

test_df.to_csv(
    TEST_FILE,
    sep="\t",
    index=False
)


# ============================================================
# FUNCTION TO PRINT CLASS STATISTICS
# ============================================================

def print_statistics(name, data):

    total = len(data)

    active = (
        data["Activity_Status"] == 1
    ).sum()

    inactive = (
        data["Activity_Status"] == 0
    ).sum()

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(f"Total:     {total:,}")
    print(f"Active:    {active:,}")
    print(f"Inactive:  {inactive:,}")

    print(
        f"Active %:  {active / total * 100:.3f}%"
    )

    print(
        f"Inactive %:{inactive / total * 100:.3f}%"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print_statistics(
    "TRAINING SET (80%)",
    train_df
)

print_statistics(
    "TESTING SET (20%)",
    test_df
)


# ============================================================
# VERIFY SPLIT
# ============================================================

print("\n" + "=" * 70)
print("SPLIT VERIFICATION")
print("=" * 70)

print(
    f"\nOriginal dataset: {len(df):,}"
)

print(
    f"Training + testing: "
    f"{len(train_df) + len(test_df):,}"
)

if len(df) == len(train_df) + len(test_df):

    print(
        "✓ All molecules accounted for."
    )

else:

    print(
        "ERROR: molecule count mismatch!"
    )


# ============================================================
# CHECK FOR DUPLICATE IDs BETWEEN SETS
# ============================================================

if "np_id" in df.columns:

    train_ids = set(
        train_df["np_id"]
    )

    test_ids = set(
        test_df["np_id"]
    )

    overlap = train_ids.intersection(
        test_ids
    )

    print(
        f"\nIngredient ID overlap: {len(overlap)}"
    )

    if len(overlap) == 0:

        print(
            "✓ No np_id occurs in both "
            "training and testing sets."
        )

    else:

        print(
            "WARNING: Duplicate np_id detected "
            "between train and test!"
        )


# ============================================================
# CHECK STRUCTURE OVERLAP
# ============================================================

if "Canonical_SMILES" in df.columns:

    train_smiles = set(
        train_df["Canonical_SMILES"]
    )

    test_smiles = set(
        test_df["Canonical_SMILES"]
    )

    structure_overlap = (
        train_smiles.intersection(
            test_smiles
        )
    )

    print(
        f"Canonical SMILES overlap: "
        f"{len(structure_overlap)}"
    )

    if len(structure_overlap) == 0:

        print(
            "✓ No identical canonical structures "
            "occur in both sets."
        )

    else:

        print(
            "WARNING: Identical chemical structures "
            "occur in both train and test."
        )


# ============================================================
# SAVE SPLIT SUMMARY
# ============================================================

SUMMARY_FILE = (
    OUTPUT_DIR /
    "split_summary.txt"
)

with open(
    SUMMARY_FILE,
    "w"
) as f:

    f.write(
        "CMAUP PHYTOCHEMICAL 80:20 TRAIN-TEST SPLIT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Input dataset: {INPUT_FILE}\n"
    )

    f.write(
        f"Random state: {RANDOM_STATE}\n"
    )

    f.write(
        f"Test size: {TEST_SIZE}\n"
    )

    f.write(
        f"Original molecules: {len(df):,}\n"
    )

    f.write(
        f"Training molecules: {len(train_df):,}\n"
    )

    f.write(
        f"Testing molecules: {len(test_df):,}\n\n"
    )

    f.write("TRAINING SET\n")
    f.write(
        f"Active: "
        f"{(train_df['Activity_Status'] == 1).sum():,}\n"
    )

    f.write(
        f"Inactive: "
        f"{(train_df['Activity_Status'] == 0).sum():,}\n"
    )

    f.write(
        f"Active percentage: "
        f"{(train_df['Activity_Status'] == 1).mean() * 100:.3f}%\n\n"
    )

    f.write("TESTING SET\n")

    f.write(
        f"Active: "
        f"{(test_df['Activity_Status'] == 1).sum():,}\n"
    )

    f.write(
        f"Inactive: "
        f"{(test_df['Activity_Status'] == 0).sum():,}\n"
    )

    f.write(
        f"Active percentage: "
        f"{(test_df['Activity_Status'] == 1).mean() * 100:.3f}%\n"
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETED")
print("=" * 70)

print("\nOutput files:")

print(
    f"Training: {TRAIN_FILE}"
)

print(
    f"Testing:  {TEST_FILE}"
)

print(
    f"Summary:  {SUMMARY_FILE}"
)

print("\nDONE.")
