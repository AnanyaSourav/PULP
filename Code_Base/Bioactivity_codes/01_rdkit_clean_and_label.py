#!/usr/bin/env python3

import pandas as pd
from rdkit import Chem
from pathlib import Path

# ============================================================
# INPUT FILES
# ============================================================

ALL_FILE = "CMAUPv2.0_download_Ingredients_All.txt"

ACTIVE_FILE = "CMAUPv2.0_download_Ingredients_onlyActive.txt"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("feature_engineering")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

FINAL_OUTPUT = (
    OUTPUT_DIR /
    "cleaned_phytochemicals_activity.tsv"
)

INVALID_OUTPUT = (
    OUTPUT_DIR /
    "invalid_smiles.tsv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR /
    "rdkit_cleaning_summary.txt"
)


# ============================================================
# READ INGREDIENTS_ALL
# ============================================================

print("=" * 70)
print("STEP 1: READING CMAUP INGREDIENTS")
print("=" * 70)

all_df = pd.read_csv(
    ALL_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)

print(
    f"Total records in Ingredients_All: "
    f"{len(all_df):,}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "np_id",
    "SMILES"
]

missing = [
    col
    for col in required_columns
    if col not in all_df.columns
]

if missing:
    raise ValueError(
        f"Required columns missing from "
        f"{ALL_FILE}: {missing}"
    )


# ============================================================
# REMOVE MISSING IDs / SMILES
# ============================================================

before_missing = len(all_df)

df = all_df[
    all_df["np_id"].notna() &
    all_df["SMILES"].notna()
].copy()

df["np_id"] = (
    df["np_id"]
    .astype(str)
    .str.strip()
)

df["SMILES"] = (
    df["SMILES"]
    .astype(str)
    .str.strip()
)

# Remove empty / invalid textual entries
df = df[
    (df["np_id"] != "") &
    (df["SMILES"] != "") &
    (df["SMILES"].str.lower() != "nan") &
    (df["SMILES"].str.lower() != "n.a.")
].copy()

removed_missing = before_missing - len(df)

print(
    f"Records removed because of missing ID/SMILES: "
    f"{removed_missing:,}"
)


# ============================================================
# RDKit SMILES VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: RDKit SMILES VALIDATION")
print("=" * 70)

valid_smiles = []
canonical_smiles = []
invalid_records = []

for index, row in df.iterrows():

    ingredient_id = row["np_id"]
    smiles = row["SMILES"]

    try:

        mol = Chem.MolFromSmiles(smiles)

    except Exception:
        mol = None

    if mol is None:

        invalid_records.append(
            {
                "np_id": ingredient_id,
                "SMILES": smiles,
                "Reason": "RDKit could not parse SMILES"
            }
        )

        valid_smiles.append(False)
        canonical_smiles.append(None)

    else:

        valid_smiles.append(True)

        canonical = Chem.MolToSmiles(
            mol,
            canonical=True
        )

        canonical_smiles.append(canonical)


# Add validation results
df["RDKit_Valid"] = valid_smiles
df["Canonical_SMILES"] = canonical_smiles


# ============================================================
# SEPARATE VALID / INVALID
# ============================================================

invalid_df = df[
    df["RDKit_Valid"] == False
].copy()

valid_df = df[
    df["RDKit_Valid"] == True
].copy()


print(
    f"Total records checked:       {len(df):,}"
)

print(
    f"Valid RDKit molecules:       {len(valid_df):,}"
)

print(
    f"Invalid SMILES removed:      {len(invalid_df):,}"
)


# ============================================================
# READ ACTIVE INGREDIENT FILE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: READING ACTIVE INGREDIENTS")
print("=" * 70)

active_df = pd.read_csv(
    ACTIVE_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)


# Check np_id
if "np_id" not in active_df.columns:
    raise ValueError(
        f"'np_id' column not found in {ACTIVE_FILE}"
    )


# ============================================================
# CREATE ACTIVE ID SET
# ============================================================

active_ids = set(
    active_df["np_id"]
    .dropna()
    .astype(str)
    .str.strip()
)

print(
    f"Unique active ingredient IDs: "
    f"{len(active_ids):,}"
)


# ============================================================
# ASSIGN ACTIVITY STATUS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: ASSIGNING ACTIVITY STATUS")
print("=" * 70)

valid_df["Activity_Status"] = (
    valid_df["np_id"]
    .isin(active_ids)
    .astype(int)
)

valid_df["Activity_Label"] = (
    valid_df["Activity_Status"]
    .map({
        1: "Active",
        0: "Inactive"
    })
)


# ============================================================
# ACTIVITY STATISTICS
# ============================================================

active_count = (
    valid_df["Activity_Status"] == 1
).sum()

inactive_count = (
    valid_df["Activity_Status"] == 0
).sum()


print(
    f"Active phytochemicals:       {active_count:,}"
)

print(
    f"Inactive phytochemicals:     {inactive_count:,}"
)

print(
    f"Total valid phytochemicals:  {len(valid_df):,}"
)

if len(valid_df) > 0:

    print(
        f"Active percentage:           "
        f"{active_count / len(valid_df) * 100:.2f}%"
    )

    print(
        f"Inactive percentage:         "
        f"{inactive_count / len(valid_df) * 100:.2f}%"
    )


# ============================================================
# REMOVE RDKit INTERNAL COLUMN
# ============================================================

# RDKit_Valid is no longer needed because
# the final dataset contains only valid molecules.

valid_df = valid_df.drop(
    columns=["RDKit_Valid"]
)


# ============================================================
# SAVE INVALID SMILES
# ============================================================

if len(invalid_df) > 0:

    invalid_df.to_csv(
        INVALID_OUTPUT,
        sep="\t",
        index=False
    )

else:

    # Create empty file with headers
    pd.DataFrame(
        columns=[
            "np_id",
            "SMILES",
            "Reason"
        ]
    ).to_csv(
        INVALID_OUTPUT,
        sep="\t",
        index=False
    )


# ============================================================
# SAVE FINAL CLEAN DATASET
# ============================================================

# Put important columns first.
# All other original CMAUP columns are retained.

priority_columns = [
    "np_id",
    "pref_name",
    "iupac_name",
    "SMILES",
    "Canonical_SMILES",
    "Activity_Status",
    "Activity_Label"
]

existing_priority = [
    col
    for col in priority_columns
    if col in valid_df.columns
]

remaining_columns = [
    col
    for col in valid_df.columns
    if col not in existing_priority
]

final_columns = (
    existing_priority +
    remaining_columns
)

valid_df = valid_df[
    final_columns
]


valid_df.to_csv(
    FINAL_OUTPUT,
    sep="\t",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_OUTPUT,
    "w"
) as f:

    f.write(
        "CMAUP PHYTOCHEMICAL RDKit CLEANING SUMMARY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Input file: {ALL_FILE}\n"
    )

    f.write(
        f"Active file: {ACTIVE_FILE}\n\n"
    )

    f.write(
        f"Original Ingredients_All records: "
        f"{len(all_df):,}\n"
    )

    f.write(
        f"Removed missing ID/SMILES: "
        f"{removed_missing:,}\n"
    )

    f.write(
        f"Records checked by RDKit: "
        f"{len(df):,}\n"
    )

    f.write(
        f"Valid RDKit molecules: "
        f"{len(valid_df):,}\n"
    )

    f.write(
        f"Invalid SMILES: "
        f"{len(invalid_df):,}\n"
    )

    f.write(
        f"Unique active IDs: "
        f"{len(active_ids):,}\n"
    )

    f.write(
        f"Active valid molecules: "
        f"{active_count:,}\n"
    )

    f.write(
        f"Inactive valid molecules: "
        f"{inactive_count:,}\n"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 COMPLETED")
print("=" * 70)

print("\nFinal cleaned dataset:")
print(f"  {FINAL_OUTPUT}")

print("\nInvalid SMILES:")
print(f"  {INVALID_OUTPUT}")

print("\nSummary:")
print(f"  {SUMMARY_OUTPUT}")

print("\nNo train/test split was performed.")
print("No fingerprints were generated.")
print("No model was trained.")

print("\nDONE.")
