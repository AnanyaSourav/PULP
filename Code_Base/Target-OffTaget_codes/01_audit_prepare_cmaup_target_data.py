#!/usr/bin/env python3

"""
01_audit_prepare_cmaup_target_data.py

CMAUPv2.0 TARGET / OFF-TARGET MODEL DEVELOPMENT
================================================

Purpose
-------
Audit and prepare the CMAUPv2.0 molecule-target data before
developing the target/off-target hypothesis-generation system.

This script DOES NOT train a prediction model.

It creates:
    1. Clean CMAUP molecule table
    2. Clean molecule-target association table
    3. Clean target annotation table
    4. Master molecule-target evidence table
    5. Target-support statistics
    6. Activity-type distributions
    7. Activity-relationship distributions
    8. Activity-unit distributions
    9. Target hierarchy completeness
   10. Molecule/target mapping QC
   11. Coverage of the existing Model-1 train/test datasets
   12. Complete audit log and manifest

IMPORTANT
---------
Absence of a target association is NOT treated as negative evidence.

The downstream target-inference system will use known CMAUP
associations as evidence.

DIRECTORY
---------
This script assumes the following structure:

PULP/
├── CMAUPv2.0_download_Ingredients_All.txt
├── CMAUPv2.0_download_Ingredient_Target_Associations_ActivityValues_References.txt
├── CMAUPv2.0_download_Targets.txt
├── CMAUPv2.0_download_Ingredients_onlyActive.txt
├── CMAUPv2.0_download_Human_Oral_Bioavailability_information_of_Ingredients_All.txt
├── feature_engineering/
│   └── split/
│       ├── train_80.tsv
│       └── test_20.tsv
└── 01_audit_prepare_cmaup_target_data.py

Output:
PULP/
└── target_model/
    └── data/
        └── 01_cmaup_audit/
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import sys


# ============================================================
# 1. DETERMINE PULP DIRECTORY AUTOMATICALLY
# ============================================================

# This script is expected to be stored directly inside PULP.
PULP_DIR = Path(__file__).resolve().parent

print("=" * 80)
print("CMAUP TARGET / OFF-TARGET DATA AUDIT")
print("=" * 80)

print(f"PULP directory:")
print(f"  {PULP_DIR}")


# ============================================================
# 2. DEFINE INPUT DIRECTORIES
# ============================================================

FEATURE_ENGINEERING_DIR = (
    PULP_DIR / "feature_engineering"
)

SPLIT_DIR = (
    FEATURE_ENGINEERING_DIR / "split"
)


# ============================================================
# 3. DEFINE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PULP_DIR
    / "target_model"
    / "data"
    / "01_cmaup_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. CMAUP SOURCE FILES
# ============================================================

MOLECULE_FILE = (
    PULP_DIR
    / "CMAUPv2.0_download_Ingredients_All.txt"
)

ASSOCIATION_FILE = (
    PULP_DIR
    / "CMAUPv2.0_download_Ingredient_Target_Associations_ActivityValues_References.txt"
)

TARGET_FILE = (
    PULP_DIR
    / "CMAUPv2.0_download_Targets.txt"
)

ACTIVE_FILE = (
    PULP_DIR
    / "CMAUPv2.0_download_Ingredients_onlyActive.txt"
)

BIOAVAILABILITY_FILE = (
    PULP_DIR
    / "CMAUPv2.0_download_Human_Oral_Bioavailability_information_of_Ingredients_All.txt"
)


# ============================================================
# 5. EXISTING BIOACTIVITY MODEL DATA
# ============================================================

TRAIN_FILE = (
    SPLIT_DIR / "train_80.tsv"
)

TEST_FILE = (
    SPLIT_DIR / "test_20.tsv"
)


# ============================================================
# 6. LOGGING
# ============================================================

LOG_FILE = (
    OUTPUT_DIR / "01_audit_log.txt"
)

log_lines = []


def log(message=""):
    """Print and store a message."""
    print(message)
    log_lines.append(str(message))


def save_log():

    LOG_FILE.write_text(
        "\n".join(log_lines),
        encoding="utf-8"
    )


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


def normalize_string_columns(df):

    df = df.copy()

    for col in df.columns:

        if (
            pd.api.types.is_object_dtype(df[col])
            or
            pd.api.types.is_string_dtype(df[col])
        ):

            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
            )

            df[col] = df[col].replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                    "NULL": pd.NA,
                    "NA": pd.NA,
                    "N/A": pd.NA
                }
            )

    return df


def read_cmaup(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file was not found:\n"
            f"{path}\n"
        )

    log(f"\nReading:")
    log(f"  {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False
    )

    df = normalize_columns(df)
    df = normalize_string_columns(df)

    log(
        f"  Rows    : {len(df):,}"
    )

    log(
        f"  Columns : {len(df.columns):,}"
    )

    log(
        f"  Headers : {list(df.columns)}"
    )

    return df


def require_columns(
    df,
    required_columns,
    dataset_name
):

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\n{dataset_name} is missing required "
            f"columns:\n{missing}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )


def missing_summary(df, dataset):

    rows = []

    for col in df.columns:

        missing_n = int(
            df[col].isna().sum()
        )

        rows.append({
            "dataset": dataset,
            "column": col,
            "total_rows": len(df),
            "missing_n": missing_n,
            "missing_percent": (
                missing_n / len(df) * 100
                if len(df) > 0
                else np.nan
            )
        })

    return pd.DataFrame(rows)


def write_tsv(df, path):

    df.to_csv(
        path,
        sep="\t",
        index=False
    )

    log(
        f"Wrote: {path}"
    )


# ============================================================
# 8. INITIAL LOG
# ============================================================

log("=" * 80)
log("CMAUP TARGET / OFF-TARGET DATA AUDIT")
log("=" * 80)

log(
    f"Timestamp: {datetime.now().isoformat()}"
)

log(
    f"PULP directory: {PULP_DIR}"
)

log(
    f"Output directory: {OUTPUT_DIR}"
)


# ============================================================
# 9. FILE AVAILABILITY
# ============================================================

log("\n" + "=" * 80)
log("FILE AVAILABILITY")
log("=" * 80)

files_to_check = {

    "Ingredients_All":
        MOLECULE_FILE,

    "Ingredient_Target_Associations":
        ASSOCIATION_FILE,

    "Targets":
        TARGET_FILE,

    "Ingredients_onlyActive":
        ACTIVE_FILE,

    "Human_Oral_Bioavailability":
        BIOAVAILABILITY_FILE,

    "Model1_train":
        TRAIN_FILE,

    "Model1_test":
        TEST_FILE
}


file_status = []

for name, path in files_to_check.items():

    exists = path.exists()

    if exists:

        size_mb = (
            path.stat().st_size
            / (1024 ** 2)
        )

    else:

        size_mb = np.nan

    file_status.append({

        "dataset":
            name,

        "path":
            str(path),

        "exists":
            exists,

        "size_MB":
            size_mb
    })

    log(
        f"{name:35s} "
        f"{'FOUND' if exists else 'NOT FOUND'}"
    )


file_status_df = pd.DataFrame(
    file_status
)

write_tsv(
    file_status_df,
    OUTPUT_DIR
    / "file_availability.tsv"
)


# ============================================================
# 10. READ MAIN CMAUP FILES
# ============================================================

log("\n" + "=" * 80)
log("READ MAIN CMAUP DATASETS")
log("=" * 80)

molecules = read_cmaup(
    MOLECULE_FILE
)

associations = read_cmaup(
    ASSOCIATION_FILE
)

targets = read_cmaup(
    TARGET_FILE
)


# ============================================================
# 11. VALIDATE MOLECULE FILE
# ============================================================

require_columns(
    molecules,
    [
        "np_id",
        "SMILES"
    ],
    "CMAUP Ingredients_All"
)


# ============================================================
# 12. VALIDATE ASSOCIATION FILE
# ============================================================

require_columns(
    associations,
    [
        "Ingredient_ID",
        "Target_ID",
        "Activity_Type",
        "Activity_Relationship",
        "Activity_Value",
        "Activity_Unit",
        "Reference_ID",
        "Reference_ID_Type"
    ],
    "CMAUP Ingredient_Target_Associations"
)


# ============================================================
# 13. VALIDATE TARGET FILE
# ============================================================

require_columns(
    targets,
    [
        "Target_ID",
        "Gene_Symbol",
        "Protein_Name",
        "Uniprot_ID",
        "ChEMBL_ID",
        "TTD_ID",
        "Target_Class_Level1",
        "Target_Class_Level2",
        "Target_Class_Level3",
        "Target_type"
    ],
    "CMAUP Targets"
)


log(
    "\nRequired-column validation: PASS"
)


# ============================================================
# 14. DATASET SIZE SUMMARY
# ============================================================

dataset_sizes = pd.DataFrame([

    {
        "dataset": "Ingredients_All",
        "rows": len(molecules),
        "columns": len(molecules.columns)
    },

    {
        "dataset":
            "Ingredient_Target_Associations",
        "rows": len(associations),
        "columns": len(associations.columns)
    },

    {
        "dataset": "Targets",
        "rows": len(targets),
        "columns": len(targets.columns)
    }

])

write_tsv(
    dataset_sizes,
    OUTPUT_DIR
    / "dataset_sizes.tsv"
)


# ============================================================
# 15. MISSING VALUE REPORTS
# ============================================================

log("\n" + "=" * 80)
log("MISSING VALUE ANALYSIS")
log("=" * 80)

for dataset_name, df in [

    ("molecules", molecules),

    ("associations", associations),

    ("targets", targets)

]:

    summary = missing_summary(
        df,
        dataset_name
    )

    write_tsv(
        summary,
        OUTPUT_DIR
        / f"{dataset_name}_missing_values.tsv"
    )

    log(f"\n{dataset_name}")

    for _, row in summary.iterrows():

        if row["missing_n"] > 0:

            log(
                f"  {row['column']:35s}"
                f"{int(row['missing_n']):>10,}"
                f" ({row['missing_percent']:.3f}%)"
            )


# ============================================================
# 16. MOLECULE ID QC
# ============================================================

log("\n" + "=" * 80)
log("MOLECULE ID QC")
log("=" * 80)

missing_np_id = int(
    molecules["np_id"].isna().sum()
)

unique_np_id = int(
    molecules["np_id"].nunique(
        dropna=True
    )
)

duplicate_np_id_rows = int(
    molecules["np_id"]
    .duplicated(keep=False)
    .sum()
)

log(
    f"Total rows             : "
    f"{len(molecules):,}"
)

log(
    f"Unique np_id            : "
    f"{unique_np_id:,}"
)

log(
    f"Missing np_id           : "
    f"{missing_np_id:,}"
)

log(
    f"Duplicated np_id rows   : "
    f"{duplicate_np_id_rows:,}"
)

molecule_id_qc = pd.DataFrame([

    {
        "metric":
            "total_rows",
        "value":
            len(molecules)
    },

    {
        "metric":
            "unique_np_id",
        "value":
            unique_np_id
    },

    {
        "metric":
            "missing_np_id",
        "value":
            missing_np_id
    },

    {
        "metric":
            "duplicated_np_id_rows",
        "value":
            duplicate_np_id_rows
    }

])

write_tsv(
    molecule_id_qc,
    OUTPUT_DIR
    / "molecule_id_qc.tsv"
)


# ============================================================
# 17. SMILES QC
# ============================================================

log("\n" + "=" * 80)
log("SMILES QC")
log("=" * 80)

missing_smiles = int(
    molecules["SMILES"].isna().sum()
)

empty_smiles = int(
    molecules["SMILES"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

unique_smiles = int(
    molecules["SMILES"].nunique(
        dropna=True
    )
)

log(
    f"Missing SMILES          : "
    f"{missing_smiles:,}"
)

log(
    f"Empty SMILES            : "
    f"{empty_smiles:,}"
)

log(
    f"Unique SMILES strings   : "
    f"{unique_smiles:,}"
)

smiles_qc = pd.DataFrame([

    {
        "metric":
            "missing_smiles",
        "value":
            missing_smiles
    },

    {
        "metric":
            "empty_smiles",
        "value":
            empty_smiles
    },

    {
        "metric":
            "unique_smiles",
        "value":
            unique_smiles
    }

])

write_tsv(
    smiles_qc,
    OUTPUT_DIR
    / "smiles_qc.tsv"
)


# ============================================================
# 18. ASSOCIATION QC
# ============================================================

log("\n" + "=" * 80)
log("ASSOCIATION QC")
log("=" * 80)

exact_duplicate_associations = int(
    associations
    .duplicated(keep=False)
    .sum()
)

association_key = [

    "Ingredient_ID",
    "Target_ID",
    "Activity_Type",
    "Activity_Relationship",
    "Activity_Value",
    "Activity_Unit",
    "Reference_ID"

]

duplicate_evidence_keys = int(
    associations
    .duplicated(
        subset=association_key,
        keep=False
    )
    .sum()
)

unique_association_molecules = int(
    associations[
        "Ingredient_ID"
    ].nunique(
        dropna=True
    )
)

unique_association_targets = int(
    associations[
        "Target_ID"
    ].nunique(
        dropna=True
    )
)

unique_references = int(
    associations[
        "Reference_ID"
    ].nunique(
        dropna=True
    )
)

log(
    f"Association records          : "
    f"{len(associations):,}"
)

log(
    f"Exact duplicate rows         : "
    f"{exact_duplicate_associations:,}"
)

log(
    f"Duplicate evidence-key rows  : "
    f"{duplicate_evidence_keys:,}"
)

log(
    f"Unique associated molecules  : "
    f"{unique_association_molecules:,}"
)

log(
    f"Unique associated targets    : "
    f"{unique_association_targets:,}"
)

log(
    f"Unique references            : "
    f"{unique_references:,}"
)


# ============================================================
# 19. TARGET ID QC
# ============================================================

log("\n" + "=" * 80)
log("TARGET ID QC")
log("=" * 80)

missing_target_id = int(
    targets["Target_ID"].isna().sum()
)

unique_target_ids = int(
    targets["Target_ID"].nunique(
        dropna=True
    )
)

duplicate_target_rows = int(
    targets["Target_ID"]
    .duplicated(keep=False)
    .sum()
)

log(
    f"Target rows             : "
    f"{len(targets):,}"
)

log(
    f"Unique Target_ID        : "
    f"{unique_target_ids:,}"
)

log(
    f"Missing Target_ID       : "
    f"{missing_target_id:,}"
)

log(
    f"Duplicated Target_ID    : "
    f"{duplicate_target_rows:,}"
)


# ============================================================
# 20. MOLECULE MAPPING
# ============================================================

log("\n" + "=" * 80)
log("ASSOCIATION → MOLECULE MAPPING")
log("=" * 80)

molecule_id_set = set(
    molecules["np_id"]
    .dropna()
    .astype(str)
)

association_molecule_set = set(
    associations["Ingredient_ID"]
    .dropna()
    .astype(str)
)

mapped_molecules = (
    association_molecule_set
    & molecule_id_set
)

unmapped_molecules = (
    association_molecule_set
    - molecule_id_set
)

log(
    f"Association molecule IDs : "
    f"{len(association_molecule_set):,}"
)

log(
    f"Mapped molecules          : "
    f"{len(mapped_molecules):,}"
)

log(
    f"Unmapped molecules        : "
    f"{len(unmapped_molecules):,}"
)

if unmapped_molecules:

    pd.DataFrame({
        "Ingredient_ID":
            sorted(unmapped_molecules)
    }).to_csv(
        OUTPUT_DIR
        / "unmapped_ingredient_ids.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# 21. TARGET MAPPING
# ============================================================

log("\n" + "=" * 80)
log("ASSOCIATION → TARGET MAPPING")
log("=" * 80)

target_id_set = set(
    targets["Target_ID"]
    .dropna()
    .astype(str)
)

association_target_set = set(
    associations["Target_ID"]
    .dropna()
    .astype(str)
)

mapped_targets = (
    association_target_set
    & target_id_set
)

unmapped_targets = (
    association_target_set
    - target_id_set
)

log(
    f"Association Target IDs : "
    f"{len(association_target_set):,}"
)

log(
    f"Mapped targets          : "
    f"{len(mapped_targets):,}"
)

log(
    f"Unmapped targets        : "
    f"{len(unmapped_targets):,}"
)

if unmapped_targets:

    pd.DataFrame({
        "Target_ID":
            sorted(unmapped_targets)
    }).to_csv(
        OUTPUT_DIR
        / "unmapped_target_ids.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# 22. ACTIVITY TYPE
# ============================================================

log("\n" + "=" * 80)
log("ACTIVITY TYPE DISTRIBUTION")
log("=" * 80)

activity_type = (
    associations[
        "Activity_Type"
    ]
    .fillna("<MISSING>")
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "Activity_Type"
    )
    .reset_index(
        name="n_associations"
    )
)

activity_type["percent"] = (
    activity_type["n_associations"]
    / len(associations)
    * 100
)

write_tsv(
    activity_type,
    OUTPUT_DIR
    / "activity_type_distribution.tsv"
)

for _, row in activity_type.iterrows():

    log(
        f"{str(row['Activity_Type']):35s}"
        f"{int(row['n_associations']):>12,}"
        f" {row['percent']:>8.3f}%"
    )


# ============================================================
# 23. ACTIVITY RELATIONSHIP
# ============================================================

log("\n" + "=" * 80)
log("ACTIVITY RELATIONSHIP DISTRIBUTION")
log("=" * 80)

activity_relationship = (
    associations[
        "Activity_Relationship"
    ]
    .fillna("<MISSING>")
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "Activity_Relationship"
    )
    .reset_index(
        name="n_associations"
    )
)

activity_relationship["percent"] = (
    activity_relationship["n_associations"]
    / len(associations)
    * 100
)

write_tsv(
    activity_relationship,
    OUTPUT_DIR
    / "activity_relationship_distribution.tsv"
)

for _, row in activity_relationship.iterrows():

    log(
        f"{str(row['Activity_Relationship']):35s}"
        f"{int(row['n_associations']):>12,}"
        f" {row['percent']:>8.3f}%"
    )


# ============================================================
# 24. ACTIVITY UNIT
# ============================================================

log("\n" + "=" * 80)
log("ACTIVITY UNIT DISTRIBUTION")
log("=" * 80)

activity_unit = (
    associations[
        "Activity_Unit"
    ]
    .fillna("<MISSING>")
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "Activity_Unit"
    )
    .reset_index(
        name="n_associations"
    )
)

activity_unit["percent"] = (
    activity_unit["n_associations"]
    / len(associations)
    * 100
)

write_tsv(
    activity_unit,
    OUTPUT_DIR
    / "activity_unit_distribution.tsv"
)

for _, row in activity_unit.iterrows():

    log(
        f"{str(row['Activity_Unit']):35s}"
        f"{int(row['n_associations']):>12,}"
        f" {row['percent']:>8.3f}%"
    )


# ============================================================
# 25. TARGET ANNOTATION COMPLETENESS
# ============================================================

log("\n" + "=" * 80)
log("TARGET ANNOTATION COMPLETENESS")
log("=" * 80)

annotation_columns = [

    "Gene_Symbol",
    "Protein_Name",
    "Uniprot_ID",
    "ChEMBL_ID",
    "TTD_ID",
    "Target_Class_Level1",
    "Target_Class_Level2",
    "Target_Class_Level3",
    "Target_type"

]

annotation_rows = []

for col in annotation_columns:

    if col not in targets.columns:
        continue

    missing = int(
        targets[col].isna().sum()
    )

    nonmissing = (
        len(targets) - missing
    )

    annotation_rows.append({

        "field":
            col,

        "total_targets":
            len(targets),

        "nonmissing":
            nonmissing,

        "missing":
            missing,

        "nonmissing_percent":
            nonmissing / len(targets) * 100
            if len(targets)
            else np.nan
    })

    log(
        f"{col:30s}"
        f" complete={nonmissing:>10,}"
        f" missing={missing:>10,}"
    )


annotation_df = pd.DataFrame(
    annotation_rows
)

write_tsv(
    annotation_df,
    OUTPUT_DIR
    / "target_annotation_completeness.tsv"
)


# ============================================================
# 26. TARGET HIERARCHY CARDINALITY
# ============================================================

log("\n" + "=" * 80)
log("TARGET HIERARCHY CARDINALITY")
log("=" * 80)

hierarchy_rows = []

for col in [

    "Target_Class_Level1",
    "Target_Class_Level2",
    "Target_Class_Level3",
    "Target_type"

]:

    if col not in targets.columns:
        continue

    n_unique = int(
        targets[col].nunique(
            dropna=True
        )
    )

    hierarchy_rows.append({

        "field":
            col,

        "unique_nonmissing_values":
            n_unique

    })

    log(
        f"{col:30s}: "
        f"{n_unique:,}"
    )


write_tsv(
    pd.DataFrame(hierarchy_rows),
    OUTPUT_DIR
    / "target_hierarchy_cardinality.tsv"
)


# ============================================================
# 27. CLEAN MOLECULE TABLE
# ============================================================

log("\n" + "=" * 80)
log("BUILD CLEAN MOLECULE TABLE")
log("=" * 80)

molecule_columns = [

    "np_id",
    "pref_name",
    "iupac_name",
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
    "SMILES"

]

molecule_columns = [
    c
    for c in molecule_columns
    if c in molecules.columns
]

clean_molecules = (
    molecules[
        molecule_columns
    ]
    .copy()
)

clean_molecules = (
    clean_molecules[
        clean_molecules[
            "np_id"
        ].notna()
    ]
    .copy()
)

clean_molecules = (
    clean_molecules
    .drop_duplicates(
        subset=["np_id"],
        keep="first"
    )
    .copy()
)

write_tsv(
    clean_molecules,
    OUTPUT_DIR
    / "cmaup_molecules_clean.tsv"
)

log(
    f"Clean molecules: "
    f"{len(clean_molecules):,}"
)


# ============================================================
# 28. CLEAN ASSOCIATION TABLE
# ============================================================

log("\n" + "=" * 80)
log("BUILD CLEAN MOLECULE-TARGET ASSOCIATION TABLE")
log("=" * 80)

association_columns = [

    "Ingredient_ID",
    "Target_ID",
    "Activity_Type",
    "Activity_Relationship",
    "Activity_Value",
    "Activity_Unit",
    "Reference_ID",
    "Reference_ID_Type"

]

clean_associations = (
    associations[
        association_columns
    ]
    .copy()
)

clean_associations = (
    clean_associations[
        clean_associations[
            "Ingredient_ID"
        ].notna()
        &
        clean_associations[
            "Target_ID"
        ].notna()
    ]
    .copy()
)

# IMPORTANT:
#
# Do NOT collapse multiple activity records here.
#
# Multiple measurements/references can represent
# independent evidence and will be required by the
# downstream evidence-weighting system.

write_tsv(
    clean_associations,
    OUTPUT_DIR
    / "cmaup_molecule_target_associations_clean.tsv"
)

log(
    f"Clean association records: "
    f"{len(clean_associations):,}"
)


# ============================================================
# 29. CLEAN TARGET TABLE
# ============================================================

log("\n" + "=" * 80)
log("BUILD CLEAN TARGET TABLE")
log("=" * 80)

target_columns = [

    "Target_ID",
    "Gene_Symbol",
    "Protein_Name",
    "Uniprot_ID",
    "ChEMBL_ID",
    "TTD_ID",
    "if_DTP",
    "if_CYP",
    "if_therapeutic_target",
    "Target_Class_Level1",
    "Target_Class_Level2",
    "Target_Class_Level3",
    "Target_Class_level_displayed",
    "Target_type"

]

target_columns = [
    c
    for c in target_columns
    if c in targets.columns
]

clean_targets = (
    targets[
        target_columns
    ]
    .copy()
)

clean_targets = (
    clean_targets[
        clean_targets[
            "Target_ID"
        ].notna()
    ]
    .drop_duplicates(
        subset=["Target_ID"],
        keep="first"
    )
    .copy()
)

write_tsv(
    clean_targets,
    OUTPUT_DIR
    / "cmaup_targets_clean.tsv"
)

log(
    f"Clean targets: "
    f"{len(clean_targets):,}"
)


# ============================================================
# 30. MASTER MOLECULE-TARGET EVIDENCE TABLE
# ============================================================

log("\n" + "=" * 80)
log("BUILD MASTER MOLECULE-TARGET EVIDENCE TABLE")
log("=" * 80)

master = clean_associations.merge(

    clean_molecules[
        [
            "np_id",
            "pref_name",
            "SMILES",
            "InChIKey"
        ]
    ],

    left_on="Ingredient_ID",

    right_on="np_id",

    how="left",

    validate="many_to_one"
)


master = master.merge(

    clean_targets,

    on="Target_ID",

    how="left",

    validate="many_to_one"
)


master["molecule_mapping_status"] = np.where(

    master["np_id"].notna(),

    "MAPPED",

    "UNMAPPED"

)


master["target_mapping_status"] = np.where(

    master["Target_Class_Level1"].notna()
    |
    master["Protein_Name"].notna()
    |
    master["Gene_Symbol"].notna(),

    "MAPPED",

    "UNMAPPED"

)


write_tsv(
    master,
    OUTPUT_DIR
    / "cmaup_molecule_target_master_evidence.tsv"
)

log(
    f"Master evidence records: "
    f"{len(master):,}"
)


# ============================================================
# 31. MASTER MAPPING QC
# ============================================================

log("\n" + "=" * 80)
log("MASTER MAPPING QC")
log("=" * 80)

mapping_qc = pd.DataFrame([

    {
        "metric":
            "total_evidence_records",

        "value":
            len(master)
    },

    {
        "metric":
            "molecule_mapped_records",

        "value":
            int(
                master[
                    "molecule_mapping_status"
                ]
                .eq("MAPPED")
                .sum()
            )
    },

    {
        "metric":
            "molecule_unmapped_records",

        "value":
            int(
                master[
                    "molecule_mapping_status"
                ]
                .eq("UNMAPPED")
                .sum()
            )
    },

    {
        "metric":
            "target_mapped_records",

        "value":
            int(
                master[
                    "target_mapping_status"
                ]
                .eq("MAPPED")
                .sum()
            )
    },

    {
        "metric":
            "target_unmapped_records",

        "value":
            int(
                master[
                    "target_mapping_status"
                ]
                .eq("UNMAPPED")
                .sum()
            )
    }

])

write_tsv(
    mapping_qc,
    OUTPUT_DIR
    / "master_mapping_qc.tsv"
)


# ============================================================
# 32. TARGETS PER MOLECULE
# ============================================================

log("\n" + "=" * 80)
log("TARGETS PER MOLECULE")
log("=" * 80)

targets_per_molecule = (

    clean_associations

    .groupby(
        "Ingredient_ID"
    )["Target_ID"]

    .nunique()

    .reset_index(
        name="n_unique_targets"
    )

)

log(
    f"Molecules with ≥1 target: "
    f"{len(targets_per_molecule):,}"
)

log(
    f"Mean targets/molecule: "
    f"{targets_per_molecule['n_unique_targets'].mean():.3f}"
)

log(
    f"Median targets/molecule: "
    f"{targets_per_molecule['n_unique_targets'].median():.3f}"
)

log(
    f"Maximum targets/molecule: "
    f"{targets_per_molecule['n_unique_targets'].max():,}"
)

write_tsv(
    targets_per_molecule,
    OUTPUT_DIR
    / "targets_per_molecule.tsv"
)


# ============================================================
# 33. MOLECULES PER TARGET
# ============================================================

log("\n" + "=" * 80)
log("MOLECULES PER TARGET")
log("=" * 80)

molecules_per_target = (

    clean_associations

    .groupby(
        "Target_ID"
    )["Ingredient_ID"]

    .nunique()

    .reset_index(
        name="n_unique_molecules"
    )

)

log(
    f"Targets with ≥1 molecule: "
    f"{len(molecules_per_target):,}"
)

log(
    f"Mean molecules/target: "
    f"{molecules_per_target['n_unique_molecules'].mean():.3f}"
)

log(
    f"Median molecules/target: "
    f"{molecules_per_target['n_unique_molecules'].median():.3f}"
)

log(
    f"Maximum molecules/target: "
    f"{molecules_per_target['n_unique_molecules'].max():,}"
)

write_tsv(
    molecules_per_target,
    OUTPUT_DIR
    / "molecules_per_target.tsv"
)


# ============================================================
# 34. REFERENCE SUPPORT PER MOLECULE-TARGET PAIR
# ============================================================

log("\n" + "=" * 80)
log("REFERENCE SUPPORT PER MOLECULE-TARGET PAIR")
log("=" * 80)

pair_reference_support = (

    clean_associations

    .groupby(
        [
            "Ingredient_ID",
            "Target_ID"
        ]
    )["Reference_ID"]

    .nunique()

    .reset_index(
        name="n_unique_references"
    )

)

write_tsv(
    pair_reference_support,
    OUTPUT_DIR
    / "molecule_target_reference_support.tsv"
)


# ============================================================
# 35. BIOACTIVITY TRAIN/TEST TARGET COVERAGE
# ============================================================

log("\n" + "=" * 80)
log("BIOACTIVITY TRAIN/TEST TARGET COVERAGE")
log("=" * 80)

split_coverage_rows = []

association_molecule_ids = set(

    clean_associations[
        "Ingredient_ID"
    ]
    .dropna()
    .astype(str)

)


for split_name, split_file in [

    ("train", TRAIN_FILE),

    ("test", TEST_FILE)

]:

    if not split_file.exists():

        log(
            f"{split_name}: file not found"
        )

        continue


    split_df = pd.read_csv(

        split_file,

        sep="\t",

        dtype=str,

        low_memory=False

    )


    split_df = normalize_columns(
        split_df
    )


    if "np_id" not in split_df.columns:

        log(
            f"{split_name}: np_id column missing"
        )

        continue


    split_ids = set(

        split_df[
            "np_id"
        ]
        .dropna()
        .astype(str)

    )


    overlap = (
        split_ids
        &
        association_molecule_ids
    )


    coverage = (

        len(overlap)
        /
        len(split_ids)
        *
        100

        if len(split_ids) > 0

        else np.nan

    )


    split_coverage_rows.append({

        "split":
            split_name,

        "n_molecules":
            len(split_ids),

        "n_molecules_with_known_target_association":
            len(overlap),

        "target_association_coverage_percent":
            coverage

    })


    log(
        f"{split_name:10s}"
        f" molecules={len(split_ids):,}"
        f" target_annotated={len(overlap):,}"
        f" coverage={coverage:.3f}%"
    )


write_tsv(

    pd.DataFrame(
        split_coverage_rows
    ),

    OUTPUT_DIR
    / "bioactivity_split_target_coverage.tsv"

)


# ============================================================
# 36. ASSOCIATED TARGET CLASS DISTRIBUTIONS
# ============================================================

log("\n" + "=" * 80)
log("ASSOCIATED TARGET CLASS DISTRIBUTIONS")
log("=" * 80)

associated_target_ids = set(

    clean_associations[
        "Target_ID"
    ]
    .dropna()
    .astype(str)

)


associated_targets = (

    clean_targets[
        clean_targets[
            "Target_ID"
        ]
        .astype(str)
        .isin(
            associated_target_ids
        )
    ]
    .copy()

)


for level in [

    "Target_Class_Level1",
    "Target_Class_Level2",
    "Target_Class_Level3"

]:

    if level not in associated_targets.columns:
        continue


    distribution = (

        associated_targets[level]

        .fillna("<MISSING>")

        .value_counts(
            dropna=False
        )

        .rename_axis(level)

        .reset_index(
            name="n_targets"
        )

    )


    distribution["percent"] = (

        distribution["n_targets"]
        /
        len(associated_targets)
        *
        100

    )


    output_name = (

        level
        .lower()
        .replace(
            " ",
            "_"
        )

    )


    write_tsv(

        distribution,

        OUTPUT_DIR
        / f"{output_name}_distribution.tsv"

    )


    log(
        f"{level}: "
        f"{len(distribution):,} categories"
    )


# ============================================================
# 37. TARGET SUPPORT DISTRIBUTION
# ============================================================

log("\n" + "=" * 80)
log("TARGET SUPPORT DISTRIBUTION")
log("=" * 80)

support_distribution = (

    molecules_per_target[
        "n_unique_molecules"
    ]

    .value_counts()

    .sort_index()

    .rename_axis(
        "n_supporting_molecules"
    )

    .reset_index(
        name="n_targets"
    )

)

write_tsv(

    support_distribution,

    OUTPUT_DIR
    / "target_support_distribution.tsv"

)


# ============================================================
# 38. LOW-SUPPORT TARGET LIST
# ============================================================

low_support_targets = (

    molecules_per_target[
        molecules_per_target[
            "n_unique_molecules"
        ] <= 3
    ]

    .merge(
        clean_targets,
        on="Target_ID",
        how="left"
    )

    .sort_values(
        [
            "n_unique_molecules",
            "Target_ID"
        ]
    )

)

write_tsv(

    low_support_targets,

    OUTPUT_DIR
    / "low_support_targets_n_le_3.tsv"

)


# ============================================================
# 39. HIGH-SUPPORT TARGET LIST
# ============================================================

high_support_targets = (

    molecules_per_target[
        molecules_per_target[
            "n_unique_molecules"
        ] >= 10
    ]

    .merge(
        clean_targets,
        on="Target_ID",
        how="left"
    )

    .sort_values(
        "n_unique_molecules",
        ascending=False
    )

)

write_tsv(

    high_support_targets,

    OUTPUT_DIR
    / "high_support_targets_n_ge_10.tsv"

)


# ============================================================
# 40. DATASET MANIFEST
# ============================================================

log("\n" + "=" * 80)
log("DATASET MANIFEST")
log("=" * 80)

manifest = {

    "timestamp":
        datetime.now().isoformat(),

    "project_directory":
        str(PULP_DIR),

    "output_directory":
        str(OUTPUT_DIR),

    "source_files": {

        "molecules":
            str(MOLECULE_FILE),

        "associations":
            str(ASSOCIATION_FILE),

        "targets":
            str(TARGET_FILE)

    },

    "source_dataset_sizes": {

        "molecules":
            int(len(molecules)),

        "associations":
            int(len(associations)),

        "targets":
            int(len(targets))

    },

    "clean_dataset_sizes": {

        "molecules":
            int(len(clean_molecules)),

        "associations":
            int(len(clean_associations)),

        "targets":
            int(len(clean_targets))

    },

    "association_coverage": {

        "unique_molecules":
            int(unique_association_molecules),

        "unique_targets":
            int(unique_association_targets),

        "unique_references":
            int(unique_references)

    },

    "mapping": {

        "mapped_molecule_ids":
            int(len(mapped_molecules)),

        "unmapped_molecule_ids":
            int(len(unmapped_molecules)),

        "mapped_target_ids":
            int(len(mapped_targets)),

        "unmapped_target_ids":
            int(len(unmapped_targets))

    },

    "methodological_rule":

        "Absence of a CMAUP target association is not "
        "treated as negative evidence.",

    "downstream_model":

        "Similarity-weighted target hypothesis generation",

    "feature_representation":

        "Existing 40-feature bioactivity representation",

    "validation_strategy":

        "Retrospective target recovery"

}


with open(

    OUTPUT_DIR
    / "dataset_manifest.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(
        manifest,
        f,
        indent=4
    )


# ============================================================
# 41. FINAL QC
# ============================================================

log("\n" + "=" * 80)
log("FINAL QC")
log("=" * 80)

warnings = []

if missing_np_id > 0:

    warnings.append(
        f"Missing molecule IDs: {missing_np_id:,}"
    )

if duplicate_np_id_rows > 0:

    warnings.append(
        f"Duplicate molecule ID rows: "
        f"{duplicate_np_id_rows:,}"
    )

if len(unmapped_molecules) > 0:

    warnings.append(
        f"Unmapped association molecule IDs: "
        f"{len(unmapped_molecules):,}"
    )

if len(unmapped_targets) > 0:

    warnings.append(
        f"Unmapped association target IDs: "
        f"{len(unmapped_targets):,}"
    )


if len(warnings) == 0:

    log(
        "OVERALL QC STATUS: PASS"
    )

else:

    log(
        "OVERALL QC STATUS: PASS WITH WARNINGS"
    )

    for warning in warnings:

        log(
            f"WARNING: {warning}"
        )


# ============================================================
# 42. SAVE LOG
# ============================================================

save_log()


# ============================================================
# 43. FINISH
# ============================================================

print("\n" + "=" * 80)
print("SCRIPT 01 COMPLETE")
print("=" * 80)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    f"\nAudit log:\n{LOG_FILE}"
)

print("\nMain files generated:")

print(
    "  cmaup_molecules_clean.tsv"
)

print(
    "  cmaup_molecule_target_associations_clean.tsv"
)

print(
    "  cmaup_targets_clean.tsv"
)

print(
    "  cmaup_molecule_target_master_evidence.tsv"
)

print(
    "  dataset_manifest.json"
)

print(
    "  01_audit_log.txt"
)

print("\nNext step:")
print(
    "Use the QC outputs from this script to design "
    "Script 02: target-evidence matrix construction."
)

print("=" * 80)
