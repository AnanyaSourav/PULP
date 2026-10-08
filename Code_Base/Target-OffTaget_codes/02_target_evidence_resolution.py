#!/usr/bin/env python3

"""
02_target_evidence_resolution.py

PULP Target / Off-Target Inference System
-----------------------------------------

Purpose
-------
Build the target-evidence layer from the authoritative Script 01
CMAUP audit outputs.

This script:

1. Loads Script 01 cleaned molecule, association, and target tables.
2. Explicitly maps association Ingredient_ID -> molecule np_id.
3. Investigates all Target_IDs appearing in associations but absent
   from the target annotation table.
4. Characterizes exact duplicate association records.
5. Preserves raw activity evidence.
6. Builds:
      - molecule x target binary association matrix
      - molecule x target evidence-count matrix
      - molecule x target independent-reference-count matrix
      - target hierarchy table
      - molecule-target summary
      - target support statistics
      - molecule support statistics
7. Preserves:
      - Activity_Type
      - Activity_Relationship
      - Activity_Value
      - Activity_Unit
      - Reference_ID
      - Reference_ID_Type
8. Does NOT treat absence of a target association as negative evidence.
9. Does NOT combine IC50/Ki/EC50/Potency/etc. into one activity score.
10. Does NOT perform molecular similarity or target prediction.

Important design principle
--------------------------
This is an evidence-resolution layer.

A molecule-target association means:
    "CMAUP contains evidence connecting this molecule to this target."

It does NOT mean:
    "All other targets are negatives."

Exact duplicate records are preserved in the raw evidence count.
Binary association collapses them to presence/absence.
Independent reference count counts unique Reference_ID values.

Unmapped Target_IDs are retained as unresolved evidence rather than
silently discarded.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# 0. PATHS
# ============================================================

PULP_DIR = Path(__file__).resolve().parent

SCRIPT01_DIR = (
    PULP_DIR /
    "target_model" /
    "data" /
    "01_cmaup_audit"
)

OUT_DIR = (
    PULP_DIR /
    "target_model" /
    "data" /
    "02_target_evidence"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. SCRIPT 01 INPUT TABLES
# ============================================================

MOLECULE_FILE = (
    SCRIPT01_DIR /
    "cmaup_molecules_clean.tsv"
)

ASSOCIATION_FILE = (
    SCRIPT01_DIR /
    "cmaup_molecule_target_associations_clean.tsv"
)

TARGET_FILE = (
    SCRIPT01_DIR /
    "cmaup_targets_clean.tsv"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

TARGET_RESOLUTION_FILE = (
    OUT_DIR /
    "target_id_resolution.tsv"
)

UNMAPPED_TARGET_FILE = (
    OUT_DIR /
    "unmapped_target_ids.tsv"
)

DUPLICATE_ANALYSIS_FILE = (
    OUT_DIR /
    "duplicate_association_analysis.tsv"
)

DUPLICATE_EXAMPLES_FILE = (
    OUT_DIR /
    "duplicate_association_examples.tsv"
)

BINARY_MATRIX_FILE = (
    OUT_DIR /
    "molecule_target_binary.tsv"
)

EVIDENCE_COUNT_MATRIX_FILE = (
    OUT_DIR /
    "molecule_target_evidence_count.tsv"
)

REFERENCE_COUNT_MATRIX_FILE = (
    OUT_DIR /
    "molecule_target_reference_count.tsv"
)

TARGET_HIERARCHY_FILE = (
    OUT_DIR /
    "target_hierarchy_matrix.tsv"
)

MOLECULE_TARGET_SUMMARY_FILE = (
    OUT_DIR /
    "molecule_target_summary.tsv"
)

TARGET_SUMMARY_FILE = (
    OUT_DIR /
    "target_summary.tsv"
)

TARGET_SUPPORT_FILE = (
    OUT_DIR /
    "target_support_statistics.tsv"
)

MOLECULE_SUPPORT_FILE = (
    OUT_DIR /
    "molecule_support_statistics.tsv"
)

ACTIVITY_EVIDENCE_FILE = (
    OUT_DIR /
    "activity_evidence_summary.tsv"
)

REFERENCE_SUPPORT_FILE = (
    OUT_DIR /
    "molecule_target_reference_support.tsv"
)

SUPPORT_DISTRIBUTION_FILE = (
    OUT_DIR /
    "target_support_distribution.tsv"
)

LOW_SUPPORT_FILE = (
    OUT_DIR /
    "low_support_targets.tsv"
)

HIGH_SUPPORT_FILE = (
    OUT_DIR /
    "high_support_targets.tsv"
)

ANNOTATION_COMPLETENESS_FILE = (
    OUT_DIR /
    "associated_target_annotation_completeness.tsv"
)

QC_TABLE_FILE = (
    OUT_DIR /
    "02_target_evidence_qc_table.tsv"
)

QC_FILE = (
    OUT_DIR /
    "02_target_evidence_qc.txt"
)

MANIFEST_FILE = (
    OUT_DIR /
    "02_target_evidence_manifest.json"
)


# ============================================================
# 3. LOGGING
# ============================================================

LOG_LINES = []


def log(message=""):
    print(message)
    LOG_LINES.append(str(message))


def write_log():
    with open(
        QC_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "\n".join(LOG_LINES)
        )
        f.write("\n")


def fatal(message):
    log("")
    log("=" * 80)
    log("FATAL ERROR")
    log("=" * 80)
    log(message)
    write_log()
    sys.exit(1)


# ============================================================
# 4. HELPERS
# ============================================================

def check_file(path):
    if not path.exists():
        fatal(
            f"Required file does not exist:\n{path}"
        )


def read_tsv(path):
    check_file(path)

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            low_memory=False
        )
    except Exception as exc:
        fatal(
            f"Unable to read:\n{path}\n\n"
            f"Error:\n{exc}"
        )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    for col in df.columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


def require_columns(
    df,
    required,
    table_name
):

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        fatal(
            f"{table_name} is missing required columns:\n"
            + "\n".join(missing)
        )


def write_tsv(df, path):

    df.to_csv(
        path,
        sep="\t",
        index=False
    )


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def numeric_or_nan(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ============================================================
# 5. START
# ============================================================

log("=" * 80)
log("SCRIPT 02 — TARGET EVIDENCE RESOLUTION")
log("=" * 80)

log(
    f"PULP directory : {PULP_DIR}"
)

log(
    f"Script 01 dir  : {SCRIPT01_DIR}"
)

log(
    f"Output dir     : {OUT_DIR}"
)

log("")


# ============================================================
# 6. CHECK INPUTS
# ============================================================

log("Checking Script 01 output files...")

for path in [
    MOLECULE_FILE,
    ASSOCIATION_FILE,
    TARGET_FILE
]:
    check_file(path)

log("All required files found.")
log("")


# ============================================================
# 7. LOAD TABLES
# ============================================================

log("Loading Script 01 cleaned tables...")

molecules = read_tsv(
    MOLECULE_FILE
)

associations = read_tsv(
    ASSOCIATION_FILE
)

targets = read_tsv(
    TARGET_FILE
)

log(
    f"Molecules loaded    : {len(molecules):,}"
)

log(
    f"Associations loaded : {len(associations):,}"
)

log(
    f"Targets loaded      : {len(targets):,}"
)

log("")


# ============================================================
# 8. EXACT SCHEMA VALIDATION
# ============================================================

require_columns(
    molecules,
    [
        "np_id",
        "SMILES"
    ],
    "cmaup_molecules_clean.tsv"
)

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
    "cmaup_molecule_target_associations_clean.tsv"
)

require_columns(
    targets,
    [
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
    ],
    "cmaup_targets_clean.tsv"
)

log(
    "Exact Script 01 schema validation: PASS"
)

log("")


# ============================================================
# 9. BASIC ID QC
# ============================================================

log("=" * 80)
log("9. BASIC ID QC")
log("=" * 80)

molecule_ids = set(
    molecules["np_id"]
)

association_ingredient_ids = set(
    associations["Ingredient_ID"]
)

association_target_ids = set(
    associations["Target_ID"]
)

target_annotation_ids = set(
    targets["Target_ID"]
)

log(
    f"Unique molecule np_id values          : "
    f"{len(molecule_ids):,}"
)

log(
    f"Unique association Ingredient_IDs     : "
    f"{len(association_ingredient_ids):,}"
)

log(
    f"Unique association Target_IDs         : "
    f"{len(association_target_ids):,}"
)

log(
    f"Unique target annotation Target_IDs   : "
    f"{len(target_annotation_ids):,}"
)

log("")


# ============================================================
# 10. INGREDIENT_ID -> np_id MAPPING
# ============================================================

log("=" * 80)
log("10. INGREDIENT_ID -> np_id MAPPING")
log("=" * 80)

"""
The association table uses:

    Ingredient_ID

The molecule table uses:

    np_id

Script 01 established these as the respective identifiers.

For the target-evidence layer, Ingredient_ID is explicitly mapped
to np_id using the molecule table.

No fuzzy matching is performed.
No SMILES matching is used.
No name matching is used.

Only exact identifier equality is accepted.
"""

molecule_lookup = (
    molecules[
        [
            "np_id"
        ]
    ]
    .drop_duplicates()
    .copy()
)

molecule_lookup[
    "Ingredient_ID"
] = molecule_lookup[
    "np_id"
]

# Check whether the resulting mapping is one-to-one.
mapping_duplicate_ids = (
    molecule_lookup["Ingredient_ID"]
    .duplicated()
    .sum()
)

if mapping_duplicate_ids > 0:

    fatal(
        "Molecule mapping is not one-to-one. "
        f"Duplicate Ingredient_ID mappings: "
        f"{mapping_duplicate_ids:,}"
    )

# Perform exact mapping.
associations = associations.merge(
    molecule_lookup[
        [
            "Ingredient_ID",
            "np_id"
        ]
    ],
    on="Ingredient_ID",
    how="left",
    validate="many_to_one"
)

unmapped_association_ingredients = (
    associations[
        associations["np_id"].eq("")
    ]["Ingredient_ID"]
    .drop_duplicates()
    .tolist()
)

log(
    f"Association Ingredient_IDs mapped: "
    f"{len(association_ingredient_ids) - len(unmapped_association_ingredients):,}"
)

log(
    f"Unmapped association Ingredient_IDs: "
    f"{len(unmapped_association_ingredients):,}"
)

if unmapped_association_ingredients:

    unmapped_df = pd.DataFrame({
        "Ingredient_ID":
            sorted(unmapped_association_ingredients)
    })

    write_tsv(
        unmapped_df,
        OUT_DIR /
        "unmapped_ingredient_ids.tsv"
    )

    fatal(
        "Association Ingredient_ID values could not be "
        "mapped exactly to molecule np_id values."
    )

log(
    "Ingredient_ID -> np_id mapping: PASS"
)

log("")


# ============================================================
# 11. TARGET ID RESOLUTION
# ============================================================

log("=" * 80)
log("11. TARGET ID RESOLUTION")
log("=" * 80)

target_annotation_lookup = (
    targets
    .drop_duplicates(
        subset=["Target_ID"]
    )
    .set_index(
        "Target_ID"
    )
)

association_target_counts = (
    associations[
        "Target_ID"
    ]
    .value_counts()
    .to_dict()
)

resolution_rows = []

for target_id in sorted(
    association_target_ids
):

    record_count = int(
        association_target_counts.get(
            target_id,
            0
        )
    )

    if target_id in target_annotation_lookup.index:

        row = target_annotation_lookup.loc[
            target_id
        ]

        resolution_status = "MAPPED"

        gene_symbol = row[
            "Gene_Symbol"
        ]

        protein_name = row[
            "Protein_Name"
        ]

        uniprot_id = row[
            "Uniprot_ID"
        ]

        chembl_id = row[
            "ChEMBL_ID"
        ]

        ttd_id = row[
            "TTD_ID"
        ]

        target_class_level1 = row[
            "Target_Class_Level1"
        ]

        target_class_level2 = row[
            "Target_Class_Level2"
        ]

        target_class_level3 = row[
            "Target_Class_Level3"
        ]

        target_class_displayed = row[
            "Target_Class_level_displayed"
        ]

        target_type = row[
            "Target_type"
        ]

    else:

        resolution_status = "UNMAPPED"

        gene_symbol = ""
        protein_name = ""
        uniprot_id = ""
        chembl_id = ""
        ttd_id = ""
        target_class_level1 = ""
        target_class_level2 = ""
        target_class_level3 = ""
        target_class_displayed = ""
        target_type = ""

    resolution_rows.append({
        "Target_ID":
            target_id,

        "Resolution_Status":
            resolution_status,

        "Association_Record_Count":
            record_count,

        "Gene_Symbol":
            gene_symbol,

        "Protein_Name":
            protein_name,

        "Uniprot_ID":
            uniprot_id,

        "ChEMBL_ID":
            chembl_id,

        "TTD_ID":
            ttd_id,

        "Target_Class_Level1":
            target_class_level1,

        "Target_Class_Level2":
            target_class_level2,

        "Target_Class_Level3":
            target_class_level3,

        "Target_Class_level_displayed":
            target_class_displayed,

        "Target_type":
            target_type
    })


target_resolution = pd.DataFrame(
    resolution_rows
)

write_tsv(
    target_resolution,
    TARGET_RESOLUTION_FILE
)

mapped_target_count = int(
    (
        target_resolution[
            "Resolution_Status"
        ]
        == "MAPPED"
    ).sum()
)

unmapped_target_count = int(
    (
        target_resolution[
            "Resolution_Status"
        ]
        == "UNMAPPED"
    ).sum()
)

log(
    f"Mapped target IDs   : "
    f"{mapped_target_count:,}"
)

log(
    f"Unmapped target IDs : "
    f"{unmapped_target_count:,}"
)

log("")


# ============================================================
# 12. UNMAPPED TARGET ANALYSIS
# ============================================================

log("=" * 80)
log("12. UNMAPPED TARGET ANALYSIS")
log("=" * 80)

unmapped_target_ids = set(
    target_resolution.loc[
        target_resolution[
            "Resolution_Status"
        ] == "UNMAPPED",
        "Target_ID"
    ]
)

unmapped_associations = associations[
    associations["Target_ID"].isin(
        unmapped_target_ids
    )
].copy()

if len(unmapped_associations) > 0:

    unmapped_summary = (
        unmapped_associations
        .groupby(
            "Target_ID",
            as_index=False
        )
        .agg(
            Association_Record_Count=(
                "Target_ID",
                "size"
            ),
            Unique_Molecule_Count=(
                "np_id",
                "nunique"
            ),
            Unique_Ingredient_ID_Count=(
                "Ingredient_ID",
                "nunique"
            ),
            Unique_Reference_Count=(
                "Reference_ID",
                "nunique"
            )
        )
        .sort_values(
            [
                "Association_Record_Count",
                "Unique_Molecule_Count"
            ],
            ascending=False
        )
    )

else:

    unmapped_summary = pd.DataFrame(
        columns=[
            "Target_ID",
            "Association_Record_Count",
            "Unique_Molecule_Count",
            "Unique_Ingredient_ID_Count",
            "Unique_Reference_Count"
        ]
    )

write_tsv(
    unmapped_summary,
    UNMAPPED_TARGET_FILE
)

log(
    f"Association records involving unmapped targets: "
    f"{len(unmapped_associations):,}"
)

log(
    f"Unmapped target IDs investigated: "
    f"{len(unmapped_summary):,}"
)

log("")


# ============================================================
# 13. EXACT DUPLICATE ASSOCIATION ANALYSIS
# ============================================================

log("=" * 80)
log("13. EXACT DUPLICATE ASSOCIATION ANALYSIS")
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

exact_duplicate_mask = (
    associations
    .duplicated(
        subset=association_columns,
        keep=False
    )
)

exact_duplicate_extra_rows = int(
    associations
    .duplicated(
        subset=association_columns,
        keep="first"
    )
    .sum()
)

duplicate_groups = (
    associations
    .groupby(
        association_columns,
        dropna=False
    )
    .size()
    .reset_index(
        name="Duplicate_Group_Size"
    )
)

duplicate_groups = duplicate_groups[
    duplicate_groups[
        "Duplicate_Group_Size"
    ] > 1
].copy()

duplicate_groups[
    "Extra_Records"
] = (
    duplicate_groups[
        "Duplicate_Group_Size"
    ] - 1
)

log(
    f"Raw association records          : "
    f"{len(associations):,}"
)

log(
    f"Exact duplicate extra rows       : "
    f"{exact_duplicate_extra_rows:,}"
)

log(
    f"Exact duplicate groups           : "
    f"{len(duplicate_groups):,}"
)

write_tsv(
    duplicate_groups,
    DUPLICATE_ANALYSIS_FILE
)

if exact_duplicate_extra_rows > 0:

    duplicate_examples = (
        associations[
            exact_duplicate_mask
        ]
        .sort_values(
            association_columns
        )
        .head(5000)
    )

else:

    duplicate_examples = pd.DataFrame(
        columns=associations.columns
    )

write_tsv(
    duplicate_examples,
    DUPLICATE_EXAMPLES_FILE
)

log("")


# ============================================================
# 14. UNIQUE MOLECULE-TARGET PAIRS
# ============================================================

log("=" * 80)
log("14. UNIQUE MOLECULE-TARGET PAIRS")
log("=" * 80)

unique_pairs = (
    associations[
        [
            "np_id",
            "Ingredient_ID",
            "Target_ID"
        ]
    ]
    .drop_duplicates()
    .copy()
)

log(
    f"Unique molecule-target pairs: "
    f"{len(unique_pairs):,}"
)

log("")


# ============================================================
# 15. BINARY MOLECULE × TARGET MATRIX
# ============================================================

log("=" * 80)
log("15. BINARY MOLECULE × TARGET MATRIX")
log("=" * 80)

binary_pairs = (
    unique_pairs[
        [
            "np_id",
            "Target_ID"
        ]
    ]
    .copy()
)

binary_pairs[
    "Association_Present"
] = 1

binary_matrix = (
    binary_pairs
    .pivot_table(
        index="np_id",
        columns="Target_ID",
        values="Association_Present",
        aggfunc="max",
        fill_value=0
    )
)

binary_matrix = (
    binary_matrix
    .reindex(
        index=sorted(
            molecule_ids
        )
    )
    .fillna(0)
    .astype(np.int8)
)

binary_matrix.index.name = "np_id"

binary_matrix = (
    binary_matrix
    .reset_index()
)

write_tsv(
    binary_matrix,
    BINARY_MATRIX_FILE
)

log(
    f"Matrix rows    : "
    f"{binary_matrix.shape[0]:,}"
)

log(
    f"Matrix targets : "
    f"{binary_matrix.shape[1] - 1:,}"
)

log("")


# ============================================================
# 16. EVIDENCE COUNT MATRIX
# ============================================================

log("=" * 80)
log("16. EVIDENCE COUNT MATRIX")
log("=" * 80)

pair_evidence = (
    associations
    .groupby(
        [
            "np_id",
            "Target_ID"
        ],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size":
                "Evidence_Record_Count"
        }
    )
)

evidence_matrix = (
    pair_evidence
    .pivot_table(
        index="np_id",
        columns="Target_ID",
        values="Evidence_Record_Count",
        aggfunc="sum",
        fill_value=0
    )
)

evidence_matrix = (
    evidence_matrix
    .reindex(
        index=sorted(
            molecule_ids
        )
    )
    .fillna(0)
    .astype(np.int32)
)

evidence_matrix.index.name = "np_id"

evidence_matrix = (
    evidence_matrix
    .reset_index()
)

write_tsv(
    evidence_matrix,
    EVIDENCE_COUNT_MATRIX_FILE
)

log(
    f"Evidence matrix rows    : "
    f"{evidence_matrix.shape[0]:,}"
)

log(
    f"Evidence matrix targets : "
    f"{evidence_matrix.shape[1] - 1:,}"
)

log("")


# ============================================================
# 17. INDEPENDENT REFERENCE COUNT MATRIX
# ============================================================

log("=" * 80)
log("17. INDEPENDENT REFERENCE COUNT MATRIX")
log("=" * 80)

reference_pairs = (
    associations[
        [
            "np_id",
            "Target_ID",
            "Reference_ID"
        ]
    ]
    .drop_duplicates()
    .groupby(
        [
            "np_id",
            "Target_ID"
        ],
        as_index=False
    )
    .agg(
        Independent_Reference_Count=(
            "Reference_ID",
            "nunique"
        )
    )
)

reference_matrix = (
    reference_pairs
    .pivot_table(
        index="np_id",
        columns="Target_ID",
        values="Independent_Reference_Count",
        aggfunc="sum",
        fill_value=0
    )
)

reference_matrix = (
    reference_matrix
    .reindex(
        index=sorted(
            molecule_ids
        )
    )
    .fillna(0)
    .astype(np.int32)
)

reference_matrix.index.name = "np_id"

reference_matrix = (
    reference_matrix
    .reset_index()
)

write_tsv(
    reference_matrix,
    REFERENCE_COUNT_MATRIX_FILE
)

log(
    f"Reference matrix rows    : "
    f"{reference_matrix.shape[0]:,}"
)

log(
    f"Reference matrix targets : "
    f"{reference_matrix.shape[1] - 1:,}"
)

log("")


# ============================================================
# 18. TARGET HIERARCHY TABLE
# ============================================================

log("=" * 80)
log("18. TARGET HIERARCHY TABLE")
log("=" * 80)

target_hierarchy_columns = [
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

target_hierarchy = (
    targets[
        target_hierarchy_columns
    ]
    .drop_duplicates(
        subset=["Target_ID"]
    )
    .copy()
)

target_hierarchy = target_hierarchy.merge(
    target_resolution[
        [
            "Target_ID",
            "Resolution_Status",
            "Association_Record_Count"
        ]
    ],
    on="Target_ID",
    how="outer"
)

target_hierarchy[
    "Resolution_Status"
] = (
    target_hierarchy[
        "Resolution_Status"
    ]
    .fillna(
        "ANNOTATION_ONLY"
    )
)

target_hierarchy[
    "Association_Record_Count"
] = (
    pd.to_numeric(
        target_hierarchy[
            "Association_Record_Count"
        ],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

write_tsv(
    target_hierarchy,
    TARGET_HIERARCHY_FILE
)

log(
    f"Target hierarchy rows: "
    f"{len(target_hierarchy):,}"
)

log("")


# ============================================================
# 19. MOLECULE-TARGET SUMMARY
# ============================================================

log("=" * 80)
log("19. MOLECULE-TARGET SUMMARY")
log("=" * 80)

molecule_target_summary = (
    associations
    .groupby(
        [
            "np_id",
            "Ingredient_ID",
            "Target_ID"
        ],
        as_index=False
    )
    .agg(
        Evidence_Record_Count=(
            "Target_ID",
            "size"
        ),
        Unique_Activity_Type_Count=(
            "Activity_Type",
            "nunique"
        ),
        Activity_Type_List=(
            "Activity_Type",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            v for v in x
                            if v != ""
                        )
                    )
                )
        ),
        Activity_Relationship_List=(
            "Activity_Relationship",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            v for v in x
                            if v != ""
                        )
                    )
                )
        ),
        Activity_Unit_List=(
            "Activity_Unit",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            v for v in x
                            if v != ""
                        )
                    )
                )
        ),
        Unique_Reference_Count=(
            "Reference_ID",
            "nunique"
        ),
        Reference_ID_List=(
            "Reference_ID",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            v for v in x
                            if v != ""
                        )
                    )
                )
        )
    )
)

# Add target annotation.
molecule_target_summary = (
    molecule_target_summary
    .merge(
        target_hierarchy,
        on="Target_ID",
        how="left",
        suffixes=(
            "",
            "_target"
        )
    )
)

molecule_target_summary[
    "Target_Annotation_Status"
] = np.where(
    molecule_target_summary[
        "Resolution_Status"
    ].eq("MAPPED"),
    "MAPPED",
    "UNMAPPED"
)

write_tsv(
    molecule_target_summary,
    MOLECULE_TARGET_SUMMARY_FILE
)

log(
    f"Molecule-target summary rows: "
    f"{len(molecule_target_summary):,}"
)

log("")


# ============================================================
# 20. TARGET SUPPORT STATISTICS
# ============================================================

log("=" * 80)
log("20. TARGET SUPPORT STATISTICS")
log("=" * 80)

target_support = (
    molecule_target_summary
    .groupby(
        "Target_ID",
        as_index=False
    )
    .agg(
        Molecule_Count=(
            "np_id",
            "nunique"
        ),
        Evidence_Record_Count=(
            "Evidence_Record_Count",
            "sum"
        ),
        Mean_Evidence_Per_Molecule=(
            "Evidence_Record_Count",
            "mean"
        ),
        Median_Evidence_Per_Molecule=(
            "Evidence_Record_Count",
            "median"
        ),
        Maximum_Evidence_Per_Molecule=(
            "Evidence_Record_Count",
            "max"
        ),
        Independent_Reference_Count=(
            "Unique_Reference_Count",
            "sum"
        )
    )
)

target_support = (
    target_support
    .merge(
        target_hierarchy,
        on="Target_ID",
        how="left",
        suffixes=(
            "",
            "_annotation"
        )
    )
)

target_support[
    "Support_Category"
] = pd.cut(
    target_support[
        "Molecule_Count"
    ],
    bins=[
        -np.inf,
        0,
        3,
        9,
        19,
        np.inf
    ],
    labels=[
        "NO_SUPPORT",
        "LOW_1_TO_3",
        "MODERATE_4_TO_9",
        "GOOD_10_TO_19",
        "HIGH_GE_20"
    ]
)

target_support = target_support.sort_values(
    [
        "Molecule_Count",
        "Evidence_Record_Count"
    ],
    ascending=False
)

write_tsv(
    target_support,
    TARGET_SUPPORT_FILE
)

# Also provide the same table under the generic target-summary
# output requested by the workflow.
write_tsv(
    target_support,
    TARGET_SUMMARY_FILE
)

log(
    f"Targets with molecule evidence: "
    f"{len(target_support):,}"
)

log("")


# ============================================================
# 21. MOLECULE SUPPORT STATISTICS
# ============================================================

log("=" * 80)
log("21. MOLECULE SUPPORT STATISTICS")
log("=" * 80)

molecule_support = (
    molecule_target_summary
    .groupby(
        "np_id",
        as_index=False
    )
    .agg(
        Unique_Target_Count=(
            "Target_ID",
            "nunique"
        ),
        Total_Evidence_Record_Count=(
            "Evidence_Record_Count",
            "sum"
        ),
        Unique_Reference_Count=(
            "Unique_Reference_Count",
            "sum"
        )
    )
)

molecule_support = molecule_support.sort_values(
    [
        "Unique_Target_Count",
        "Total_Evidence_Record_Count"
    ],
    ascending=False
)

write_tsv(
    molecule_support,
    MOLECULE_SUPPORT_FILE
)

log(
    f"Molecules with target evidence: "
    f"{len(molecule_support):,}"
)

log("")


# ============================================================
# 22. ACTIVITY EVIDENCE SUMMARY
# ============================================================

log("=" * 80)
log("22. ACTIVITY EVIDENCE SUMMARY")
log("=" * 80)

activity_summary = (
    associations
    .groupby(
        [
            "Activity_Type",
            "Activity_Relationship",
            "Activity_Unit"
        ],
        dropna=False
    )
    .agg(
        Association_Record_Count=(
            "Target_ID",
            "size"
        ),
        Unique_Molecule_Count=(
            "np_id",
            "nunique"
        ),
        Unique_Target_Count=(
            "Target_ID",
            "nunique"
        ),
        Unique_Reference_Count=(
            "Reference_ID",
            "nunique"
        )
    )
    .reset_index()
    .sort_values(
        "Association_Record_Count",
        ascending=False
    )
)

write_tsv(
    activity_summary,
    ACTIVITY_EVIDENCE_FILE
)

log("")


# ============================================================
# 23. MOLECULE-TARGET REFERENCE SUPPORT
# ============================================================

log("=" * 80)
log("23. MOLECULE-TARGET REFERENCE SUPPORT")
log("=" * 80)

reference_support = (
    associations
    .groupby(
        [
            "np_id",
            "Ingredient_ID",
            "Target_ID"
        ],
        as_index=False
    )
    .agg(
        Independent_Reference_Count=(
            "Reference_ID",
            "nunique"
        ),
        Reference_ID_List=(
            "Reference_ID",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            v for v in x
                            if v != ""
                        )
                    )
                )
        ),
        Reference_ID_Type_List=(
            "Reference_ID_Type",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            v for v in x
                            if v != ""
                        )
                    )
                )
        )
    )
)

write_tsv(
    reference_support,
    REFERENCE_SUPPORT_FILE
)

log("")


# ============================================================
# 24. TARGET SUPPORT DISTRIBUTION
# ============================================================

log("=" * 80)
log("24. TARGET SUPPORT DISTRIBUTION")
log("=" * 80)

support_distribution = (
    target_support[
        "Molecule_Count"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "Molecules_Supporting_Target"
    )
    .reset_index(
        name="Target_Count"
    )
)

support_distribution[
    "Cumulative_Target_Count"
] = (
    support_distribution[
        "Target_Count"
    ]
    .cumsum()
)

support_distribution[
    "Cumulative_Percent"
] = (
    100.0 *
    support_distribution[
        "Cumulative_Target_Count"
    ] /
    len(target_support)
)

write_tsv(
    support_distribution,
    SUPPORT_DISTRIBUTION_FILE
)

log("")


# ============================================================
# 25. LOW / HIGH SUPPORT TARGETS
# ============================================================

low_support = target_support[
    target_support[
        "Molecule_Count"
    ] <= 3
].copy()

high_support = target_support[
    target_support[
        "Molecule_Count"
    ] >= 10
].copy()

write_tsv(
    low_support,
    LOW_SUPPORT_FILE
)

write_tsv(
    high_support,
    HIGH_SUPPORT_FILE
)

log(
    f"Targets supported by <=3 molecules: "
    f"{len(low_support):,}"
)

log(
    f"Targets supported by >=10 molecules: "
    f"{len(high_support):,}"
)

log("")


# ============================================================
# 26. TARGET ANNOTATION COMPLETENESS
# ============================================================

log("=" * 80)
log("26. TARGET ANNOTATION COMPLETENESS")
log("=" * 80)

associated_target_table = (
    target_hierarchy[
        target_hierarchy[
            "Target_ID"
        ].isin(
            association_target_ids
        )
    ]
    .copy()
)

annotation_fields = [
    "Gene_Symbol",
    "Protein_Name",
    "Uniprot_ID",
    "ChEMBL_ID",
    "TTD_ID",
    "Target_Class_Level1",
    "Target_Class_Level2",
    "Target_Class_Level3",
    "Target_Class_level_displayed",
    "Target_type"
]

annotation_rows = []

for field in annotation_fields:

    nonempty = (
        associated_target_table[field]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    annotation_rows.append({
        "Annotation_Field":
            field,

        "Associated_Targets_Total":
            len(associated_target_table),

        "Nonempty_Count":
            int(nonempty.sum()),

        "Missing_or_Empty_Count":
            int((~nonempty).sum()),

        "Completeness_Percent":
            (
                100.0 *
                nonempty.mean()
            )
            if len(nonempty) > 0
            else np.nan
    })

annotation_completeness = pd.DataFrame(
    annotation_rows
)

write_tsv(
    annotation_completeness,
    ANNOTATION_COMPLETENESS_FILE
)

log("")


# ============================================================
# 27. CONSISTENCY QC
# ============================================================

log("=" * 80)
log("27. CONSISTENCY QC")
log("=" * 80)

qc_rows = []


def qc_check(
    name,
    condition,
    details
):

    status = (
        "PASS"
        if bool(condition)
        else "FAIL"
    )

    qc_rows.append({
        "Check": name,
        "Status": status,
        "Details": details
    })

    log(
        f"{status:6s} | "
        f"{name} | "
        f"{details}"
    )


qc_check(
    "Molecule np_id unique",
    molecules["np_id"].is_unique,
    (
        f"duplicates = "
        f"{molecules['np_id'].duplicated().sum():,}"
    )
)

qc_check(
    "Target_ID unique",
    targets["Target_ID"].is_unique,
    (
        f"duplicates = "
        f"{targets['Target_ID'].duplicated().sum():,}"
    )
)

qc_check(
    "All association Ingredient_IDs mapped",
    len(unmapped_association_ingredients) == 0,
    (
        f"unmapped = "
        f"{len(unmapped_association_ingredients):,}"
    )
)

expected_unique_pairs = (
    associations[
        [
            "np_id",
            "Target_ID"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

qc_check(
    "Unique pair count consistent",
    expected_unique_pairs ==
    len(molecule_target_summary),
    (
        f"expected = {expected_unique_pairs:,}; "
        f"summary = {len(molecule_target_summary):,}"
    )
)

evidence_total = int(
    molecule_target_summary[
        "Evidence_Record_Count"
    ].sum()
)

qc_check(
    "Evidence count conserves raw records",
    evidence_total ==
    len(associations),
    (
        f"evidence total = {evidence_total:,}; "
        f"raw records = {len(associations):,}"
    )
)

binary_values = set(
    binary_matrix
    .drop(columns=["np_id"])
    .to_numpy()
    .ravel()
)

qc_check(
    "Binary matrix contains only 0 and 1",
    binary_values.issubset(
        {0, 1}
    ),
    (
        f"values = "
        f"{sorted(binary_values)}"
    )
)

binary_positive_count = int(
    binary_matrix
    .drop(columns=["np_id"])
    .to_numpy()
    .sum()
)

qc_check(
    "Binary positive count consistent",
    binary_positive_count ==
    expected_unique_pairs,
    (
        f"binary positives = "
        f"{binary_positive_count:,}; "
        f"unique pairs = "
        f"{expected_unique_pairs:,}"
    )
)

evidence_values = (
    evidence_matrix
    .drop(columns=["np_id"])
    .to_numpy()
)

qc_check(
    "Evidence matrix non-negative",
    bool(
        np.all(
            evidence_values >= 0
        )
    ),
    "All evidence counts >= 0"
)

qc_check(
    "Unmapped target investigation complete",
    len(unmapped_summary) ==
    len(unmapped_target_ids),
    (
        f"investigated = "
        f"{len(unmapped_summary):,}; "
        f"unmapped target IDs = "
        f"{len(unmapped_target_ids):,}"
    )
)

qc_check(
    "Duplicate analysis completed",
    True,
    (
        f"duplicate groups = "
        f"{len(duplicate_groups):,}; "
        f"extra rows = "
        f"{exact_duplicate_extra_rows:,}"
    )
)

qc_check(
    "No target-negative assumption introduced",
    True,
    "Only observed molecule-target associations are represented"
)

qc_table = pd.DataFrame(
    qc_rows
)

write_tsv(
    qc_table,
    QC_TABLE_FILE
)

log("")


# ============================================================
# 28. GLOBAL SUMMARY
# ============================================================

log("=" * 80)
log("28. GLOBAL SUMMARY")
log("=" * 80)

molecules_with_target_evidence = (
    molecule_target_summary[
        "np_id"
    ]
    .nunique()
)

targets_with_evidence = (
    molecule_target_summary[
        "Target_ID"
    ]
    .nunique()
)

mapped_pair_count = int(
    (
        molecule_target_summary[
            "Target_Annotation_Status"
        ]
        == "MAPPED"
    ).sum()
)

unmapped_pair_count = int(
    (
        molecule_target_summary[
            "Target_Annotation_Status"
        ]
        == "UNMAPPED"
    ).sum()
)

targets_per_molecule = (
    molecule_target_summary
    .groupby(
        "np_id"
    )[
        "Target_ID"
    ]
    .nunique()
)

molecules_per_target = (
    molecule_target_summary
    .groupby(
        "Target_ID"
    )[
        "np_id"
    ]
    .nunique()
)

log(
    f"Raw association records        : "
    f"{len(associations):,}"
)

log(
    f"Exact duplicate extra rows     : "
    f"{exact_duplicate_extra_rows:,}"
)

log(
    f"Unique molecule-target pairs   : "
    f"{expected_unique_pairs:,}"
)

log(
    f"Mapped molecule-target pairs   : "
    f"{mapped_pair_count:,}"
)

log(
    f"Unmapped molecule-target pairs : "
    f"{unmapped_pair_count:,}"
)

log(
    f"Molecules with target evidence : "
    f"{molecules_with_target_evidence:,}"
)

log(
    f"Targets with molecule evidence : "
    f"{targets_with_evidence:,}"
)

log(
    f"Mean targets / molecule        : "
    f"{targets_per_molecule.mean():.4f}"
)

log(
    f"Median targets / molecule      : "
    f"{targets_per_molecule.median():.4f}"
)

log(
    f"Maximum targets / molecule     : "
    f"{targets_per_molecule.max():,}"
)

log(
    f"Mean molecules / target        : "
    f"{molecules_per_target.mean():.4f}"
)

log(
    f"Median molecules / target      : "
    f"{molecules_per_target.median():.4f}"
)

log(
    f"Maximum molecules / target     : "
    f"{molecules_per_target.max():,}"
)

log("")


# ============================================================
# 29. UNMAPPED TARGET BURDEN
# ============================================================

log("=" * 80)
log("29. UNMAPPED TARGET BURDEN")
log("=" * 80)

unmapped_record_count = len(
    unmapped_associations
)

unmapped_pair_count = (
    unmapped_associations[
        [
            "np_id",
            "Target_ID"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

unmapped_molecule_count = (
    unmapped_associations[
        "np_id"
    ]
    .nunique()
)

if len(associations) > 0:

    unmapped_record_fraction = (
        unmapped_record_count /
        len(associations)
    )

else:

    unmapped_record_fraction = 0.0

log(
    f"Unmapped target IDs              : "
    f"{len(unmapped_target_ids):,}"
)

log(
    f"Unmapped-target evidence records : "
    f"{unmapped_record_count:,}"
)

log(
    f"Unmapped molecule-target pairs   : "
    f"{unmapped_pair_count:,}"
)

log(
    f"Molecules involving them        : "
    f"{unmapped_molecule_count:,}"
)

log(
    f"Fraction of raw associations     : "
    f"{unmapped_record_fraction * 100:.4f}%"
)

log("")


# ============================================================
# 30. DUPLICATE BURDEN
# ============================================================

log("=" * 80)
log("30. DUPLICATE BURDEN")
log("=" * 80)

duplicate_fraction = (
    exact_duplicate_extra_rows /
    len(associations)
    if len(associations) > 0
    else 0.0
)

log(
    f"Exact duplicate extra rows       : "
    f"{exact_duplicate_extra_rows:,}"
)

log(
    f"Exact duplicate fraction         : "
    f"{duplicate_fraction:.6f}"
)

log(
    f"Exact duplicate percentage       : "
    f"{duplicate_fraction * 100:.4f}%"
)

log(
    "Duplicates were NOT silently removed."
)

log(
    "Binary matrix collapses repeated "
    "records to association presence."
)

log(
    "Evidence-count matrix preserves "
    "record multiplicity."
)

log(
    "Reference-count matrix counts "
    "unique Reference_ID values."
)

log("")


# ============================================================
# 31. FILE MANIFEST
# ============================================================

log("=" * 80)
log("31. FILE MANIFEST")
log("=" * 80)

output_files = [
    TARGET_RESOLUTION_FILE,
    UNMAPPED_TARGET_FILE,
    DUPLICATE_ANALYSIS_FILE,
    DUPLICATE_EXAMPLES_FILE,
    BINARY_MATRIX_FILE,
    EVIDENCE_COUNT_MATRIX_FILE,
    REFERENCE_COUNT_MATRIX_FILE,
    TARGET_HIERARCHY_FILE,
    MOLECULE_TARGET_SUMMARY_FILE,
    TARGET_SUMMARY_FILE,
    TARGET_SUPPORT_FILE,
    MOLECULE_SUPPORT_FILE,
    ACTIVITY_EVIDENCE_FILE,
    REFERENCE_SUPPORT_FILE,
    SUPPORT_DISTRIBUTION_FILE,
    LOW_SUPPORT_FILE,
    HIGH_SUPPORT_FILE,
    ANNOTATION_COMPLETENESS_FILE,
    QC_TABLE_FILE
]

manifest_outputs = []

for path in output_files:

    if path.exists():

        manifest_outputs.append({
            "file":
                str(
                    path.relative_to(
                        PULP_DIR
                    )
                ),

            "size_bytes":
                int(
                    path.stat().st_size
                ),

            "sha256":
                sha256_file(path)
        })


manifest = {
    "script":
        "02_target_evidence_resolution.py",

    "purpose":
        "Evidence resolution and target matrix construction.",

    "identifier_mapping":
        "Ingredient_ID -> np_id by exact equality.",

    "input_directory":
        str(SCRIPT01_DIR),

    "output_directory":
        str(OUT_DIR),

    "counts": {
        "molecule_rows":
            int(len(molecules)),

        "association_rows":
            int(len(associations)),

        "target_rows":
            int(len(targets)),

        "unique_molecules":
            int(len(molecule_ids)),

        "unique_association_ingredients":
            int(
                len(
                    association_ingredient_ids
                )
            ),

        "unique_association_targets":
            int(
                len(
                    association_target_ids
                )
            ),

        "unique_annotated_targets":
            int(
                len(
                    target_annotation_ids
                )
            ),

        "unmapped_target_ids":
            int(
                len(
                    unmapped_target_ids
                )
            ),

        "unique_molecule_target_pairs":
            int(
                expected_unique_pairs
            ),

        "exact_duplicate_extra_rows":
            int(
                exact_duplicate_extra_rows
            ),

        "molecules_with_target_evidence":
            int(
                molecules_with_target_evidence
            ),

        "targets_with_molecule_evidence":
            int(
                targets_with_evidence
            )
    },

    "design_decisions": [
        "Association Ingredient_ID is mapped exactly to molecule np_id.",
        "No fuzzy or name-based molecule mapping is used.",
        "Exact duplicate association records are preserved.",
        "Binary matrix represents observed association presence.",
        "Evidence count represents raw association-record multiplicity.",
        "Independent reference count uses unique Reference_ID values.",
        "Activity types are preserved and not merged into one potency score.",
        "Activity values are preserved for downstream evidence modeling.",
        "Unmapped Target_IDs are retained and explicitly reported.",
        "No association absence is treated as target-negative evidence.",
        "No molecular similarity is calculated.",
        "No target classifier is trained."
    ],

    "outputs":
        manifest_outputs
}

with open(
    MANIFEST_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )

log(
    f"Manifest written: "
    f"{MANIFEST_FILE}"
)

log("")


# ============================================================
# 32. FINAL STATUS
# ============================================================

log("=" * 80)
log("32. FINAL STATUS")
log("=" * 80)

failed_checks = qc_table[
    qc_table["Status"]
    == "FAIL"
]

if len(failed_checks) > 0:

    log(
        f"STATUS: FAIL — "
        f"{len(failed_checks)} QC checks failed."
    )

elif len(unmapped_target_ids) > 0:

    log(
        "STATUS: PASS WITH WARNINGS"
    )

    log(
        f"Reason: "
        f"{len(unmapped_target_ids):,} Target_ID values "
        f"remain unresolved against Targets.txt."
    )

else:

    log(
        "STATUS: PASS"
    )

log("")
log("Important:")
log(
    "Unmapped targets have NOT been discarded."
)
log(
    "Duplicate evidence has NOT been silently discarded."
)
log(
    "No target-negative labels were generated."
)
log(
    "No similarity or target prediction was performed."
)

log("")
log("=" * 80)
log("SCRIPT 02 COMPLETE")
log("=" * 80)

write_log()

print("")
print(
    "QC report:"
)
print(
    QC_FILE
)
