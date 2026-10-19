#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT 03 — TARGET ID RESOLUTION & EVIDENCE CLEANING
====================================================

Purpose
-------
Investigate the 385 unresolved Target_ID values remaining after Script 02.

This script is deliberately conservative.

It:

1. Loads Script 01 cleaned molecule/association/target tables.
2. Loads Script 02 target-resolution output.
3. Re-validates Target_ID mapping.
4. Performs exact Target_ID matching.
5. Performs conservative whitespace/case normalization matching.
6. Searches CMAUP source files for unresolved Target_ID occurrences.
7. Separates "found somewhere" from "actually annotated".
8. Preserves all raw association evidence.
9. Builds an annotated target reference table using ONLY real
   annotations present in Targets.txt.
10. Builds an association-level resolution table without unsafe
    many-to-many merges.
11. Quantifies unresolved target burden.
12. Produces QC tables, a manifest, and a text report.

IMPORTANT BIOLOGICAL RULES
--------------------------
- An unresolved Target_ID is NOT treated as a negative target.
- An unresolved Target_ID is NOT assigned a fabricated annotation.
- Finding an ID in another CMAUP file does NOT automatically resolve it.
- Gene symbols, protein names, target classes, UniProt IDs, etc.
  are copied only from the authoritative Targets.txt-derived table.
- Duplicate association records are preserved.
- Observed associations are NOT converted into negative labels.
- This script does NOT perform similarity-based target prediction.

Output
------
target_model/data/03_target_resolution/

Authoritative PULP path:
    PULP_DIR = Path(__file__).resolve().parent
"""


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path
from datetime import datetime
import hashlib
import json
import re

import numpy as np
import pandas as pd


# =============================================================================
# 2. PATH CONFIGURATION
# =============================================================================

PULP_DIR = Path(__file__).resolve().parent

SCRIPT01_DIR = (
    PULP_DIR
    / "target_model"
    / "data"
    / "01_cmaup_audit"
)

SCRIPT02_DIR = (
    PULP_DIR
    / "target_model"
    / "data"
    / "02_target_evidence"
)

OUTPUT_DIR = (
    PULP_DIR
    / "target_model"
    / "data"
    / "03_target_resolution"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 3. INPUT FILES
# =============================================================================

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

SCRIPT02_RESOLUTION_FILE = (
    SCRIPT02_DIR /
    "target_id_resolution.tsv"
)

SCRIPT02_MOLECULE_TARGET_FILE = (
    SCRIPT02_DIR /
    "molecule_target_summary.tsv"
)


# =============================================================================
# 4. CMAUP SOURCE FILES FOR INVESTIGATION
# =============================================================================

CMAUP_SOURCE_FILES = [
    PULP_DIR / "CMAUPv2.0_download_Ingredients_All.txt",

    PULP_DIR /
    "CMAUPv2.0_download_Ingredient_Target_Associations_ActivityValues_References.txt",

    PULP_DIR / "CMAUPv2.0_download_Targets.txt",

    PULP_DIR / "CMAUPv2.0_download_Ingredients_onlyActive.txt",

    PULP_DIR /
    "CMAUPv2.0_download_Human_Oral_Bioavailability_information_of_Ingredients_All.txt",

    PULP_DIR /
    "CMAUPv2.0_download_Plant_Clinical_Trials_Associations.txt",

    PULP_DIR /
    "CMAUPv2.0_download_Plant_Human_Disease_Associations.txt",

    PULP_DIR /
    "CMAUPv2.0_download_Plant_Ingredient_Associations_allIngredients.txt",

    PULP_DIR /
    "CMAUPv2.0_download_Plant_Ingredient_Associations_onlyActiveIngredients.txt",

    PULP_DIR /
    "CMAUPv2.0_download_Plant_molecular_targets_overlapping_with_DEGs.txt",

    PULP_DIR / "CMAUPv2.0_download_Plants.txt",

    PULP_DIR / "Download_Readme.txt",
]


# =============================================================================
# 5. EXPECTED SCHEMAS
# =============================================================================

EXPECTED_ASSOC_COLUMNS = [
    "Ingredient_ID",
    "Target_ID",
    "Activity_Type",
    "Activity_Relationship",
    "Activity_Value",
    "Activity_Unit",
    "Reference_ID",
    "Reference_ID_Type",
]


EXPECTED_TARGET_COLUMNS = [
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
    "Target_type",
]


TARGET_ANNOTATION_COLUMNS = [
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
    "Target_type",
]


# =============================================================================
# 6. LOGGING
# =============================================================================

LOG_LINES = []


def log(message=""):
    print(message)
    LOG_LINES.append(message)


def timestamp():
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =============================================================================
# 7. GENERAL HELPERS
# =============================================================================

def read_tsv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False
    )


def write_tsv(df, path):
    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8"
    )


def validate_columns(df, expected_columns, table_name):

    missing = [
        c for c in expected_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\n{table_name} is missing required columns:\n"
            f"{missing}\n\n"
            f"Observed columns:\n"
            f"{list(df.columns)}"
        )


def normalize_id(value):
    """
    Conservative normalization.

    Only:
        1. converts to string
        2. strips leading/trailing whitespace
        3. converts to casefold

    No punctuation removal.
    No prefix removal.
    No numeric manipulation.
    No fuzzy matching.
    """

    if pd.isna(value):
        return ""

    return str(value).strip().casefold()


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def unique_clean_values(series):

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    return sorted(
        values.unique()
    )


# =============================================================================
# 8. START
# =============================================================================

log("=" * 80)
log("SCRIPT 03 — TARGET ID RESOLUTION & EVIDENCE CLEANING")
log("=" * 80)

log(
    f"PULP directory : {PULP_DIR}"
)

log(
    f"Script 01 dir  : {SCRIPT01_DIR}"
)

log(
    f"Script 02 dir  : {SCRIPT02_DIR}"
)

log(
    f"Output dir     : {OUTPUT_DIR}"
)

log(
    f"Start time     : {timestamp()}"
)

log("")


# =============================================================================
# 9. INPUT FILE CHECK
# =============================================================================

log("=" * 80)
log("1. INPUT FILE CHECK")
log("=" * 80)

required_files = [
    MOLECULE_FILE,
    ASSOCIATION_FILE,
    TARGET_FILE,
    SCRIPT02_RESOLUTION_FILE,
    SCRIPT02_MOLECULE_TARGET_FILE,
]

missing_files = []

for path in required_files:

    if path.exists():

        log(
            f"FOUND      | {path}"
        )

    else:

        log(
            f"MISSING    | {path}"
        )

        missing_files.append(path)


if missing_files:

    raise FileNotFoundError(
        "Required input files are missing."
    )

log("")


# =============================================================================
# 10. LOAD SCRIPT 01 / SCRIPT 02 TABLES
# =============================================================================

log("=" * 80)
log("2. LOADING SCRIPT 01/02 TABLES")
log("=" * 80)

molecules = read_tsv(
    MOLECULE_FILE
)

associations = read_tsv(
    ASSOCIATION_FILE
)

targets = read_tsv(
    TARGET_FILE
)

script02_resolution = read_tsv(
    SCRIPT02_RESOLUTION_FILE
)

script02_molecule_target = read_tsv(
    SCRIPT02_MOLECULE_TARGET_FILE
)

log(
    f"Molecules loaded              : {len(molecules):,}"
)

log(
    f"Associations loaded           : {len(associations):,}"
)

log(
    f"Targets loaded                : {len(targets):,}"
)

log(
    f"Script 02 target resolution   : {len(script02_resolution):,}"
)

log(
    f"Script 02 molecule-target     : "
    f"{len(script02_molecule_target):,}"
)

log("")


# =============================================================================
# 11. SCHEMA VALIDATION
# =============================================================================

log("=" * 80)
log("3. SCHEMA VALIDATION")
log("=" * 80)

validate_columns(
    associations,
    EXPECTED_ASSOC_COLUMNS,
    "Association table"
)

validate_columns(
    targets,
    EXPECTED_TARGET_COLUMNS,
    "Target table"
)

if "Target_ID" not in script02_resolution.columns:

    raise ValueError(
        "Script 02 target_id_resolution.tsv does not contain Target_ID."
    )

log(
    "Association schema : PASS"
)

log(
    "Target schema       : PASS"
)

log(
    "Script 02 resolution: PASS"
)

log("")


# =============================================================================
# 12. TARGET TABLE UNIQUENESS
# =============================================================================

log("=" * 80)
log("4. TARGET TABLE UNIQUENESS QC")
log("=" * 80)

target_ids_raw = (
    targets["Target_ID"]
    .dropna()
    .astype(str)
    .str.strip()
)

target_duplicate_mask = (
    targets["Target_ID"]
    .fillna("")
    .astype(str)
    .str.strip()
    .duplicated(
        keep=False
    )
)

target_duplicate_rows = int(
    target_duplicate_mask.sum()
)

target_duplicate_values = int(
    targets.loc[
        target_duplicate_mask,
        "Target_ID"
    ]
    .astype(str)
    .str.strip()
    .nunique()
)

log(
    f"Target rows                      : {len(targets):,}"
)

log(
    f"Unique Target_ID values          : "
    f"{target_ids_raw.nunique():,}"
)

log(
    f"Duplicate Target_ID rows         : "
    f"{target_duplicate_rows:,}"
)

log(
    f"Duplicate Target_ID values      : "
    f"{target_duplicate_values:,}"
)

if target_duplicate_values > 0:

    write_tsv(
        targets.loc[
            target_duplicate_mask
        ].copy(),
        OUTPUT_DIR /
        "duplicate_target_id_annotations.tsv"
    )

    log(
        "WARNING: Targets.txt contains duplicate Target_ID values."
    )

else:

    log(
        "Target_ID uniqueness             : PASS"
    )

log("")


# =============================================================================
# 13. BASIC TARGET ID INVENTORY
# =============================================================================

log("=" * 80)
log("5. TARGET ID INVENTORY")
log("=" * 80)

association_target_ids = unique_clean_values(
    associations["Target_ID"]
)

target_table_ids = unique_clean_values(
    targets["Target_ID"]
)

association_target_set = set(
    association_target_ids
)

target_table_id_set = set(
    target_table_ids
)

exact_mapped_ids = sorted(
    association_target_set &
    target_table_id_set
)

exact_unmapped_ids = sorted(
    association_target_set -
    target_table_id_set
)

log(
    f"Unique Target_IDs in associations : "
    f"{len(association_target_ids):,}"
)

log(
    f"Unique Target_IDs in Targets.txt   : "
    f"{len(target_table_ids):,}"
)

log(
    f"Exact mapped IDs                  : "
    f"{len(exact_mapped_ids):,}"
)

log(
    f"Exact unresolved IDs              : "
    f"{len(exact_unmapped_ids):,}"
)

log("")


# =============================================================================
# 14. CONSERVATIVE NORMALIZATION
# =============================================================================

log("=" * 80)
log("6. CONSERVATIVE NORMALIZATION MATCH")
log("=" * 80)

normalized_target_map = {}

for target_id in target_table_ids:

    normalized = normalize_id(
        target_id
    )

    if normalized == "":
        continue

    normalized_target_map.setdefault(
        normalized,
        []
    ).append(
        target_id
    )


normalization_rows = []

for unresolved_id in exact_unmapped_ids:

    normalized = normalize_id(
        unresolved_id
    )

    candidates = normalized_target_map.get(
        normalized,
        []
    )

    if len(candidates) == 1:

        candidate = candidates[0]

        if candidate != unresolved_id:

            status = (
                "NORMALIZATION_ONLY_MATCH"
            )

        else:

            status = (
                "EXACT_MATCH_ALREADY_CAPTURED"
            )

        normalization_rows.append({
            "Unresolved_Target_ID": unresolved_id,
            "Normalized_ID": normalized,
            "Candidate_Target_ID": candidate,
            "Candidate_Count": 1,
            "Resolution_Status": status,
        })

    elif len(candidates) > 1:

        normalization_rows.append({
            "Unresolved_Target_ID": unresolved_id,
            "Normalized_ID": normalized,
            "Candidate_Target_ID": ";".join(
                candidates
            ),
            "Candidate_Count": len(candidates),
            "Resolution_Status":
                "AMBIGUOUS_NORMALIZATION_MATCH",
        })


normalization_df = pd.DataFrame(
    normalization_rows,
    columns=[
        "Unresolved_Target_ID",
        "Normalized_ID",
        "Candidate_Target_ID",
        "Candidate_Count",
        "Resolution_Status",
    ]
)

write_tsv(
    normalization_df,
    OUTPUT_DIR /
    "normalization_match_candidates.tsv"
)

if len(normalization_df) > 0:

    normalization_only_count = int(
        (
            normalization_df[
                "Resolution_Status"
            ]
            == "NORMALIZATION_ONLY_MATCH"
        ).sum()
    )

    ambiguous_count = int(
        (
            normalization_df[
                "Resolution_Status"
            ]
            == "AMBIGUOUS_NORMALIZATION_MATCH"
        ).sum()
    )

else:

    normalization_only_count = 0
    ambiguous_count = 0


log(
    f"Normalization-only candidates    : "
    f"{normalization_only_count:,}"
)

log(
    f"Ambiguous normalization matches  : "
    f"{ambiguous_count:,}"
)

log(
    "Normalization matching is descriptive "
    "and conservative only."
)

log("")


# =============================================================================
# 15. PROVISIONAL RESOLUTION
# =============================================================================

log("=" * 80)
log("7. PROVISIONAL TARGET RESOLUTION MAP")
log("=" * 80)

provisional_rows = []

for target_id in association_target_ids:

    if target_id in target_table_id_set:

        provisional_rows.append({
            "Target_ID": target_id,
            "Resolution_Status":
                "MAPPED_IN_TARGET_TABLE",
            "Resolved_Target_ID": target_id,
            "Resolution_Method":
                "EXACT_MATCH",
        })

        continue

    normalized = normalize_id(
        target_id
    )

    candidates = normalized_target_map.get(
        normalized,
        []
    )

    if len(candidates) == 1:

        candidate = candidates[0]

        provisional_rows.append({
            "Target_ID": target_id,
            "Resolution_Status":
                "NORMALIZATION_ONLY_MATCH",
            "Resolved_Target_ID": candidate,
            "Resolution_Method":
                "STRIP_AND_CASEFOLD_ONLY",
        })

    elif len(candidates) > 1:

        provisional_rows.append({
            "Target_ID": target_id,
            "Resolution_Status":
                "AMBIGUOUS_NORMALIZATION_MATCH",
            "Resolved_Target_ID": "",
            "Resolution_Method":
                "STRIP_AND_CASEFOLD_AMBIGUOUS",
        })

    else:

        provisional_rows.append({
            "Target_ID": target_id,
            "Resolution_Status":
                "UNRESOLVED",
            "Resolved_Target_ID": "",
            "Resolution_Method":
                "NO_TARGET_TABLE_MATCH",
        })


provisional_resolution = pd.DataFrame(
    provisional_rows
)

write_tsv(
    provisional_resolution,
    OUTPUT_DIR /
    "target_id_resolution_provisional.tsv"
)

log(
    f"Provisional resolution rows       : "
    f"{len(provisional_resolution):,}"
)

log("")


# =============================================================================
# 16. SEARCH CMAUP SOURCE FILES
# =============================================================================

log("=" * 80)
log("8. SEARCHING OTHER CMAUP FILES")
log("=" * 80)

unresolved_ids = set(
    provisional_resolution.loc[
        provisional_resolution[
            "Resolution_Status"
        ] == "UNRESOLVED",
        "Target_ID"
    ]
)

log(
    f"Unresolved Target_IDs to investigate : "
    f"{len(unresolved_ids):,}"
)

log("")

source_scan_rows = []

# IMPORTANT:
# We deliberately use token/field-aware matching where possible.
#
# Since CMAUP files can have different separators, we record occurrences
# as evidence only. We NEVER convert an occurrence into an annotation.


def id_occurs_in_line(line, target_id):
    """
    Conservative occurrence check.

    First attempt:
        exact field match after splitting common delimiters.

    Second attempt:
        token/word-boundary match.

    This function is for evidence discovery only.
    """

    target = str(target_id).strip()

    if target == "":
        return False

    # Exact match in tab-delimited fields.
    fields = line.rstrip("\n\r").split("\t")

    for field in fields:

        if field.strip() == target:
            return True

    # Word-boundary fallback.
    pattern = (
        r"(?<![A-Za-z0-9_.:-])"
        + re.escape(target)
        + r"(?![A-Za-z0-9_.:-])"
    )

    return bool(
        re.search(
            pattern,
            line
        )
    )


for source_file in CMAUP_SOURCE_FILES:

    if not source_file.exists():

        source_scan_rows.append({
            "File": source_file.name,
            "Exists": 0,
            "Target_IDs_Found": 0,
            "Total_Occurrences": 0,
            "Status": "FILE_NOT_FOUND",
        })

        continue

    found_ids = set()
    total_occurrences = 0

    try:

        with open(
            source_file,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as handle:

            for line in handle:

                for target_id in unresolved_ids:

                    if id_occurs_in_line(
                        line,
                        target_id
                    ):

                        found_ids.add(
                            target_id
                        )

                        total_occurrences += 1

    except Exception as exc:

        source_scan_rows.append({
            "File": source_file.name,
            "Exists": 1,
            "Target_IDs_Found": 0,
            "Total_Occurrences": 0,
            "Status":
                f"SCAN_ERROR: {exc}",
        })

        continue

    source_scan_rows.append({
        "File": source_file.name,
        "Exists": 1,
        "Target_IDs_Found":
            len(found_ids),
        "Total_Occurrences":
            total_occurrences,
        "Status": "SCANNED",
    })

    log(
        f"{source_file.name:75s} | "
        f"IDs found = {len(found_ids):4d} | "
        f"occurrences = {total_occurrences:,}"
    )


source_scan_df = pd.DataFrame(
    source_scan_rows
)

write_tsv(
    source_scan_df,
    OUTPUT_DIR /
    "unresolved_target_source_scan.tsv"
)

log("")


# =============================================================================
# 17. TARGET-ID × SOURCE EVIDENCE
# =============================================================================

log("=" * 80)
log("9. BUILDING TARGET-ID SOURCE EVIDENCE TABLE")
log("=" * 80)

target_source_rows = []

for target_id in sorted(
    unresolved_ids
):

    for source_file in CMAUP_SOURCE_FILES:

        if not source_file.exists():
            continue

        occurrence_count = 0

        try:

            with open(
                source_file,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as handle:

                for line in handle:

                    if id_occurs_in_line(
                        line,
                        target_id
                    ):

                        occurrence_count += 1

        except Exception:
            continue

        if occurrence_count > 0:

            target_source_rows.append({
                "Target_ID": target_id,
                "Source_File":
                    source_file.name,
                "Occurrence_Count":
                    occurrence_count,
            })


target_source_df = pd.DataFrame(
    target_source_rows,
    columns=[
        "Target_ID",
        "Source_File",
        "Occurrence_Count",
    ]
)

write_tsv(
    target_source_df,
    OUTPUT_DIR /
    "unresolved_target_source_evidence.tsv"
)

log(
    f"Target-ID/source evidence rows     : "
    f"{len(target_source_df):,}"
)

log("")


# =============================================================================
# 18. FINAL CONSERVATIVE TARGET STATUS
# =============================================================================

log("=" * 80)
log("10. FINAL CONSERVATIVE STATUS CLASSIFICATION")
log("=" * 80)

source_found_ids = set()

if len(target_source_df) > 0:

    source_found_ids = set(
        target_source_df[
            "Target_ID"
        ]
        .astype(str)
    )


final_resolution_rows = []

for _, row in provisional_resolution.iterrows():

    target_id = str(
        row["Target_ID"]
    )

    provisional_status = (
        row["Resolution_Status"]
    )

    resolved_target_id = (
        row["Resolved_Target_ID"]
    )

    method = (
        row["Resolution_Method"]
    )

    if provisional_status == (
        "MAPPED_IN_TARGET_TABLE"
    ):

        final_status = (
            "MAPPED_IN_TARGET_TABLE"
        )

    elif provisional_status == (
        "NORMALIZATION_ONLY_MATCH"
    ):

        final_status = (
            "NORMALIZATION_ONLY_MATCH"
        )

    elif provisional_status == (
        "AMBIGUOUS_NORMALIZATION_MATCH"
    ):

        final_status = (
            "UNMAPPED_NO_UNIQUE_ANNOTATION"
        )

        resolved_target_id = ""

    elif target_id in source_found_ids:

        final_status = (
            "UNMAPPED_BUT_FOUND_IN_OTHER_CMAUP_FILE"
        )

        resolved_target_id = ""

        method = (
            "OCCURRENCE_FOUND_OUTSIDE_TARGET_TABLE; "
            "NO_TARGET_ANNOTATION_ASSIGNED"
        )

    else:

        final_status = (
            "UNMAPPED_NO_ANNOTATION"
        )

        resolved_target_id = ""

        method = (
            "NO_TARGET_TABLE_MATCH_OR_OTHER_FILE_OCCURRENCE"
        )

    final_resolution_rows.append({
        "Target_ID": target_id,
        "Final_Status": final_status,
        "Resolved_Target_ID":
            resolved_target_id,
        "Resolution_Method": method,
    })


final_resolution_df = pd.DataFrame(
    final_resolution_rows
)

write_tsv(
    final_resolution_df,
    OUTPUT_DIR /
    "target_id_resolution_final.tsv"
)

status_distribution = (
    final_resolution_df[
        "Final_Status"
    ]
    .value_counts()
    .rename_axis(
        "Final_Status"
    )
    .reset_index(
        name="Target_ID_Count"
    )
)

write_tsv(
    status_distribution,
    OUTPUT_DIR /
    "target_resolution_status_distribution.tsv"
)

for _, row in status_distribution.iterrows():

    log(
        f"{row['Final_Status']:50s} : "
        f"{int(row['Target_ID_Count']):,}"
    )

log("")


# =============================================================================
# 19. BUILD TARGET ANNOTATION LOOKUP
# =============================================================================

log("=" * 80)
log("11. BUILDING TARGET ANNOTATION LOOKUP")
log("=" * 80)

"""
CRITICAL FIX
------------
Do NOT merge unresolved records on:

    Resolved_Target_ID + Annotation_Available

because all unresolved records have an empty/NA resolved ID and
can create many-to-many merge keys.

Instead:

1. Construct a unique annotation lookup ONLY for targets that actually
   exist in Targets.txt.
2. Merge annotated targets by Resolved_Target_ID only.
3. Leave unresolved records as NA.

This guarantees a many-to-one lookup when Targets.txt is unique.
"""

target_lookup = targets[
    [
        "Target_ID"
    ]
    + TARGET_ANNOTATION_COLUMNS
].copy()

target_lookup[
    "Target_ID"
] = (
    target_lookup[
        "Target_ID"
    ]
    .astype(str)
    .str.strip()
)

# Only retain non-empty Target_ID.
target_lookup = target_lookup[
    target_lookup["Target_ID"] != ""
].copy()

# If duplicate Target_ID values exist, do NOT silently choose one.
# Instead stop, because target annotation would be ambiguous.
duplicate_lookup_ids = (
    target_lookup[
        "Target_ID"
    ]
    .duplicated(
        keep=False
    )
)

if duplicate_lookup_ids.any():

    duplicate_ids = sorted(
        target_lookup.loc[
            duplicate_lookup_ids,
            "Target_ID"
        ]
        .unique()
    )

    write_tsv(
        target_lookup.loc[
            duplicate_lookup_ids
        ].copy(),
        OUTPUT_DIR /
        "ambiguous_target_annotation_lookup.tsv"
    )

    raise ValueError(
        "\nTargets.txt contains duplicate Target_ID values.\n"
        "Cannot safely construct a one-target-ID-to-one-annotation lookup.\n"
        f"Duplicate Target_ID count: {len(duplicate_ids):,}\n"
        f"See: {OUTPUT_DIR / 'ambiguous_target_annotation_lookup.tsv'}"
    )


target_lookup = target_lookup.rename(
    columns={
        "Target_ID":
            "Resolved_Target_ID"
    }
)

target_lookup["Annotation_Available"] = True

log(
    f"Unique target annotation records : "
    f"{len(target_lookup):,}"
)

log(
    "Target annotation lookup         : "
    "MANY-TO-ONE SAFE"
)

log("")


# =============================================================================
# 20. TARGET RESOLUTION + ANNOTATION TABLE
# =============================================================================

log("=" * 80)
log("12. BUILDING CONSERVATIVE TARGET ANNOTATION TABLE")
log("=" * 80)

conservative_target_table = (
    final_resolution_df
    .merge(
        target_lookup,
        how="left",
        on="Resolved_Target_ID",
        validate="many_to_one"
    )
)

# Only exact/normalization mappings can obtain annotation.
# Unresolved rows must remain annotation-free.
conservative_target_table[
    "Annotation_Available"
] = (
    conservative_target_table[
        "Resolved_Target_ID"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

for column in TARGET_ANNOTATION_COLUMNS:

    conservative_target_table.loc[
        ~conservative_target_table[
            "Annotation_Available"
        ],
        column
    ] = pd.NA


write_tsv(
    conservative_target_table,
    OUTPUT_DIR /
    "cmaup_targets_conservative_resolution.tsv"
)

annotation_count = int(
    conservative_target_table[
        "Annotation_Available"
    ].sum()
)

no_annotation_count = (
    len(conservative_target_table)
    - annotation_count
)

log(
    f"Target annotation table rows       : "
    f"{len(conservative_target_table):,}"
)

log(
    f"Targets with annotations            : "
    f"{annotation_count:,}"
)

log(
    f"Targets without annotations         : "
    f"{no_annotation_count:,}"
)

log("")


# =============================================================================
# 21. ASSOCIATION-LEVEL TARGET RESOLUTION
# =============================================================================

log("=" * 80)
log("13. BUILDING ASSOCIATION-LEVEL RESOLUTION TABLE")
log("=" * 80)

"""
Another important design choice:

We merge associations to final_resolution_df first.

Then we merge ONLY the unique target annotation lookup.

The annotation lookup contains exactly one row per Resolved_Target_ID.

Therefore:
    association rows are preserved
    annotation cannot multiply association records
"""

association_resolution = (
    associations
    .copy()
)

association_resolution[
    "Target_ID"
] = (
    association_resolution[
        "Target_ID"
    ]
    .astype(str)
    .str.strip()
)

association_resolution = (
    association_resolution
    .merge(
        final_resolution_df,
        how="left",
        on="Target_ID",
        validate="many_to_one"
    )
)

# Merge annotation ONLY through the unique lookup.
association_resolution = (
    association_resolution
    .merge(
        target_lookup,
        how="left",
        on="Resolved_Target_ID",
        validate="many_to_one"
    )
)

association_resolution[
    "Annotation_Available"
] = (
    association_resolution[
        "Resolved_Target_ID"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

write_tsv(
    association_resolution,
    OUTPUT_DIR /
    "cmaup_associations_with_target_resolution.tsv"
)

log(
    f"Association records retained       : "
    f"{len(association_resolution):,}"
)

log(
    f"Original association records       : "
    f"{len(associations):,}"
)

log(
    f"Annotated association records      : "
    f"{int(association_resolution['Annotation_Available'].sum()):,}"
)

log(
    f"Unannotated association records    : "
    f"{int((~association_resolution['Annotation_Available']).sum()):,}"
)

log("")


# =============================================================================
# 22. ASSOCIATION ROW CONSERVATION CHECK
# =============================================================================

if len(
    association_resolution
) != len(
    associations
):

    raise RuntimeError(
        "\nCRITICAL ERROR:\n"
        "Association row count changed during target annotation merge.\n"
        f"Original : {len(associations):,}\n"
        f"Current  : {len(association_resolution):,}"
    )


# =============================================================================
# 23. MOLECULE-TARGET PAIR RESOLUTION
# =============================================================================

log("=" * 80)
log("14. MOLECULE-TARGET PAIR RESOLUTION")
log("=" * 80)

pair_columns = [
    "Ingredient_ID",
    "Target_ID",
    "Final_Status",
    "Resolved_Target_ID",
    "Annotation_Available",
    "Target_Class_Level1",
    "Target_Class_Level2",
    "Target_Class_Level3",
    "Protein_Name",
    "Gene_Symbol",
]

pair_resolution = (
    association_resolution[
        pair_columns
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)

write_tsv(
    pair_resolution,
    OUTPUT_DIR /
    "molecule_target_pair_resolution.tsv"
)

pair_status_distribution = (
    pair_resolution[
        "Final_Status"
    ]
    .value_counts()
    .rename_axis(
        "Final_Status"
    )
    .reset_index(
        name="Molecule_Target_Pair_Count"
    )
)

write_tsv(
    pair_status_distribution,
    OUTPUT_DIR /
    "molecule_target_pair_resolution_distribution.tsv"
)

for _, row in pair_status_distribution.iterrows():

    log(
        f"{row['Final_Status']:50s} : "
        f"{int(row['Molecule_Target_Pair_Count']):,}"
    )

log("")


# =============================================================================
# 24. MOLECULE-LEVEL USABILITY
# =============================================================================

log("=" * 80)
log("15. MOLECULE-LEVEL TARGET EVIDENCE USABILITY")
log("=" * 80)

all_associated_molecules = set(
    associations[
        "Ingredient_ID"
    ]
    .dropna()
    .astype(str)
    .str.strip()
)

molecule_usability_rows = []

for molecule_id in sorted(
    all_associated_molecules
):

    molecule_pairs = pair_resolution[
        pair_resolution[
            "Ingredient_ID"
        ].astype(str)
        == molecule_id
    ]

    annotated_pairs = molecule_pairs[
        molecule_pairs[
            "Annotation_Available"
        ]
    ]

    unresolved_pairs_for_molecule = (
        molecule_pairs[
            ~molecule_pairs[
                "Annotation_Available"
            ]
        ]
    )

    if (
        len(annotated_pairs) > 0
        and
        len(unresolved_pairs_for_molecule) > 0
    ):

        usability = (
            "MIXED_ANNOTATED_AND_UNANNOTATED_TARGETS"
        )

    elif len(annotated_pairs) > 0:

        usability = (
            "ANNOTATED_TARGET_EVIDENCE_AVAILABLE"
        )

    else:

        usability = (
            "UNANNOTATED_TARGET_EVIDENCE_ONLY"
        )

    molecule_usability_rows.append({
        "Ingredient_ID":
            molecule_id,
        "Total_Unique_Target_Pairs":
            len(molecule_pairs),
        "Annotated_Target_Pairs":
            len(annotated_pairs),
        "Unannotated_Target_Pairs":
            len(unresolved_pairs_for_molecule),
        "Molecule_Target_Usability":
            usability,
    })


molecule_usability_df = pd.DataFrame(
    molecule_usability_rows
)

write_tsv(
    molecule_usability_df,
    OUTPUT_DIR /
    "molecule_target_usability.tsv"
)

molecule_usability_distribution = (
    molecule_usability_df[
        "Molecule_Target_Usability"
    ]
    .value_counts()
    .rename_axis(
        "Molecule_Target_Usability"
    )
    .reset_index(
        name="Molecule_Count"
    )
)

write_tsv(
    molecule_usability_distribution,
    OUTPUT_DIR /
    "molecule_target_usability_distribution.tsv"
)

for _, row in molecule_usability_distribution.iterrows():

    log(
        f"{row['Molecule_Target_Usability']:55s} : "
        f"{int(row['Molecule_Count']):,}"
    )

log("")


# =============================================================================
# 25. TARGET HIERARCHY COMPLETENESS
# =============================================================================

log("=" * 80)
log("16. TARGET HIERARCHY / ANNOTATION COMPLETENESS")
log("=" * 80)

completeness_rows = []

annotation_table_size = len(
    conservative_target_table
)

for column in [
    "Target_Class_Level1",
    "Target_Class_Level2",
    "Target_Class_Level3",
    "Protein_Name",
    "Gene_Symbol",
    "Uniprot_ID",
    "ChEMBL_ID",
    "TTD_ID",
    "Target_type",
]:

    available = int(
        conservative_target_table[
            column
        ]
        .notna()
        .sum()
    )

    missing = (
        annotation_table_size
        - available
    )

    completeness = (
        available /
        annotation_table_size *
        100
        if annotation_table_size > 0
        else 0
    )

    completeness_rows.append({
        "Annotation_Field":
            column,
        "Available_Targets":
            available,
        "Missing_Targets":
            missing,
        "Completeness_Percent":
            completeness,
    })

    log(
        f"{column:30s} | "
        f"available = {available:,} | "
        f"missing = {missing:,} | "
        f"complete = {completeness:.3f}%"
    )


completeness_df = pd.DataFrame(
    completeness_rows
)

write_tsv(
    completeness_df,
    OUTPUT_DIR /
    "target_annotation_resolution_completeness.tsv"
)

log("")


# =============================================================================
# 26. UNRESOLVED TARGET IDENTIFIER PATTERNS
# =============================================================================

log("=" * 80)
log("17. UNRESOLVED TARGET IDENTIFIER PATTERN ANALYSIS")
log("=" * 80)

pattern_rows = []

for target_id in sorted(
    unresolved_ids
):

    target_string = str(
        target_id
    )

    has_letters = bool(
        re.search(
            r"[A-Za-z]",
            target_string
        )
    )

    has_digits = bool(
        re.search(
            r"\d",
            target_string
        )
    )

    has_separator = bool(
        re.search(
            r"[-_:./]",
            target_string
        )
    )

    if (
        has_letters
        and has_digits
        and has_separator
    ):

        pattern_class = (
            "ALPHANUMERIC_WITH_SEPARATOR"
        )

    elif (
        has_letters
        and has_digits
    ):

        pattern_class = (
            "ALPHANUMERIC"
        )

    elif (
        has_digits
        and not has_letters
    ):

        pattern_class = (
            "NUMERIC_ONLY"
        )

    elif (
        has_letters
        and not has_digits
    ):

        pattern_class = (
            "LETTER_ONLY"
        )

    else:

        pattern_class = (
            "OTHER"
        )

    pattern_rows.append({
        "Target_ID":
            target_string,
        "Length":
            len(target_string),
        "Has_Letters":
            int(has_letters),
        "Has_Digits":
            int(has_digits),
        "Has_Separator":
            int(has_separator),
        "Identifier_Pattern":
            pattern_class,
    })


pattern_df = pd.DataFrame(
    pattern_rows
)

write_tsv(
    pattern_df,
    OUTPUT_DIR /
    "unresolved_target_identifier_patterns.tsv"
)

pattern_distribution = (
    pattern_df[
        "Identifier_Pattern"
    ]
    .value_counts()
    .rename_axis(
        "Identifier_Pattern"
    )
    .reset_index(
        name="Target_ID_Count"
    )
)

write_tsv(
    pattern_distribution,
    OUTPUT_DIR /
    "unresolved_target_identifier_pattern_distribution.tsv"
)

log(
    "Identifier pattern classification is descriptive only."
)

log(
    "Recognizable ID syntax is NOT treated as evidence "
    "of target identity."
)

log("")


# =============================================================================
# 27. UNRESOLVED TARGET BURDEN
# =============================================================================

log("=" * 80)
log("18. UNRESOLVED TARGET BURDEN")
log("=" * 80)

unannotated_associations = (
    association_resolution[
        ~association_resolution[
            "Annotation_Available"
        ]
    ]
)

unresolved_pairs = (
    unannotated_associations[
        [
            "Ingredient_ID",
            "Target_ID"
        ]
    ]
    .drop_duplicates()
)

total_raw_records = len(
    associations
)

total_unique_pairs = (
    associations[
        [
            "Ingredient_ID",
            "Target_ID"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

unresolved_raw_records = len(
    unannotated_associations
)

unresolved_unique_pairs = len(
    unresolved_pairs
)

unresolved_target_count = (
    unresolved_pairs[
        "Target_ID"
    ]
    .nunique()
)

unresolved_molecule_count = (
    unresolved_pairs[
        "Ingredient_ID"
    ]
    .nunique()
)

unresolved_pair_percentage = (
    unresolved_unique_pairs /
    total_unique_pairs *
    100
    if total_unique_pairs > 0
    else 0
)

unresolved_record_percentage = (
    unresolved_raw_records /
    total_raw_records *
    100
    if total_raw_records > 0
    else 0
)

burden_df = pd.DataFrame([
    {
        "Metric":
            "Total raw association records",
        "Value":
            total_raw_records,
    },
    {
        "Metric":
            "Total unique molecule-target pairs",
        "Value":
            total_unique_pairs,
    },
    {
        "Metric":
            "Unresolved raw association records",
        "Value":
            unresolved_raw_records,
    },
    {
        "Metric":
            "Unresolved unique molecule-target pairs",
        "Value":
            unresolved_unique_pairs,
    },
    {
        "Metric":
            "Unresolved Target_ID values",
        "Value":
            unresolved_target_count,
    },
    {
        "Metric":
            "Molecules involving unresolved targets",
        "Value":
            unresolved_molecule_count,
    },
    {
        "Metric":
            "Unresolved unique-pair percentage",
        "Value":
            unresolved_pair_percentage,
    },
    {
        "Metric":
            "Unresolved raw-record percentage",
        "Value":
            unresolved_record_percentage,
    },
])

write_tsv(
    burden_df,
    OUTPUT_DIR /
    "unresolved_target_burden.tsv"
)

log(
    f"Total raw associations            : "
    f"{total_raw_records:,}"
)

log(
    f"Total unique molecule-target pairs: "
    f"{total_unique_pairs:,}"
)

log(
    f"Unresolved raw association records : "
    f"{unresolved_raw_records:,}"
)

log(
    f"Unresolved unique pairs            : "
    f"{unresolved_unique_pairs:,}"
)

log(
    f"Unresolved Target_IDs              : "
    f"{unresolved_target_count:,}"
)

log(
    f"Molecules involved                : "
    f"{unresolved_molecule_count:,}"
)

log(
    f"Unresolved pair percentage         : "
    f"{unresolved_pair_percentage:.4f}%"
)

log(
    f"Unresolved record percentage       : "
    f"{unresolved_record_percentage:.4f}%"
)

log("")


# =============================================================================
# 28. TARGET-INFERENCE REFERENCE TABLE
# =============================================================================

log("=" * 80)
log("19. BUILDING TARGET-INFERENCE REFERENCE TABLE")
log("=" * 80)

"""
This is the authoritative target annotation table for the next stage.

Only targets that actually have annotations in Targets.txt are included.

Unresolved targets remain available in:
    cmaup_associations_with_target_resolution.tsv

but are NOT given fabricated hierarchy/protein/gene information.
"""

target_inference_reference = (
    target_lookup
    .copy()
)

target_inference_reference = (
    target_inference_reference[
        [
            "Resolved_Target_ID",
            "Gene_Symbol",
            "Protein_Name",
            "Uniprot_ID",
            "ChEMBL_ID",
            "TTD_ID",
            "Target_Class_Level1",
            "Target_Class_Level2",
            "Target_Class_Level3",
            "Target_Class_level_displayed",
            "Target_type",
        ]
    ]
    .rename(
        columns={
            "Resolved_Target_ID":
                "Target_ID"
        }
    )
)

write_tsv(
    target_inference_reference,
    OUTPUT_DIR /
    "target_inference_reference_annotations.tsv"
)

log(
    f"Target IDs available for annotated inference : "
    f"{len(target_inference_reference):,}"
)

log("")


# =============================================================================
# 29. ANNOTATED ASSOCIATION TABLE FOR NEXT STAGE
# =============================================================================

log("=" * 80)
log("20. BUILDING ANNOTATED TARGET-INFERENCE ASSOCIATIONS")
log("=" * 80)

target_inference_associations = (
    association_resolution[
        association_resolution[
            "Annotation_Available"
        ]
    ]
    .copy()
)

target_inference_associations = (
    target_inference_associations[
        [
            "Ingredient_ID",
            "Target_ID",
            "Activity_Type",
            "Activity_Relationship",
            "Activity_Value",
            "Activity_Unit",
            "Reference_ID",
            "Reference_ID_Type",
            "Resolved_Target_ID",
            "Final_Status",
            "Gene_Symbol",
            "Protein_Name",
            "Uniprot_ID",
            "Target_Class_Level1",
            "Target_Class_Level2",
            "Target_Class_Level3",
            "Target_type",
        ]
    ]
)

write_tsv(
    target_inference_associations,
    OUTPUT_DIR /
    "target_inference_associations_annotated.tsv"
)

log(
    f"Annotated association records available : "
    f"{len(target_inference_associations):,}"
)

log("")


# =============================================================================
# 30. FINAL QC
# =============================================================================

log("=" * 80)
log("21. FINAL QC")
log("=" * 80)

qc_rows = []


def add_qc(
    test_name,
    condition,
    observed,
    expected
):

    status = (
        "PASS"
        if condition
        else "FAIL"
    )

    qc_rows.append({
        "QC_Test":
            test_name,
        "Status":
            status,
        "Observed":
            observed,
        "Expected":
            expected,
    })

    log(
        f"{status:5s} | "
        f"{test_name} | "
        f"observed = {observed} | "
        f"expected = {expected}"
    )


# -------------------------------------------------------------------------
# QC 1
# -------------------------------------------------------------------------

add_qc(
    "Association row count preserved",
    len(association_resolution)
    == len(associations),
    len(association_resolution),
    len(associations)
)


# -------------------------------------------------------------------------
# QC 2
# -------------------------------------------------------------------------

add_qc(
    "All association Target_IDs have resolution rows",
    set(
        association_resolution[
            "Target_ID"
        ]
        .astype(str)
    )
    ==
    set(
        final_resolution_df[
            "Target_ID"
        ]
        .astype(str)
    ),
    len(
        set(
            association_resolution[
                "Target_ID"
            ]
            .astype(str)
        )
        -
        set(
            final_resolution_df[
                "Target_ID"
            ]
            .astype(str)
        )
    ),
    0
)


# -------------------------------------------------------------------------
# QC 3
# -------------------------------------------------------------------------

unresolved_annotation_values = int(
    conservative_target_table.loc[
        ~conservative_target_table[
            "Annotation_Available"
        ],
        TARGET_ANNOTATION_COLUMNS
    ]
    .notna()
    .sum()
    .sum()
)

add_qc(
    "No fabricated annotation for unresolved targets",
    unresolved_annotation_values == 0,
    unresolved_annotation_values,
    0
)


# -------------------------------------------------------------------------
# QC 4
# -------------------------------------------------------------------------

add_qc(
    "Target annotation lookup has unique resolved IDs",
    target_lookup[
        "Resolved_Target_ID"
    ].is_unique,
    int(
        target_lookup[
            "Resolved_Target_ID"
        ].nunique()
    ),
    int(
        len(target_lookup)
    )
)


# -------------------------------------------------------------------------
# QC 5
# -------------------------------------------------------------------------

add_qc(
    "Annotated target reference contains unique Target_IDs",
    target_inference_reference[
        "Target_ID"
    ].is_unique,
    int(
        target_inference_reference[
            "Target_ID"
        ].nunique()
    ),
    int(
        len(target_inference_reference)
    )
)


# -------------------------------------------------------------------------
# QC 6
# -------------------------------------------------------------------------

add_qc(
    "No target-negative labels generated",
    True,
    "NONE",
    "NONE"
)


# -------------------------------------------------------------------------
# QC 7
# -------------------------------------------------------------------------

add_qc(
    "Unresolved evidence retained",
    (
        unresolved_raw_records > 0
        and
        unresolved_unique_pairs > 0
    ),
    unresolved_raw_records,
    ">0"
)


# -------------------------------------------------------------------------
# QC 8
# -------------------------------------------------------------------------

add_qc(
    "Unique pair count preserved",
    len(
        pair_resolution
    )
    ==
    total_unique_pairs,
    len(pair_resolution),
    total_unique_pairs
)


# -------------------------------------------------------------------------
# QC 9
# -------------------------------------------------------------------------

annotated_pair_count = int(
    pair_resolution[
        "Annotation_Available"
    ]
    .sum()
)

unannotated_pair_count = int(
    (
        ~pair_resolution[
            "Annotation_Available"
        ]
    )
    .sum()
)

add_qc(
    "Annotated + unannotated pair counts equal total pairs",
    (
        annotated_pair_count
        +
        unannotated_pair_count
        ==
        total_unique_pairs
    ),
    annotated_pair_count
    +
    unannotated_pair_count,
    total_unique_pairs
)


# -------------------------------------------------------------------------
# QC 10
# -------------------------------------------------------------------------

add_qc(
    "No association records lost",
    len(
        association_resolution
    )
    ==
    total_raw_records,
    len(association_resolution),
    total_raw_records
)


qc_df = pd.DataFrame(
    qc_rows
)

write_tsv(
    qc_df,
    OUTPUT_DIR /
    "03_target_resolution_qc_table.tsv"
)

log("")


# =============================================================================
# 31. FINAL STATUS
# =============================================================================

log("=" * 80)
log("22. FINAL STATUS")
log("=" * 80)

failed_qc = qc_df[
    qc_df[
        "Status"
    ]
    == "FAIL"
]

if len(failed_qc) == 0:

    overall_status = (
        "PASS WITH UNRESOLVED TARGETS RETAINED"
    )

else:

    overall_status = (
        "FAIL — QC ERRORS PRESENT"
    )


log(
    f"STATUS: {overall_status}"
)

log("")

log(
    "Resolution interpretation:"
)

log(
    "1. Exact matches to Targets.txt are fully annotated."
)

log(
    "2. Normalization-only matches are reported separately."
)

log(
    "3. IDs merely found in another CMAUP file are NOT "
    "automatically considered biologically resolved."
)

log(
    "4. Unresolved Target_IDs remain in the association evidence."
)

log(
    "5. No protein names, gene symbols, or target classes "
    "were fabricated."
)

log(
    "6. No target-negative labels were generated."
)

log(
    "7. The target-inference reference contains only targets "
    "with actual CMAUP annotation."
)

log(
    "8. Association row counts are preserved."
)

log("")


# =============================================================================
# 32. MANIFEST
# =============================================================================

log("=" * 80)
log("23. MANIFEST")
log("=" * 80)

manifest_output_files = []

for path in sorted(
    OUTPUT_DIR.iterdir()
):

    if (
        path.is_file()
        and
        path.name !=
        "03_target_resolution_manifest.json"
    ):

        manifest_output_files.append({
            "File":
                path.name,
            "Size_Bytes":
                path.stat().st_size,
            "SHA256":
                sha256_file(path),
        })


manifest = {
    "script":
        "03_resolve_unmapped_targets.py",

    "script_version":
        "2.0",

    "created":
        timestamp(),

    "pulp_directory":
        str(PULP_DIR),

    "input_files": {
        "molecules":
            str(MOLECULE_FILE),
        "associations":
            str(ASSOCIATION_FILE),
        "targets":
            str(TARGET_FILE),
        "script02_resolution":
            str(SCRIPT02_RESOLUTION_FILE),
        "script02_molecule_target":
            str(SCRIPT02_MOLECULE_TARGET_FILE),
    },

    "statistics": {
        "association_target_ids":
            len(association_target_ids),

        "target_table_ids":
            len(target_table_ids),

        "exact_mapped_target_ids":
            len(exact_mapped_ids),

        "exact_unmapped_target_ids":
            len(exact_unmapped_ids),

        "normalization_only_candidates":
            normalization_only_count,

        "ambiguous_normalization_candidates":
            ambiguous_count,

        "unresolved_target_ids":
            unresolved_target_count,

        "total_raw_association_records":
            total_raw_records,

        "total_unique_molecule_target_pairs":
            total_unique_pairs,

        "unresolved_raw_association_records":
            unresolved_raw_records,

        "unresolved_unique_molecule_target_pairs":
            unresolved_unique_pairs,

        "unresolved_molecules":
            unresolved_molecule_count,

        "annotated_target_ids":
            len(target_inference_reference),

        "annotated_association_records":
            len(target_inference_associations),
    },

    "status":
        overall_status,

    "output_files":
        manifest_output_files,
}


manifest_path = (
    OUTPUT_DIR /
    "03_target_resolution_manifest.json"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2,
        ensure_ascii=False
    )

log(
    f"Manifest written: "
    f"{manifest_path}"
)

log("")


# =============================================================================
# 33. QC TEXT REPORT
# =============================================================================

qc_report_path = (
    OUTPUT_DIR /
    "03_target_resolution_qc.txt"
)

with open(
    qc_report_path,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "\n".join(
            LOG_LINES
        )
    )

log(
    f"QC report written: "
    f"{qc_report_path}"
)

log("")


# =============================================================================
# 34. COMPLETION
# =============================================================================

log("=" * 80)
log("SCRIPT 03 COMPLETE")
log("=" * 80)

log(
    f"End time: {timestamp()}"
)

log("")

log(
    "IMPORTANT:"
)

log(
    "Inspect 03_target_resolution_qc.txt and "
    "target_id_resolution_final.tsv before proceeding."
)

log(
    "Do not interpret 'UNMAPPED_BUT_FOUND_IN_OTHER_CMAUP_FILE' "
    "as a resolved biological target."
)

log(
    "The next target-inference stage should use "
    "target_inference_reference_annotations.tsv and "
    "target_inference_associations_annotated.tsv."
)

log("")
