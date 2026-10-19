#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
SCRIPT 06 — VALIDATE 40D TARGET / POSSIBLE OFF-TARGET RECOVERY
==============================================================================

Purpose
-------
Validate the Model-1 40D representation as a similarity-based target
and possible off-target recovery system.

Core scientific logic
---------------------
For a known CMAUP molecule:

    molecule
        |
        v
    locked 40D representation
        |
        v
    cosine similarity against CMAUP 40D reference
        |
        v
    nearest known phytochemicals
        |
        +------------------------------+
        |                              |
        v                              v
    high-similarity evidence       lower-similarity evidence
        |                              |
        v                              v
    known annotated targets        known annotated targets
        |                              |
        v                              v
    candidate relevant targets     possible off-targets

Important
---------
1. Only actual CMAUP target annotations are used.
2. UNMAPPED_BUT_FOUND_IN_OTHER_CMAUP_FILE is NOT treated as an annotation.
3. No target names/classes/genes are fabricated.
4. Morgan/MACCS fingerprints are NOT used.
5. The locked Model-1 40D representation is used.
6. Script-05 matrices may contain an additional 'split' metadata column.
7. 'split' is explicitly excluded from the 40D feature matrix.
8. This script performs validation/recovery, not new model training.
9. Similarity is geometric similarity in the locked 40D space; it is not
   itself proof of biological binding.
==============================================================================

Authoritative inputs
--------------------
Script 04:
    target_model/feature_space/04_cmaup_40d_reference/

Script 05:
    target_model/similarity/05_cmaup_40d_similarity/

Script 03:
    target_model/data/03_target_resolution/

Outputs
-------
    target_model/validation/06_40d_target_offtarget_validation/

Main outputs include:
    06_validation_summary.tsv
    06_neighbor_recovery.tsv
    06_target_recovery.tsv
    06_offtarget_recovery.tsv
    06_molecule_recovery_summary.tsv
    06_similarity_threshold_summary.tsv
    06_qc.tsv
    06_qc.txt
    06_manifest.json
    06_feature_alignment.tsv
    06_target_annotation_qc.tsv
    06_input_provenance.tsv

==============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import sklearn


# =============================================================================
# CONFIGURATION
# =============================================================================

PULP = Path("/lustre/home/sklab202/sourav/PULP")

SCRIPT04_DIR = (
    PULP
    / "target_model"
    / "feature_space"
    / "04_cmaup_40d_reference"
)

SCRIPT05_DIR = (
    PULP
    / "target_model"
    / "similarity"
    / "05_cmaup_40d_similarity"
)

SCRIPT03_DIR = (
    PULP
    / "target_model"
    / "data"
    / "03_target_resolution"
)

OUTPUT_DIR = (
    PULP
    / "target_model"
    / "validation"
    / "06_40d_target_offtarget_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Authoritative files
# -----------------------------------------------------------------------------

REFERENCE_FILE = SCRIPT04_DIR / "cmaup_40d_reference.tsv"
SCRIPT04_MANIFEST = SCRIPT04_DIR / "04_cmaup_40d_reference_manifest.json"

STANDARDIZED_FILE = SCRIPT05_DIR / "cmaup_40d_standardized_matrix.tsv"
COSINE_INDEX_FILE = SCRIPT05_DIR / "cmaup_40d_cosine_neighbor_index.joblib"
SCRIPT05_MANIFEST = SCRIPT05_DIR / "05_cmaup_40d_similarity_manifest.json"

TARGET_ANNOTATIONS_FILE = (
    SCRIPT03_DIR / "target_inference_reference_annotations.tsv"
)

ANNOTATED_ASSOCIATIONS_FILE = (
    SCRIPT03_DIR / "target_inference_associations_annotated.tsv"
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

OUTPUT_QC = OUTPUT_DIR / "06_qc.tsv"
OUTPUT_QC_TXT = OUTPUT_DIR / "06_qc.txt"
OUTPUT_MANIFEST = OUTPUT_DIR / "06_manifest.json"

OUTPUT_FEATURE_ALIGNMENT = OUTPUT_DIR / "06_feature_alignment.tsv"
OUTPUT_TARGET_QC = OUTPUT_DIR / "06_target_annotation_qc.tsv"
OUTPUT_INPUT_PROVENANCE = OUTPUT_DIR / "06_input_provenance.tsv"

OUTPUT_VALIDATION_SUMMARY = OUTPUT_DIR / "06_validation_summary.tsv"
OUTPUT_NEIGHBOR_RECOVERY = OUTPUT_DIR / "06_neighbor_recovery.tsv"
OUTPUT_TARGET_RECOVERY = OUTPUT_DIR / "06_target_recovery.tsv"
OUTPUT_OFFTARGET_RECOVERY = OUTPUT_DIR / "06_offtarget_recovery.tsv"
OUTPUT_MOLECULE_SUMMARY = OUTPUT_DIR / "06_molecule_recovery_summary.tsv"
OUTPUT_THRESHOLD_SUMMARY = OUTPUT_DIR / "06_similarity_threshold_summary.tsv"


# =============================================================================
# LOCKED 40D FEATURE ORDER
# =============================================================================

MOLECULAR_FEATURES = [
    "FractionCSP3",
    "SPS",
    "NumAtomStereoCenters",
    "SMR_VSA5",
    "BCUT2D_MRLOW",
    "BCUT2D_CHGLO",
    "VSA_EState6",
    "NumAromaticRings",
    "SMR_VSA7",
    "fr_Al_OH",
]

THREE_D_FEATURES = [
    "PBF",
    "SpherocityIndex",
    "PMI1",
    "InertialShapeFactor",
    "PMI2",
    "NPR2",
    "RadiusOfGyration",
    "Eccentricity",
    "NPR1",
]

MOLFORMER_FEATURES = [
    "molformer_0279",
    "molformer_0491",
    "molformer_0407",
    "molformer_0047",
    "molformer_0230",
    "molformer_0638",
    "molformer_0393",
    "molformer_0129",
    "molformer_0659",
    "molformer_0747",
    "molformer_0298",
    "molformer_0742",
    "molformer_0177",
    "molformer_0119",
    "molformer_0400",
    "molformer_0367",
    "molformer_0485",
    "molformer_0536",
    "molformer_0418",
    "molformer_0455",
    "molformer_0312",
]

FEATURES_40D = (
    MOLECULAR_FEATURES
    + THREE_D_FEATURES
    + MOLFORMER_FEATURES
)

assert len(MOLECULAR_FEATURES) == 10
assert len(THREE_D_FEATURES) == 9
assert len(MOLFORMER_FEATURES) == 21
assert len(FEATURES_40D) == 40


# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Number of nearest reference molecules retained per query.
TOP_K_NEIGHBORS = 20

# Similarity boundary used only for interpretation.
# It does NOT create biological annotations.
HIGH_SIMILARITY_THRESHOLD = 0.90

# Lower similarity boundary for retaining possible related evidence.
LOW_SIMILARITY_THRESHOLD = 0.70

# Similarity below this is retained only in the complete neighbor table.
# It is not classified as a possible off-target candidate.
OFFTARGET_MIN_SIMILARITY = 0.70

# Number of nearest neighbors used for target-level evidence aggregation.
TARGET_TOP_K = 20

# Self-neighbor exclusion during leave-one-out validation.
EXCLUDE_SELF = True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_kv(label: str, value: Any) -> None:
    print(f"{label:<42}: {value}")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {label}: {path}"
        )

    print(f"FOUND : {label}")
    print(f"        {path}")
    print(f"SHA256: {sha256_file(path)}")


def clean_id_series(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
    )


def numeric_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df[columns].copy()

    for c in columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def cosine_similarity_matrix(
    query_vector: np.ndarray,
    reference_matrix: np.ndarray,
) -> np.ndarray:

    q = np.asarray(query_vector, dtype=np.float64)
    X = np.asarray(reference_matrix, dtype=np.float64)

    q_norm = np.linalg.norm(q)

    if not np.isfinite(q_norm) or q_norm == 0:
        raise ValueError("Query vector has invalid or zero norm.")

    x_norm = np.linalg.norm(X, axis=1)

    denominator = x_norm * q_norm

    result = np.zeros(X.shape[0], dtype=np.float64)

    valid = (
        np.isfinite(denominator)
        & (denominator > 0)
    )

    result[valid] = (
        X[valid] @ q
    ) / denominator[valid]

    result[~np.isfinite(result)] = 0.0

    return result


def get_column(df: pd.DataFrame, candidates: List[str], required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise KeyError(
            f"None of the expected columns found: {candidates}"
        )

    return None


def safe_unique_nonblank(values) -> List[str]:
    out = []

    for value in values:
        s = str(value).strip()

        if s and s.lower() not in {"nan", "none", "<na>"}:
            out.append(s)

    return sorted(set(out))


def first_nonblank(series: pd.Series) -> str:
    for value in series:
        s = str(value).strip()

        if s and s.lower() not in {"nan", "none", "<na>"}:
            return s

    return ""


# =============================================================================
# QC LOGGER
# =============================================================================

class QCLogger:

    def __init__(self):
        self.rows = []

    def add(
        self,
        section: str,
        check: str,
        observed: Any,
        expected: Any,
        status: str,
        detail: str = "",
    ):
        self.rows.append(
            {
                "section": section,
                "check": check,
                "observed": observed,
                "expected": expected,
                "status": status,
                "detail": detail,
            }
        )

        prefix = "PASS" if status == "PASS" else status

        print(
            f"{prefix:<5} | {check} | "
            f"observed = {observed} | expected = {expected}"
        )

    def dataframe(self):
        return pd.DataFrame(self.rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    start_time = now_iso()
    qc = QCLogger()

    manifest = {
        "script": "06_validate_40d_target_offtarget_recovery.py",
        "version": "corrected_split_metadata_handling",
        "start_time": start_time,
        "parameters": {
            "top_k_neighbors": TOP_K_NEIGHBORS,
            "target_top_k": TARGET_TOP_K,
            "high_similarity_threshold": HIGH_SIMILARITY_THRESHOLD,
            "low_similarity_threshold": LOW_SIMILARITY_THRESHOLD,
            "offtarget_min_similarity": OFFTARGET_MIN_SIMILARITY,
            "exclude_self": EXCLUDE_SELF,
        },
        "feature_space": {
            "dimension": 40,
            "molecular_features": MOLECULAR_FEATURES,
            "three_d_features": THREE_D_FEATURES,
            "molformer_features": MOLFORMER_FEATURES,
            "feature_order": FEATURES_40D,
        },
        "model_training_performed": False,
        "new_features_generated": False,
        "new_imputation_performed": False,
    }

    # =========================================================================
    # 1. INPUT FILE CHECK
    # =========================================================================

    print_header(
        "SCRIPT 06 — VALIDATE 40D TARGET / POSSIBLE OFF-TARGET RECOVERY"
    )

    print_kv("PULP directory", PULP)
    print_kv("Output directory", OUTPUT_DIR)
    print_kv("Start time", start_time)
    print_kv("Python", platform.python_version())
    print_kv("NumPy", np.__version__)
    print_kv("pandas", pd.__version__)
    print_kv("scikit-learn", sklearn.__version__)

    print_header("1. INPUT FILE CHECK")

    required_inputs = [
        (REFERENCE_FILE, "Script-04 40D reference"),
        (SCRIPT04_MANIFEST, "Script-04 manifest"),
        (STANDARDIZED_FILE, "Script-05 standardized matrix"),
        (SCRIPT05_MANIFEST, "Script-05 manifest"),
        (TARGET_ANNOTATIONS_FILE, "Target annotations"),
        (
            ANNOTATED_ASSOCIATIONS_FILE,
            "Annotated target associations",
        ),
    ]

    for path, label in required_inputs:
        require_file(path, label)

    # Cosine index is useful but the validation below does not blindly trust
    # its feature count. The actual standardized matrix is authoritative for
    # this validation.
    if COSINE_INDEX_FILE.exists():
        print(f"FOUND : Script-05 cosine index")
        print(f"        {COSINE_INDEX_FILE}")
        print(f"SHA256: {sha256_file(COSINE_INDEX_FILE)}")
        cosine_index_available = True
    else:
        print(
            f"OPTIONAL Script-05 cosine index not found: "
            f"{COSINE_INDEX_FILE}"
        )
        cosine_index_available = False

    manifest["input_files"] = []

    for path, label in required_inputs:
        manifest["input_files"].append(
            {
                "label": label,
                "path": str(path.relative_to(PULP)),
                "exists": True,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "required": True,
            }
        )

    if COSINE_INDEX_FILE.exists():
        manifest["input_files"].append(
            {
                "label": "Script-05 cosine index",
                "path": str(COSINE_INDEX_FILE.relative_to(PULP)),
                "exists": True,
                "sha256": sha256_file(COSINE_INDEX_FILE),
                "size_bytes": COSINE_INDEX_FILE.stat().st_size,
                "required": False,
            }
        )

    # =========================================================================
    # 2. LOAD SCRIPT-04 REFERENCE
    # =========================================================================

    print_header("2. LOADING AUTHORITATIVE SCRIPT-04 40D REFERENCE")

    reference = pd.read_csv(
        REFERENCE_FILE,
        sep="\t",
        low_memory=False,
    )

    reference["np_id"] = clean_id_series(reference["np_id"])

    reference_feature_missing = [
        f for f in FEATURES_40D
        if f not in reference.columns
    ]

    reference_extra_features = [
        c for c in reference.columns
        if c not in FEATURES_40D
        and c != "np_id"
    ]

    print_kv("Reference rows", len(reference))

    qc.add(
        "reference",
        "Reference row count",
        len(reference),
        60190,
        "PASS" if len(reference) == 60190 else "FAIL",
    )

    qc.add(
        "reference",
        "Reference np_id column present",
        "np_id" in reference.columns,
        True,
        "PASS" if "np_id" in reference.columns else "FAIL",
    )

    blank_ids = int((reference["np_id"] == "").sum())

    qc.add(
        "reference",
        "Reference np_id nonblank",
        blank_ids,
        0,
        "PASS" if blank_ids == 0 else "FAIL",
    )

    duplicate_ids = int(
        reference["np_id"].duplicated().sum()
    )

    qc.add(
        "reference",
        "Reference np_id unique",
        duplicate_ids,
        0,
        "PASS" if duplicate_ids == 0 else "FAIL",
    )

    qc.add(
        "reference",
        "All 40 expected features present",
        reference_feature_missing,
        [],
        "PASS" if not reference_feature_missing else "FAIL",
    )

    if reference_feature_missing:
        raise RuntimeError(
            "Script-04 reference is missing locked 40D features."
        )

    # =========================================================================
    # 3. LOAD SCRIPT-05 STANDARDIZED MATRIX
    # =========================================================================

    print_header("3. LOADING SCRIPT-05 STANDARDIZED 40D MATRIX")

    standardized = pd.read_csv(
        STANDARDIZED_FILE,
        sep="\t",
        low_memory=False,
    )

    # -------------------------------------------------------------------------
    # CRITICAL CORRECTION
    #
    # Script 05 contains:
    #
    # split + 40 feature columns
    #
    # 'split' is metadata and is NOT a 40D feature.
    #
    # We therefore explicitly separate metadata from the locked feature space.
    # -------------------------------------------------------------------------

    standardized_id_col = get_column(
        standardized,
        ["np_id", "NP_ID", "ID", "id"],
        required=True,
    )

    standardized["np_id"] = clean_id_series(
        standardized[standardized_id_col]
    )

    metadata_columns = []

    for c in ["np_id", "split"]:
        if c in standardized.columns:
            metadata_columns.append(c)

    standardized_feature_columns = [
        c for c in standardized.columns
        if c not in metadata_columns
    ]

    standardized_missing = [
        f for f in FEATURES_40D
        if f not in standardized.columns
    ]

    standardized_extra = [
        c for c in standardized_feature_columns
        if c not in FEATURES_40D
    ]

    print_kv(
        "Standardized total columns",
        len(standardized.columns),
    )

    print_kv(
        "Standardized metadata columns",
        metadata_columns,
    )

    print_kv(
        "Standardized actual feature columns",
        len(standardized_feature_columns),
    )

    print_kv(
        "Standardized extra metadata/features",
        standardized_extra,
    )

    qc.add(
        "standardized_matrix",
        "Standardized actual feature count",
        len(standardized_feature_columns),
        40,
        "PASS"
        if len(standardized_feature_columns) == 40
        else "FAIL",
        detail=(
            "The 'split' column is treated as metadata and excluded "
            "from the 40D feature space."
        ),
    )

    qc.add(
        "standardized_matrix",
        "Standardized locked feature completeness",
        standardized_missing,
        [],
        "PASS" if not standardized_missing else "FAIL",
    )

    if standardized_missing:
        raise RuntimeError(
            "Script-05 standardized matrix is missing one or more "
            "locked 40D features."
        )

    if standardized_extra:
        raise RuntimeError(
            "Unexpected non-metadata feature columns detected in "
            "Script-05 standardized matrix: "
            + ", ".join(standardized_extra)
        )

    standardized_order = [
        c for c in standardized.columns
        if c in FEATURES_40D
    ]

    qc.add(
        "standardized_matrix",
        "Standardized locked feature order",
        standardized_order,
        FEATURES_40D,
        "PASS"
        if standardized_order == FEATURES_40D
        else "FAIL",
    )

    if standardized_order != FEATURES_40D:
        raise ValueError(
            "Script-05 40D feature order does not match the locked "
            "Model-1 40D feature order."
        )

    qc.add(
        "standardized_matrix",
        "Standardized row count",
        len(standardized),
        len(reference),
        "PASS"
        if len(standardized) == len(reference)
        else "FAIL",
    )

    reference_ids = set(reference["np_id"])
    standardized_ids = set(standardized["np_id"])

    missing_in_standardized = reference_ids - standardized_ids
    extra_in_standardized = standardized_ids - reference_ids

    qc.add(
        "standardized_matrix",
        "Reference IDs missing from standardized matrix",
        len(missing_in_standardized),
        0,
        "PASS" if not missing_in_standardized else "FAIL",
    )

    qc.add(
        "standardized_matrix",
        "Standardized IDs absent from reference",
        len(extra_in_standardized),
        0,
        "PASS" if not extra_in_standardized else "FAIL",
    )

    if missing_in_standardized or extra_in_standardized:
        raise RuntimeError(
            "Script-04 and Script-05 molecule universes do not match."
        )

    # Reorder Script-05 matrix exactly to Script-04 reference order.
    standardized = (
        standardized
        .set_index("np_id")
        .loc[reference["np_id"]]
        .reset_index()
    )

    # =========================================================================
    # 4. FEATURE ALIGNMENT QC
    # =========================================================================

    print_header("4. LOCKED 40D FEATURE ALIGNMENT QC")

    alignment_rows = []

    for i, feature in enumerate(FEATURES_40D, start=1):

        if feature in MOLECULAR_FEATURES:
            family = "Molecular"
        elif feature in THREE_D_FEATURES:
            family = "3D-QSAR"
        else:
            family = "MoLFormer"

        alignment_rows.append(
            {
                "feature_position": i,
                "feature_name": feature,
                "feature_family": family,
                "script04_present": feature in reference.columns,
                "script05_present": feature in standardized.columns,
                "script05_position": (
                    standardized.columns.get_loc(feature)
                    if feature in standardized.columns
                    else -1
                ),
                "locked_order": i,
            }
        )

    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(
        OUTPUT_FEATURE_ALIGNMENT,
        sep="\t",
        index=False,
    )

    print_kv(
        "Locked feature count",
        len(FEATURES_40D),
    )

    print_kv(
        "Molecular feature count",
        len(MOLECULAR_FEATURES),
    )

    print_kv(
        "3D-QSAR feature count",
        len(THREE_D_FEATURES),
    )

    print_kv(
        "MoLFormer feature count",
        len(MOLFORMER_FEATURES),
    )

    # =========================================================================
    # 5. NUMERICAL QC
    # =========================================================================

    print_header("5. NUMERICAL QC OF LOCKED 40D SPACE")

    reference_40d = numeric_frame(
        reference,
        FEATURES_40D,
    )

    standardized_40d = numeric_frame(
        standardized,
        FEATURES_40D,
    )

    ref_nan = int(reference_40d.isna().sum().sum())
    ref_inf = int(
        np.isinf(
            reference_40d.to_numpy(dtype=np.float64)
        ).sum()
    )

    std_nan = int(standardized_40d.isna().sum().sum())
    std_inf = int(
        np.isinf(
            standardized_40d.to_numpy(dtype=np.float64)
        ).sum()
    )

    print_kv("Reference NaN count", ref_nan)
    print_kv("Reference Inf count", ref_inf)
    print_kv("Standardized NaN count", std_nan)
    print_kv("Standardized Inf count", std_inf)

    qc.add(
        "numerical",
        "Reference 40D NaN count",
        ref_nan,
        0,
        "PASS" if ref_nan == 0 else "FAIL",
    )

    qc.add(
        "numerical",
        "Reference 40D Inf count",
        ref_inf,
        0,
        "PASS" if ref_inf == 0 else "FAIL",
    )

    qc.add(
        "numerical",
        "Standardized 40D NaN count",
        std_nan,
        0,
        "PASS" if std_nan == 0 else "FAIL",
    )

    qc.add(
        "numerical",
        "Standardized 40D Inf count",
        std_inf,
        0,
        "PASS" if std_inf == 0 else "FAIL",
    )

    if (
        ref_nan > 0
        or ref_inf > 0
        or std_nan > 0
        or std_inf > 0
    ):
        raise RuntimeError(
            "Invalid numerical values detected in the final 40D "
            "representation."
        )

    # =========================================================================
    # 6. LOAD TARGET ANNOTATIONS
    # =========================================================================

    print_header("6. LOADING TARGET ANNOTATIONS")

    target_annotations = pd.read_csv(
        TARGET_ANNOTATIONS_FILE,
        sep="\t",
        low_memory=False,
    )

    target_annotations.columns = [
        str(c).strip()
        for c in target_annotations.columns
    ]

    target_id_annotation_col = get_column(
        target_annotations,
        ["Target_ID", "target_id", "TargetID"],
        required=True,
    )

    target_annotations["Target_ID"] = clean_id_series(
        target_annotations[target_id_annotation_col]
    )

    target_annotations = target_annotations[
        target_annotations["Target_ID"] != ""
    ].copy()

    annotation_duplicates = int(
        target_annotations["Target_ID"].duplicated().sum()
    )

    print_kv(
        "Target annotation rows",
        len(target_annotations),
    )

    print_kv(
        "Unique Target_IDs",
        target_annotations["Target_ID"].nunique(),
    )

    print_kv(
        "Duplicate Target_ID rows",
        annotation_duplicates,
    )

    qc.add(
        "target_annotations",
        "Target annotation IDs unique",
        annotation_duplicates,
        0,
        "PASS"
        if annotation_duplicates == 0
        else "FAIL",
    )

    if annotation_duplicates:
        raise RuntimeError(
            "Target annotation table contains duplicate Target_ID "
            "records. Script 03 promised a unique annotation lookup."
        )

    # =========================================================================
    # 7. LOAD ANNOTATED ASSOCIATIONS
    # =========================================================================

    print_header("7. LOADING ANNOTATED TARGET ASSOCIATIONS")

    associations = pd.read_csv(
        ANNOTATED_ASSOCIATIONS_FILE,
        sep="\t",
        low_memory=False,
    )

    associations.columns = [
        str(c).strip()
        for c in associations.columns
    ]

    molecule_id_col = get_column(
        associations,
        ["np_id", "NP_ID", "Molecule_ID", "Ingredient_ID"],
        required=True,
    )

    association_target_col = get_column(
        associations,
        ["Target_ID", "target_id", "TargetID"],
        required=True,
    )

    associations["np_id"] = clean_id_series(
        associations[molecule_id_col]
    )

    associations["Target_ID"] = clean_id_series(
        associations[association_target_col]
    )

    associations = associations[
        (associations["np_id"] != "")
        & (associations["Target_ID"] != "")
    ].copy()

    print_kv(
        "Annotated association rows",
        len(associations),
    )

    print_kv(
        "Unique molecules with annotated associations",
        associations["np_id"].nunique(),
    )

    print_kv(
        "Unique annotated targets",
        associations["Target_ID"].nunique(),
    )

    annotation_target_ids = set(
        target_annotations["Target_ID"]
    )

    association_target_ids = set(
        associations["Target_ID"]
    )

    missing_annotations = (
        association_target_ids
        - annotation_target_ids
    )

    qc.add(
        "target_annotations",
        "All association Target_IDs have annotations",
        len(missing_annotations),
        0,
        "PASS" if len(missing_annotations) == 0 else "FAIL",
    )

    if missing_annotations:
        raise RuntimeError(
            "Annotated association table contains Target_IDs that "
            "are absent from the target annotation reference."
        )

    # =========================================================================
    # 8. BUILD TARGET ANNOTATION LOOKUP
    # =========================================================================

    print_header("8. BUILDING TARGET ANNOTATION LOOKUP")

    preferred_annotation_columns = [
        "Target_Class_Level1",
        "Target_Class_Level2",
        "Target_Class_Level3",
        "Protein_Name",
        "Gene_Symbol",
        "Uniprot_ID",
        "ChEMBL_ID",
        "TTD_ID",
        "Target_type",
    ]

    available_annotation_columns = [
        c
        for c in preferred_annotation_columns
        if c in target_annotations.columns
    ]

    print_kv(
        "Annotation fields available",
        available_annotation_columns,
    )

    target_lookup = (
        target_annotations[
            ["Target_ID"] + available_annotation_columns
        ]
        .drop_duplicates("Target_ID")
        .set_index("Target_ID")
        .to_dict("index")
    )

    # =========================================================================
    # 9. BUILD MOLECULE -> TARGET MAP
    # =========================================================================

    print_header("9. BUILDING MOLECULE → TARGET EVIDENCE MAP")

    molecule_target_map: Dict[str, List[str]] = {}

    for molecule_id, group in associations.groupby("np_id"):

        target_ids = safe_unique_nonblank(
            group["Target_ID"]
        )

        molecule_target_map[molecule_id] = target_ids

    print_kv(
        "Molecules with target evidence",
        len(molecule_target_map),
    )

    print_kv(
        "Target-bearing molecule-target pairs",
        sum(
            len(v)
            for v in molecule_target_map.values()
        ),
    )

    # =========================================================================
    # 10. ALIGN STANDARDIZED MATRIX
    # =========================================================================

    print_header("10. BUILDING AUTHORITATIVE STANDARDIZED 40D MATRIX")

    X = standardized_40d.to_numpy(
        dtype=np.float64
    )

    print_kv("40D matrix shape", X.shape)

    qc.add(
        "40d_matrix",
        "40D row count",
        X.shape[0],
        60190,
        "PASS" if X.shape[0] == 60190 else "FAIL",
    )

    qc.add(
        "40d_matrix",
        "40D column count",
        X.shape[1],
        40,
        "PASS" if X.shape[1] == 40 else "FAIL",
    )

    # =========================================================================
    # 11. NORMALIZE FOR COSINE SEARCH
    # =========================================================================

    print_header("11. PREPARING COSINE SIMILARITY SPACE")

    norms = np.linalg.norm(
        X,
        axis=1,
    )

    zero_norm_count = int(
        np.sum(norms == 0)
    )

    invalid_norm_count = int(
        np.sum(~np.isfinite(norms))
    )

    print_kv(
        "Zero-norm reference vectors",
        zero_norm_count,
    )

    print_kv(
        "Invalid-norm reference vectors",
        invalid_norm_count,
    )

    qc.add(
        "cosine_space",
        "Zero-norm reference vectors",
        zero_norm_count,
        0,
        "PASS" if zero_norm_count == 0 else "FAIL",
    )

    qc.add(
        "cosine_space",
        "Invalid-norm reference vectors",
        invalid_norm_count,
        0,
        "PASS" if invalid_norm_count == 0 else "FAIL",
    )

    if zero_norm_count or invalid_norm_count:
        raise RuntimeError(
            "Reference contains invalid vectors for cosine similarity."
        )

    X_normalized = X / norms[:, None]

    # =========================================================================
    # 12. BUILD INDEX MAPPING
    # =========================================================================

    print_header("12. BUILDING REFERENCE INDEX")

    reference_ids_ordered = reference["np_id"].tolist()

    id_to_row = {
        molecule_id: i
        for i, molecule_id
        in enumerate(reference_ids_ordered)
    }

    qc.add(
        "reference_index",
        "ID-to-row mapping size",
        len(id_to_row),
        60190,
        "PASS"
        if len(id_to_row) == 60190
        else "FAIL",
    )

    # =========================================================================
    # 13. LEAVE-ONE-OUT TARGET/OFF-TARGET RECOVERY
    # =========================================================================

    print_header(
        "13. LEAVE-ONE-OUT 40D TARGET / POSSIBLE OFF-TARGET RECOVERY"
    )

    validation_rows = []
    neighbor_rows = []
    target_recovery_rows = []
    offtarget_rows = []
    molecule_summary_rows = []

    validation_molecules = sorted(
        set(molecule_target_map.keys())
        & set(reference_ids_ordered)
    )

    print_kv(
        "Target-bearing molecules in 40D reference",
        len(validation_molecules),
    )

    total_target_bearing = len(validation_molecules)

    if total_target_bearing == 0:
        raise RuntimeError(
            "No target-bearing molecules overlap the 40D reference."
        )

    for counter, query_id in enumerate(
        validation_molecules,
        start=1,
    ):

        query_row = id_to_row[query_id]

        query_vector = X_normalized[query_row]

        similarities = X_normalized @ query_vector

        if EXCLUDE_SELF:
            similarities[query_row] = -np.inf

        # Select top K by similarity.
        k = min(
            TOP_K_NEIGHBORS,
            len(similarities) - (
                1 if EXCLUDE_SELF else 0
            ),
        )

        top_indices = np.argpartition(
            -similarities,
            kth=k - 1,
        )[:k]

        top_indices = top_indices[
            np.argsort(
                -similarities[top_indices]
            )
        ]

        query_targets = set(
            molecule_target_map.get(
                query_id,
                [],
            )
        )

        # ---------------------------------------------------------------------
        # Neighbor-level evidence
        # ---------------------------------------------------------------------

        recovered_target_ids = set()

        high_similarity_neighbor_count = 0
        moderate_similarity_neighbor_count = 0

        for rank, idx in enumerate(
            top_indices,
            start=1,
        ):

            neighbor_id = reference_ids_ordered[idx]
            similarity = float(similarities[idx])

            if similarity >= HIGH_SIMILARITY_THRESHOLD:
                similarity_band = "HIGH"
                high_similarity_neighbor_count += 1

            elif similarity >= LOW_SIMILARITY_THRESHOLD:
                similarity_band = "MODERATE"
                moderate_similarity_neighbor_count += 1

            else:
                similarity_band = "LOW"

            neighbor_targets = molecule_target_map.get(
                neighbor_id,
                [],
            )

            overlap_targets = (
                query_targets
                & set(neighbor_targets)
            )

            recovered_target_ids.update(
                neighbor_targets
            )

            neighbor_rows.append(
                {
                    "query_np_id": query_id,
                    "neighbor_rank": rank,
                    "neighbor_np_id": neighbor_id,
                    "cosine_similarity": similarity,
                    "similarity_band": similarity_band,
                    "query_known_target_count": len(query_targets),
                    "neighbor_known_target_count": len(
                        neighbor_targets
                    ),
                    "shared_target_count": len(
                        overlap_targets
                    ),
                    "shared_target_ids": ";".join(
                        sorted(overlap_targets)
                    ),
                    "neighbor_has_annotated_target": (
                        len(neighbor_targets) > 0
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Target-level aggregation
        # ---------------------------------------------------------------------

        target_evidence: Dict[str, List[float]] = {}

        for idx in top_indices:

            neighbor_id = reference_ids_ordered[idx]
            similarity = float(similarities[idx])

            if not np.isfinite(similarity):
                continue

            neighbor_targets = molecule_target_map.get(
                neighbor_id,
                [],
            )

            for target_id in neighbor_targets:

                target_evidence.setdefault(
                    target_id,
                    [],
                ).append(
                    similarity
                )

        target_records_for_query = []

        for target_id, sim_values in target_evidence.items():

            max_similarity = max(sim_values)
            mean_similarity = float(
                np.mean(sim_values)
            )

            supporting_neighbor_count = len(
                sim_values
            )

            annotation = target_lookup.get(
                target_id,
                {},
            )

            if max_similarity >= HIGH_SIMILARITY_THRESHOLD:
                evidence_class = "HIGH_SIMILARITY_TARGET"

            elif max_similarity >= LOW_SIMILARITY_THRESHOLD:
                evidence_class = "MODERATE_SIMILARITY_TARGET"

            else:
                evidence_class = "LOW_SIMILARITY_TARGET"

            target_records_for_query.append(
                {
                    "query_np_id": query_id,
                    "Target_ID": target_id,
                    "max_similarity": max_similarity,
                    "mean_similarity": mean_similarity,
                    "supporting_neighbor_count":
                        supporting_neighbor_count,
                    "evidence_class": evidence_class,
                    **{
                        c: annotation.get(c, "")
                        for c in available_annotation_columns
                    },
                }
            )

        target_records_for_query.sort(
            key=lambda x: (
                -x["max_similarity"],
                -x["supporting_neighbor_count"],
                x["Target_ID"],
            )
        )

        # ---------------------------------------------------------------------
        # Relevant target evidence
        # ---------------------------------------------------------------------

        for rank, record in enumerate(
            target_records_for_query,
            start=1,
        ):

            if record["max_similarity"] >= HIGH_SIMILARITY_THRESHOLD:

                target_recovery_rows.append(
                    {
                        "target_rank": rank,
                        **record,
                        "interpretation":
                            "HIGH_SIMILARITY_TARGET",
                    }
                )

        # ---------------------------------------------------------------------
        # Possible off-target evidence
        #
        # Definition used here:
        # target evidence carried by similar molecules, but NOT among the
        # query's already-known target set, and with similarity lower than
        # the strongest high-similarity target evidence.
        #
        # This is intentionally conservative.
        # ---------------------------------------------------------------------

        highest_relevant_similarity = (
            max(
                [
                    r["max_similarity"]
                    for r in target_records_for_query
                    if (
                        r["Target_ID"] in query_targets
                        and
                        r["max_similarity"]
                        >= HIGH_SIMILARITY_THRESHOLD
                    )
                ],
                default=HIGH_SIMILARITY_THRESHOLD,
            )
        )

        for rank, record in enumerate(
            target_records_for_query,
            start=1,
        ):

            target_id = record["Target_ID"]
            max_similarity = record["max_similarity"]

            if target_id in query_targets:
                continue

            if max_similarity < OFFTARGET_MIN_SIMILARITY:
                continue

            if max_similarity >= highest_relevant_similarity:
                continue

            offtarget_rows.append(
                {
                    "offtarget_rank": rank,
                    **record,
                    "interpretation":
                        "POSSIBLE_OFF_TARGET",
                    "offtarget_rule":
                        "annotated target of a similar molecule; "
                        "similarity lower than strongest relevant "
                        "target evidence",
                }
            )

        # ---------------------------------------------------------------------
        # Query-level recovery metrics
        # ---------------------------------------------------------------------

        high_target_ids = set(
            r["Target_ID"]
            for r in target_records_for_query
            if r["max_similarity"]
            >= HIGH_SIMILARITY_THRESHOLD
        )

        recovered_known_targets = (
            query_targets
            & high_target_ids
        )

        target_recall = (
            len(recovered_known_targets)
            / len(query_targets)
            if query_targets
            else np.nan
        )

        top_neighbor_target_overlap = 0

        for idx in top_indices:

            neighbor_id = reference_ids_ordered[idx]

            neighbor_targets = set(
                molecule_target_map.get(
                    neighbor_id,
                    [],
                )
            )

            if query_targets & neighbor_targets:
                top_neighbor_target_overlap += 1

        best_similarity = (
            float(
                similarities[top_indices[0]]
            )
            if len(top_indices) > 0
            else np.nan
        )

        validation_rows.append(
            {
                "query_np_id": query_id,
                "known_target_count": len(query_targets),
                "recovered_known_target_count":
                    len(recovered_known_targets),
                "known_target_recall_at_top_k":
                    target_recall,
                "best_neighbor_similarity":
                    best_similarity,
                "high_similarity_neighbor_count":
                    high_similarity_neighbor_count,
                "moderate_similarity_neighbor_count":
                    moderate_similarity_neighbor_count,
                "target_bearing_neighbors":
                    sum(
                        1
                        for idx in top_indices
                        if len(
                            molecule_target_map.get(
                                reference_ids_ordered[idx],
                                [],
                            )
                        ) > 0
                    ),
                "neighbors_sharing_at_least_one_known_target":
                    top_neighbor_target_overlap,
                "candidate_high_similarity_target_count":
                    len(high_target_ids),
                "possible_offtarget_count":
                    sum(
                        1
                        for r in target_records_for_query
                        if (
                            r["Target_ID"] not in query_targets
                            and
                            r["max_similarity"]
                            >= OFFTARGET_MIN_SIMILARITY
                            and
                            r["max_similarity"]
                            < highest_relevant_similarity
                        )
                    ),
            }
        )

        molecule_summary_rows.append(
            {
                "np_id": query_id,
                "known_target_count": len(query_targets),
                "best_40d_neighbor_similarity":
                    best_similarity,
                "known_target_recall":
                    target_recall,
                "high_similarity_target_count":
                    len(high_target_ids),
                "possible_offtarget_count":
                    sum(
                        1
                        for r in target_records_for_query
                        if (
                            r["Target_ID"] not in query_targets
                            and
                            r["max_similarity"]
                            >= OFFTARGET_MIN_SIMILARITY
                            and
                            r["max_similarity"]
                            < highest_relevant_similarity
                        )
                    ),
            }
        )

        if (
            counter % 500 == 0
            or counter == total_target_bearing
        ):
            print(
                f"Processed target-bearing molecules: "
                f"{counter}/{total_target_bearing}"
            )

    # =========================================================================
    # 14. BUILD DATAFRAMES
    # =========================================================================

    print_header("14. BUILDING RECOVERY TABLES")

    validation_df = pd.DataFrame(
        validation_rows
    )

    neighbor_df = pd.DataFrame(
        neighbor_rows
    )

    target_recovery_df = pd.DataFrame(
        target_recovery_rows
    )

    offtarget_df = pd.DataFrame(
        offtarget_rows
    )

    molecule_summary_df = pd.DataFrame(
        molecule_summary_rows
    )

    validation_df.to_csv(
        OUTPUT_VALIDATION_SUMMARY,
        sep="\t",
        index=False,
    )

    neighbor_df.to_csv(
        OUTPUT_NEIGHBOR_RECOVERY,
        sep="\t",
        index=False,
    )

    target_recovery_df.to_csv(
        OUTPUT_TARGET_RECOVERY,
        sep="\t",
        index=False,
    )

    offtarget_df.to_csv(
        OUTPUT_OFFTARGET_RECOVERY,
        sep="\t",
        index=False,
    )

    molecule_summary_df.to_csv(
        OUTPUT_MOLECULE_SUMMARY,
        sep="\t",
        index=False,
    )

    print_kv(
        "Validation molecules",
        len(validation_df),
    )

    print_kv(
        "Neighbor recovery rows",
        len(neighbor_df),
    )

    print_kv(
        "High-similarity target evidence rows",
        len(target_recovery_df),
    )

    print_kv(
        "Possible off-target evidence rows",
        len(offtarget_df),
    )

    # =========================================================================
    # 15. TARGET ANNOTATION QC
    # =========================================================================

    print_header("15. TARGET ANNOTATION QC")

    target_qc_rows = []

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

        if column not in target_annotations.columns:
            continue

        values = (
            target_annotations[column]
            .astype(str)
            .str.strip()
        )

        missing = int(
            values.isin(
                ["", "nan", "None", "<NA>"]
            ).sum()
        )

        available = len(values) - missing

        target_qc_rows.append(
            {
                "field": column,
                "target_rows": len(values),
                "available": available,
                "missing": missing,
                "completeness_percent":
                    (
                        100.0 * available / len(values)
                        if len(values)
                        else np.nan
                    ),
            }
        )

    target_qc_df = pd.DataFrame(
        target_qc_rows
    )

    target_qc_df.to_csv(
        OUTPUT_TARGET_QC,
        sep="\t",
        index=False,
    )

    # =========================================================================
    # 16. SIMILARITY THRESHOLD SUMMARY
    # =========================================================================

    print_header("16. SIMILARITY THRESHOLD SUMMARY")

    if len(neighbor_df) > 0:

        threshold_rows = []

        for threshold in [
            0.95,
            0.90,
            0.85,
            0.80,
            0.75,
            0.70,
            0.60,
        ]:

            selected = neighbor_df[
                neighbor_df["cosine_similarity"]
                >= threshold
            ]

            threshold_rows.append(
                {
                    "similarity_threshold":
                        threshold,
                    "neighbor_rows":
                        len(selected),
                    "unique_queries":
                        selected["query_np_id"].nunique(),
                    "unique_neighbors":
                        selected["neighbor_np_id"].nunique(),
                    "target_bearing_neighbor_rows":
                        int(
                            selected[
                                "neighbor_has_annotated_target"
                            ].sum()
                        ),
                }
            )

        threshold_df = pd.DataFrame(
            threshold_rows
        )

    else:

        threshold_df = pd.DataFrame(
            columns=[
                "similarity_threshold",
                "neighbor_rows",
                "unique_queries",
                "unique_neighbors",
                "target_bearing_neighbor_rows",
            ]
        )

    threshold_df.to_csv(
        OUTPUT_THRESHOLD_SUMMARY,
        sep="\t",
        index=False,
    )

    # =========================================================================
    # 17. VALIDATION METRICS
    # =========================================================================

    print_header("17. VALIDATION METRICS")

    if len(validation_df) > 0:

        mean_target_recall = float(
            validation_df[
                "known_target_recall_at_top_k"
            ].mean()
        )

        median_best_similarity = float(
            validation_df[
                "best_neighbor_similarity"
            ].median()
        )

        fraction_with_high_neighbor = float(
            (
                validation_df[
                    "high_similarity_neighbor_count"
                ] > 0
            ).mean()
        )

        fraction_with_target_neighbor = float(
            (
                validation_df[
                    "target_bearing_neighbors"
                ] > 0
            ).mean()
        )

    else:

        mean_target_recall = np.nan
        median_best_similarity = np.nan
        fraction_with_high_neighbor = np.nan
        fraction_with_target_neighbor = np.nan

    print_kv(
        "Mean known-target recall",
        mean_target_recall,
    )

    print_kv(
        "Median best-neighbor similarity",
        median_best_similarity,
    )

    print_kv(
        "Fraction with >=1 high-similarity neighbor",
        fraction_with_high_neighbor,
    )

    print_kv(
        "Fraction with >=1 target-bearing neighbor",
        fraction_with_target_neighbor,
    )

    # =========================================================================
    # 18. FINAL QC
    # =========================================================================

    print_header("18. FINAL QC")

    qc.add(
        "final",
        "Reference and standardized molecule count",
        f"{len(reference)} / {len(standardized)}",
        "60190 / 60190",
        "PASS"
        if len(reference) == 60190
        and len(standardized) == 60190
        else "FAIL",
    )

    qc.add(
        "final",
        "Locked 40D feature count",
        len(FEATURES_40D),
        40,
        "PASS" if len(FEATURES_40D) == 40 else "FAIL",
    )

    qc.add(
        "final",
        "Script-05 split metadata excluded from features",
        "split" in standardized.columns
        and "split" not in standardized_order,
        True,
        "PASS"
        if (
            "split" in standardized.columns
            and "split" not in standardized_order
        )
        else "FAIL",
    )

    qc.add(
        "final",
        "40D feature order locked",
        standardized_order == FEATURES_40D,
        True,
        "PASS"
        if standardized_order == FEATURES_40D
        else "FAIL",
    )

    qc.add(
        "final",
        "No new imputation performed",
        True,
        True,
        "PASS",
    )

    qc.add(
        "final",
        "No new features generated",
        True,
        True,
        "PASS",
    )

    qc.add(
        "final",
        "No Morgan fingerprints used",
        True,
        True,
        "PASS",
    )

    qc.add(
        "final",
        "Only annotated CMAUP targets used",
        len(missing_annotations),
        0,
        "PASS"
        if len(missing_annotations) == 0
        else "FAIL",
    )

    qc.add(
        "final",
        "Target recovery table generated",
        len(target_recovery_df),
        ">0",
        "PASS"
        if len(target_recovery_df) > 0
        else "WARN",
    )

    qc.add(
        "final",
        "Possible off-target table generated",
        len(offtarget_df),
        ">=0",
        "PASS",
    )

    # =========================================================================
    # 19. INPUT PROVENANCE
    # =========================================================================

    print_header("19. INPUT PROVENANCE")

    provenance_rows = []

    provenance_files = [
        ("script04_reference", REFERENCE_FILE, True),
        ("script04_manifest", SCRIPT04_MANIFEST, True),
        ("script05_standardized_matrix", STANDARDIZED_FILE, True),
        ("script05_manifest", SCRIPT05_MANIFEST, True),
        ("target_annotations", TARGET_ANNOTATIONS_FILE, True),
        (
            "annotated_target_associations",
            ANNOTATED_ASSOCIATIONS_FILE,
            True,
        ),
        (
            "script05_cosine_index",
            COSINE_INDEX_FILE,
            False,
        ),
    ]

    for label, path, required in provenance_files:

        provenance_rows.append(
            {
                "input_label": label,
                "path": str(path.relative_to(PULP)),
                "exists": path.exists(),
                "required": required,
                "sha256":
                    sha256_file(path)
                    if path.exists()
                    else "",
                "size_bytes":
                    path.stat().st_size
                    if path.exists()
                    else "",
            }
        )

    provenance_df = pd.DataFrame(
        provenance_rows
    )

    provenance_df.to_csv(
        OUTPUT_INPUT_PROVENANCE,
        sep="\t",
        index=False,
    )

    # =========================================================================
    # 20. FINAL MANIFEST
    # =========================================================================

    print_header("20. WRITING MANIFEST")

    qc_df = qc.dataframe()

    qc_df.to_csv(
        OUTPUT_QC,
        sep="\t",
        index=False,
    )

    failed_qc = qc_df[
        qc_df["status"] == "FAIL"
    ]

    final_status = (
        "PASS"
        if len(failed_qc) == 0
        else "FAIL"
    )

    manifest.update(
        {
            "finish_time": now_iso(),
            "final_status": final_status,
            "reference_rows": len(reference),
            "standardized_rows": len(standardized),
            "reference_dimension": 40,
            "standardized_dimension": 40,
            "standardized_metadata_columns": metadata_columns,
            "split_metadata_present": (
                "split" in standardized.columns
            ),
            "split_metadata_used_as_feature": False,
            "validation_molecules": len(validation_df),
            "neighbor_recovery_rows": len(neighbor_df),
            "high_similarity_target_rows":
                len(target_recovery_df),
            "possible_offtarget_rows":
                len(offtarget_df),
            "mean_known_target_recall":
                mean_target_recall,
            "median_best_neighbor_similarity":
                median_best_similarity,
            "fraction_with_high_similarity_neighbor":
                fraction_with_high_neighbor,
            "fraction_with_target_bearing_neighbor":
                fraction_with_target_neighbor,
            "target_annotation_columns":
                available_annotation_columns,
            "target_annotation_source":
                str(
                    TARGET_ANNOTATIONS_FILE.relative_to(PULP)
                ),
            "association_source":
                str(
                    ANNOTATED_ASSOCIATIONS_FILE.relative_to(PULP)
                ),
            "similarity_method":
                "cosine similarity in locked 40D standardized space",
            "offtarget_definition":
                "annotated target of a reference molecule with "
                "similarity >= 0.70 but lower than the strongest "
                "high-similarity relevant target evidence and not "
                "already a known target of the query molecule",
            "biological_annotation_policy":
                "Only actual CMAUP target annotations are used; "
                "unresolved target IDs are excluded from biological "
                "annotation inference.",
            "model_training_performed":
                False,
            "morgan_fingerprints_used":
                False,
        }
    )

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
            allow_nan=False,
        )

    # =========================================================================
    # 21. HUMAN-READABLE QC
    # =========================================================================

    print_header("21. WRITING HUMAN-READABLE QC LOG")

    with OUTPUT_QC_TXT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "==============================================================================\n"
        )
        handle.write(
            "SCRIPT 06 — VALIDATE 40D TARGET / POSSIBLE OFF-TARGET RECOVERY\n"
        )
        handle.write(
            "==============================================================================\n\n"
        )

        handle.write(
            f"Start time : {start_time}\n"
        )

        handle.write(
            f"Finish time: {now_iso()}\n\n"
        )

        handle.write(
            "40D FEATURE SPACE\n"
        )

        handle.write(
            "-----------------\n"
        )

        handle.write(
            f"Molecular features : {len(MOLECULAR_FEATURES)}\n"
        )

        handle.write(
            f"3D-QSAR features   : {len(THREE_D_FEATURES)}\n"
        )

        handle.write(
            f"MoLFormer features : {len(MOLFORMER_FEATURES)}\n"
        )

        handle.write(
            f"Total features     : {len(FEATURES_40D)}\n\n"
        )

        handle.write(
            "IMPORTANT SCRIPT-05 COMPATIBILITY RULE\n"
        )

        handle.write(
            "--------------------------------------\n"
        )

        handle.write(
            "The Script-05 standardized matrix may contain a 'split'\n"
        )

        handle.write(
            "column. This is metadata and is NOT part of the 40D\n"
        )

        handle.write(
            "feature representation.\n\n"
        )

        handle.write(
            "VALIDATION\n"
        )

        handle.write(
            "----------\n"
        )

        handle.write(
            f"Reference molecules : {len(reference)}\n"
        )

        handle.write(
            f"Validation molecules: {len(validation_df)}\n"
        )

        handle.write(
            f"Neighbor rows       : {len(neighbor_df)}\n"
        )

        handle.write(
            f"High-similarity target rows: "
            f"{len(target_recovery_df)}\n"
        )

        handle.write(
            f"Possible off-target rows: "
            f"{len(offtarget_df)}\n\n"
        )

        handle.write(
            f"Mean known-target recall: "
            f"{mean_target_recall}\n"
        )

        handle.write(
            f"Median best similarity: "
            f"{median_best_similarity}\n"
        )

        handle.write(
            f"Fraction with high-similarity neighbor: "
            f"{fraction_with_high_neighbor}\n"
        )

        handle.write(
            f"Fraction with target-bearing neighbor: "
            f"{fraction_with_target_neighbor}\n\n"
        )

        handle.write(
            "BIOLOGICAL INTERPRETATION POLICY\n"
        )

        handle.write(
            "--------------------------------\n"
        )

        handle.write(
            "40D similarity identifies chemically/representation-wise\n"
        )

        handle.write(
            "similar known molecules. Target annotations are inherited\n"
        )

        handle.write(
            "only from actual CMAUP annotated target evidence.\n\n"
        )

        handle.write(
            "Possible off-targets are similarity-based hypotheses,\n"
        )

        handle.write(
            "not experimentally confirmed off-target interactions.\n\n"
        )

        handle.write(
            "No Morgan fingerprints are used.\n"
        )

        handle.write(
            "No new feature generation is performed.\n"
        )

        handle.write(
            "No new imputation is performed.\n"
        )

        handle.write(
            "No target annotations are fabricated.\n\n"
        )

        handle.write(
            "FINAL STATUS\n"
        )

        handle.write(
            "------------\n"
        )

        handle.write(
            final_status + "\n"
        )

    # =========================================================================
    # 22. OUTPUT INVENTORY
    # =========================================================================

    print_header("22. OUTPUT INVENTORY")

    output_files = [
        OUTPUT_VALIDATION_SUMMARY,
        OUTPUT_NEIGHBOR_RECOVERY,
        OUTPUT_TARGET_RECOVERY,
        OUTPUT_OFFTARGET_RECOVERY,
        OUTPUT_MOLECULE_SUMMARY,
        OUTPUT_THRESHOLD_SUMMARY,
        OUTPUT_FEATURE_ALIGNMENT,
        OUTPUT_TARGET_QC,
        OUTPUT_INPUT_PROVENANCE,
        OUTPUT_QC,
        OUTPUT_QC_TXT,
        OUTPUT_MANIFEST,
    ]

    for path in output_files:

        if path.exists():

            print(
                f"{path.name:<50} "
                f"{path.stat().st_size:>12,d} bytes"
            )

    # =========================================================================
    # 23. FINAL STATUS
    # =========================================================================

    print_header("FINAL STATUS")

    print_kv(
        "Status",
        final_status,
    )

    print_kv(
        "CMAUP 40D reference molecules",
        len(reference),
    )

    print_kv(
        "Locked feature dimension",
        40,
    )

    print_kv(
        "Molecular features",
        10,
    )

    print_kv(
        "3D-QSAR features",
        9,
    )

    print_kv(
        "MoLFormer features",
        21,
    )

    print_kv(
        "Validation molecules",
        len(validation_df),
    )

    print_kv(
        "High-similarity target evidence rows",
        len(target_recovery_df),
    )

    print_kv(
        "Possible off-target evidence rows",
        len(offtarget_df),
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "The 'split' column in Script-05 is metadata only and is NOT"
    )
    print(
        "counted as a 40D feature."
    )
    print()
    print(
        "Target annotations come only from Script-03 annotated CMAUP"
    )
    print(
        "target evidence."
    )
    print()
    print(
        "Possible off-targets are similarity-based candidates and"
    )
    print(
        "must not be described as experimentally confirmed targets."
    )
    print()
    print(
        f"QC report : {OUTPUT_QC_TXT}"
    )
    print(
        f"Manifest   : {OUTPUT_MANIFEST}"
    )

    return 0 if final_status == "PASS" else 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        exit_code = main()

        sys.exit(exit_code)

    except Exception as exc:

        print_header("FATAL ERROR")

        print(
            repr(exc)
        )

        traceback.print_exc()

        # Write a failure manifest so the failed run is explicitly recorded.
        failure_manifest = {
            "script":
                "06_validate_40d_target_offtarget_recovery.py",
            "version":
                "corrected_split_metadata_handling",
            "final_status":
                "FAIL",
            "failure_time":
                now_iso(),
            "error_type":
                type(exc).__name__,
            "error":
                str(exc),
            "feature_dimension":
                40,
            "feature_order":
                FEATURES_40D,
            "split_metadata_policy":
                "The Script-05 split column is metadata and is excluded "
                "from the 40D feature space.",
            "traceback":
                traceback.format_exc(),
        }

        failure_file = (
            OUTPUT_DIR
            / "06_validation_FAILURE.json"
        )

        with failure_file.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                failure_manifest,
                handle,
                indent=2,
            )

        sys.exit(1)
