#!/usr/bin/env python3
"""
===============================================================================
SCRIPT 04 — CMAUP 40D REFERENCE REPRESENTATION
CORRECTED VERSION

Purpose
-------
Build the exact 40-dimensional Model-1 representation for the 60,190 molecules
in the authoritative Model-1 train/test universe.

Critical correction
-------------------
The raw molecular-descriptor TSVs contain missing values. Those raw NaNs are
EXPECTED at the descriptor-generation stage and must NOT be treated as a
fatal QC failure.

Model-1 preprocessing already performed TRAIN-ONLY median imputation after
the 20% missingness screen. The authoritative post-preprocessing artifact is:

    feature_engineering/final_228_features/
        train_80_final_228_features.tsv
        test_20_final_228_features.tsv
        feature_list.txt
        removed_features.txt
        feature_removal_log.tsv
        preprocessing_statistics.tsv
        qc_summary.txt

This script therefore obtains the 10 molecular + 9 3D features from those
already-preprocessed Model-1 final-feature tables. It does NOT invent a new
imputation rule.

The raw molecular/3D descriptor files are still loaded and audited for
row/ID/source provenance.

MoLFormer
---------
The exact Model-1 768D NPY arrays are used. The 21 selected dimensions are
selected by the exact manifest names, e.g. molformer_0279 -> NPY column 279.

Excluded records
----------------
CMAUP has 60,222 records; the authoritative Model-1 train/test union has
60,190 records. The 32 records outside that union receive NO feature vector.

Bioactivity model
-----------------
The calibrated Model-1 model is optional. If it is absent, the script does
NOT invent probabilities. The reference table records NaN probability and
MODEL_FILE_NOT_FOUND status. If the model exists and can be validated, it is
used to score the exact 40D matrix.

No new model is trained.
No new descriptors are generated.
No excluded molecule is imputed or fabricated.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. PATHS
# =============================================================================

PULP_DIR = Path(__file__).resolve().parent

OUT_DIR = (
    PULP_DIR
    / "target_model"
    / "feature_space"
    / "04_cmaup_40d_reference"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.now().astimezone()

CMAUP_MOLECULES = (
    PULP_DIR / "target_model" / "data" / "01_cmaup_audit"
    / "cmaup_molecules_clean.tsv"
)

TRAIN_SPLIT = PULP_DIR / "feature_engineering" / "split" / "train_80.tsv"
TEST_SPLIT = PULP_DIR / "feature_engineering" / "split" / "test_20.tsv"

TRAIN_MOL_DESC = (
    PULP_DIR / "feature_engineering" / "molecular_descriptors"
    / "train_80_molecular_descriptors.tsv"
)
TEST_MOL_DESC = (
    PULP_DIR / "feature_engineering" / "molecular_descriptors"
    / "test_20_molecular_descriptors.tsv"
)

TRAIN_3D = (
    PULP_DIR / "feature_engineering" / "3d_qsar_descriptors"
    / "train_80_3d_qsar_descriptors.tsv"
)
TEST_3D = (
    PULP_DIR / "feature_engineering" / "3d_qsar_descriptors"
    / "test_20_3d_qsar_descriptors.tsv"
)

# Authoritative Model-1 post-preprocessing artifact.
FINAL_FEATURE_DIR = PULP_DIR / "feature_engineering" / "final_228_features"

TRAIN_FINAL_FEATURES = (
    FINAL_FEATURE_DIR / "train_80_final_228_features.tsv"
)
TEST_FINAL_FEATURES = (
    FINAL_FEATURE_DIR / "test_20_final_228_features.tsv"
)
FINAL_FEATURE_LIST = FINAL_FEATURE_DIR / "feature_list.txt"
FINAL_REMOVED_FEATURES = FINAL_FEATURE_DIR / "removed_features.txt"
FINAL_REMOVAL_LOG = FINAL_FEATURE_DIR / "feature_removal_log.tsv"
FINAL_PREPROCESSING_STATS = (
    FINAL_FEATURE_DIR / "preprocessing_statistics.tsv"
)
FINAL_QC_SUMMARY = FINAL_FEATURE_DIR / "qc_summary.txt"

TRAIN_MOLFORMER = (
    PULP_DIR / "feature_engineering" / "molformer_embeddings"
    / "train_80_molformer_embeddings.npy"
)
TEST_MOLFORMER = (
    PULP_DIR / "feature_engineering" / "molformer_embeddings"
    / "test_20_molformer_embeddings.npy"
)

FEATURE_MANIFEST = (
    PULP_DIR / "feature_engineering" / "model_development" / "tables"
    / "table_03_final_40_feature_manifest.tsv"
)

MODEL_PATH = (
    PULP_DIR / "models" / "final_calibrated_bioactivity_model.joblib"
)

# Known Model-1 invalid-SMILES/QC artifacts.
INVALID_SMILES_ARTIFACTS = [
    PULP_DIR / "feature_engineering" / "invalid_smiles.tsv",
    PULP_DIR / "target_model" / "data" / "01_cmaup_audit"
    / "smiles_qc.tsv",
]

EXPECTED_CMAUP = 60222
EXPECTED_TRAIN = 48152
EXPECTED_TEST = 12038
EXPECTED_REFERENCE = 60190
EXPECTED_EXCLUDED = 32
EXPECTED_MOLFORMER_DIM = 768


# =============================================================================
# Exact validated 40-feature identities
# =============================================================================

EXPECTED_MOLECULAR = [
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

EXPECTED_3D = [
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

EXPECTED_MOLFORMER = [
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

EXPECTED_40 = EXPECTED_MOLECULAR + EXPECTED_3D + EXPECTED_MOLFORMER


# =============================================================================
# Utility functions
# =============================================================================

def log(message: str = "") -> None:
    print(message, flush=True)


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PULP_DIR))
    except ValueError:
        return str(path)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"REQUIRED INPUT MISSING [{label}]: {path}"
        )
    if not path.is_file():
        raise RuntimeError(
            f"REQUIRED INPUT IS NOT A FILE [{label}]: {path}"
        )


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def safe(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            v = float(v)
            return None if not math.isfinite(v) else v
        if isinstance(v, float):
            return None if not math.isfinite(v) else v
        if isinstance(v, Path):
            return str(v)
        return v

    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe(obj), handle, indent=2, sort_keys=True)


def find_column(
    columns: List[str],
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:
    exact = {str(c): str(c) for c in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]

    lowered = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not identify required column. "
            f"Candidates={candidates}; available={columns}"
        )
    return None


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def ensure_unique_ids(
    df: pd.DataFrame,
    id_col: str,
    table_name: str,
) -> None:
    if id_col not in df.columns:
        raise KeyError(
            f"{table_name}: missing ID column '{id_col}'."
        )

    duplicated = df[id_col].duplicated(keep=False)
    if duplicated.any():
        ids = (
            df.loc[duplicated, id_col]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise RuntimeError(
            f"{table_name}: duplicate IDs detected. "
            f"Count={len(ids)}; examples={ids[:10]}"
        )


def numeric_qc(
    df: pd.DataFrame,
    features: List[str],
    table_name: str,
) -> Dict[str, Any]:
    values = df[features].apply(pd.to_numeric, errors="coerce")
    arr = values.to_numpy(dtype=np.float64)

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    return {
        "table": table_name,
        "feature_count": len(features),
        "NaN_count": nan_count,
        "Inf_count": inf_count,
        "PASS": bool(nan_count == 0 and inf_count == 0),
    }


def load_tsv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_split(path: Path, label: str) -> pd.DataFrame:
    df = load_tsv(path, label)
    id_col = find_column(
        list(df.columns),
        ["np_id", "NP_ID", "Ingredient_ID", "ingredient_id"],
    )
    if id_col != "np_id":
        df = df.rename(columns={id_col: "np_id"})

    df["np_id"] = normalize_ids(df["np_id"])
    ensure_unique_ids(df, "np_id", label)

    if df["np_id"].isin(["", "nan", "None"]).any():
        raise RuntimeError(f"{label}: missing/invalid np_id values.")

    return df


# =============================================================================
# Manifest
# =============================================================================

def parse_feature_manifest() -> Tuple[pd.DataFrame, List[str]]:
    manifest = load_tsv(FEATURE_MANIFEST, "final 40-feature manifest")

    feature_col = find_column(
        list(manifest.columns),
        [
            "feature",
            "Feature",
            "feature_name",
            "Feature_Name",
            "name",
            "Name",
        ],
    )

    manifest[feature_col] = (
        manifest[feature_col].astype(str).str.strip()
    )
    manifest = manifest.loc[manifest[feature_col].ne("")].copy()

    if manifest[feature_col].duplicated().any():
        dup = (
            manifest.loc[
                manifest[feature_col].duplicated(keep=False),
                feature_col,
            ].tolist()
        )
        raise RuntimeError(
            f"Manifest contains duplicate features: {dup}"
        )

    order_col = find_column(
        list(manifest.columns),
        [
            "order",
            "Order",
            "feature_order",
            "Feature_Order",
            "rank",
            "Rank",
            "position",
            "Position",
        ],
        required=False,
    )

    if order_col is not None:
        order_numeric = pd.to_numeric(
            manifest[order_col], errors="coerce"
        )
        if order_numeric.notna().all():
            manifest = (
                manifest.assign(__order=order_numeric)
                .sort_values("__order", kind="stable")
                .drop(columns="__order")
                .reset_index(drop=True)
            )

    features = manifest[feature_col].tolist()

    if len(features) != 40:
        raise RuntimeError(
            f"Final manifest must contain exactly 40 features; "
            f"observed {len(features)}."
        )

    if set(features) != set(EXPECTED_40):
        missing = sorted(set(EXPECTED_40) - set(features))
        unexpected = sorted(set(features) - set(EXPECTED_40))
        raise RuntimeError(
            "Final 40-feature manifest does not match the validated "
            "Model-1 feature identities.\n"
            f"Missing={missing}\nUnexpected={unexpected}"
        )

    write_tsv(
        manifest,
        OUT_DIR / "validated_40_feature_manifest.tsv",
    )

    return manifest, features


# =============================================================================
# Model-1 post-preprocessing artifact
# =============================================================================

def validate_final_preprocessed_feature_artifacts(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load the actual Model-1 final 182-feature matrices.

    This is the critical correction: raw descriptor NaNs are NOT imputed here.
    We reuse the exact post-preprocessing Model-1 artifact, whose logged
    preprocessing included TRAIN-ONLY median imputation.
    """

    require_file(
        TRAIN_FINAL_FEATURES,
        "Model-1 final preprocessed TRAIN feature table",
    )
    require_file(
        TEST_FINAL_FEATURES,
        "Model-1 final preprocessed TEST feature table",
    )

    train_final = pd.read_csv(
        TRAIN_FINAL_FEATURES, sep="\t", low_memory=False
    )
    test_final = pd.read_csv(
        TEST_FINAL_FEATURES, sep="\t", low_memory=False
    )

    train_id_col = find_column(
        list(train_final.columns),
        ["np_id", "NP_ID", "Ingredient_ID", "ingredient_id"],
    )
    test_id_col = find_column(
        list(test_final.columns),
        ["np_id", "NP_ID", "Ingredient_ID", "ingredient_id"],
    )

    if train_id_col != "np_id":
        train_final = train_final.rename(
            columns={train_id_col: "np_id"}
        )
    if test_id_col != "np_id":
        test_final = test_final.rename(
            columns={test_id_col: "np_id"}
        )

    train_final["np_id"] = normalize_ids(train_final["np_id"])
    test_final["np_id"] = normalize_ids(test_final["np_id"])

    ensure_unique_ids(
        train_final, "np_id",
        "Model-1 final preprocessed TRAIN feature table",
    )
    ensure_unique_ids(
        test_final, "np_id",
        "Model-1 final preprocessed TEST feature table",
    )

    expected_train_ids = train["np_id"].tolist()
    expected_test_ids = test["np_id"].tolist()

    if set(train_final["np_id"]) != set(expected_train_ids):
        raise RuntimeError(
            "Final preprocessed TRAIN feature ID set does not match "
            "train_80.tsv."
        )

    if set(test_final["np_id"]) != set(expected_test_ids):
        raise RuntimeError(
            "Final preprocessed TEST feature ID set does not match "
            "test_20.tsv."
        )

    train_final = (
        train_final.set_index("np_id", drop=False)
        .loc[expected_train_ids]
        .reset_index(drop=True)
    )
    test_final = (
        test_final.set_index("np_id", drop=False)
        .loc[expected_test_ids]
        .reset_index(drop=True)
    )

    required_19 = EXPECTED_MOLECULAR + EXPECTED_3D

    missing_train = [
        f for f in required_19 if f not in train_final.columns
    ]
    missing_test = [
        f for f in required_19 if f not in test_final.columns
    ]

    if missing_train or missing_test:
        raise RuntimeError(
            "The Model-1 final preprocessed feature artifact does not "
            "contain all selected molecular/3D features.\n"
            f"TRAIN missing={missing_train}\n"
            f"TEST missing={missing_test}"
        )

    train_selected = train_final[required_19].copy()
    test_selected = test_final[required_19].copy()

    for f in required_19:
        train_selected[f] = pd.to_numeric(
            train_selected[f], errors="coerce"
        )
        test_selected[f] = pd.to_numeric(
            test_selected[f], errors="coerce"
        )

    train_qc = numeric_qc(
        train_selected, required_19,
        "TRAIN Model-1 post-preprocessed selected molecular+3D",
    )
    test_qc = numeric_qc(
        test_selected, required_19,
        "TEST Model-1 post-preprocessed selected molecular+3D",
    )

    if not train_qc["PASS"] or not test_qc["PASS"]:
        raise RuntimeError(
            "Model-1 post-preprocessed selected molecular/3D features "
            "contain NaN/Inf. The existing Model-1 final artifact is "
            "therefore not usable for exact 40D reconstruction."
        )

    # Preserve the actual final feature artifact for provenance.
    write_tsv(
        pd.DataFrame([train_qc, test_qc]),
        OUT_DIR / "preprocessed_19_feature_qc.tsv",
    )

    preprocessing_artifacts = [
        FINAL_FEATURE_LIST,
        FINAL_REMOVED_FEATURES,
        FINAL_REMOVAL_LOG,
        FINAL_PREPROCESSING_STATS,
        FINAL_QC_SUMMARY,
        TRAIN_FINAL_FEATURES,
        TEST_FINAL_FEATURES,
    ]

    artifact_rows = []
    for path in preprocessing_artifacts:
        require_file(path, "Model-1 preprocessing artifact")
        artifact_rows.append(
            {
                "artifact": relpath(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    write_tsv(
        pd.DataFrame(artifact_rows),
        OUT_DIR / "model1_preprocessing_artifacts.tsv",
    )

    info = {
        "preprocessing_source": relpath(FINAL_FEATURE_DIR),
        "preprocessing_method": (
            "REUSED ACTUAL MODEL-1 POST-PREPROCESSING FEATURE TABLES"
        ),
        "raw_descriptor_nan_handling": (
            "raw NaNs are allowed; they are not fatal at this stage"
        ),
        "model1_logged_method": (
            "TRAIN-ONLY MEDIAN IMPUTATION after 20% missingness screen"
        ),
        "new_imputation_performed_by_script": False,
        "train_final_rows": len(train_final),
        "test_final_rows": len(test_final),
        "selected_features": required_19,
    }

    return train_selected, test_selected, info


# =============================================================================
# Raw descriptor provenance/QC
# =============================================================================

def audit_raw_descriptor_table(
    path: Path,
    split_df: pd.DataFrame,
    selected_features: List[str],
    label: str,
) -> Dict[str, Any]:
    df = load_tsv(path, label)

    id_col = find_column(
        list(df.columns),
        ["np_id", "NP_ID", "Ingredient_ID", "ingredient_id"],
    )
    if id_col != "np_id":
        df = df.rename(columns={id_col: "np_id"})

    df["np_id"] = normalize_ids(df["np_id"])
    ensure_unique_ids(df, "np_id", label)

    expected = split_df["np_id"].tolist()

    if set(df["np_id"]) != set(expected):
        missing = sorted(set(expected) - set(df["np_id"]))
        extra = sorted(set(df["np_id"]) - set(expected))
        raise RuntimeError(
            f"{label}: ID set mismatch. "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    aligned = (
        df.set_index("np_id", drop=False)
        .loc[expected]
        .reset_index(drop=True)
    )

    missing_features = [
        f for f in selected_features if f not in aligned.columns
    ]
    if missing_features:
        raise RuntimeError(
            f"{label}: selected features missing: {missing_features}"
        )

    selected = aligned[selected_features].apply(
        pd.to_numeric, errors="coerce"
    )
    arr = selected.to_numpy(dtype=np.float64)

    return {
        "label": label,
        "rows": len(aligned),
        "columns": len(aligned.columns),
        "selected_feature_count": len(selected_features),
        "raw_selected_NaN_count": int(np.isnan(arr).sum()),
        "raw_selected_Inf_count": int(np.isinf(arr).sum()),
        "raw_descriptor_qc": "PASS",
        "raw_NaN_is_fatal": False,
    }


# =============================================================================
# MoLFormer
# =============================================================================

def load_molformer(
    path: Path,
    expected_rows: int,
    label: str,
) -> np.ndarray:
    require_file(path, label)

    arr = np.load(path, allow_pickle=False)

    if arr.ndim != 2:
        raise RuntimeError(
            f"{label}: expected 2D array; observed {arr.shape}"
        )

    if arr.shape != (expected_rows, EXPECTED_MOLFORMER_DIM):
        raise RuntimeError(
            f"{label}: expected shape "
            f"({expected_rows}, {EXPECTED_MOLFORMER_DIM}); "
            f"observed {arr.shape}"
        )

    if not np.issubdtype(arr.dtype, np.number):
        raise RuntimeError(
            f"{label}: non-numeric dtype {arr.dtype}"
        )

    arr = arr.astype(np.float64, copy=False)

    return arr


def molformer_index(feature: str) -> int:
    m = re.fullmatch(r"molformer_(\d+)", feature, re.IGNORECASE)
    if m is None:
        raise ValueError(
            f"Cannot parse MoLFormer feature index: {feature}"
        )

    idx = int(m.group(1))
    if not 0 <= idx < EXPECTED_MOLFORMER_DIM:
        raise ValueError(
            f"MoLFormer feature index out of range: {feature} -> {idx}"
        )
    return idx


# =============================================================================
# Excluded-record cross-check
# =============================================================================

def exact_id_matches_in_artifact(
    artifact: Path,
    excluded_ids: set[str],
) -> set[str]:
    if not artifact.exists():
        return set()

    matched: set[str] = set()

    try:
        if artifact.suffix.lower() in {".tsv", ".csv", ".txt"}:
            sep = "\t" if artifact.suffix.lower() in {".tsv", ".txt"} else ","

            try:
                df = pd.read_csv(
                    artifact,
                    sep=sep,
                    dtype=str,
                    low_memory=False,
                )

                id_cols = [
                    c for c in df.columns
                    if str(c).strip().lower()
                    in {
                        "np_id",
                        "ingredient_id",
                        "molecule_id",
                        "id",
                    }
                ]

                for col in id_cols:
                    vals = (
                        df[col].dropna().astype(str).str.strip()
                    )
                    matched.update(set(vals).intersection(excluded_ids))

            except Exception:
                pass

            # Conservative exact token matching for text artifacts.
            try:
                text = artifact.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
                for value in excluded_ids:
                    if re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(value)}"
                        rf"(?![A-Za-z0-9_])",
                        text,
                    ):
                        matched.add(value)
            except Exception:
                pass

    except Exception:
        pass

    return matched


# =============================================================================
# Optional model
# =============================================================================

def recursively_find_predictor(
    obj: Any,
    depth: int = 0,
    max_depth: int = 6,
) -> Optional[Any]:
    if obj is None or depth > max_depth:
        return None

    try:
        if hasattr(obj, "predict_proba"):
            return obj
    except Exception:
        pass

    if isinstance(obj, dict):
        for value in obj.values():
            found = recursively_find_predictor(
                value, depth + 1, max_depth
            )
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = recursively_find_predictor(
                value, depth + 1, max_depth
            )
            if found is not None:
                return found
        return None

    for attr in [
        "estimator",
        "base_estimator",
        "classifier",
        "model",
        "calibrated_classifiers_",
        "classifier_",
    ]:
        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        found = recursively_find_predictor(
            value, depth + 1, max_depth
        )
        if found is not None:
            return found

    return None


def recursively_find_feature_names(
    obj: Any,
    depth: int = 0,
    max_depth: int = 6,
) -> Optional[List[str]]:
    if obj is None or depth > max_depth:
        return None

    for attr in [
        "feature_names_in_",
        "feature_names",
        "features",
        "predictor_names",
    ]:
        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        if value is None:
            continue

        try:
            values = list(value)
            if all(isinstance(x, str) for x in values):
                return values
        except Exception:
            pass

    if isinstance(obj, dict):
        for key in [
            "feature_names",
            "feature_names_in_",
            "features",
            "predictor_names",
        ]:
            if key in obj:
                try:
                    values = list(obj[key])
                    if all(isinstance(x, str) for x in values):
                        return values
                except Exception:
                    pass

        for value in obj.values():
            found = recursively_find_feature_names(
                value, depth + 1, max_depth
            )
            if found is not None:
                return found

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = recursively_find_feature_names(
                value, depth + 1, max_depth
            )
            if found is not None:
                return found

    else:
        for attr in [
            "estimator",
            "base_estimator",
            "classifier",
            "model",
            "calibrated_classifiers_",
            "classifier_",
        ]:
            try:
                value = getattr(obj, attr)
            except Exception:
                continue

            found = recursively_find_feature_names(
                value, depth + 1, max_depth
            )
            if found is not None:
                return found

    return None


def score_optional_model(
    X: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "model_path": relpath(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": False,
        "prediction_status": "NOT_ATTEMPTED",
    }

    if not MODEL_PATH.exists():
        info["prediction_status"] = "MODEL_FILE_NOT_FOUND"
        return None, info

    info["model_sha256"] = sha256_file(MODEL_PATH)

    try:
        import joblib
    except ImportError as exc:
        info["prediction_status"] = "JOBLIB_IMPORT_FAILED"
        info["error"] = repr(exc)
        return None, info

    try:
        model = joblib.load(MODEL_PATH)
        info["model_loaded"] = True
        info["loaded_object_type"] = type(model).__name__

        predictor = recursively_find_predictor(model)
        if predictor is None:
            raise RuntimeError(
                "No predict_proba-capable estimator found."
            )

        feature_names = recursively_find_feature_names(model)

        if feature_names is not None:
            info["model_feature_names_found"] = True
            info["model_feature_count"] = len(feature_names)

            if len(feature_names) != 40:
                raise RuntimeError(
                    f"Model reports {len(feature_names)} features, not 40."
                )

            if set(feature_names) != set(X.columns):
                raise RuntimeError(
                    "Deployment model feature names do not match "
                    "the exact 40-feature reference space."
                )

            X_model = X.loc[:, feature_names]
        else:
            info["model_feature_names_found"] = False
            n_features = getattr(
                predictor, "n_features_in_", None
            )
            if n_features is None:
                n_features = getattr(
                    model, "n_features_in_", None
                )

            if n_features is not None:
                info["model_feature_count"] = int(n_features)
                if int(n_features) != 40:
                    raise RuntimeError(
                        f"Model reports {n_features} input features, not 40."
                    )

            X_model = X

        proba = np.asarray(
            predictor.predict_proba(X_model),
            dtype=np.float64,
        )

        if proba.ndim != 2 or proba.shape[0] != len(X):
            raise RuntimeError(
                f"Invalid predict_proba shape: {proba.shape}"
            )

        classes = getattr(predictor, "classes_", None)
        if classes is None:
            classes = getattr(model, "classes_", None)
        if classes is None:
            classes = np.array([0, 1])

        classes = np.asarray(classes)

        active_idx = np.where(classes == 1)[0]
        if len(active_idx) != 1:
            raise RuntimeError(
                f"Could not identify exactly one Active class=1; "
                f"classes={classes.tolist()}"
            )

        active_idx = int(active_idx[0])
        active_probability = proba[:, active_idx]

        if not np.isfinite(active_probability).all():
            raise RuntimeError(
                "Deployment model returned NaN/Inf probabilities."
            )

        info["classes"] = classes.tolist()
        info["active_probability_column"] = active_idx
        info["prediction_status"] = "PASS"

        return active_probability, info

    except Exception as exc:
        info["prediction_status"] = "PREDICTION_FAILED"
        info["error"] = repr(exc)
        return None, info


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    section("SCRIPT 04 — CMAUP 40D REFERENCE REPRESENTATION — CORRECTED")

    log(f"PULP directory : {PULP_DIR}")
    log(f"Output directory : {OUT_DIR}")
    log(f"Start time : {START_TIME.isoformat()}")

    provenance: Dict[str, Any] = {
        "script": "04_build_cmaup_40d_reference.py",
        "version": "corrected_exact_model1_preprocessed_artifact",
        "start_time": START_TIME.isoformat(),
        "feature_space_dimension": 40,
        "model_training_performed": False,
        "new_features_generated": False,
        "excluded_records_feature_generated": 0,
        "raw_descriptor_nan_is_fatal": False,
        "preprocessing_rule": (
            "Reuse actual Model-1 post-preprocessing feature tables; "
            "do not invent a new imputation rule."
        ),
    }

    # -------------------------------------------------------------------------
    # 1. Required inputs
    # -------------------------------------------------------------------------

    section("1. INPUT FILE CHECK")

    required_inputs = {
        "cmaup_molecules_clean": CMAUP_MOLECULES,
        "train_split": TRAIN_SPLIT,
        "test_split": TEST_SPLIT,
        "train_raw_molecular_descriptors": TRAIN_MOL_DESC,
        "test_raw_molecular_descriptors": TEST_MOL_DESC,
        "train_raw_3d_descriptors": TRAIN_3D,
        "test_raw_3d_descriptors": TEST_3D,
        "train_model1_final_features": TRAIN_FINAL_FEATURES,
        "test_model1_final_features": TEST_FINAL_FEATURES,
        "model1_feature_list": FINAL_FEATURE_LIST,
        "model1_removed_features": FINAL_REMOVED_FEATURES,
        "model1_feature_removal_log": FINAL_REMOVAL_LOG,
        "model1_preprocessing_statistics": FINAL_PREPROCESSING_STATS,
        "model1_qc_summary": FINAL_QC_SUMMARY,
        "train_molformer_embeddings": TRAIN_MOLFORMER,
        "test_molformer_embeddings": TEST_MOLFORMER,
        "final_40_feature_manifest": FEATURE_MANIFEST,
    }

    provenance_rows = []

    for label, path in required_inputs.items():
        require_file(path, label)
        digest = sha256_file(path)

        log(f"FOUND: {relpath(path)}")
        log(f"SHA256: {digest}")

        provenance_rows.append(
            {
                "input_label": label,
                "path": relpath(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "required": True,
            }
        )

    if MODEL_PATH.exists():
        model_digest = sha256_file(MODEL_PATH)
        log(f"FOUND OPTIONAL MODEL: {relpath(MODEL_PATH)}")
        log(f"MODEL SHA256: {model_digest}")
        provenance_rows.append(
            {
                "input_label": "calibrated_bioactivity_model",
                "path": relpath(MODEL_PATH),
                "exists": True,
                "size_bytes": MODEL_PATH.stat().st_size,
                "sha256": model_digest,
                "required": False,
            }
        )
    else:
        log(f"OPTIONAL MODEL NOT FOUND: {MODEL_PATH}")
        provenance_rows.append(
            {
                "input_label": "calibrated_bioactivity_model",
                "path": relpath(MODEL_PATH),
                "exists": False,
                "size_bytes": None,
                "sha256": None,
                "required": False,
            }
        )

    write_tsv(
        pd.DataFrame(provenance_rows),
        OUT_DIR / "input_file_provenance.tsv",
    )
    provenance["input_files"] = provenance_rows

    # -------------------------------------------------------------------------
    # 2. Exact 40-feature manifest
    # -------------------------------------------------------------------------

    section("2. VALIDATING EXACT 40-FEATURE MANIFEST")

    manifest_df, manifest_features = parse_feature_manifest()

    if manifest_features != EXPECTED_40:
        log(
            "NOTE: manifest order differs from the fixed family order. "
            "The final matrix will follow the manifest order."
        )

    molecular_features = [
        f for f in manifest_features if f in EXPECTED_MOLECULAR
    ]
    three_d_features = [
        f for f in manifest_features if f in EXPECTED_3D
    ]
    molformer_features = [
        f for f in manifest_features if f in EXPECTED_MOLFORMER
    ]

    if len(molecular_features) != 10:
        raise RuntimeError(
            f"Expected 10 molecular features; observed "
            f"{len(molecular_features)}"
        )
    if len(three_d_features) != 9:
        raise RuntimeError(
            f"Expected 9 3D features; observed {len(three_d_features)}"
        )
    if len(molformer_features) != 21:
        raise RuntimeError(
            f"Expected 21 MoLFormer features; observed "
            f"{len(molformer_features)}"
        )

    family_rows = []
    for order, feature in enumerate(manifest_features, start=1):
        if feature in EXPECTED_MOLECULAR:
            family = "molecular"
            source = "Model-1 final preprocessed 228-feature table"
            index = None
        elif feature in EXPECTED_3D:
            family = "3D-QSAR"
            source = "Model-1 final preprocessed 228-feature table"
            index = None
        else:
            family = "MoLFormer"
            source = "Model-1 768D NPY"
            index = molformer_index(feature)

        family_rows.append(
            {
                "final_order": order,
                "feature": feature,
                "representation": family,
                "source": source,
                "embedding_index": index,
            }
        )

    feature_provenance = pd.DataFrame(family_rows)
    write_tsv(
        feature_provenance,
        OUT_DIR / "40_feature_provenance.tsv",
    )

    log(f"Molecular features : {len(molecular_features)}")
    log(f"3D-QSAR features   : {len(three_d_features)}")
    log(f"MoLFormer features : {len(molformer_features)}")
    log(f"Total features     : {len(manifest_features)}")

    # -------------------------------------------------------------------------
    # 3. CMAUP master table
    # -------------------------------------------------------------------------

    section("3. LOADING CMAUP MASTER MOLECULE TABLE")

    cmaup = load_tsv(
        CMAUP_MOLECULES,
        "CMAUP molecule table",
    )

    np_id_col = find_column(
        list(cmaup.columns),
        ["np_id", "NP_ID", "Ingredient_ID", "ingredient_id"],
    )
    if np_id_col != "np_id":
        cmaup = cmaup.rename(columns={np_id_col: "np_id"})

    cmaup["np_id"] = normalize_ids(cmaup["np_id"])
    ensure_unique_ids(cmaup, "np_id", "CMAUP molecule table")

    if len(cmaup) != EXPECTED_CMAUP:
        raise RuntimeError(
            f"CMAUP row count expected {EXPECTED_CMAUP}; "
            f"observed {len(cmaup)}"
        )

    if cmaup["np_id"].isin(["", "nan", "None"]).any():
        raise RuntimeError("CMAUP contains missing/invalid np_id values.")

    log(f"CMAUP rows : {len(cmaup):,}")
    log(f"CMAUP columns : {len(cmaup.columns)}")

    # -------------------------------------------------------------------------
    # 4. Authoritative split
    # -------------------------------------------------------------------------

    section("4. LOADING AUTHORITATIVE MODEL-1 TRAIN/TEST SPLITS")

    train = load_split(TRAIN_SPLIT, "Model-1 TRAIN split")
    test = load_split(TEST_SPLIT, "Model-1 TEST split")

    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"TRAIN row count expected {EXPECTED_TRAIN}; observed {len(train)}"
        )
    if len(test) != EXPECTED_TEST:
        raise RuntimeError(
            f"TEST row count expected {EXPECTED_TEST}; observed {len(test)}"
        )

    train_ids = set(train["np_id"])
    test_ids = set(test["np_id"])

    overlap = train_ids.intersection(test_ids)
    if overlap:
        raise RuntimeError(
            f"TRAIN/TEST overlap detected: {len(overlap)}"
        )

    split_union = train_ids.union(test_ids)

    if len(split_union) != EXPECTED_REFERENCE:
        raise RuntimeError(
            f"Split union expected {EXPECTED_REFERENCE}; "
            f"observed {len(split_union)}"
        )

    cmaup_ids = set(cmaup["np_id"])

    missing_from_cmaup = sorted(split_union - cmaup_ids)
    if missing_from_cmaup:
        raise RuntimeError(
            "Model-1 split contains IDs absent from CMAUP: "
            f"{len(missing_from_cmaup)}; examples={missing_from_cmaup[:10]}"
        )

    excluded_ids = sorted(cmaup_ids - split_union)

    if len(excluded_ids) != EXPECTED_EXCLUDED:
        raise RuntimeError(
            f"Expected {EXPECTED_EXCLUDED} excluded CMAUP records; "
            f"observed {len(excluded_ids)}"
        )

    log(f"TRAIN : {len(train):,}")
    log(f"TEST  : {len(test):,}")
    log(f"Union : {len(split_union):,}")
    log(f"Excluded CMAUP records : {len(excluded_ids):,}")
    log("Train/test separation : PASS")

    # -------------------------------------------------------------------------
    # 5. Exclusion table and exact cross-check
    # -------------------------------------------------------------------------

    section("5. EXCLUDED CMAUP RECORDS — NO FEATURE GENERATION")

    excluded = cmaup.loc[
        cmaup["np_id"].isin(excluded_ids)
    ].copy()

    excluded["exclusion_reason"] = (
        "OUTSIDE_AUTHORITATIVE_MODEL1_TRAIN_TEST_UNION"
    )
    excluded["reference_feature_status"] = "NOT_GENERATED"
    excluded["included_in_40d_reference"] = False
    excluded["feature_generation_performed"] = False

    crosscheck_rows = []
    confirmed = set()

    for artifact in INVALID_SMILES_ARTIFACTS:
        matched = exact_id_matches_in_artifact(
            artifact, set(excluded_ids)
        )

        if artifact.exists():
            crosscheck_rows.append(
                {
                    "artifact": relpath(artifact),
                    "exists": True,
                    "sha256": sha256_file(artifact),
                    "matched_excluded_ids": len(matched),
                    "matched_ids": ",".join(sorted(matched)),
                }
            )
            confirmed.update(matched)
        else:
            crosscheck_rows.append(
                {
                    "artifact": relpath(artifact),
                    "exists": False,
                    "sha256": None,
                    "matched_excluded_ids": 0,
                    "matched_ids": "",
                }
            )

    excluded["model1_invalid_qc_crosscheck"] = (
        excluded["np_id"].isin(confirmed)
    )

    excluded["exclusion_crosscheck_status"] = np.where(
        excluded["model1_invalid_qc_crosscheck"],
        "FOUND_IN_MODEL1_INVALID_QC_ARTIFACT",
        "NOT_CONFIRMED_BY_LISTED_INVALID_QC_ARTIFACTS",
    )

    write_tsv(
        excluded,
        OUT_DIR / "cmaup_reference_exclusions.tsv",
    )
    write_tsv(
        pd.DataFrame(crosscheck_rows),
        OUT_DIR / "excluded_record_qc_crosscheck.tsv",
    )

    log(
        "Excluded IDs confirmed by listed invalid-SMILES/QC artifacts: "
        f"{len(confirmed)}/{len(excluded_ids)}"
    )
    log(
        "IMPORTANT: exclusion itself is authoritative from the Model-1 "
        "split; invalid-QC matching is provenance only."
    )

    # -------------------------------------------------------------------------
    # 6. Raw descriptor audit
    # -------------------------------------------------------------------------

    section("6. RAW DESCRIPTOR AUDIT — NaNs ARE NOT FATAL")

    raw_mol_train_qc = audit_raw_descriptor_table(
        TRAIN_MOL_DESC,
        train,
        EXPECTED_MOLECULAR,
        "TRAIN raw molecular descriptors",
    )
    raw_mol_test_qc = audit_raw_descriptor_table(
        TEST_MOL_DESC,
        test,
        EXPECTED_MOLECULAR,
        "TEST raw molecular descriptors",
    )
    raw_3d_train_qc = audit_raw_descriptor_table(
        TRAIN_3D,
        train,
        EXPECTED_3D,
        "TRAIN raw 3D-QSAR descriptors",
    )
    raw_3d_test_qc = audit_raw_descriptor_table(
        TEST_3D,
        test,
        EXPECTED_3D,
        "TEST raw 3D-QSAR descriptors",
    )

    raw_qc_df = pd.DataFrame([
        raw_mol_train_qc,
        raw_mol_test_qc,
        raw_3d_train_qc,
        raw_3d_test_qc,
    ])
    write_tsv(
        raw_qc_df,
        OUT_DIR / "raw_descriptor_selected_feature_qc.tsv",
    )

    for row in raw_qc_df.itertuples(index=False):
        log(
            f"{row.label}: raw NaN={row.raw_selected_NaN_count}, "
            f"raw Inf={row.raw_selected_Inf_count}, "
            f"QC={row.raw_descriptor_qc}"
        )

    # -------------------------------------------------------------------------
    # 7. Exact Model-1 preprocessing reuse
    # -------------------------------------------------------------------------

    section(
        "7. REUSING EXACT MODEL-1 POST-PREPROCESSING "
        "FOR 19 MOLECULAR + 3D FEATURES"
    )

    train_19, test_19, preprocessing_info = (
        validate_final_preprocessed_feature_artifacts(
            train, test
        )
    )

    log(
        "Preprocessing source : "
        f"{preprocessing_info['preprocessing_source']}"
    )
    log(
        "New imputation performed by Script 04 : NO"
    )
    log(
        "Model-1 preprocessing : TRAIN-ONLY MEDIAN IMPUTATION"
    )
    log(
        "Post-preprocessing TRAIN selected 19 features : PASS"
    )
    log(
        "Post-preprocessing TEST selected 19 features : PASS"
    )

    # -------------------------------------------------------------------------
    # 8. MoLFormer
    # -------------------------------------------------------------------------

    section("8. LOADING MODEL-1 MOLFORMER EMBEDDINGS")

    train_mf = load_molformer(
        TRAIN_MOLFORMER,
        len(train),
        "TRAIN MoLFormer embeddings",
    )
    test_mf = load_molformer(
        TEST_MOLFORMER,
        len(test),
        "TEST MoLFormer embeddings",
    )

    train_mf_nan = int(np.isnan(train_mf).sum())
    train_mf_inf = int(np.isinf(train_mf).sum())
    test_mf_nan = int(np.isnan(test_mf).sum())
    test_mf_inf = int(np.isinf(test_mf).sum())

    if any([
        train_mf_nan,
        train_mf_inf,
        test_mf_nan,
        test_mf_inf,
    ]):
        raise RuntimeError(
            "NaN/Inf detected in Model-1 MoLFormer embeddings."
        )

    mf_indices = [
        molformer_index(f) for f in molformer_features
    ]

    train_mf_selected = pd.DataFrame(
        train_mf[:, mf_indices],
        columns=molformer_features,
    )
    test_mf_selected = pd.DataFrame(
        test_mf[:, mf_indices],
        columns=molformer_features,
    )

    mf_qc = pd.DataFrame([
        {
            "split": "TRAIN",
            "rows": len(train_mf),
            "embedding_dimension": train_mf.shape[1],
            "selected_dimensions": len(mf_indices),
            "NaN_count": train_mf_nan,
            "Inf_count": train_mf_inf,
            "PASS": train_mf_nan == 0 and train_mf_inf == 0,
        },
        {
            "split": "TEST",
            "rows": len(test_mf),
            "embedding_dimension": test_mf.shape[1],
            "selected_dimensions": len(mf_indices),
            "NaN_count": test_mf_nan,
            "Inf_count": test_mf_inf,
            "PASS": test_mf_nan == 0 and test_mf_inf == 0,
        },
    ])

    write_tsv(
        mf_qc,
        OUT_DIR / "molformer_alignment_qc.tsv",
    )

    mf_alignment = pd.DataFrame([
        {
            "feature": f,
            "npy_column_index_zero_based": molformer_index(f),
        }
        for f in molformer_features
    ])
    write_tsv(
        mf_alignment,
        OUT_DIR / "molformer_selected_dimension_manifest.tsv",
    )

    log(f"TRAIN MoLFormer shape : {train_mf.shape}")
    log(f"TEST MoLFormer shape  : {test_mf.shape}")
    log(f"Selected dimensions   : {len(mf_indices)}")

    # -------------------------------------------------------------------------
    # 9. Build exact 40D train/test matrices
    # -------------------------------------------------------------------------

    section("9. BUILDING EXACT MODEL-1 40D TRAIN/TEST MATRICES")

    train_parts = {
        **{f: train_19[f].to_numpy(dtype=np.float64)
           for f in EXPECTED_MOLECULAR + EXPECTED_3D},
        **{f: train_mf_selected[f].to_numpy(dtype=np.float64)
           for f in EXPECTED_MOLFORMER},
    }

    test_parts = {
        **{f: test_19[f].to_numpy(dtype=np.float64)
           for f in EXPECTED_MOLECULAR + EXPECTED_3D},
        **{f: test_mf_selected[f].to_numpy(dtype=np.float64)
           for f in EXPECTED_MOLFORMER},
    }

    train_40 = pd.DataFrame(
        {feature: train_parts[feature] for feature in manifest_features}
    )
    test_40 = pd.DataFrame(
        {feature: test_parts[feature] for feature in manifest_features}
    )

    if train_40.shape != (len(train), 40):
        raise RuntimeError(
            f"TRAIN 40D shape mismatch: {train_40.shape}"
        )
    if test_40.shape != (len(test), 40):
        raise RuntimeError(
            f"TEST 40D shape mismatch: {test_40.shape}"
        )

    train_40_qc = numeric_qc(
        train_40, manifest_features, "TRAIN 40D"
    )
    test_40_qc = numeric_qc(
        test_40, manifest_features, "TEST 40D"
    )

    if not train_40_qc["PASS"] or not test_40_qc["PASS"]:
        raise RuntimeError(
            "Final 40D train/test matrix contains NaN/Inf."
        )

    write_tsv(
        pd.DataFrame([train_40_qc, test_40_qc]),
        OUT_DIR / "train_test_40d_numeric_qc.tsv",
    )

    log(f"TRAIN 40D shape : {train_40.shape}")
    log(f"TEST 40D shape  : {test_40.shape}")
    log("40D train/test numerical QC : PASS")

    # -------------------------------------------------------------------------
    # 10. Build reference with SMILES
    # -------------------------------------------------------------------------

    section("10. BUILDING 60,190-MOLECULE CMAUP 40D REFERENCE")

    if "SMILES" not in cmaup.columns:
        raise RuntimeError(
            "CMAUP molecule table does not contain SMILES."
        )

    cmaup_smiles = (
        cmaup[["np_id", "SMILES"]]
        .drop_duplicates("np_id")
        .set_index("np_id")["SMILES"]
    )

    train_reference = pd.DataFrame({
        "np_id": train["np_id"].tolist(),
        "SMILES": train["np_id"].map(cmaup_smiles).tolist(),
    })
    test_reference = pd.DataFrame({
        "np_id": test["np_id"].tolist(),
        "SMILES": test["np_id"].map(cmaup_smiles).tolist(),
    })

    if train_reference["SMILES"].isna().any():
        raise RuntimeError("TRAIN reference has missing CMAUP SMILES.")
    if test_reference["SMILES"].isna().any():
        raise RuntimeError("TEST reference has missing CMAUP SMILES.")

    train_reference = pd.concat(
        [train_reference.reset_index(drop=True),
         train_40.reset_index(drop=True)],
        axis=1,
    )
    test_reference = pd.concat(
        [test_reference.reset_index(drop=True),
         test_40.reset_index(drop=True)],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # 11. Optional calibrated bioactivity probability
    # -------------------------------------------------------------------------

    section("11. OPTIONAL CALIBRATED BIOACTIVITY PROBABILITY")

    X_reference = pd.concat(
        [
            train_reference[manifest_features],
            test_reference[manifest_features],
        ],
        axis=0,
        ignore_index=True,
    )

    probabilities, model_info = score_optional_model(
        X_reference
    )

    if probabilities is None:
        train_reference["bioactivity_probability"] = np.nan
        test_reference["bioactivity_probability"] = np.nan
        log(
            "Bioactivity probability status : "
            f"{model_info['prediction_status']}"
        )
    else:
        train_reference["bioactivity_probability"] = (
            probabilities[:len(train_reference)]
        )
        test_reference["bioactivity_probability"] = (
            probabilities[len(train_reference):]
        )
        log("Bioactivity probability generation : PASS")

    write_json(
        OUT_DIR / "bioactivity_model_qc.json",
        model_info,
    )

    # -------------------------------------------------------------------------
    # 12. Combine
    # -------------------------------------------------------------------------

    section("12. COMBINING TRAIN + TEST REFERENCE")

    train_reference["Model1_split"] = "TRAIN"
    test_reference["Model1_split"] = "TEST"

    train_reference["Model1_reference_status"] = (
        "INCLUDED_VALID_REFERENCE"
    )
    test_reference["Model1_reference_status"] = (
        "INCLUDED_VALID_REFERENCE"
    )

    train_reference["reference_feature_source"] = (
        "ACTUAL_MODEL1_POSTPREPROCESSED_ARTIFACTS"
    )
    test_reference["reference_feature_source"] = (
        "ACTUAL_MODEL1_POSTPREPROCESSED_ARTIFACTS"
    )

    reference = pd.concat(
        [train_reference, test_reference],
        axis=0,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # 13. Exact reference coverage
    # -------------------------------------------------------------------------

    section("13. REFERENCE COVERAGE QC")

    reference_ids = set(reference["np_id"])

    if len(reference) != EXPECTED_REFERENCE:
        raise RuntimeError(
            f"Reference row count expected {EXPECTED_REFERENCE}; "
            f"observed {len(reference)}"
        )

    if len(reference_ids) != EXPECTED_REFERENCE:
        raise RuntimeError(
            "Reference contains duplicate np_id values."
        )

    if reference_ids != split_union:
        missing = sorted(split_union - reference_ids)
        extra = sorted(reference_ids - split_union)
        raise RuntimeError(
            "Final reference ID set mismatch.\n"
            f"Missing={len(missing)}\nExtra={len(extra)}"
        )

    if reference_ids.intersection(set(excluded_ids)):
        raise RuntimeError(
            "Excluded CMAUP records were included in the final reference."
        )

    log("Reference ID coverage : PASS")
    log(f"Reference molecule count : {len(reference):,}")
    log("Excluded 32 records feature-generated : NO")

    # -------------------------------------------------------------------------
    # 14. Final feature QC
    # -------------------------------------------------------------------------

    section("14. FINAL 40D FEATURE QC")

    final_qc_rows = [
        numeric_qc(
            train_reference,
            manifest_features,
            "TRAIN reference",
        ),
        numeric_qc(
            test_reference,
            manifest_features,
            "TEST reference",
        ),
        numeric_qc(
            reference,
            manifest_features,
            "ALL reference",
        ),
    ]

    final_qc = pd.DataFrame(final_qc_rows)

    if not final_qc["PASS"].all():
        raise RuntimeError(
            "Final 40D feature QC failed:\n"
            f"{final_qc.to_string(index=False)}"
        )

    write_tsv(
        final_qc,
        OUT_DIR / "final_40d_feature_qc.tsv",
    )

    if list(reference[manifest_features].columns) != manifest_features:
        raise RuntimeError(
            "Final 40D feature order does not match manifest."
        )

    if len(set(manifest_features)) != 40:
        raise RuntimeError(
            "Final feature names are not unique."
        )

    log("Final 40D NaN/Inf QC : PASS")
    log("Final 40D feature order : PASS")
    log("Final 40D feature identities : PASS")

    # -------------------------------------------------------------------------
    # 15. Feature family QC
    # -------------------------------------------------------------------------

    section("15. FEATURE FAMILY QC")

    family_qc = pd.DataFrame([
        {
            "representation": "molecular",
            "feature_count": len(molecular_features),
            "features": ",".join(molecular_features),
            "PASS": len(molecular_features) == 10,
        },
        {
            "representation": "3D-QSAR",
            "feature_count": len(three_d_features),
            "features": ",".join(three_d_features),
            "PASS": len(three_d_features) == 9,
        },
        {
            "representation": "MoLFormer",
            "feature_count": len(molformer_features),
            "features": ",".join(molformer_features),
            "PASS": len(molformer_features) == 21,
        },
        {
            "representation": "TOTAL",
            "feature_count": len(manifest_features),
            "features": ",".join(manifest_features),
            "PASS": len(manifest_features) == 40,
        },
    ])

    write_tsv(
        family_qc,
        OUT_DIR / "feature_family_qc.tsv",
    )

    # -------------------------------------------------------------------------
    # 16. Train/test separation
    # -------------------------------------------------------------------------

    section("16. TRAIN/TEST SEPARATION QC")

    train_reference_ids = set(train_reference["np_id"])
    test_reference_ids = set(test_reference["np_id"])
    train_test_overlap = train_reference_ids.intersection(
        test_reference_ids
    )

    split_qc = pd.DataFrame([{
        "train_count": len(train_reference_ids),
        "test_count": len(test_reference_ids),
        "overlap_count": len(train_test_overlap),
        "union_count": len(
            train_reference_ids.union(test_reference_ids)
        ),
        "expected_union": EXPECTED_REFERENCE,
        "PASS": (
            len(train_test_overlap) == 0
            and len(
                train_reference_ids.union(test_reference_ids)
            ) == EXPECTED_REFERENCE
        ),
    }])

    write_tsv(
        split_qc,
        OUT_DIR / "train_test_separation_qc.tsv",
    )

    if len(train_test_overlap):
        raise RuntimeError(
            "Train/test overlap detected in final reference."
        )

    log("Train/test separation : PASS")

    # -------------------------------------------------------------------------
    # 17. Main reference outputs
    # -------------------------------------------------------------------------

    section("17. WRITING MAIN 40D REFERENCE")

    reference_columns = (
        ["np_id", "SMILES"]
        + manifest_features
        + [
            "bioactivity_probability",
            "Model1_split",
            "Model1_reference_status",
            "reference_feature_source",
        ]
    )

    reference = reference[reference_columns]

    main_reference_path = OUT_DIR / "cmaup_40d_reference.tsv"

    write_tsv(
        reference,
        main_reference_path,
    )

    write_tsv(
        train_reference[reference_columns],
        OUT_DIR / "cmaup_40d_reference_train.tsv",
    )
    write_tsv(
        test_reference[reference_columns],
        OUT_DIR / "cmaup_40d_reference_test.tsv",
    )

    # Feature-only matrix for similarity engine.
    feature_matrix = reference[manifest_features].copy()
    write_tsv(
        feature_matrix,
        OUT_DIR / "cmaup_40d_feature_matrix.tsv",
    )

    # ID + feature matrix is convenient for downstream similarity indexing.
    id_feature_matrix = pd.concat(
        [
            reference[["np_id", "SMILES"]].reset_index(drop=True),
            feature_matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    write_tsv(
        id_feature_matrix,
        OUT_DIR / "cmaup_40d_id_feature_matrix.tsv",
    )

    log(f"Written : {main_reference_path}")
    log(f"Rows    : {len(reference):,}")
    log(f"Columns : {len(reference.columns)}")

    # -------------------------------------------------------------------------
    # 18. Missingness provenance
    # -------------------------------------------------------------------------

    section("18. RAW VS POST-PROCESSED MISSINGNESS PROVENANCE")

    raw_selected_rows = []
    for row in [
        raw_mol_train_qc,
        raw_mol_test_qc,
        raw_3d_train_qc,
        raw_3d_test_qc,
    ]:
        raw_selected_rows.append({
            "table": row["label"],
            "stage": "RAW_DESCRIPTOR",
            "NaN_count_selected_features": row[
                "raw_selected_NaN_count"
            ],
            "Inf_count_selected_features": row[
                "raw_selected_Inf_count"
            ],
            "NaN_is_fatal": False,
        })

    post_selected_rows = []
    for name, df in [
        ("TRAIN", train_19),
        ("TEST", test_19),
    ]:
        arr = df.to_numpy(dtype=np.float64)
        post_selected_rows.append({
            "table": name,
            "stage": "MODEL1_POST_PREPROCESSING",
            "NaN_count_selected_features": int(np.isnan(arr).sum()),
            "Inf_count_selected_features": int(np.isinf(arr).sum()),
            "NaN_is_fatal": True,
        })

    missingness_df = pd.DataFrame(
        raw_selected_rows + post_selected_rows
    )

    write_tsv(
        missingness_df,
        OUT_DIR / "raw_vs_postprocessed_missingness.tsv",
    )

    # -------------------------------------------------------------------------
    # 19. Numerical summary
    # -------------------------------------------------------------------------

    section("19. NUMERICAL FEATURE SUMMARY")

    numerical_rows = []

    for feature in manifest_features:
        values = reference[feature].to_numpy(dtype=np.float64)

        numerical_rows.append({
            "feature": feature,
            "representation": (
                "molecular"
                if feature in EXPECTED_MOLECULAR
                else (
                    "3D-QSAR"
                    if feature in EXPECTED_3D
                    else "MoLFormer"
                )
            ),
            "n": len(values),
            "missing": int(np.isnan(values).sum()),
            "inf": int(np.isinf(values).sum()),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "unique_values": int(
                pd.Series(values).nunique(dropna=True)
            ),
        })

    write_tsv(
        pd.DataFrame(numerical_rows),
        OUT_DIR / "40d_feature_numerical_summary.tsv",
    )

    # -------------------------------------------------------------------------
    # 20. Alignment provenance
    # -------------------------------------------------------------------------

    section("20. ALIGNMENT PROVENANCE")

    alignment = pd.DataFrame([
        {
            "component": "CMAUP molecule table",
            "rows": len(cmaup),
            "ordering": "CMAUP source-table order",
        },
        {
            "component": "Model-1 TRAIN split",
            "rows": len(train),
            "ordering": "authoritative train_80.tsv row order",
        },
        {
            "component": "Model-1 TEST split",
            "rows": len(test),
            "ordering": "authoritative test_20.tsv row order",
        },
        {
            "component": "Model-1 final TRAIN feature table",
            "rows": len(train_19),
            "ordering": "reordered exactly to train_80.tsv np_id order",
        },
        {
            "component": "Model-1 final TEST feature table",
            "rows": len(test_19),
            "ordering": "reordered exactly to test_20.tsv np_id order",
        },
        {
            "component": "TRAIN MoLFormer NPY",
            "rows": len(train_mf),
            "ordering": (
                "actual Model-1 NPY row order attached to "
                "authoritative train_80.tsv order by row-count/order provenance"
            ),
        },
        {
            "component": "TEST MoLFormer NPY",
            "rows": len(test_mf),
            "ordering": (
                "actual Model-1 NPY row order attached to "
                "authoritative test_20.tsv order by row-count/order provenance"
            ),
        },
    ])

    write_tsv(
        alignment,
        OUT_DIR / "alignment_provenance.tsv",
    )

    # -------------------------------------------------------------------------
    # 21. Comprehensive QC table
    # -------------------------------------------------------------------------

    section("21. COMPLETE QC SUMMARY")

    qc_rows = [
        {
            "qc": "CMAUP molecule count",
            "observed": len(cmaup),
            "expected": EXPECTED_CMAUP,
            "status": "PASS" if len(cmaup) == EXPECTED_CMAUP else "FAIL",
        },
        {
            "qc": "TRAIN count",
            "observed": len(train),
            "expected": EXPECTED_TRAIN,
            "status": "PASS" if len(train) == EXPECTED_TRAIN else "FAIL",
        },
        {
            "qc": "TEST count",
            "observed": len(test),
            "expected": EXPECTED_TEST,
            "status": "PASS" if len(test) == EXPECTED_TEST else "FAIL",
        },
        {
            "qc": "Reference count",
            "observed": len(reference),
            "expected": EXPECTED_REFERENCE,
            "status": "PASS"
            if len(reference) == EXPECTED_REFERENCE else "FAIL",
        },
        {
            "qc": "Excluded CMAUP records",
            "observed": len(excluded_ids),
            "expected": EXPECTED_EXCLUDED,
            "status": "PASS"
            if len(excluded_ids) == EXPECTED_EXCLUDED else "FAIL",
        },
        {
            "qc": "TRAIN/TEST overlap",
            "observed": len(train_test_overlap),
            "expected": 0,
            "status": "PASS"
            if len(train_test_overlap) == 0 else "FAIL",
        },
        {
            "qc": "40 feature count",
            "observed": len(manifest_features),
            "expected": 40,
            "status": "PASS"
            if len(manifest_features) == 40 else "FAIL",
        },
        {
            "qc": "Molecular feature count",
            "observed": len(molecular_features),
            "expected": 10,
            "status": "PASS"
            if len(molecular_features) == 10 else "FAIL",
        },
        {
            "qc": "3D-QSAR feature count",
            "observed": len(three_d_features),
            "expected": 9,
            "status": "PASS"
            if len(three_d_features) == 9 else "FAIL",
        },
        {
            "qc": "MoLFormer feature count",
            "observed": len(molformer_features),
            "expected": 21,
            "status": "PASS"
            if len(molformer_features) == 21 else "FAIL",
        },
        {
            "qc": "TRAIN final 40D NaN/Inf",
            "observed": int(
                final_qc.iloc[0]["NaN_count"]
                + final_qc.iloc[0]["Inf_count"]
            ),
            "expected": 0,
            "status": "PASS"
            if final_qc.iloc[0]["PASS"] else "FAIL",
        },
        {
            "qc": "TEST final 40D NaN/Inf",
            "observed": int(
                final_qc.iloc[1]["NaN_count"]
                + final_qc.iloc[1]["Inf_count"]
            ),
            "expected": 0,
            "status": "PASS"
            if final_qc.iloc[1]["PASS"] else "FAIL",
        },
        {
            "qc": "ALL reference final 40D NaN/Inf",
            "observed": int(
                final_qc.iloc[2]["NaN_count"]
                + final_qc.iloc[2]["Inf_count"]
            ),
            "expected": 0,
            "status": "PASS"
            if final_qc.iloc[2]["PASS"] else "FAIL",
        },
        {
            "qc": "Excluded records feature-generated",
            "observed": 0,
            "expected": 0,
            "status": "PASS",
        },
        {
            "qc": "Reference IDs equal split union",
            "observed": int(reference_ids == split_union),
            "expected": 1,
            "status": "PASS"
            if reference_ids == split_union else "FAIL",
        },
        {
            "qc": "Raw descriptor NaNs treated as fatal",
            "observed": 0,
            "expected": 0,
            "status": "PASS",
        },
        {
            "qc": "New imputation performed",
            "observed": 0,
            "expected": 0,
            "status": "PASS",
        },
    ]

    qc_table = pd.DataFrame(qc_rows)
    write_tsv(
        qc_table,
        OUT_DIR / "04_cmaup_40d_reference_qc.tsv",
    )

    failed = qc_table.loc[
        qc_table["status"] == "FAIL"
    ]

    if not failed.empty:
        raise RuntimeError(
            "One or more final QC checks failed:\n"
            f"{failed.to_string(index=False)}"
        )

    # -------------------------------------------------------------------------
    # 22. Manifest
    # -------------------------------------------------------------------------

    section("22. WRITING DATASET MANIFEST")

    finish_time = datetime.now().astimezone()

    provenance.update({
        "finish_time": finish_time.isoformat(),
        "duration_seconds": (
            finish_time - START_TIME
        ).total_seconds(),
        "cmaup_total_records": len(cmaup),
        "model1_train_records": len(train),
        "model1_test_records": len(test),
        "model1_reference_records": len(reference),
        "cmaup_excluded_records": len(excluded_ids),
        "model1_excluded_records_feature_generated": 0,
        "feature_count": 40,
        "molecular_feature_count": len(molecular_features),
        "three_d_feature_count": len(three_d_features),
        "molformer_feature_count": len(molformer_features),
        "molformer_embedding_dimension": EXPECTED_MOLFORMER_DIM,
        "reference_feature_order": manifest_features,
        "feature_manifest": relpath(FEATURE_MANIFEST),
        "preprocessing_source": relpath(FINAL_FEATURE_DIR),
        "preprocessing_info": preprocessing_info,
        "bioactivity_model": model_info,
        "exclusion_policy": (
            "CMAUP records outside the authoritative Model-1 train/test "
            "union receive no feature vector."
        ),
        "raw_descriptor_nan_policy": (
            "Raw descriptor NaNs are permitted because they are handled "
            "by the already-completed Model-1 preprocessing artifact."
        ),
        "new_imputation_performed": False,
        "excluded_records_imputed": False,
        "model_training_performed": False,
        "new_features_generated": False,
        "final_status": "PASS",
    })

    write_json(
        OUT_DIR / "04_cmaup_40d_reference_manifest.json",
        provenance,
    )

    # -------------------------------------------------------------------------
    # 23. Human-readable QC
    # -------------------------------------------------------------------------

    section("23. WRITING HUMAN-READABLE QC LOG")

    qc_lines = [
        "SCRIPT 04 — CMAUP 40D REFERENCE REPRESENTATION — CORRECTED",
        "=" * 78,
        f"PULP directory : {PULP_DIR}",
        f"Output directory : {OUT_DIR}",
        f"Start time : {START_TIME.isoformat()}",
        f"Finish time : {finish_time.isoformat()}",
        "",
        "CORE COUNTS",
        f"  CMAUP records                  : {len(cmaup):,}",
        f"  Model-1 TRAIN                  : {len(train):,}",
        f"  Model-1 TEST                   : {len(test):,}",
        f"  Model-1 reference              : {len(reference):,}",
        f"  Excluded CMAUP records         : {len(excluded_ids):,}",
        "",
        "FEATURE COMPOSITION",
        f"  Molecular                      : {len(molecular_features)}",
        f"  3D-QSAR                        : {len(three_d_features)}",
        f"  MoLFormer                      : {len(molformer_features)}",
        f"  TOTAL                          : {len(manifest_features)}",
        "",
        "CRITICAL CORRECTION",
        "  Raw molecular/3D descriptor NaNs are NOT fatal.",
        "  Exact Model-1 post-preprocessing feature tables are reused.",
        "  Model-1 preprocessing used TRAIN-ONLY MEDIAN IMPUTATION.",
        "  Script 04 performs NO new imputation.",
        "  Script 04 performs NO new descriptor generation.",
        "",
        "EXCLUSION POLICY",
        "  32 records outside the authoritative Model-1 universe",
        "  receive NO 40D feature vector.",
        "  They are retained only in the exclusion/QC table.",
        "",
        "FINAL QC",
        "  Reference IDs = split union    : PASS",
        "  Train/test separation           : PASS",
        "  40 feature identities           : PASS",
        "  40 feature order                : PASS",
        "  Final 40D NaN/Inf QC            : PASS",
        "  Excluded records feature-made   : NO",
        "  New imputation                  : NO",
        "  New model training              : NO",
        "",
        "BIOACTIVITY PROBABILITY",
        f"  Status : {model_info.get('prediction_status')}",
        "",
        "FINAL STATUS: PASS",
    ]

    write_text_path = OUT_DIR / "04_cmaup_40d_reference_qc.txt"
    write_text_path.write_text(
        "\n".join(qc_lines) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 24. Output inventory
    # -------------------------------------------------------------------------

    section("24. OUTPUT INVENTORY")

    output_paths = sorted(
        p for p in OUT_DIR.iterdir()
        if p.is_file()
    )

    inventory_rows = []
    for path in output_paths:
        inventory_rows.append({
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
        log(
            f"{path.name:55s} "
            f"{path.stat().st_size:>12,d} bytes"
        )

    write_tsv(
        pd.DataFrame(inventory_rows),
        OUT_DIR / "output_inventory.tsv",
    )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    section("FINAL STATUS")

    log("PASS")
    log(f"CMAUP total records       : {len(cmaup):,}")
    log(f"Model-1 reference records : {len(reference):,}")
    log(f"Excluded records           : {len(excluded_ids):,}")
    log(f"Final features             : {len(manifest_features)}")
    log(f"Molecular features         : {len(molecular_features)}")
    log(f"3D-QSAR features           : {len(three_d_features)}")
    log(f"MoLFormer features         : {len(molformer_features)}")
    log(f"Main reference             : {main_reference_path}")
    log(
        "Exclusion table            : "
        f"{OUT_DIR / 'cmaup_reference_exclusions.tsv'}"
    )
    log(
        "QC table                   : "
        f"{OUT_DIR / '04_cmaup_40d_reference_qc.tsv'}"
    )
    log(
        "Manifest                   : "
        f"{OUT_DIR / '04_cmaup_40d_reference_manifest.json'}"
    )

    return 0


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    try:
        sys.exit(main())

    except Exception as exc:
        section("FATAL ERROR")
        log(repr(exc))
        log("")
        log("Traceback:")
        traceback.print_exc()

        failure_payload = {
            "script": "04_build_cmaup_40d_reference.py",
            "status": "FAILED",
            "time": datetime.now().astimezone().isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

        try:
            write_json(
                OUT_DIR / "04_cmaup_40d_reference_FAILURE.json",
                failure_payload,
            )
        except Exception:
            pass

        sys.exit(1)
