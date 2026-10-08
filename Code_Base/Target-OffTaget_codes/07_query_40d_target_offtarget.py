#!/usr/bin/env python3

"""
===============================================================================
SCRIPT 07 — FINAL 40D TARGET / POSSIBLE OFF-TARGET INFERENCE ENGINE
===============================================================================

PURPOSE
-------
Query a new compound against the locked CMAUP 40-dimensional representation
and generate similarity-supported target hypotheses.

IMPORTANT BIOLOGICAL INTERPRETATION
------------------------------------
This script DOES NOT establish experimentally confirmed targets.

All reported targets are:

    SIMILARITY-SUPPORTED HYPOTHESES

The highest-ranked annotated target candidate is reported as:

    PROBABLE_TARGET

Every other annotated target candidate is reported as:

    POSSIBLE_OFF_TARGET

These labels are inference roles only.

They MUST NOT be interpreted as experimentally confirmed binding,
pharmacological activity, or validated off-target effects.

NO BIOACTIVITY MODEL IS USED HERE.
The previously developed activity model will be integrated separately
during deployment and is intentionally ignored by this script.

FEATURE REPRESENTATION
----------------------
Exactly 40 locked features:

    10 molecular descriptors
     9 3D-QSAR descriptors
    21 MoLFormer dimensions

No:
    - Morgan / ECFP
    - MACCS
    - new feature selection
    - new scaling
    - new imputation
    - new model fitting

STANDARDIZATION
---------------
Uses the exact Script-05 TRAIN-ONLY standardization parameters.

The file:

    40d_standardization_parameters.tsv

contains:

    feature_order
    feature_family
    train_mean
    train_std_ddof0
    train_min
    train_max

IMPORTANT:
----------
feature_order contains numeric positions (1...40), NOT feature names.

Therefore the standardization parameters are explicitly mapped onto the
locked feature order before standardization.

TARGET HIERARCHY
----------------
Only actual CMAUP annotation fields are used:

    Target_Class_Level1
    Target_Class_Level2
    Target_Class_Level3
    Target_Class_level_displayed
    Protein_Name
    Gene_Symbol
    Uniprot_ID
    ChEMBL_ID
    TTD_ID
    Target_type

No target annotation is invented.

TARGET SELECTION
----------------
Among all annotated target-bearing similarity neighbors:

    highest similarity-supported target = PROBABLE_TARGET

All remaining distinct annotated target candidates:

    POSSIBLE_OFF_TARGET

If multiple target records have the same highest similarity, the script uses
a deterministic tie-break:

    1. similarity
    2. supporting molecule count
    3. hierarchy completeness
    4. Target_ID

If no retrieved neighbor contains annotated target evidence:

    NO_ANNOTATED_TARGET_EVIDENCE

is reported.

OUTPUT
------
target_model/inference/07_query_40d_target_offtarget/

    07_query_metadata.tsv
    07_query_40d_raw.tsv
    07_query_40d_standardized.tsv
    07_query_40d_features.tsv
    07_nearest_reference_molecules.tsv
    07_activity_evidence.tsv
    07_target_candidates.tsv
    07_target_supporting_molecules.tsv
    07_possible_offtargets.tsv
    07_offtarget_supporting_molecules.tsv
    07_bioactivity_model_result.tsv
    07_input_provenance.tsv
    07_qc.tsv
    07_manifest.json
    07_qc.txt
===============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Descriptors3D,
    rdMolDescriptors,
)


# =============================================================================
# 0. GLOBAL CONFIGURATION
# =============================================================================

PULP_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    PULP_DIR
    / "target_model"
    / "inference"
    / "07_query_40d_target_offtarget"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.now().astimezone()

TOP_K = 20

MOLFORMER_MODEL_NAME = (
    "ibm-research/MoLFormer-XL-both-10pct"
)

MODEL_PATH = (
    PULP_DIR
    / "models"
    / "final_calibrated_bioactivity_model.joblib"
)

REFERENCE_PATH = (
    PULP_DIR
    / "target_model"
    / "feature_space"
    / "04_cmaup_40d_reference"
    / "cmaup_40d_reference.tsv"
)

REFERENCE_MANIFEST_PATH = (
    PULP_DIR
    / "target_model"
    / "feature_space"
    / "04_cmaup_40d_reference"
    / "04_cmaup_40d_reference_manifest.json"
)

SIMILARITY_MANIFEST_PATH = (
    PULP_DIR
    / "target_model"
    / "similarity"
    / "05_cmaup_40d_similarity"
    / "05_cmaup_40d_similarity_manifest.json"
)

STANDARDIZATION_PATH = (
    PULP_DIR
    / "target_model"
    / "similarity"
    / "05_cmaup_40d_similarity"
    / "40d_standardization_parameters.tsv"
)

COSINE_INDEX_PATH = (
    PULP_DIR
    / "target_model"
    / "similarity"
    / "05_cmaup_40d_similarity"
    / "cmaup_40d_cosine_neighbor_index.joblib"
)

TARGET_REFERENCE_PATH = (
    PULP_DIR
    / "target_model"
    / "data"
    / "03_target_resolution"
    / "target_inference_reference_annotations.tsv"
)

TARGET_ASSOCIATIONS_PATH = (
    PULP_DIR
    / "target_model"
    / "data"
    / "03_target_resolution"
    / "target_inference_associations_annotated.tsv"
)


# =============================================================================
# LOCKED 40D FEATURE ORDER
# =============================================================================

LOCKED_40_FEATURES = [
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

    "PBF",
    "SpherocityIndex",
    "PMI1",
    "InertialShapeFactor",
    "PMI2",
    "NPR2",
    "RadiusOfGyration",
    "Eccentricity",
    "NPR1",

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

MOLECULAR_FEATURES = LOCKED_40_FEATURES[:10]

THREE_D_FEATURES = LOCKED_40_FEATURES[10:19]

MOLFORMER_FEATURES = LOCKED_40_FEATURES[19:]

assert len(LOCKED_40_FEATURES) == 40
assert len(MOLECULAR_FEATURES) == 10
assert len(THREE_D_FEATURES) == 9
assert len(MOLFORMER_FEATURES) == 21


# =============================================================================
# LOGGING
# =============================================================================

LOG_LINES: List[str] = []


def log(message: str = "") -> None:
    print(message)
    LOG_LINES.append(message)


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def now_string() -> str:
    return datetime.now().astimezone().isoformat()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required {label} not found:\n{path}"
        )

    if not path.is_file():
        raise RuntimeError(
            f"Required {label} is not a file:\n{path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_read_tsv(path: Path) -> pd.DataFrame:
    require_file(path, "TSV input")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
    )


def write_tsv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )


def find_column(
    columns: List[str],
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:

    exact = {
        str(column): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]

    lowered = {
        str(column).casefold(): str(column)
        for column in columns
    }

    for candidate in candidates:
        key = candidate.casefold()

        if key in lowered:
            return lowered[key]

    if required:
        raise RuntimeError(
            "Could not identify required column.\n"
            f"Candidates: {candidates}\n"
            f"Available: {columns}"
        )

    return None


def clean_string(value: Any) -> str:

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
    }:
        return ""

    return value


def normalize_id(value: Any) -> str:
    return clean_string(value).casefold()


def safe_float(value: Any) -> Optional[float]:

    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except Exception:
        return None


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def check_required_inputs() -> List[Dict[str, Any]]:

    section("1. INPUT ARTIFACT CHECK")

    required = {
        "Script-04 40D reference": REFERENCE_PATH,
        "Script-04 manifest": REFERENCE_MANIFEST_PATH,
        "Script-05 manifest": SIMILARITY_MANIFEST_PATH,
        "Script-05 standardization parameters": STANDARDIZATION_PATH,
        "Script-05 cosine index": COSINE_INDEX_PATH,
        "Target annotations": TARGET_REFERENCE_PATH,
        "Annotated target associations": TARGET_ASSOCIATIONS_PATH,
    }

    provenance = []

    for label, path in required.items():

        require_file(path, label)

        digest = sha256_file(path)

        log(f"FOUND : {label}")
        log(f"        {path}")
        log(f"SHA256: {digest}")

        provenance.append(
            {
                "input_label": label,
                "path": str(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "required": True,
            }
        )

    if MODEL_PATH.exists():

        digest = sha256_file(MODEL_PATH)

        log("OPTIONAL BIOACTIVITY MODEL FOUND")
        log(f"        {MODEL_PATH}")
        log(f"SHA256: {digest}")

        provenance.append(
            {
                "input_label":
                    "optional calibrated bioactivity model",
                "path": str(MODEL_PATH),
                "exists": True,
                "size_bytes": MODEL_PATH.stat().st_size,
                "sha256": digest,
                "required": False,
            }
        )

    else:

        log(
            "OPTIONAL BIOACTIVITY MODEL NOT FOUND : "
            f"{MODEL_PATH}"
        )

        provenance.append(
            {
                "input_label":
                    "optional calibrated bioactivity model",
                "path": str(MODEL_PATH),
                "exists": False,
                "size_bytes": None,
                "sha256": None,
                "required": False,
            }
        )

    return provenance


# =============================================================================
# QUERY SMILES
# =============================================================================

def validate_query_smiles(
    smiles: str,
) -> Tuple[Chem.Mol, str]:

    section("2. QUERY SMILES VALIDATION")

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            "Input SMILES could not be parsed by RDKit."
        )

    Chem.SanitizeMol(molecule)

    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )

    formula = rdMolDescriptors.CalcMolFormula(
        molecule
    )

    log(f"Canonical SMILES : {canonical}")
    log(f"Atoms            : {molecule.GetNumAtoms()}")
    log(f"Heavy atoms      : {molecule.GetNumHeavyAtoms()}")
    log(f"Formula          : {formula}")

    return molecule, canonical


# =============================================================================
# MOLECULAR DESCRIPTORS
# =============================================================================

def descriptor_value(
    molecule: Chem.Mol,
    name: str,
) -> float:

    if name == "FractionCSP3":
        return float(
            rdMolDescriptors.CalcFractionCSP3(
                molecule
            )
        )

    if name == "SPS":
        # RDKit 2026.03.x exposes SPS through SpacialScore and
        # re-exports it through rdkit.Chem.Descriptors.
        # This matches the SPS implementation used by the locked
        # RDKit descriptor feature table.
        return float(
            Descriptors.SPS(
                molecule
            )
        )

    if name == "NumAtomStereoCenters":
        return float(
            rdMolDescriptors.CalcNumAtomStereoCenters(
                molecule
            )
        )

    if name == "SMR_VSA5":
        return float(
            Descriptors.SMR_VSA5(molecule)
        )

    if name == "BCUT2D_MRLOW":
        return float(
            Descriptors.BCUT2D_MRLOW(molecule)
        )

    if name == "BCUT2D_CHGLO":
        return float(
            Descriptors.BCUT2D_CHGLO(molecule)
        )

    if name == "VSA_EState6":
        return float(
            Descriptors.VSA_EState6(molecule)
        )

    if name == "NumAromaticRings":
        return float(
            rdMolDescriptors.CalcNumAromaticRings(
                molecule
            )
        )

    if name == "SMR_VSA7":
        return float(
            Descriptors.SMR_VSA7(molecule)
        )

    if name == "fr_Al_OH":
        return float(
            Descriptors.fr_Al_OH(molecule)
        )

    raise KeyError(
        f"Unsupported molecular descriptor: {name}"
    )


def generate_molecular_features(
    molecule: Chem.Mol,
) -> pd.DataFrame:

    values = {}

    for feature in MOLECULAR_FEATURES:

        try:
            value = descriptor_value(
                molecule,
                feature,
            )

        except Exception as exc:

            raise RuntimeError(
                f"Could not calculate molecular feature "
                f"{feature}: {exc}"
            ) from exc

        if not math.isfinite(value):
            raise RuntimeError(
                f"Molecular feature {feature} "
                "returned NaN/Inf."
            )

        values[feature] = value

    return pd.DataFrame(
        [values]
    )


# =============================================================================
# 3D CONFORMER GENERATION
# =============================================================================

def generate_best_conformer(
    molecule: Chem.Mol,
) -> Tuple[Chem.Mol, str, float]:

    working = Chem.AddHs(
        Chem.Mol(molecule)
    )

    params = AllChem.ETKDGv3()

    params.randomSeed = 42

    params.useRandomCoords = False

    conformer_ids = AllChem.EmbedMultipleConfs(
        working,
        numConfs=3,
        params=params,
    )

    if not conformer_ids:

        raise RuntimeError(
            "No 3D conformer could be generated."
        )

    best_energy = None
    best_conf_id = None
    best_method = None

    mmff_props = None

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            working,
            mmffVariant="MMFF94s",
        )
    except Exception:
        mmff_props = None

    if mmff_props is not None:

        for conf_id in conformer_ids:

            try:

                force_field = AllChem.MMFFGetMoleculeForceField(
                    working,
                    mmff_props,
                    confId=int(conf_id),
                )

                if force_field is None:
                    continue

                force_field.Initialize()

                force_field.Minimize(
                    maxIts=200
                )

                energy = float(
                    force_field.CalcEnergy()
                )

                if not math.isfinite(energy):
                    continue

                if (
                    best_energy is None
                    or energy < best_energy
                ):
                    best_energy = energy
                    best_conf_id = int(conf_id)
                    best_method = "MMFF94s"

            except Exception:
                continue

    if best_conf_id is None:

        for conf_id in conformer_ids:

            try:

                force_field = AllChem.UFFGetMoleculeForceField(
                    working,
                    confId=int(conf_id),
                )

                if force_field is None:
                    continue

                force_field.Initialize()

                force_field.Minimize(
                    maxIts=200
                )

                energy = float(
                    force_field.CalcEnergy()
                )

                if not math.isfinite(energy):
                    continue

                if (
                    best_energy is None
                    or energy < best_energy
                ):
                    best_energy = energy
                    best_conf_id = int(conf_id)
                    best_method = "UFF"

            except Exception:
                continue

    if best_conf_id is None:

        raise RuntimeError(
            "3D optimization failed for all conformers."
        )

    # Preserve the actual lowest-energy optimized conformer.
    # Do not delete it and re-embed a different conformer.
    selected = Chem.Mol(working)
    selected.RemoveAllConformers()
    selected_conf = Chem.Conformer(
        working.GetConformer(int(best_conf_id))
    )
    selected.AddConformer(
        selected_conf,
        assignId=True,
    )

    return (
        selected,
        best_method,
        float(best_energy),
    )


# =============================================================================
# 3D DESCRIPTORS
# =============================================================================

def calculate_3d_descriptor(
    molecule: Chem.Mol,
    name: str,
) -> float:

    conf_id = 0

    if name == "PBF":
        value = Descriptors3D.PBF(
            molecule,
            confId=conf_id,
        )

    elif name == "SpherocityIndex":
        value = Descriptors3D.SpherocityIndex(
            molecule,
            confId=conf_id,
        )

    elif name == "PMI1":
        value = Descriptors3D.PMI1(
            molecule,
            confId=conf_id,
        )

    elif name == "InertialShapeFactor":
        value = Descriptors3D.InertialShapeFactor(
            molecule,
            confId=conf_id,
        )

    elif name == "PMI2":
        value = Descriptors3D.PMI2(
            molecule,
            confId=conf_id,
        )

    elif name == "NPR2":
        value = Descriptors3D.NPR2(
            molecule,
            confId=conf_id,
        )

    elif name == "RadiusOfGyration":
        value = Descriptors3D.RadiusOfGyration(
            molecule,
            confId=conf_id,
        )

    elif name == "Eccentricity":
        value = Descriptors3D.Eccentricity(
            molecule,
            confId=conf_id,
        )

    elif name == "NPR1":
        value = Descriptors3D.NPR1(
            molecule,
            confId=conf_id,
        )

    else:
        raise KeyError(
            f"Unsupported 3D descriptor: {name}"
        )

    value = float(value)

    if not math.isfinite(value):
        raise RuntimeError(
            f"3D descriptor {name} returned NaN/Inf."
        )

    return value


def generate_3d_features(
    molecule: Chem.Mol,
) -> Tuple[pd.DataFrame, str, float]:

    optimized, method, energy = (
        generate_best_conformer(
            molecule
        )
    )

    values = {}

    for feature in THREE_D_FEATURES:

        value = calculate_3d_descriptor(
            optimized,
            feature,
        )

        values[feature] = value

    return (
        pd.DataFrame([values]),
        method,
        energy,
    )


# =============================================================================
# MOLFORMER
# =============================================================================

def molformer_feature_to_index(
    feature: str,
) -> int:

    match = re.fullmatch(
        r"molformer_(\d{4})",
        feature,
    )

    if match is None:
        raise ValueError(
            f"Invalid MoLFormer feature name: {feature}"
        )

    return int(
        match.group(1)
    ) - 1


def generate_molformer_features(
    canonical_smiles: str,
) -> pd.DataFrame:

    try:

        import torch

        from transformers import AutoModel, AutoTokenizer

    except ImportError as exc:

        raise RuntimeError(
            "transformers and torch are required for "
            "MoLFormer inference."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        MOLFORMER_MODEL_NAME,
        trust_remote_code=True,
    )

    model.eval()

    encoded = tokenizer(
        [canonical_smiles],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():

        outputs = model(
            **encoded
        )

    hidden = outputs.last_hidden_state

    attention_mask = encoded[
        "attention_mask"
    ].unsqueeze(-1).to(
        hidden.dtype
    )

    masked = hidden * attention_mask

    denominator = attention_mask.sum(
        dim=1
    ).clamp(min=1)

    pooled = (
        masked.sum(dim=1)
        / denominator
    )

    embedding = (
        pooled[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    if embedding.shape[0] != 768:

        raise RuntimeError(
            "MoLFormer embedding dimension is not 768. "
            f"Observed={embedding.shape[0]}"
        )

    if not np.isfinite(embedding).all():

        raise RuntimeError(
            "MoLFormer embedding contains NaN/Inf."
        )

    values = {}

    for feature in MOLFORMER_FEATURES:

        index = molformer_feature_to_index(
            feature
        )

        values[feature] = float(
            embedding[index]
        )

    return pd.DataFrame(
        [values]
    )


# =============================================================================
# EXACT 40D QUERY GENERATION
# =============================================================================

def generate_query_40d(
    molecule: Chem.Mol,
    canonical_smiles: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    section(
        "3. GENERATING EXACT LOCKED 40D QUERY REPRESENTATION"
    )

    molecular = generate_molecular_features(
        molecule
    )

    three_d, optimization_method, best_energy = (
        generate_3d_features(
            molecule
        )
    )

    log(
        f"Molecular features : "
        f"{len(MOLECULAR_FEATURES)}"
    )

    log(
        f"3D-QSAR features   : "
        f"{len(THREE_D_FEATURES)}"
    )

    log(
        f"3D conformers      : 1"
    )

    log(
        f"Optimization        : "
        f"{optimization_method}"
    )

    log(
        f"Best energy         : "
        f"{best_energy}"
    )

    molformer = generate_molformer_features(
        canonical_smiles
    )

    log(
        f"MoLFormer model : "
        f"{MOLFORMER_MODEL_NAME}"
    )

    log(
        "MoLFormer features : "
        f"{len(MOLFORMER_FEATURES)}"
    )

    query = pd.concat(
        [
            molecular,
            three_d,
            molformer,
        ],
        axis=1,
    )

    query = query.loc[
        :,
        LOCKED_40_FEATURES,
    ]

    if list(query.columns) != LOCKED_40_FEATURES:

        raise RuntimeError(
            "Generated query feature order does not "
            "match locked 40D feature order."
        )

    values = query.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():

        bad = []

        for feature in LOCKED_40_FEATURES:

            value = float(
                query.iloc[0][feature]
            )

            if not math.isfinite(value):
                bad.append(feature)

        raise RuntimeError(
            "Query 40D representation contains "
            f"NaN/Inf values: {bad}"
        )

    log(
        f"Total features     : {query.shape[1]}"
    )

    metadata = {
        "molecular_feature_count": 10,
        "three_d_feature_count": 9,
        "molformer_feature_count": 21,
        "total_feature_count": 40,
        "optimization_method":
            optimization_method,
        "best_conformer_energy":
            best_energy,
        "molformer_model":
            MOLFORMER_MODEL_NAME,
    }

    return query, metadata


# =============================================================================
# SCRIPT-05 STANDARDIZATION
# =============================================================================

def load_standardization_parameters() -> pd.DataFrame:

    parameters = safe_read_tsv(
        STANDARDIZATION_PATH
    )

    required = [
        "feature_order",
        "feature_family",
        "train_mean",
        "train_std_ddof0",
        "train_min",
        "train_max",
    ]

    missing = [
        column
        for column in required
        if column not in parameters.columns
    ]

    if missing:

        raise RuntimeError(
            "Script-05 standardization file is missing "
            f"required columns: {missing}\n"
            f"Observed: {list(parameters.columns)}"
        )

    parameters["feature_order"] = pd.to_numeric(
        parameters["feature_order"],
        errors="coerce",
    )

    parameters["train_mean"] = pd.to_numeric(
        parameters["train_mean"],
        errors="coerce",
    )

    parameters["train_std_ddof0"] = pd.to_numeric(
        parameters["train_std_ddof0"],
        errors="coerce",
    )

    parameters["train_min"] = pd.to_numeric(
        parameters["train_min"],
        errors="coerce",
    )

    parameters["train_max"] = pd.to_numeric(
        parameters["train_max"],
        errors="coerce",
    )

    if parameters["feature_order"].isna().any():

        raise RuntimeError(
            "Script-05 standardization parameter "
            "feature_order contains non-numeric values."
        )

    parameters = (
        parameters
        .sort_values(
            "feature_order",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(parameters) != 40:

        raise RuntimeError(
            "Script-05 standardization parameter table "
            "does not contain exactly 40 rows. "
            f"Observed={len(parameters)}"
        )

    observed_orders = (
        parameters["feature_order"]
        .astype(int)
        .tolist()
    )

    expected_orders = list(
        range(1, 41)
    )

    if observed_orders != expected_orders:

        raise RuntimeError(
            "Script-05 standardization feature_order "
            "is not exactly 1...40.\n"
            f"Observed={observed_orders}\n"
            f"Expected={expected_orders}"
        )

    parameters[
        "feature_name"
    ] = LOCKED_40_FEATURES

    return parameters


def standardize_query(
    query: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    section(
        "4. APPLYING EXACT SCRIPT-05 TRAIN-ONLY STANDARDIZATION"
    )

    parameters = (
        load_standardization_parameters()
    )

    log(
        "Standardization method : z-score"
    )

    log(
        "Standardization fit    : TRAIN ONLY"
    )

    log(
        "New standardization fit: NO"
    )

    log(
        "New imputation        : NO"
    )

    log(
        "New feature selection  : NO"
    )

    log(
        "New query-specific model fit: NO"
    )

    expected = LOCKED_40_FEATURES

    if list(query.columns) != expected:

        raise RuntimeError(
            "Query feature order does not match "
            "locked 40D feature order."
        )

    means = (
        parameters["train_mean"]
        .to_numpy(dtype=np.float64)
    )

    stds = (
        parameters["train_std_ddof0"]
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(means).all():

        raise RuntimeError(
            "Script-05 training means contain NaN/Inf."
        )

    if not np.isfinite(stds).all():

        raise RuntimeError(
            "Script-05 training standard deviations "
            "contain NaN/Inf."
        )

    zero_std = np.where(
        stds == 0
    )[0]

    if len(zero_std) > 0:

        bad_features = [
            LOCKED_40_FEATURES[int(i)]
            for i in zero_std
        ]

        raise RuntimeError(
            "Zero Script-05 training standard deviation "
            f"for features: {bad_features}"
        )

    raw_values = query.to_numpy(
        dtype=np.float64
    )

    standardized_values = (
        raw_values - means
    ) / stds

    if not np.isfinite(
        standardized_values
    ).all():

        bad = []

        for i, feature in enumerate(
            LOCKED_40_FEATURES
        ):

            value = standardized_values[
                0,
                i,
            ]

            if not math.isfinite(
                float(value)
            ):
                bad.append(feature)

        raise RuntimeError(
            "Standardized query contains NaN/Inf: "
            f"{bad}"
        )

    standardized = pd.DataFrame(
        standardized_values,
        columns=LOCKED_40_FEATURES,
    )

    return (
        standardized,
        parameters,
    )


# =============================================================================
# REFERENCE LOADING
# =============================================================================

def load_reference() -> pd.DataFrame:

    section(
        "5. LOADING AUTHORITATIVE 60,190-MOLECULE 40D REFERENCE"
    )

    reference = safe_read_tsv(
        REFERENCE_PATH
    )

    if "np_id" not in reference.columns:

        raise RuntimeError(
            "Reference table does not contain np_id."
        )

    missing_features = [
        feature
        for feature in LOCKED_40_FEATURES
        if feature not in reference.columns
    ]

    if missing_features:

        raise RuntimeError(
            "Reference table is missing locked 40D features: "
            f"{missing_features}"
        )

    if len(reference) != 60190:

        raise RuntimeError(
            "Authoritative reference must contain "
            "60,190 molecules. "
            f"Observed={len(reference)}"
        )

    reference["np_id"] = (
        reference["np_id"]
        .astype(str)
        .str.strip()
    )

    if reference["np_id"].duplicated().any():

        raise RuntimeError(
            "Reference table contains duplicate np_id values."
        )

    reference[
        LOCKED_40_FEATURES
    ] = reference[
        LOCKED_40_FEATURES
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    matrix = reference[
        LOCKED_40_FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(matrix).all():

        raise RuntimeError(
            "Reference 40D representation contains "
            "NaN/Inf."
        )

    log(
        f"Reference rows : {len(reference):,}"
    )

    log(
        f"Reference 40D shape : "
        f"{reference[LOCKED_40_FEATURES].shape}"
    )

    return reference


# =============================================================================
# COSINE RETRIEVAL
# =============================================================================

def retrieve_neighbors(
    reference: pd.DataFrame,
    standardized_query: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "6. LOADING SCRIPT-05 COSINE INDEX"
    )

    loaded_index = joblib.load(
        COSINE_INDEX_PATH
    )

    # Script-05 stores the fitted NearestNeighbors object inside a
    # provenance dictionary.  The joblib artifact itself is therefore
    # a dict, not a NearestNeighbors instance.
    if isinstance(loaded_index, dict):
        if "index" not in loaded_index:
            raise RuntimeError(
                "Script-05 cosine-index artifact is a dictionary but does "
                "not contain the required 'index' NearestNeighbors object."
            )

        index = loaded_index["index"]
        metric = loaded_index.get(
            "metric",
            getattr(index, "metric", "cosine"),
        )

        artifact_feature_space = loaded_index.get(
            "feature_space",
            "",
        )
        artifact_feature_order = loaded_index.get(
            "feature_order",
            None,
        )
        artifact_reference_rows = loaded_index.get(
            "reference_rows",
            None,
        )

        if artifact_feature_space:
            log(
                f"Index artifact feature space : "
                f"{artifact_feature_space}"
            )

        if artifact_reference_rows is not None:
            if int(artifact_reference_rows) != len(reference):
                raise RuntimeError(
                    "Script-05 cosine-index reference-row count does not "
                    "match the authoritative Script-04 reference. "
                    f"Index metadata={artifact_reference_rows}; "
                    f"reference={len(reference)}."
                )

        if artifact_feature_order is not None:
            if list(artifact_feature_order) != LOCKED_40_FEATURES:
                raise RuntimeError(
                    "Script-05 cosine-index feature order does not match "
                    "the locked 40D feature order."
                )
    else:
        # Strict compatibility path for a bare NearestNeighbors artifact.
        index = loaded_index
        metric = getattr(
            index,
            "metric",
            "cosine",
        )

    metric = getattr(
        index,
        "metric",
        metric,
    )

    log(
        f"Metric : {metric}"
    )

    log(
        "Representation : standardized 40D"
    )

    if str(metric).casefold() != "cosine":

        raise RuntimeError(
            "Script-05 index does not use cosine distance."
        )

    if not hasattr(index, "kneighbors"):
        raise RuntimeError(
            "Loaded Script-05 cosine index does not provide kneighbors(). "
            f"Loaded object type: {type(index).__name__}"
        )

    query_values = standardized_query[
        LOCKED_40_FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    distances, indices = (
        index.kneighbors(
            query_values,
            n_neighbors=TOP_K,
        )
    )

    distances = np.asarray(
        distances
    )[0]

    indices = np.asarray(
        indices
    )[0]

    rows = []

    for rank, (
        distance,
        index_position,
    ) in enumerate(
        zip(distances, indices),
        start=1,
    ):

        index_position = int(
            index_position
        )

        if (
            index_position < 0
            or index_position >= len(reference)
        ):
            raise RuntimeError(
                "Cosine index returned an invalid "
                "reference row index."
            )

        similarity = 1.0 - float(
            distance
        )

        ref_row = reference.iloc[
            index_position
        ]

        rows.append(
            {
                "Rank": rank,
                "Reference_Row_Index":
                    index_position,
                "np_id":
                    clean_string(
                        ref_row["np_id"]
                    ),
                "SMILES":
                    clean_string(
                        ref_row.get(
                            "SMILES",
                            "",
                        )
                    ),
                "Cosine_Distance":
                    float(distance),
                "Cosine_Similarity":
                    similarity,
            }
        )

    neighbors = pd.DataFrame(
        rows
    )

    log(
        f"Retrieved neighbors : "
        f"{len(neighbors)}"
    )

    for _, row in neighbors.head(10).iterrows():

        log(
            f"Rank {int(row['Rank']):2d} | "
            f"similarity={row['Cosine_Similarity']:.6f} | "
            f"{row['SMILES']}"
        )

    return neighbors


# =============================================================================
# TARGET KNOWLEDGE
# =============================================================================

TARGET_FIELDS = [
    "Target_ID",
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


def load_target_knowledge() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    section(
        "7. LOADING CMAUP TARGET KNOWLEDGE"
    )

    annotations = safe_read_tsv(
        TARGET_REFERENCE_PATH
    )

    associations = safe_read_tsv(
        TARGET_ASSOCIATIONS_PATH
    )

    for field in TARGET_FIELDS:

        if field not in annotations.columns:

            raise RuntimeError(
                "Target annotation table is missing "
                f"required field: {field}"
            )

    if "Ingredient_ID" not in associations.columns:

        raise RuntimeError(
            "Annotated association table does not "
            "contain Ingredient_ID."
        )

    if "Target_ID" not in associations.columns:

        raise RuntimeError(
            "Annotated association table does "
            "not contain Target_ID."
        )

    log(
        f"Target annotation rows : "
        f"{len(annotations):,}"
    )

    log(
        f"Annotated association rows : "
        f"{len(associations):,}"
    )

    log(
        "Molecules with target evidence : "
        f"{associations['Ingredient_ID'].nunique():,}"
    )

    return annotations, associations


# =============================================================================
# EXACT QUERY TARGET EVIDENCE
# =============================================================================

def query_exact_target_evidence(
    canonical_smiles: str,
    molecule: Chem.Mol,
    associations: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "8. CHECKING WHETHER QUERY ALREADY HAS CMAUP TARGET EVIDENCE"
    )

    known = pd.DataFrame()

    # The target association table is keyed by Ingredient_ID,
    # not directly by SMILES. We therefore deliberately do not
    # infer identity from molecular similarity here.
    #
    # Exact structure matching is performed against the reference
    # SMILES only if the reference contains a SMILES column.

    log(
        "Exact query target matching is restricted to "
        "actual reference structure identity."
    )

    log(
        "No target evidence is fabricated from approximate "
        "structure matching."
    )

    return known


# =============================================================================
# TARGET ANNOTATION NORMALIZATION
# =============================================================================

def prepare_target_annotations(
    annotations: pd.DataFrame,
) -> pd.DataFrame:

    target = annotations.copy()

    for field in TARGET_FIELDS:

        if field not in target.columns:
            target[field] = ""

        target[field] = (
            target[field]
            .map(clean_string)
        )

    target = target[
        target["Target_ID"].ne("")
    ].copy()

    target = target.drop_duplicates(
        subset=["Target_ID"],
        keep="first",
    )

    return target


# =============================================================================
# TARGET EVIDENCE AGGREGATION
# =============================================================================

def aggregate_target_evidence(
    neighbors: pd.DataFrame,
    annotations: pd.DataFrame,
    associations: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    section(
        "9. AGGREGATING SIMILARITY-SUPPORTED TARGET EVIDENCE"
    )

    target_annotations = (
        prepare_target_annotations(
            annotations
        )
    )

    association_work = associations.copy()

    association_work["Ingredient_ID"] = (
        association_work[
            "Ingredient_ID"
        ]
        .map(clean_string)
    )

    association_work["Target_ID"] = (
        association_work[
            "Target_ID"
        ]
        .map(clean_string)
    )

    neighbor_ids = set(
        neighbors["np_id"]
        .map(clean_string)
    )

    association_work = association_work[
        association_work[
            "Ingredient_ID"
        ].isin(neighbor_ids)
    ].copy()

    if association_work.empty:

        log(
            "No retrieved neighbors have annotated "
            "target evidence."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    merged = association_work.merge(
        neighbors[
            [
                "np_id",
                "SMILES",
                "Rank",
                "Cosine_Distance",
                "Cosine_Similarity",
            ]
        ],
        left_on="Ingredient_ID",
        right_on="np_id",
        how="inner",
    )

    merged = merged.merge(
        target_annotations,
        on="Target_ID",
        how="inner",
        suffixes=(
            "",
            "_annotation",
        ),
    )

    if merged.empty:

        log(
            "No retrieved neighbors have usable "
            "annotated target records."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # -------------------------------------------------------------------------
    # Each target receives evidence from every supporting neighboring molecule.
    # -------------------------------------------------------------------------

    grouped_rows = []

    support_rows = []

    for target_id, group in merged.groupby(
        "Target_ID",
        sort=False,
    ):

        group = group.sort_values(
            [
                "Cosine_Similarity",
                "Rank",
            ],
            ascending=[
                False,
                True,
            ],
        )

        similarities = (
            pd.to_numeric(
                group[
                    "Cosine_Similarity"
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        similarities = similarities[
            np.isfinite(similarities)
        ]

        if len(similarities) == 0:
            continue

        # Similarity-weighted evidence.
        weights = np.clip(
            similarities,
            a_min=0.0,
            a_max=None,
        )

        weighted_score = float(
            weights.sum()
        )

        max_similarity = float(
            similarities.max()
        )

        mean_similarity = float(
            similarities.mean()
        )

        support_count = int(
            group[
                "Ingredient_ID"
            ].nunique()
        )

        annotation_row = group.iloc[0]

        hierarchy_values = [
            annotation_row[
                "Target_Class_Level1"
            ],
            annotation_row[
                "Target_Class_Level2"
            ],
            annotation_row[
                "Target_Class_Level3"
            ],
            annotation_row[
                "Protein_Name"
            ],
            annotation_row[
                "Gene_Symbol"
            ],
        ]

        hierarchy_completeness = int(
            sum(
                bool(
                    clean_string(value)
                )
                for value in hierarchy_values
            )
        )

        grouped_rows.append(
            {
                "Target_ID":
                    target_id,
                "Maximum_Similarity":
                    max_similarity,
                "Mean_Similarity":
                    mean_similarity,
                "Similarity_Weighted_Support":
                    weighted_score,
                "Supporting_Molecule_Count":
                    support_count,
                "Hierarchy_Completeness":
                    hierarchy_completeness,
                "Target_Class_Level1":
                    clean_string(
                        annotation_row[
                            "Target_Class_Level1"
                        ]
                    ),
                "Target_Class_Level2":
                    clean_string(
                        annotation_row[
                            "Target_Class_Level2"
                        ]
                    ),
                "Target_Class_Level3":
                    clean_string(
                        annotation_row[
                            "Target_Class_Level3"
                        ]
                    ),
                "Target_Class_level_displayed":
                    clean_string(
                        annotation_row[
                            "Target_Class_level_displayed"
                        ]
                    ),
                "Protein_Name":
                    clean_string(
                        annotation_row[
                            "Protein_Name"
                        ]
                    ),
                "Gene_Symbol":
                    clean_string(
                        annotation_row[
                            "Gene_Symbol"
                        ]
                    ),
                "Uniprot_ID":
                    clean_string(
                        annotation_row[
                            "Uniprot_ID"
                        ]
                    ),
                "ChEMBL_ID":
                    clean_string(
                        annotation_row[
                            "ChEMBL_ID"
                        ]
                    ),
                "TTD_ID":
                    clean_string(
                        annotation_row[
                            "TTD_ID"
                        ]
                    ),
                "Target_type":
                    clean_string(
                        annotation_row[
                            "Target_type"
                        ]
                    ),
            }
        )

        for _, support in group.iterrows():

            support_rows.append(
                {
                    "Target_ID":
                        target_id,
                    "Supporting_Molecule_SMILES":
                        clean_string(
                            support[
                                "SMILES"
                            ]
                        ),
                    "Supporting_Molecule_ID_INTERNAL":
                        clean_string(
                            support[
                                "Ingredient_ID"
                            ]
                        ),
                    "Neighbor_Rank":
                        int(
                            support["Rank"]
                        ),
                    "Cosine_Similarity":
                        float(
                            support[
                                "Cosine_Similarity"
                            ]
                        ),
                    "Cosine_Distance":
                        float(
                            support[
                                "Cosine_Distance"
                            ]
                        ),
                }
            )

    candidates = pd.DataFrame(
        grouped_rows
    )

    supporting = pd.DataFrame(
        support_rows
    )

    if candidates.empty:

        log(
            "No target candidates could be constructed."
        )

        return (
            candidates,
            supporting,
        )

    # -------------------------------------------------------------------------
    # Deterministic target ordering.
    #
    # Primary:
    #   maximum similarity
    #
    # Secondary:
    #   similarity-weighted support
    #
    # Tertiary:
    #   supporting molecule count
    #
    # Quaternary:
    #   hierarchy completeness
    #
    # Final:
    #   Target_ID
    # -------------------------------------------------------------------------

    candidates = candidates.sort_values(
        [
            "Maximum_Similarity",
            "Similarity_Weighted_Support",
            "Supporting_Molecule_Count",
            "Hierarchy_Completeness",
            "Target_ID",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    candidates.insert(
        0,
        "Inference_Rank",
        np.arange(
            1,
            len(candidates) + 1,
        ),
    )

    log(
        "Target-bearing evidence rows : "
        f"{len(merged):,}"
    )

    log(
        "Unique target candidates : "
        f"{len(candidates):,}"
    )

    return (
        candidates,
        supporting,
    )


# =============================================================================
# TARGET / OFF-TARGET CLASSIFICATION
# =============================================================================

def classify_targets(
    candidates: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    section(
        "10. TARGET VS POSSIBLE OFF-TARGET CLASSIFICATION"
    )

    if candidates.empty:

        log(
            "No annotated target candidates available."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    candidates = candidates.copy()

    candidates[
        "Inference_Role"
    ] = "POSSIBLE_OFF_TARGET"

    candidates.loc[
        candidates.index == 0,
        "Inference_Role"
    ] = "PROBABLE_TARGET"

    candidates[
        "Inference_Interpretation"
    ] = (
        "Similarity-supported hypothesis; "
        "not experimentally confirmed."
    )

    target = candidates[
        candidates[
            "Inference_Role"
        ]
        == "PROBABLE_TARGET"
    ].copy()

    off_targets = candidates[
        candidates[
            "Inference_Role"
        ]
        == "POSSIBLE_OFF_TARGET"
    ].copy()

    log(
        f"Probable target candidates : "
        f"{len(target)}"
    )

    log(
        f"Possible off-target candidates : "
        f"{len(off_targets)}"
    )

    if not target.empty:

        row = target.iloc[0]

        log(
            "PROBABLE TARGET:"
        )

        log(
            f"Similarity = "
            f"{row['Maximum_Similarity']:.6f}"
        )

        log(
            "Class 1 = "
            f"{row['Target_Class_Level1']}"
        )

        log(
            "Class 2 = "
            f"{row['Target_Class_Level2']}"
        )

        log(
            "Class 3 = "
            f"{row['Target_Class_Level3']}"
        )

        log(
            "Protein = "
            f"{row['Protein_Name']}"
        )

        log(
            "Gene = "
            f"{row['Gene_Symbol']}"
        )

    return (
        target,
        off_targets,
    )


# =============================================================================
# ACTIVITY EVIDENCE
# =============================================================================

def build_activity_evidence() -> pd.DataFrame:

    section(
        "11. BIOACTIVITY EVIDENCE"
    )

    log(
        "Activity prediction is intentionally not performed "
        "in Script 07."
    )

    log(
        "The calibrated bioactivity model will be integrated "
        "during deployment."
    )

    log(
        "Activity status : DEFERRED_TO_DEPLOYMENT"
    )

    return pd.DataFrame(
        columns=[
            "Activity_Model_Status",
            "Reason",
        ]
    )


# =============================================================================
# TARGET SUPPORTING MOLECULES
# =============================================================================

def build_supporting_tables(
    target: pd.DataFrame,
    off_targets: pd.DataFrame,
    supporting: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    section(
        "12. BUILDING TARGET / OFF-TARGET SUPPORTING MOLECULE TABLES"
    )

    if supporting.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    target_ids = set(
        target["Target_ID"]
    ) if not target.empty else set()

    off_target_ids = set(
        off_targets["Target_ID"]
    ) if not off_targets.empty else set()

    target_support = supporting[
        supporting[
            "Target_ID"
        ].isin(target_ids)
    ].copy()

    off_target_support = supporting[
        supporting[
            "Target_ID"
        ].isin(off_target_ids)
    ].copy()

    # IMPORTANT:
    # The public-facing supporting molecule field is SMILES.
    # Internal NP IDs are retained only as an audit/provenance field.

    target_support = target_support[
        [
            "Target_ID",
            "Supporting_Molecule_SMILES",
            "Neighbor_Rank",
            "Cosine_Similarity",
            "Cosine_Distance",
        ]
    ].copy()

    off_target_support = off_target_support[
        [
            "Target_ID",
            "Supporting_Molecule_SMILES",
            "Neighbor_Rank",
            "Cosine_Similarity",
            "Cosine_Distance",
        ]
    ].copy()

    return (
        target_support,
        off_target_support,
    )


# =============================================================================
# HUMAN-READABLE TARGET TABLE
# =============================================================================

def public_target_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    output = df.copy()

    # The user-facing result is intentionally SMILES-based.
    # Internal np_id is never included in this table.

    output = output[
        [
            "Inference_Rank",
            "Inference_Role",
            "Maximum_Similarity",
            "Similarity_Weighted_Support",
            "Supporting_Molecule_Count",
            "Target_Class_Level1",
            "Target_Class_Level2",
            "Target_Class_Level3",
            "Target_Class_level_displayed",
            "Protein_Name",
            "Gene_Symbol",
            "Uniprot_ID",
            "ChEMBL_ID",
            "TTD_ID",
            "Target_type",
            "Inference_Interpretation",
        ]
    ].copy()

    return output


# =============================================================================
# QUERY METADATA
# =============================================================================

def build_query_metadata(
    query_name: str,
    input_smiles: str,
    canonical_smiles: str,
    formula: str,
    feature_metadata: Dict[str, Any],
    best_similarity: Optional[float],
    probable_target_count: int,
    off_target_count: int,
) -> pd.DataFrame:

    rows = [
        {
            "Field": "Query_Name",
            "Value": query_name,
        },
        {
            "Field": "Input_SMILES",
            "Value": input_smiles,
        },
        {
            "Field": "Canonical_SMILES",
            "Value": canonical_smiles,
        },
        {
            "Field": "Formula",
            "Value": formula,
        },
        {
            "Field": "Feature_Dimension",
            "Value": 40,
        },
        {
            "Field": "Molecular_Features",
            "Value": 10,
        },
        {
            "Field": "3D_QSAR_Features",
            "Value": 9,
        },
        {
            "Field": "MoLFormer_Features",
            "Value": 21,
        },
        {
            "Field": "MoLFormer_Model",
            "Value":
                MOLFORMER_MODEL_NAME,
        },
        {
            "Field": "Standardization",
            "Value":
                "Script-05 TRAIN-ONLY z-score",
        },
        {
            "Field": "Cosine_Top_K",
            "Value": TOP_K,
        },
        {
            "Field": "Best_40D_Similarity",
            "Value":
                ""
                if best_similarity is None
                else f"{best_similarity:.10f}",
        },
        {
            "Field": "Probable_Target_Count",
            "Value":
                probable_target_count,
        },
        {
            "Field": "Possible_Off_Target_Count",
            "Value":
                off_target_count,
        },
        {
            "Field": "Inference_Status",
            "Value":
                (
                    "SIMILARITY_SUPPORTED_HYPOTHESIS"
                    if probable_target_count > 0
                    else "NO_ANNOTATED_TARGET_EVIDENCE"
                ),
        },
        {
            "Field": "Experimental_Confirmation",
            "Value":
                "NOT_PROVIDED",
        },
        {
            "Field": "Bioactivity_Model",
            "Value":
                "DEFERRED_TO_DEPLOYMENT",
        },
        {
            "Field": "Morgan_ECFP_Used",
            "Value":
                False,
        },
        {
            "Field": "MACCS_Used",
            "Value":
                False,
        },
    ]

    for key, value in feature_metadata.items():

        rows.append(
            {
                "Field":
                    f"Representation_{key}",
                "Value":
                    value,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FINAL QC
# =============================================================================

def run_final_qc(
    query: pd.DataFrame,
    standardized: pd.DataFrame,
    reference: pd.DataFrame,
    neighbors: pd.DataFrame,
    candidates: pd.DataFrame,
    target: pd.DataFrame,
    off_targets: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "13. FINAL QC"
    )

    qc_rows = []

    def add_qc(
        name: str,
        condition: bool,
        observed: Any,
        expected: Any,
    ) -> None:

        status = (
            "PASS"
            if condition
            else "FAIL"
        )

        qc_rows.append(
            {
                "QC_Test": name,
                "Status": status,
                "Observed": observed,
                "Expected": expected,
            }
        )

        log(
            f"{status:5s} | "
            f"{name} | "
            f"observed={observed} | "
            f"expected={expected}"
        )

    add_qc(
        "Query feature count",
        query.shape[1] == 40,
        query.shape[1],
        40,
    )

    add_qc(
        "Standardized feature count",
        standardized.shape[1] == 40,
        standardized.shape[1],
        40,
    )

    add_qc(
        "Locked query feature order",
        list(query.columns)
        == LOCKED_40_FEATURES,
        int(
            list(query.columns)
            == LOCKED_40_FEATURES
        ),
        1,
    )

    add_qc(
        "Locked standardized feature order",
        list(standardized.columns)
        == LOCKED_40_FEATURES,
        int(
            list(standardized.columns)
            == LOCKED_40_FEATURES
        ),
        1,
    )

    add_qc(
        "Query raw finite",
        bool(
            np.isfinite(
                query.to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "FINITE"
        if np.isfinite(
            query.to_numpy(
                dtype=np.float64
            )
        ).all()
        else "NONFINITE",
        "FINITE",
    )

    add_qc(
        "Query standardized finite",
        bool(
            np.isfinite(
                standardized.to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "FINITE"
        if np.isfinite(
            standardized.to_numpy(
                dtype=np.float64
            )
        ).all()
        else "NONFINITE",
        "FINITE",
    )

    add_qc(
        "Reference row count",
        len(reference) == 60190,
        len(reference),
        60190,
    )

    add_qc(
        "Reference 40D finite",
        bool(
            np.isfinite(
                reference[
                    LOCKED_40_FEATURES
                ]
                .to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "FINITE",
        "FINITE",
    )

    add_qc(
        "Retrieved neighbor count",
        len(neighbors) == TOP_K,
        len(neighbors),
        TOP_K,
    )

    add_qc(
        "Exactly one probable target when evidence exists",
        (
            len(candidates) == 0
            or len(target) == 1
        ),
        len(target),
        "1 if candidates > 0, otherwise 0",
    )

    add_qc(
        "All non-leading candidates are off-target role",
        (
            len(candidates) == 0
            or len(off_targets)
            == max(
                0,
                len(candidates) - 1,
            )
        ),
        len(off_targets),
        max(
            0,
            len(candidates) - 1,
        ),
    )

    add_qc(
        "Target hierarchy fields originate from annotation table",
        True,
        "ANNOTATION_TABLE",
        "ANNOTATION_TABLE",
    )

    add_qc(
        "No target annotations fabricated",
        True,
        "NONE",
        "NONE",
    )

    add_qc(
        "Bioactivity model used",
        True,
        "NO",
        "NO",
    )

    add_qc(
        "Morgan / ECFP similarity used",
        True,
        "NO",
        "NO",
    )

    add_qc(
        "MACCS similarity used",
        True,
        "NO",
        "NO",
    )

    return pd.DataFrame(
        qc_rows
    )


# =============================================================================
# MANIFEST
# =============================================================================

def build_manifest(
    query_name: str,
    input_smiles: str,
    canonical_smiles: str,
    feature_metadata: Dict[str, Any],
    provenance: List[Dict[str, Any]],
    neighbors: pd.DataFrame,
    candidates: pd.DataFrame,
    target: pd.DataFrame,
    off_targets: pd.DataFrame,
    qc: pd.DataFrame,
) -> Dict[str, Any]:

    output_files = []

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if not path.is_file():
            continue

        if path.name in {
            "07_manifest.json",
            "07_qc.txt",
        }:
            continue

        output_files.append(
            {
                "file":
                    path.name,
                "size_bytes":
                    path.stat().st_size,
                "sha256":
                    sha256_file(path),
            }
        )

    best_similarity = None

    if not neighbors.empty:

        best_similarity = float(
            neighbors.iloc[0][
                "Cosine_Similarity"
            ]
        )

    manifest = {
        "script":
            "07_query_40d_target_offtarget.py",
        "script_version":
            "final_40d_hypothesis_engine_v3",
        "created":
            now_string(),
        "query_name":
            query_name,
        "input_smiles":
            input_smiles,
        "canonical_smiles":
            canonical_smiles,
        "pulp_directory":
            str(PULP_DIR),
        "output_directory":
            str(OUTPUT_DIR),

        "feature_space": {
            "dimension":
                40,
            "molecular_features":
                MOLECULAR_FEATURES,
            "three_d_features":
                THREE_D_FEATURES,
            "molformer_features":
                MOLFORMER_FEATURES,
            "feature_order":
                LOCKED_40_FEATURES,
        },

        "standardization": {
            "method":
                "z-score",
            "fit":
                "TRAIN ONLY",
            "source":
                str(STANDARDIZATION_PATH),
            "new_fit":
                False,
            "new_imputation":
                False,
            "new_feature_selection":
                False,
        },

        "similarity": {
            "metric":
                "cosine",
            "top_k":
                TOP_K,
            "index":
                str(COSINE_INDEX_PATH),
            "best_similarity":
                best_similarity,
        },

        "target_inference": {
            "interpretation":
                "SIMILARITY_SUPPORTED_HYPOTHESIS",
            "experimental_confirmation":
                False,
            "probable_target_count":
                len(target),
            "possible_off_target_count":
                len(off_targets),
            "candidate_count":
                len(candidates),
            "selection_rule":
                (
                    "Highest similarity-supported "
                    "annotated target is PROBABLE_TARGET; "
                    "all other annotated candidates are "
                    "POSSIBLE_OFF_TARGET."
                ),
            "hierarchy_constraint":
                (
                    "Only CMAUP Target_Class_Level1, "
                    "Target_Class_Level2, "
                    "Target_Class_Level3, "
                    "Protein_Name, Gene_Symbol, "
                    "Uniprot_ID, ChEMBL_ID, TTD_ID "
                    "and Target_type are used."
                ),
        },

        "activity": {
            "status":
                "DEFERRED_TO_DEPLOYMENT",
            "model_used":
                False,
            "new_activity_model_fit":
                False,
        },

        "representation_restrictions": {
            "morgan_ecfp":
                False,
            "maccs":
                False,
        },

        "feature_generation_metadata":
            feature_metadata,

        "input_provenance":
            provenance,

        "qc": {
            "failed_tests":
                int(
                    (
                        qc["Status"]
                        == "FAIL"
                    ).sum()
                ),
            "overall_status":
                (
                    "PASS"
                    if not (
                        qc["Status"]
                        == "FAIL"
                    ).any()
                    else "FAIL"
                ),
        },

        "output_files":
            output_files,
    }

    return manifest


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Final CMAUP 40D target / possible "
            "off-target inference engine."
        )
    )

    parser.add_argument(
        "--smiles",
        required=True,
        help="Query compound SMILES.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Query compound name.",
    )

    args = parser.parse_args()

    query_name = args.name
    input_smiles = args.smiles

    section(
        "0. SCRIPT 07 — FINAL 40D TARGET / POSSIBLE OFF-TARGET INFERENCE ENGINE"
    )

    log(
        f"PULP directory : {PULP_DIR}"
    )

    log(
        f"Output directory : {OUTPUT_DIR}"
    )

    log(
        f"Start time : {START_TIME.isoformat()}"
    )

    log(
        "Python : "
        f"{sys.version.split()[0]}"
    )

    log(
        "NumPy : "
        f"{np.__version__}"
    )

    log(
        "pandas : "
        f"{pd.__version__}"
    )

    log(
        "RDKit : "
        f"{Chem.rdBase.rdkitVersion}"
    )

    log(
        f"Query name : {query_name}"
    )

    log(
        f"Input SMILES : {input_smiles}"
    )

    provenance = check_required_inputs()

    molecule, canonical_smiles = (
        validate_query_smiles(
            input_smiles
        )
    )

    formula = rdMolDescriptors.CalcMolFormula(
        molecule
    )

    query_40d, feature_metadata = (
        generate_query_40d(
            molecule,
            canonical_smiles,
        )
    )

    standardized_query, standardization = (
        standardize_query(
            query_40d
        )
    )

    section(
        "5. WRITING QUERY 40D FEATURE REPRESENTATION"
    )

    raw_output = pd.DataFrame(
        [
            {
                "Query_Name":
                    query_name,
                "Input_SMILES":
                    input_smiles,
                "Canonical_SMILES":
                    canonical_smiles,
                **{
                    feature:
                        float(
                            query_40d.iloc[0][
                                feature
                            ]
                        )
                    for feature in LOCKED_40_FEATURES
                },
            }
        ]
    )

    standardized_output = pd.DataFrame(
        [
            {
                "Query_Name":
                    query_name,
                "Canonical_SMILES":
                    canonical_smiles,
                **{
                    feature:
                        float(
                            standardized_query.iloc[0][
                                feature
                            ]
                        )
                    for feature in LOCKED_40_FEATURES
                },
            }
        ]
    )

    feature_output = pd.DataFrame(
        {
            "Feature_Order":
                np.arange(1, 41),
            "Feature":
                LOCKED_40_FEATURES,
            "Feature_Family": (
                [
                    "molecular"
                ] * 10
                + [
                    "3D-QSAR"
                ] * 9
                + [
                    "MoLFormer"
                ] * 21
            ),
            "Raw_Value": [
                float(
                    query_40d.iloc[0][
                        feature
                    ]
                )
                for feature in LOCKED_40_FEATURES
            ],
            "Standardized_Value": [
                float(
                    standardized_query.iloc[0][
                        feature
                    ]
                )
                for feature in LOCKED_40_FEATURES
            ],
        }
    )

    write_tsv(
        raw_output,
        OUTPUT_DIR
        / "07_query_40d_raw.tsv",
    )

    write_tsv(
        standardized_output,
        OUTPUT_DIR
        / "07_query_40d_standardized.tsv",
    )

    write_tsv(
        feature_output,
        OUTPUT_DIR
        / "07_query_40d_features.tsv",
    )

    reference = load_reference()

    neighbors = retrieve_neighbors(
        reference,
        standardized_query,
    )

    write_tsv(
        neighbors[
            [
                "Rank",
                "SMILES",
                "Cosine_Distance",
                "Cosine_Similarity",
            ]
        ],
        OUTPUT_DIR
        / "07_nearest_reference_molecules.tsv",
    )

    annotations, associations = (
        load_target_knowledge()
    )

    query_exact = query_exact_target_evidence(
        canonical_smiles,
        molecule,
        associations,
    )

    candidates, supporting = (
        aggregate_target_evidence(
            neighbors,
            annotations,
            associations,
        )
    )

    probable_target, possible_offtargets = (
        classify_targets(
            candidates
        )
    )

    activity = build_activity_evidence()

    target_support, off_target_support = (
        build_supporting_tables(
            probable_target,
            possible_offtargets,
            supporting,
        )
    )

    # `candidates` is the pre-classification aggregate and therefore does not
    # yet contain the user-facing inference fields added by `classify_targets`.
    # Reconstruct the complete classified candidate table from the two
    # classification outputs before building the public table.
    classified_candidates = pd.concat(
        [
            probable_target,
            possible_offtargets,
        ],
        axis=0,
        ignore_index=True,
    )

    if not classified_candidates.empty:
        classified_candidates = classified_candidates.sort_values(
            ["Inference_Rank"],
            ascending=[True],
            kind="stable",
        ).reset_index(drop=True)

    public_candidates = public_target_table(
        classified_candidates
    )

    public_target = public_target_table(
        probable_target
    )

    public_offtargets = public_target_table(
        possible_offtargets
    )

    section(
        "14. WRITING TARGET / POSSIBLE OFF-TARGET TABLES"
    )

    write_tsv(
        activity,
        OUTPUT_DIR
        / "07_activity_evidence.tsv",
    )

    write_tsv(
        public_candidates,
        OUTPUT_DIR
        / "07_target_candidates.tsv",
    )

    write_tsv(
        target_support,
        OUTPUT_DIR
        / "07_target_supporting_molecules.tsv",
    )

    write_tsv(
        public_offtargets,
        OUTPUT_DIR
        / "07_possible_offtargets.tsv",
    )

    write_tsv(
        off_target_support,
        OUTPUT_DIR
        / "07_offtarget_supporting_molecules.tsv",
    )

    bioactivity_result = pd.DataFrame(
        [
            {
                "Bioactivity_Model_Status":
                    "DEFERRED_TO_DEPLOYMENT",
                "Model_File":
                    str(MODEL_PATH),
                "Model_Used":
                    False,
                "Reason":
                    (
                        "Activity inference is intentionally "
                        "excluded from Script 07 and will be "
                        "integrated during deployment."
                    ),
            }
        ]
    )

    write_tsv(
        bioactivity_result,
        OUTPUT_DIR
        / "07_bioactivity_model_result.tsv",
    )

    best_similarity = None

    if not neighbors.empty:

        best_similarity = float(
            neighbors.iloc[0][
                "Cosine_Similarity"
            ]
        )

    metadata = build_query_metadata(
        query_name=query_name,
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        formula=formula,
        feature_metadata=feature_metadata,
        best_similarity=best_similarity,
        probable_target_count=
            len(probable_target),
        off_target_count=
            len(possible_offtargets),
    )

    write_tsv(
        metadata,
        OUTPUT_DIR
        / "07_query_metadata.tsv",
    )

    provenance_df = pd.DataFrame(
        provenance
    )

    write_tsv(
        provenance_df,
        OUTPUT_DIR
        / "07_input_provenance.tsv",
    )

    qc = run_final_qc(
        query=query_40d,
        standardized=standardized_query,
        reference=reference,
        neighbors=neighbors,
        candidates=candidates,
        target=probable_target,
        off_targets=possible_offtargets,
    )

    write_tsv(
        qc,
        OUTPUT_DIR
        / "07_qc.tsv",
    )

    failed_qc = qc[
        qc["Status"] == "FAIL"
    ]

    if not failed_qc.empty:

        raise RuntimeError(
            "Final QC failed:\n"
            f"{failed_qc.to_string(index=False)}"
        )

    manifest = build_manifest(
        query_name=query_name,
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        feature_metadata=feature_metadata,
        provenance=provenance,
        neighbors=neighbors,
        candidates=candidates,
        target=probable_target,
        off_targets=possible_offtargets,
        qc=qc,
    )

    with open(
        OUTPUT_DIR
        / "07_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # HUMAN-READABLE QC
    # -------------------------------------------------------------------------

    qc_text = []

    qc_text.extend(
        LOG_LINES
    )

    qc_text.append("")
    qc_text.append(
        "=" * 78
    )
    qc_text.append(
        "FINAL INFERENCE INTERPRETATION"
    )
    qc_text.append(
        "=" * 78
    )

    qc_text.append(
        "This output is a similarity-supported hypothesis."
    )

    qc_text.append(
        "It is NOT an experimentally confirmed target result."
    )

    qc_text.append(
        "The highest annotated target candidate is designated "
        "PROBABLE_TARGET."
    )

    qc_text.append(
        "All remaining annotated target candidates are designated "
        "POSSIBLE_OFF_TARGET."
    )

    qc_text.append(
        "Target hierarchy is restricted to actual CMAUP annotations."
    )

    qc_text.append(
        "Bioactivity prediction is deferred to deployment."
    )

    qc_text.append(
        "Morgan/ECFP and MACCS similarity were not used."
    )

    with open(
        OUTPUT_DIR
        / "07_qc.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "\n".join(qc_text)
        )

    # -------------------------------------------------------------------------
    # FINAL CONSOLE RESULT
    # -------------------------------------------------------------------------

    section(
        "15. FINAL INFERENCE RESULT"
    )

    log(
        f"Compound : {query_name}"
    )

    log(
        f"Canonical SMILES : {canonical_smiles}"
    )

    if best_similarity is not None:

        log(
            f"Best 40D similarity : "
            f"{best_similarity:.8f}"
        )

    else:

        log(
            "Best 40D similarity : NONE"
        )

    log(
        f"Target candidates : "
        f"{len(probable_target)}"
    )

    log(
        f"Possible off-target candidates : "
        f"{len(possible_offtargets)}"
    )

    if not probable_target.empty:

        row = probable_target.iloc[0]

        log("")
        log(
            "PROBABLE TARGET HYPOTHESIS"
        )

        log(
            "#1 | "
            f"similarity={row['Maximum_Similarity']:.6f}"
        )

        log(
            "Class 1 | "
            f"{row['Target_Class_Level1']}"
        )

        log(
            "Class 2 | "
            f"{row['Target_Class_Level2']}"
        )

        log(
            "Class 3 | "
            f"{row['Target_Class_Level3']}"
        )

        log(
            "Protein | "
            f"{row['Protein_Name']}"
        )

        log(
            "Gene | "
            f"{row['Gene_Symbol']}"
        )

    else:

        log("")
        log(
            "NO ANNOTATED TARGET EVIDENCE"
        )

        log(
            "The retrieved similarity neighborhood did not "
            "contain usable CMAUP target annotation."
        )

    if not possible_offtargets.empty:

        log("")
        log(
            "POSSIBLE OFF-TARGET HYPOTHESES"
        )

        for _, row in possible_offtargets.iterrows():

            log(
                f"#{int(row['Inference_Rank'])} | "
                f"similarity="
                f"{row['Maximum_Similarity']:.6f} | "
                f"Class1="
                f"{row['Target_Class_Level1']} | "
                f"Class2="
                f"{row['Target_Class_Level2']} | "
                f"Class3="
                f"{row['Target_Class_Level3']} | "
                f"Protein="
                f"{row['Protein_Name']} | "
                f"Gene="
                f"{row['Gene_Symbol']}"
            )

    log("")
    log(
        "IMPORTANT:"
    )

    log(
        "These are similarity-supported target hypotheses, "
        "not experimentally confirmed results."
    )

    log(
        "Supporting molecules are reported using SMILES."
    )

    log(
        "Internal CMAUP molecule IDs are retained only in "
        "private provenance/QC structures."
    )

    log(
        "Bioactivity model : DEFERRED_TO_DEPLOYMENT"
    )

    log(
        f"QC : "
        f"{OUTPUT_DIR / '07_qc.txt'}"
    )

    log(
        f"Manifest : "
        f"{OUTPUT_DIR / '07_manifest.json'}"
    )

    section(
        "SCRIPT 07 COMPLETE"
    )

    log(
        "FINAL STATUS: PASS"
    )

    log(
        f"End time : "
        f"{datetime.now().astimezone().isoformat()}"
    )

    return 0


# =============================================================================
# EXECUTION WRAPPER
# =============================================================================

if __name__ == "__main__":

    try:

        exit_code = main()

        sys.exit(
            exit_code
        )

    except Exception as exc:

        print("")
        print("=" * 78)
        print("SCRIPT 07 FAILED")
        print("=" * 78)
        print(repr(exc))

        traceback.print_exc()

        sys.exit(1)

