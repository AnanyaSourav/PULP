
#!/usr/bin/env python3
"""
==============================================================================
SCRIPT 05 — BATCH EXTERNAL VALIDATION
==============================================================================

Purpose
-------
Evaluate the previously trained and calibrated PULP bioactivity model on an
external labelled dataset without retraining or changing its threshold.

Input
-----
labels_and_smiles.txt
Format: label<TAB>SMILES
Labels: 1 = active, 0 = inactive

Model
-----
feature_engineering/model_development/models/
    final_calibrated_bioactivity_model.joblib

Feature representation
----------------------
10 molecular descriptors
9 three-dimensional descriptors
21 selected MoLFormer embedding features
Total: 40 features

Outputs
-------
external_validation/
    02_external_predictions.tsv
    02_external_feature_matrix.tsv
    02_external_metrics.json
    02_external_metrics.txt
    02_confusion_matrix.tsv
    02_canonical_structure_groups.tsv
    02_processing_errors.tsv
    02_external_validation_checkpoint.tsv

IMPORTANT
---------
- No retraining.
- No external-set threshold optimization.
- The saved threshold must equal 0.046.
- All input rows are retained in the prediction output.
==============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import hashlib
import logging
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem import Fragments

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

import torch
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

ROOT = Path("/lustre/home/sklab202/sourav/PULP")

INPUT_FILE = ROOT / "Case_Study_external_validation_phytochemicals"

MODEL_FILE = (
    ROOT
    / "feature_engineering/model_development/models/"
      "final_calibrated_bioactivity_model.joblib"
)

PREPROCESSING_FILE = (
    ROOT
    / "feature_engineering/final_228_features/"
      "preprocessing_statistics.tsv"
)

OUTPUT_DIR = ROOT / "external_validation"

EXPECTED_THRESHOLD = 0.046126477738876355
THRESHOLD_TOLERANCE = 1e-12

RANDOM_SEED = 42
N_CONFORMERS = 3
MAX_OPT_ITERS = 500
MAX_TOKEN_LENGTH = 512

CHECKPOINT_EVERY = 25

MOLFORMER_MODEL = "ibm-research/MoLFormer-XL-both-10pct"

OUTPUT_PREDICTIONS = OUTPUT_DIR / "02_external_predictions.tsv"
OUTPUT_FEATURES = OUTPUT_DIR / "02_external_feature_matrix.tsv"
OUTPUT_METRICS_JSON = OUTPUT_DIR / "02_external_metrics.json"
OUTPUT_METRICS_TXT = OUTPUT_DIR / "02_external_metrics.txt"
OUTPUT_CONFUSION = OUTPUT_DIR / "02_confusion_matrix.tsv"
OUTPUT_GROUPS = OUTPUT_DIR / "02_canonical_structure_groups.tsv"
OUTPUT_ERRORS = OUTPUT_DIR / "02_processing_errors.tsv"
OUTPUT_CHECKPOINT = OUTPUT_DIR / "02_external_validation_checkpoint.tsv"
LOG_FILE = OUTPUT_DIR / "02_external_validation.log"

# Suppress noisy RDKit warnings; exceptions are still recorded.
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger("script05")


# =============================================================================
# 2. EXACT FEATURE NAMES USED BY THE MODEL
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

THREED_FEATURES = [
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

EXPECTED_FEATURES = (
    MOLECULAR_FEATURES + THREED_FEATURES + MOLFORMER_FEATURES
)

assert len(MOLECULAR_FEATURES) == 10
assert len(THREED_FEATURES) == 9
assert len(MOLFORMER_FEATURES) == 21
assert len(EXPECTED_FEATURES) == 40


# =============================================================================
# 3. INPUT READING
# =============================================================================

def read_external_dataset(path: Path) -> pd.DataFrame:
    """
    Read a two-column tab-separated file:
        label<TAB>SMILES

    A header is not required. Blank lines and comment lines beginning with #
    are skipped. Every data record receives a stable 1-based input row ID.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []
    input_errors = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")

            # Also accept whitespace-separated label/SMILES records if a
            # particular line does not contain a tab.
            if len(fields) != 2:
                fields = line.split(None, 1)

            if len(fields) != 2:
                input_errors.append({
                    "source_line": line_number,
                    "raw_line": raw_line.rstrip("\n"),
                    "error": "Expected exactly two fields: label and SMILES",
                })
                continue

            label_text = fields[0].strip()
            smiles = fields[1].strip()

            # Permit a conventional optional header.
            if (
                line_number == 1
                and label_text.lower() in {"label", "activity", "class"}
                and smiles.lower() in {"smiles", "canonical_smiles"}
            ):
                continue

            try:
                label = int(label_text)
            except ValueError:
                input_errors.append({
                    "source_line": line_number,
                    "raw_line": raw_line.rstrip("\n"),
                    "error": f"Invalid label: {label_text!r}",
                })
                continue

            if label not in (0, 1):
                input_errors.append({
                    "source_line": line_number,
                    "raw_line": raw_line.rstrip("\n"),
                    "error": f"Label must be 0 or 1; received {label}",
                })
                continue

            if not smiles:
                input_errors.append({
                    "source_line": line_number,
                    "raw_line": raw_line.rstrip("\n"),
                    "error": "Empty SMILES",
                })
                continue

            records.append({
                "input_row_id": len(records) + 1,
                "source_line": line_number,
                "true_label": label,
                "original_smiles": smiles,
            })

    if input_errors:
        pd.DataFrame(input_errors).to_csv(
            OUTPUT_DIR / "02_input_read_errors.tsv",
            sep="\t",
            index=False,
        )

    if not records:
        raise ValueError("No valid labelled SMILES records were found.")

    df = pd.DataFrame(records)

    log.info("Valid input rows: %d", len(df))
    log.info("Active labels (1): %d", int((df["true_label"] == 1).sum()))
    log.info("Inactive labels (0): %d", int((df["true_label"] == 0).sum()))
    log.info("Input parsing errors: %d", len(input_errors))

    return df


# =============================================================================
# 4. MODEL AND PREPROCESSING LOADING
# =============================================================================

def load_model_bundle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise TypeError(
            "Expected the saved model artifact to be a dictionary."
        )

    required = {"model", "feature_order", "threshold"}
    missing = required - set(artifact.keys())

    if missing:
        raise KeyError(
            f"Model artifact is missing required keys: {sorted(missing)}"
        )

    model = artifact["model"]
    feature_order = list(artifact["feature_order"])
    threshold = float(artifact["threshold"])

    if not np.isfinite(threshold):
        raise ValueError(f"Saved threshold is not finite: {threshold}")

    if not math.isclose(
        threshold,
        EXPECTED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=THRESHOLD_TOLERANCE,
    ):
        raise ValueError(
            "Threshold verification failed. "
            f"Expected {EXPECTED_THRESHOLD}, found {threshold}. "
            "Validation stopped; inspect the saved artifact before proceeding."
        )

    if len(feature_order) != 40:
        raise ValueError(
            f"Expected 40 saved features, found {len(feature_order)}."
        )

    if len(set(feature_order)) != len(feature_order):
        raise ValueError("The saved feature_order contains duplicate names.")

    if set(feature_order) != set(EXPECTED_FEATURES):
        missing_features = sorted(set(feature_order) - set(EXPECTED_FEATURES))
        unexpected_features = sorted(set(EXPECTED_FEATURES) - set(feature_order))

        raise ValueError(
            "Feature definitions do not match the saved model.\n"
            f"Features in artifact but not in this script: {missing_features}\n"
            f"Features in this script but not in artifact: {unexpected_features}"
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError("Saved model does not support predict_proba().")

    model_classes = list(model.classes_)

    if 1 not in model_classes or 0 not in model_classes:
        raise ValueError(
            f"Expected binary classes [0, 1]; found {model_classes}"
        )

    if getattr(model, "n_features_in_", 40) != 40:
        raise ValueError(
            f"Model expects {model.n_features_in_} features, not 40."
        )

    log.info("Model: %s", artifact.get("model_name", type(model).__name__))
    log.info("Estimator type: %s", type(model).__name__)
    log.info("Feature count: %d", len(feature_order))
    log.info("Saved classification threshold: %.12f", threshold)
    log.info("Model classes: %s", model_classes)

    return artifact, model, feature_order, threshold


def load_training_medians(path: Path) -> dict:
    """
    Load imputation medians from the training preprocessing file.
    Never calculate medians from the external validation data.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Training preprocessing statistics not found: {path}"
        )

    stats = pd.read_csv(path, sep="\t")

    if "Feature" not in stats.columns:
        raise ValueError(
            "Preprocessing statistics must contain a 'Feature' column."
        )

    median_col = None

    for candidate in ("Train_Median", "TRAIN_Median"):
        if candidate in stats.columns:
            median_col = candidate
            break

    if median_col is None:
        raise ValueError(
            "Could not find Train_Median or TRAIN_Median in preprocessing TSV."
        )

    medians = {}

    for _, row in stats.iterrows():
        name = str(row["Feature"])
        try:
            value = float(row[median_col])
        except (TypeError, ValueError):
            continue

        if np.isfinite(value):
            medians[name] = value

    missing = sorted(set(MOLECULAR_FEATURES + THREED_FEATURES) - set(medians))

    if missing:
        raise ValueError(
            "Training medians are missing for handcrafted descriptors: "
            + ", ".join(missing)
        )

    log.info("Loaded %d training imputation medians.", len(medians))

    return medians


# =============================================================================
# 5. SMILES AND MOLECULAR DESCRIPTORS
# =============================================================================

def prepare_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("RDKit could not parse this SMILES.")

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )

    if not canonical_smiles:
        raise ValueError("Could not generate canonical isomeric SMILES.")

    return mol, canonical_smiles


def calculate_molecular_features(mol) -> dict:
    """
    Calculate the ten molecular descriptors specified by the saved model.
    """

    feature_values = {}

    feature_values["FractionCSP3"] = (
        rdMolDescriptors.CalcFractionCSP3(mol)
    )

    # SPS is part of RDKit's descriptor collection in compatible versions.
    if not hasattr(Descriptors, "SPS"):
        raise AttributeError(
            "This RDKit installation does not provide Descriptors.SPS. "
            "Use the same RDKit environment as the original inference script."
        )

    feature_values["SPS"] = Descriptors.SPS(mol)

    feature_values["NumAtomStereoCenters"] = (
        rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    )

    feature_values["SMR_VSA5"] = Descriptors.SMR_VSA5(mol)
    feature_values["BCUT2D_MRLOW"] = Descriptors.BCUT2D_MRLOW(mol)
    feature_values["BCUT2D_CHGLO"] = Descriptors.BCUT2D_CHGLO(mol)
    feature_values["VSA_EState6"] = Descriptors.VSA_EState6(mol)
    feature_values["NumAromaticRings"] = (
        rdMolDescriptors.CalcNumAromaticRings(mol)
    )
    feature_values["SMR_VSA7"] = Descriptors.SMR_VSA7(mol)

    # RDKit fragment descriptor naming varies slightly by version.
    fragment_function = getattr(Fragments, "fr_Al_OH", None)

    if fragment_function is None:
        raise AttributeError(
            "This RDKit installation does not provide Fragments.fr_Al_OH."
        )

    feature_values["fr_Al_OH"] = fragment_function(mol)

    return feature_values


# =============================================================================
# 6. THREE-DIMENSIONAL CONFORMER GENERATION AND DESCRIPTORS
# =============================================================================

def generate_best_conformer(mol):
    """
    Generate up to three conformers and select the lowest finite MMFF94s
    energy. Fall back to UFF if MMFF is unavailable or no usable MMFF energy
    is obtained.
    """

    mol_h = Chem.AddHs(Chem.Mol(mol))

    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.pruneRmsThresh = 0.5
    params.numThreads = 0

    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            mol_h,
            numConfs=N_CONFORMERS,
            params=params,
        )
    )

    if not conformer_ids:
        raise RuntimeError("RDKit could not generate a 3D conformer.")

    mmff_props = AllChem.MMFFGetMoleculeProperties(
        mol_h,
        mmffVariant="MMFF94s",
    )

    candidates = []

    if mmff_props is not None:
        for conf_id in conformer_ids:
            try:
                result = AllChem.MMFFOptimizeMolecule(
                    mol_h,
                    mmffVariant="MMFF94s",
                    maxIters=MAX_OPT_ITERS,
                    confId=int(conf_id),
                )

                if result < 0:
                    continue

                ff = AllChem.MMFFGetMoleculeForceField(
                    mol_h,
                    mmff_props,
                    confId=int(conf_id),
                )

                if ff is None:
                    continue

                energy = float(ff.CalcEnergy())

                if np.isfinite(energy):
                    candidates.append((energy, int(conf_id)))

            except Exception:
                continue

    method = "MMFF94s"

    if not candidates:
        method = "UFF"
        candidates = []

        for conf_id in conformer_ids:
            try:
                result = AllChem.UFFOptimizeMolecule(
                    mol_h,
                    maxIters=MAX_OPT_ITERS,
                    confId=int(conf_id),
                )

                if result < 0:
                    continue

                ff = AllChem.UFFGetMoleculeForceField(
                    mol_h,
                    confId=int(conf_id),
                )

                if ff is None:
                    continue

                energy = float(ff.CalcEnergy())

                if np.isfinite(energy):
                    candidates.append((energy, int(conf_id)))

            except Exception:
                continue

    if not candidates:
        raise RuntimeError(
            "Neither MMFF94s nor UFF produced a usable optimized conformer."
        )

    best_energy, best_conf_id = min(candidates, key=lambda item: item[0])

    best_mol = Chem.Mol(mol_h)
    best_conf = best_mol.GetConformer(best_conf_id)

    # Preserve only the selected conformer in the returned molecule.
    selected_mol = Chem.Mol(best_mol)
    selected_mol.RemoveAllConformers()
    selected_mol.AddConformer(Chem.Conformer(best_conf), assignId=True)

    return selected_mol, method, float(best_energy), len(conformer_ids)


def calculate_3d_features(mol3d) -> dict:
    """
    Calculate the nine 3D descriptors using RDKit.
    """

    feature_values = {}

    # These functions require a molecule containing 3D coordinates.
    feature_values["PBF"] = rdMolDescriptors.CalcPBF(mol3d)
    feature_values["SpherocityIndex"] = (
        rdMolDescriptors.CalcSpherocityIndex(mol3d)
    )
    feature_values["PMI1"] = rdMolDescriptors.CalcPMI1(mol3d)
    feature_values["InertialShapeFactor"] = (
        rdMolDescriptors.CalcInertialShapeFactor(mol3d)
    )
    feature_values["PMI2"] = rdMolDescriptors.CalcPMI2(mol3d)
    feature_values["NPR2"] = rdMolDescriptors.CalcNPR2(mol3d)
    feature_values["RadiusOfGyration"] = (
        rdMolDescriptors.CalcRadiusOfGyration(mol3d)
    )
    feature_values["Eccentricity"] = (
        rdMolDescriptors.CalcEccentricity(mol3d)
    )
    feature_values["NPR1"] = rdMolDescriptors.CalcNPR1(mol3d)

    return feature_values


def clean_handcrafted_features(features: dict, medians: dict):
    """
    Replace missing/non-finite handcrafted descriptor values with their
    corresponding training medians.
    """

    cleaned = {}
    imputed_names = []

    for name in MOLECULAR_FEATURES + THREED_FEATURES:
        value = features.get(name, np.nan)

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan

        if not np.isfinite(value):
            if name not in medians:
                raise ValueError(
                    f"Cannot impute {name}: no training median is available."
                )

            value = float(medians[name])
            imputed_names.append(name)

        cleaned[name] = value

    return cleaned, imputed_names


# =============================================================================
# 7. MoLFORMER INITIALIZATION AND EMBEDDING
# =============================================================================

def load_molformer():
    """
    Load the same pretrained MoLFormer model used by the single-SMILES
    inference workflow.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    log.info("Loading MoLFormer: %s", MOLFORMER_MODEL)
    log.info("MoLFormer device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_MODEL,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        MOLFORMER_MODEL,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def generate_molformer_embedding(
    canonical_smiles: str,
    tokenizer,
    model,
    device,
) -> np.ndarray:
    """
    Generate a 768-dimensional masked-mean-pooled MoLFormer embedding.
    """

    encoded = tokenizer(
        canonical_smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)

        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)

        masked_hidden = hidden * attention_mask
        token_sum = masked_hidden.sum(dim=1)

        token_count = attention_mask.sum(dim=1).clamp(min=1)
        embedding = token_sum / token_count

    embedding = embedding[0].detach().cpu().numpy().astype(np.float32)

    if embedding.shape != (768,):
        raise ValueError(
            f"Expected 768-dimensional embedding; got {embedding.shape}."
        )

    if not np.isfinite(embedding).all():
        raise ValueError("MoLFormer embedding contains non-finite values.")

    return embedding


def select_molformer_features(embedding: np.ndarray) -> dict:
    """
    Map the saved molformer_#### feature names to their embedding indices.
    """

    selected = {}

    for feature_name in MOLFORMER_FEATURES:
        try:
            embedding_index = int(feature_name.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Unexpected MoLFormer feature name: {feature_name}"
            ) from exc

        if not 0 <= embedding_index < len(embedding):
            raise IndexError(
                f"Embedding index {embedding_index} is out of range."
            )

        selected[feature_name] = float(embedding[embedding_index])

    return selected


# =============================================================================
# 8. COMPLETE FEATURE GENERATION FOR ONE INPUT ROW
# =============================================================================

def featurize_one_smiles(
    smiles: str,
    medians: dict,
    tokenizer,
    molformer_model,
    device,
):
    """
    Return:
        canonical_smiles,
        complete 40-feature dictionary,
        metadata dictionary
    """

    mol, canonical_smiles = prepare_molecule(smiles)

    molecular = calculate_molecular_features(mol)

    mol3d, force_field, energy, conformer_count = (
        generate_best_conformer(mol)
    )

    three_d = calculate_3d_features(mol3d)

    handcrafted = {}
    handcrafted.update(molecular)
    handcrafted.update(three_d)

    handcrafted, imputed = clean_handcrafted_features(
        handcrafted,
        medians,
    )

    embedding = generate_molformer_embedding(
        canonical_smiles,
        tokenizer,
        molformer_model,
        device,
    )

    molformer_features = select_molformer_features(embedding)

    complete_features = {}
    complete_features.update(handcrafted)
    complete_features.update(molformer_features)

    if set(complete_features) != set(EXPECTED_FEATURES):
        raise ValueError(
            "Generated features do not match the expected 40-feature set."
        )

    for name, value in complete_features.items():
        if not np.isfinite(float(value)):
            raise ValueError(
                f"Feature {name} remains non-finite after preprocessing."
            )

    metadata = {
        "conformer_count": int(conformer_count),
        "force_field": force_field,
        "best_conformer_energy": float(energy),
        "imputed_features": ",".join(imputed),
    }

    return canonical_smiles, complete_features, metadata


# =============================================================================
# 9. METRICS
# =============================================================================

def calculate_metrics(y_true, y_pred, p_active):
    """
    Calculate binary classification metrics at the saved threshold.
    AUROC/AUPRC use probabilities, not hard class predictions.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p_active = np.asarray(p_active, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        float(tn / (tn + fp))
        if (tn + fp) > 0
        else None
    )

    sensitivity = (
        float(tp / (tp + fn))
        if (tp + fn) > 0
        else None
    )

    metrics = {
        "n_evaluated": int(len(y_true)),
        "n_active_true": int((y_true == 1).sum()),
        "n_inactive_true": int((y_true == 0).sum()),
        "n_predicted_active": int((y_pred == 1).sum()),
        "n_predicted_inactive": int((y_pred == 0).sum()),
        "threshold": EXPECTED_THRESHOLD,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(y_true, y_pred)
        ),
        "brier_score": float(brier_score_loss(y_true, p_active)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, p_active))
        metrics["average_precision_auprc"] = float(
            average_precision_score(y_true, p_active)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision_auprc"] = None

    return metrics


def safe_json_value(value):
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


# =============================================================================
# 10. CANONICAL STRUCTURE GROUPING
# =============================================================================

def make_canonical_group_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize predictions by canonical isomeric SMILES.

    This report preserves label conflicts rather than silently resolving them.
    The main row-level metrics remain the primary external validation result.
    """

    successful = predictions[
        (predictions["status"] == "predicted")
        & predictions["canonical_smiles"].notna()
    ].copy()

    if successful.empty:
        return pd.DataFrame()

    grouped_rows = []

    for canonical, group in successful.groupby(
        "canonical_smiles",
        sort=False,
        dropna=False,
    ):
        labels = sorted(group["true_label"].astype(int).unique())
        probabilities = group["p_active"].astype(float).to_numpy()

        grouped_rows.append({
            "canonical_smiles": canonical,
            "n_input_rows": int(len(group)),
            "true_labels_present": ",".join(map(str, labels)),
            "label_conflict": bool(len(labels) > 1),
            "consistent_true_label": (
                int(labels[0]) if len(labels) == 1 else np.nan
            ),
            "mean_p_active": float(np.mean(probabilities)),
            "min_p_active": float(np.min(probabilities)),
            "max_p_active": float(np.max(probabilities)),
            "predicted_classes_present": ",".join(
                map(
                    str,
                    sorted(group["predicted_class"].astype(int).unique()),
                )
            ),
            "input_row_ids": ",".join(
                map(str, group["input_row_id"].astype(int).tolist())
            ),
        })

    return pd.DataFrame(grouped_rows)


# =============================================================================
# 11. MAIN VALIDATION WORKFLOW
# =============================================================================

def main():
    start_time = time.time()
    start_iso = datetime.now().astimezone().isoformat()

    log.info("=" * 78)
    log.info("SCRIPT 05 — BATCH EXTERNAL VALIDATION")
    log.info("=" * 78)
    log.info("Project root: %s", ROOT)
    log.info("Input file: %s", INPUT_FILE)
    log.info("Output directory: %s", OUTPUT_DIR)
    log.info("Start time: %s", start_iso)

    # -------------------------------------------------------------------------
    # Load input, model and training preprocessing values
    # -------------------------------------------------------------------------

    df = read_external_dataset(INPUT_FILE)

    artifact, model, feature_order, threshold = load_model_bundle(
        MODEL_FILE
    )

    medians = load_training_medians(PREPROCESSING_FILE)

    # Checkpoint output is deliberately separate from final predictions.
    # This initial implementation starts a fresh run; an existing checkpoint
    # is not automatically trusted or loaded as a completed result.

    tokenizer, molformer_model, device = load_molformer()

    # -------------------------------------------------------------------------
    # Result storage
    # -------------------------------------------------------------------------

    prediction_rows = []
    feature_rows = []
    error_rows = []

    # Exact raw SMILES cache: repeated literal input strings reuse the same
    # computed features, reducing unnecessary conformer/embedding work.
    #
    # Different SMILES strings that canonicalize to the same structure are
    # processed independently to avoid silently changing the original
    # single-molecule inference procedure.
    feature_cache = {}

    total = len(df)

    for position, record in enumerate(df.to_dict("records"), start=1):
        row_id = int(record["input_row_id"])
        true_label = int(record["true_label"])
        original_smiles = str(record["original_smiles"])

        log.info(
            "Processing row %d/%d (input_row_id=%d)",
            position,
            total,
            row_id,
        )

        result = {
            "input_row_id": row_id,
            "source_line": int(record["source_line"]),
            "true_label": true_label,
            "original_smiles": original_smiles,
            "canonical_smiles": None,
            "p_active": np.nan,
            "predicted_class": np.nan,
            "threshold": threshold,
            "status": "failed",
            "error": "",
            "conformer_count": np.nan,
            "force_field": "",
            "best_conformer_energy": np.nan,
            "imputed_features": "",
        }

        try:
            # Reuse features only for exact repeated raw SMILES.
            if original_smiles in feature_cache:
                canonical_smiles, complete_features, metadata = (
                    feature_cache[original_smiles]
                )
            else:
                canonical_smiles, complete_features, metadata = (
                    featurize_one_smiles(
                        original_smiles,
                        medians,
                        tokenizer,
                        molformer_model,
                        device,
                    )
                )

                feature_cache[original_smiles] = (
                    canonical_smiles,
                    complete_features,
                    metadata,
                )

            # Build features in the exact order saved with the model.
            feature_vector = [
                float(complete_features[name])
                for name in feature_order
            ]

            X_one = pd.DataFrame(
                [feature_vector],
                columns=feature_order,
            )

            if X_one.shape != (1, 40):
                raise ValueError(
                    f"Unexpected feature matrix shape: {X_one.shape}"
                )

            if not np.isfinite(X_one.to_numpy(dtype=float)).all():
                raise ValueError(
                    "The final model input contains non-finite values."
                )

            probability_matrix = model.predict_proba(X_one)

            classes = list(model.classes_)
            active_index = classes.index(1)

            p_active = float(probability_matrix[0, active_index])

            if not np.isfinite(p_active):
                raise ValueError("Model returned a non-finite probability.")

            # Match the existing single-SMILES inference rule exactly.
            predicted_class = int(p_active >= threshold)

            result.update({
                "canonical_smiles": canonical_smiles,
                "p_active": p_active,
                "predicted_class": predicted_class,
                "status": "predicted",
                "error": "",
                "conformer_count": metadata["conformer_count"],
                "force_field": metadata["force_field"],
                "best_conformer_energy": metadata[
                    "best_conformer_energy"
                ],
                "imputed_features": metadata["imputed_features"],
            })

            feature_row = {
                "input_row_id": row_id,
                "source_line": int(record["source_line"]),
                "true_label": true_label,
                "original_smiles": original_smiles,
                "canonical_smiles": canonical_smiles,
            }

            # The 40 feature columns retain the exact model feature order.
            for feature_name in feature_order:
                feature_row[feature_name] = float(
                    complete_features[feature_name]
                )

            feature_rows.append(feature_row)

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            result["error"] = error_message

            error_rows.append({
                "input_row_id": row_id,
                "source_line": int(record["source_line"]),
                "true_label": true_label,
                "original_smiles": original_smiles,
                "error": error_message,
                "traceback": traceback.format_exc(),
            })

            log.exception(
                "Failed row %d (input_row_id=%d)",
                position,
                row_id,
            )

        prediction_rows.append(result)

        # Write checkpoints regularly so progress survives interruption.
        if position % CHECKPOINT_EVERY == 0 or position == total:
            checkpoint_df = pd.DataFrame(prediction_rows)
            checkpoint_df.to_csv(
                OUTPUT_CHECKPOINT,
                sep="\t",
                index=False,
            )

            pd.DataFrame(error_rows).to_csv(
                OUTPUT_ERRORS,
                sep="\t",
                index=False,
            )

            log.info(
                "Checkpoint saved: %d/%d rows; successful=%d; failed=%d",
                position,
                total,
                sum(r["status"] == "predicted" for r in prediction_rows),
                sum(r["status"] == "failed" for r in prediction_rows),
            )

    # -------------------------------------------------------------------------
    # Save final row-level predictions and feature matrix
    # -------------------------------------------------------------------------

    predictions = pd.DataFrame(prediction_rows)
    features_df = pd.DataFrame(feature_rows)

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        sep="\t",
        index=False,
    )

    if not features_df.empty:
        features_df.to_csv(
            OUTPUT_FEATURES,
            sep="\t",
            index=False,
        )
    else:
        # Save an empty but correctly named feature matrix.
        feature_columns = [
            "input_row_id",
            "source_line",
            "true_label",
            "original_smiles",
            "canonical_smiles",
        ] + feature_order

        pd.DataFrame(columns=feature_columns).to_csv(
            OUTPUT_FEATURES,
            sep="\t",
            index=False,
        )

    pd.DataFrame(error_rows).to_csv(
        OUTPUT_ERRORS,
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # External-validation metrics on all successfully predicted input rows
    # -------------------------------------------------------------------------

    successful = predictions[
        predictions["status"] == "predicted"
    ].copy()

    if successful.empty:
        raise RuntimeError(
            "No rows were successfully predicted. "
            "Check the log and processing-error file."
        )

    y_true = successful["true_label"].astype(int).to_numpy()
    y_pred = successful["predicted_class"].astype(int).to_numpy()
    p_active = successful["p_active"].astype(float).to_numpy()

    row_metrics = calculate_metrics(y_true, y_pred, p_active)

    # -------------------------------------------------------------------------
    # Canonical-structure grouping and label conflict reporting
    # -------------------------------------------------------------------------

    group_table = make_canonical_group_table(predictions)

    group_table.to_csv(
        OUTPUT_GROUPS,
        sep="\t",
        index=False,
    )

    if not group_table.empty:
        n_unique_canonical = int(len(group_table))
        n_conflict_groups = int(group_table["label_conflict"].sum())
        n_consistent_groups = int(
            (~group_table["label_conflict"]).sum()
        )
    else:
        n_unique_canonical = 0
        n_conflict_groups = 0
        n_consistent_groups = 0

    # A separate molecule-level result uses one mean probability per
    # canonical structure, and only structures whose observed labels agree.
    # This avoids assigning an arbitrary ground-truth label to conflicting
    # duplicate structures.
    molecule_metrics = None

    if not group_table.empty:
        consistent_groups = group_table[
            ~group_table["label_conflict"]
        ].copy()

        if (
            not consistent_groups.empty
            and consistent_groups["consistent_true_label"].nunique() == 2
        ):
            group_true = (
                consistent_groups["consistent_true_label"]
                .astype(int)
                .to_numpy()
            )
            group_prob = (
                consistent_groups["mean_p_active"]
                .astype(float)
                .to_numpy()
            )
            group_pred = (group_prob >= threshold).astype(int)

            molecule_metrics = calculate_metrics(
                group_true,
                group_pred,
                group_prob,
            )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual_inactive_0", "Actual_active_1"],
        columns=["Predicted_inactive_0", "Predicted_active_1"],
    )

    confusion_df.to_csv(
        OUTPUT_CONFUSION,
        sep="\t",
        index=True,
        index_label="actual_class",
    )

    # -------------------------------------------------------------------------
    # Summary metadata
    # -------------------------------------------------------------------------

    elapsed_seconds = time.time() - start_time

    metrics_bundle = {
        "validation_run": {
            "script": "05_batch_external_validation.py",
            "start_time": start_iso,
            "end_time": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": float(elapsed_seconds),
            "project_root": str(ROOT),
            "input_file": str(INPUT_FILE),
            "model_file": str(MODEL_FILE),
            "preprocessing_file": str(PREPROCESSING_FILE),
            "output_directory": str(OUTPUT_DIR),
            "random_seed": RANDOM_SEED,
            "n_conformers": N_CONFORMERS,
            "molformer_model": MOLFORMER_MODEL,
            "device": str(device),
        },
        "model": {
            "model_name": artifact.get("model_name"),
            "estimator_type": type(model).__name__,
            "calibration": artifact.get("calibration"),
            "saved_threshold": float(threshold),
            "expected_threshold": float(EXPECTED_THRESHOLD),
            "threshold_verified": True,
            "feature_count": int(len(feature_order)),
            "feature_order": feature_order,
            "model_classes": list(map(int, model.classes_)),
            "training_n": artifact.get("training_n"),
            "training_active": artifact.get("training_active"),
            "training_inactive": artifact.get("training_inactive"),
            "representation": artifact.get("representation"),
        },
        "input_audit": {
            "n_valid_input_rows": int(len(df)),
            "n_successfully_predicted": int(len(successful)),
            "n_failed_predictions": int(
                (predictions["status"] != "predicted").sum()
            ),
            "n_active_labels": int((df["true_label"] == 1).sum()),
            "n_inactive_labels": int((df["true_label"] == 0).sum()),
            "n_unique_raw_smiles": int(df["original_smiles"].nunique()),
            "n_unique_canonical_smiles_predicted": n_unique_canonical,
            "n_consistent_canonical_groups": n_consistent_groups,
            "n_conflicting_canonical_groups": n_conflict_groups,
        },
        "row_level_external_validation": row_metrics,
        "canonical_molecule_level_validation": {
            "description": (
                "One mean probability per canonical structure; groups with "
                "conflicting observed labels are excluded. This is a "
                "secondary analysis, not the primary row-level result."
            ),
            "metrics": molecule_metrics,
            "n_canonical_groups_excluded_for_label_conflict": (
                n_conflict_groups
            ),
        },
        "output_files": {
            "predictions": str(OUTPUT_PREDICTIONS),
            "feature_matrix": str(OUTPUT_FEATURES),
            "metrics_json": str(OUTPUT_METRICS_JSON),
            "metrics_text": str(OUTPUT_METRICS_TXT),
            "confusion_matrix": str(OUTPUT_CONFUSION),
            "canonical_structure_groups": str(OUTPUT_GROUPS),
            "processing_errors": str(OUTPUT_ERRORS),
            "checkpoint": str(OUTPUT_CHECKPOINT),
            "log": str(LOG_FILE),
        },
    }

    with OUTPUT_METRICS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(
            safe_json_value(metrics_bundle),
            handle,
            indent=2,
            allow_nan=False,
        )

    # Human-readable report
    lines = [
        "=" * 78,
        "SCRIPT 05 — BATCH EXTERNAL VALIDATION REPORT",
        "=" * 78,
        "",
        f"Input file: {INPUT_FILE}",
        f"Model file: {MODEL_FILE}",
        f"Preprocessing file: {PREPROCESSING_FILE}",
        f"Start time: {start_iso}",
        f"Elapsed seconds: {elapsed_seconds:.2f}",
        "",
        "MODEL SETTINGS",
        "-" * 78,
        f"Model: {artifact.get('model_name', type(model).__name__)}",
        f"Estimator: {type(model).__name__}",
        f"Threshold: {threshold:.12f}",
        "Threshold verified against 0.046: YES",
        f"Feature count: {len(feature_order)}",
        "",
        "INPUT AND PROCESSING SUMMARY",
        "-" * 78,
        f"Valid input rows: {len(df)}",
        f"Successfully predicted: {len(successful)}",
        f"Failed predictions: {(predictions['status'] != 'predicted').sum()}",
        f"Input active labels: {(df['true_label'] == 1).sum()}",
        f"Input inactive labels: {(df['true_label'] == 0).sum()}",
        f"Unique raw SMILES: {df['original_smiles'].nunique()}",
        f"Unique canonical structures predicted: {n_unique_canonical}",
        f"Consistent canonical groups: {n_consistent_groups}",
        f"Canonical groups with conflicting labels: {n_conflict_groups}",
        "",
        "PRIMARY RESULTS — ROW-LEVEL EXTERNAL VALIDATION",
        "-" * 78,
    ]

    for key, value in row_metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")

    lines.extend([
        "",
        "CONFUSION MATRIX",
        "-" * 78,
        f"TN: {int(cm[0, 0])}",
        f"FP: {int(cm[0, 1])}",
        f"FN: {int(cm[1, 0])}",
        f"TP: {int(cm[1, 1])}",
        "",
        "SECONDARY RESULTS — CANONICAL MOLECULE-LEVEL ANALYSIS",
        "-" * 78,
        "Conflicting-label canonical groups are excluded from this analysis.",
    ])

    if molecule_metrics is None:
        lines.append(
            "Metrics not calculated: insufficient consistent canonical "
            "groups or only one true class remains."
        )
    else:
        for key, value in molecule_metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")

    lines.extend([
        "",
        "OUTPUT FILES",
        "-" * 78,
        f"Predictions: {OUTPUT_PREDICTIONS}",
        f"40-feature matrix: {OUTPUT_FEATURES}",
        f"Metrics JSON: {OUTPUT_METRICS_JSON}",
        f"Confusion matrix: {OUTPUT_CONFUSION}",
        f"Canonical groups: {OUTPUT_GROUPS}",
        f"Processing errors: {OUTPUT_ERRORS}",
        f"Checkpoint: {OUTPUT_CHECKPOINT}",
        f"Log: {LOG_FILE}",
        "",
        "Interpretation note: all row-level metrics include repeated input rows",
        "that were successfully predicted. Duplicate and conflicting-label",
        "structures are documented separately; no rows were silently removed",
        "from the prediction table.",
        "=" * 78,
    ])

    OUTPUT_METRICS_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Final console summary
    # -------------------------------------------------------------------------

    log.info("=" * 78)
    log.info("EXTERNAL VALIDATION FINISHED")
    log.info("Successfully predicted: %d / %d", len(successful), len(df))
    log.info(
        "Failed rows: %d",
        int((predictions["status"] != "predicted").sum()),
    )
    log.info("Threshold: %.6f", threshold)
    log.info("Accuracy: %.4f", row_metrics["accuracy"])
    log.info("Balanced accuracy: %.4f", row_metrics["balanced_accuracy"])
    log.info("ROC-AUC: %s", row_metrics["roc_auc"])
    log.info("AUPRC: %s", row_metrics["average_precision_auprc"])
    log.info("MCC: %.4f", row_metrics["matthews_correlation_coefficient"])
    log.info("Predictions: %s", OUTPUT_PREDICTIONS)
    log.info("Metrics: %s", OUTPUT_METRICS_TXT)
    log.info("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.exception("SCRIPT 05 FAILED: %s", exc)
        raise

