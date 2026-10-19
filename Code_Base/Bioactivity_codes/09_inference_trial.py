#!/usr/bin/env python3

# =============================================================================
# SCRIPT 09 — TRIAL INFERENCE FOR NEW PHYTOCHEMICAL
# =============================================================================
#
# INPUT:
#     One SMILES string
#
# EXAMPLE:
#     python 09_inference_trial.py \
#         --smiles "CCO" \
#         --name "Ethanol"
#
# OUTPUT:
#     inference_trial/
#       prediction_summary.tsv
#       molecular_features.tsv
#       qsar_3d_features.tsv
#       molformer_768_embedding.tsv
#       molformer_selected_21.tsv
#       final_40_feature_matrix.tsv
#       inference_report.json
#
# FINAL MODEL:
#     Extra Trees
#     Platt/sigmoid calibrated
#
# FINAL REPRESENTATION:
#     10 molecular
#      9 3D-QSAR
#     21 MoLFormer
#     ----------------
#     40 predictors
#
# IMPORTANT:
#     This is a screening / hypothesis-generation model.
#     It does not experimentally determine bioactivity.
#
# =============================================================================


import os
import sys
import json
import math
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors

import torch

from transformers import AutoTokenizer
from transformers import AutoModel


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_FILE = (
    "feature_engineering/model_development/models/"
    "final_calibrated_bioactivity_model.joblib"
)

PREPROCESSING_FILE = (
    "feature_engineering/final_228_features/"
    "preprocessing_statistics.tsv"
)

MOLFORMER_MODEL = (
    "ibm-research/MoLFormer-XL-both-10pct"
)

OUTPUT_DIR = "inference_trial"

RANDOM_SEED = 42

N_CONFORMERS = 3

MAX_OPT_ITERS = 500

MAX_TOKEN_LENGTH = 512


# =============================================================================
# SELECTED FEATURES
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


# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(text):

    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def section(text):

    print()
    print("-" * 78)
    print(text)
    print("-" * 78)


# =============================================================================
# LOAD TRAINING MEDIANS
# =============================================================================

def load_training_medians():

    if not os.path.exists(PREPROCESSING_FILE):

        raise FileNotFoundError(
            f"Missing preprocessing file:\n{PREPROCESSING_FILE}"
        )

    df = pd.read_csv(
        PREPROCESSING_FILE,
        sep="\t"
    )

    if "Feature" not in df.columns:

        raise RuntimeError(
            "Feature column missing from preprocessing statistics."
        )

    if "Train_Median" in df.columns:

        median_col = "Train_Median"

    elif "TRAIN_Median" in df.columns:

        median_col = "TRAIN_Median"

    else:

        raise RuntimeError(
            "Could not identify training median column."
        )

    medians = {}

    for _, row in df.iterrows():

        try:

            value = float(row[median_col])

            if np.isfinite(value):

                medians[
                    str(row["Feature"])
                ] = value

        except Exception:

            pass

    return medians


# =============================================================================
# SMILES
# =============================================================================

def prepare_molecule(smiles):

    smiles = str(smiles).strip()

    if not smiles:

        raise ValueError(
            "Empty SMILES."
        )

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        raise ValueError(
            "RDKit could not parse the supplied SMILES."
        )

    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )

    return mol, canonical


# =============================================================================
# MOLECULAR DESCRIPTORS
# =============================================================================

def calculate_molecular_features(mol):

    values = {}

    values["FractionCSP3"] = (
        rdMolDescriptors.CalcFractionCSP3(mol)
    )

    # RDKit 2026 descriptor
    if hasattr(
        Descriptors,
        "SPS"
    ):

        values["SPS"] = (
            Descriptors.SPS(mol)
        )

    elif hasattr(
        rdMolDescriptors,
        "CalcSPS"
    ):

        values["SPS"] = (
            rdMolDescriptors.CalcSPS(mol)
        )

    else:

        raise RuntimeError(
            "SPS descriptor unavailable in this RDKit version."
        )

    values["NumAtomStereoCenters"] = (
        rdMolDescriptors.CalcNumAtomStereoCenters(
            mol
        )
    )

    values["SMR_VSA5"] = (
        Descriptors.SMR_VSA5(mol)
    )

    values["BCUT2D_MRLOW"] = (
        Descriptors.BCUT2D_MRLOW(mol)
    )

    values["BCUT2D_CHGLO"] = (
        Descriptors.BCUT2D_CHGLO(mol)
    )

    values["VSA_EState6"] = (
        Descriptors.VSA_EState6(mol)
    )

    values["NumAromaticRings"] = (
        rdMolDescriptors.CalcNumAromaticRings(
            mol
        )
    )

    values["SMR_VSA7"] = (
        Descriptors.SMR_VSA7(mol)
    )

    values["fr_Al_OH"] = (
        Descriptors.fr_Al_OH(mol)
    )

    return values


# =============================================================================
# 3D CONFORMER GENERATION
# =============================================================================

def generate_best_conformer(mol):

    mol3d = Chem.AddHs(
        Chem.Mol(mol)
    )

    params = AllChem.ETKDGv3()

    params.randomSeed = RANDOM_SEED

    params.pruneRmsThresh = 0.5

    params.numThreads = 0

    try:

        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol3d,
                numConfs=N_CONFORMERS,
                params=params
            )
        )

    except Exception:

        conf_ids = []

    if len(conf_ids) == 0:

        raise RuntimeError(
            "NO_CONFORMER: 3D conformer generation failed."
        )

    best_conf = None

    best_energy = np.inf

    best_method = None

    # -------------------------------------------------------------------------
    # MMFF94s
    # -------------------------------------------------------------------------

    try:

        props = (
            AllChem.MMFFGetMoleculeProperties(
                mol3d,
                mmffVariant="MMFF94s"
            )
        )

    except Exception:

        props = None

    if props is not None:

        for cid in conf_ids:

            try:

                ff = (
                    AllChem.MMFFGetMoleculeForceField(
                        mol3d,
                        props,
                        confId=int(cid)
                    )
                )

                if ff is None:

                    continue

                ff.Initialize()

                ff.Minimize(
                    maxIts=MAX_OPT_ITERS
                )

                energy = ff.CalcEnergy()

                if (
                    np.isfinite(energy)
                    and
                    energy < best_energy
                ):

                    best_energy = float(
                        energy
                    )

                    best_conf = int(
                        cid
                    )

                    best_method = (
                        "MMFF94s"
                    )

            except Exception:

                pass

    # -------------------------------------------------------------------------
    # UFF FALLBACK
    # -------------------------------------------------------------------------

    if best_conf is None:

        for cid in conf_ids:

            try:

                ff = (
                    AllChem.UFFGetMoleculeForceField(
                        mol3d,
                        confId=int(cid)
                    )
                )

                if ff is None:

                    continue

                ff.Initialize()

                ff.Minimize(
                    maxIts=MAX_OPT_ITERS
                )

                energy = ff.CalcEnergy()

                if (
                    np.isfinite(energy)
                    and
                    energy < best_energy
                ):

                    best_energy = float(
                        energy
                    )

                    best_conf = int(
                        cid
                    )

                    best_method = (
                        "UFF"
                    )

            except Exception:

                pass

    if best_conf is None:

        raise RuntimeError(
            "OPTIMIZATION_FAILED: neither MMFF94s nor UFF succeeded."
        )

    # -------------------------------------------------------------------------
    # COPY BEST CONFORMER
    # -------------------------------------------------------------------------

    selected_conf = Chem.Conformer(
        mol3d.GetConformer(
            best_conf
        )
    )

    best_mol = Chem.Mol(
        mol3d
    )

    best_mol.RemoveAllConformers()

    best_mol.AddConformer(
        selected_conf,
        assignId=True
    )

    return {

        "mol": best_mol,

        "method": best_method,

        "energy": best_energy,

        "conformers_generated":
            len(conf_ids)
    }


# =============================================================================
# 3D-QSAR FEATURES
# =============================================================================

def calculate_3d_features(mol3d):

    values = {}

    values["PBF"] = (
        Descriptors3D.PBF(mol3d)
    )

    values["SpherocityIndex"] = (
        Descriptors3D.SpherocityIndex(
            mol3d
        )
    )

    values["PMI1"] = (
        Descriptors3D.PMI1(
            mol3d
        )
    )

    values["InertialShapeFactor"] = (
        Descriptors3D.InertialShapeFactor(
            mol3d
        )
    )

    values["PMI2"] = (
        Descriptors3D.PMI2(
            mol3d
        )
    )

    values["NPR2"] = (
        Descriptors3D.NPR2(
            mol3d
        )
    )

    values["RadiusOfGyration"] = (
        Descriptors3D.RadiusOfGyration(
            mol3d
        )
    )

    values["Eccentricity"] = (
        Descriptors3D.Eccentricity(
            mol3d
        )
    )

    values["NPR1"] = (
        Descriptors3D.NPR1(
            mol3d
        )
    )

    return values


# =============================================================================
# FEATURE CLEANING
# =============================================================================

def clean_handcrafted_features(
    features,
    training_medians
):

    output = {}

    imputed = []

    for feature, value in features.items():

        try:

            value = float(
                value
            )

        except Exception:

            value = np.nan

        if not np.isfinite(
            value
        ):

            if feature not in training_medians:

                raise RuntimeError(
                    f"{feature} is invalid and "
                    "no training median is available."
                )

            value = (
                training_medians[
                    feature
                ]
            )

            imputed.append(
                feature
            )

        output[
            feature
        ] = value

    return output, imputed


# =============================================================================
# MOLFORMER
# =============================================================================

def load_molformer():

    section(
        "LOADING MOLFORMER"
    )

    print(
        "Model:",
        MOLFORMER_MODEL
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "Device:",
        device
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MOLFORMER_MODEL,
            trust_remote_code=True
        )
    )

    model = (
        AutoModel.from_pretrained(
            MOLFORMER_MODEL,
            trust_remote_code=True
        )
    )

    model.to(
        device
    )

    model.eval()

    return tokenizer, model, device


# =============================================================================
# MOLFORMER EMBEDDING
# =============================================================================

def generate_molformer_embedding(
    canonical_smiles,
    tokenizer,
    model,
    device
):

    inputs = tokenizer(

        [canonical_smiles],

        padding=True,

        truncation=True,

        max_length=MAX_TOKEN_LENGTH,

        return_tensors="pt"
    )

    inputs = {

        key: value.to(
            device
        )

        for key, value
        in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    hidden = (
        outputs.last_hidden_state
    )

    attention_mask = (
        inputs[
            "attention_mask"
        ]
    )

    # -------------------------------------------------------------------------
    # EXACT MASKED MEAN POOLING
    # -------------------------------------------------------------------------

    mask = (

        attention_mask
        .unsqueeze(-1)
        .expand(
            hidden.size()
        )
        .float()
    )

    masked_hidden = (
        hidden * mask
    )

    summed = (
        masked_hidden.sum(
            dim=1
        )
    )

    counts = (
        mask.sum(
            dim=1
        )
        .clamp(
            min=1e-9
        )
    )

    embedding = (
        summed / counts
    )

    embedding = (

        embedding[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    if (
        embedding.shape[0]
        !=
        768
    ):

        raise RuntimeError(

            "Unexpected MoLFormer dimension: "
            f"{embedding.shape[0]} "
            "(expected 768)"
        )

    return embedding


# =============================================================================
# SELECT MOLFORMER DIMENSIONS
# =============================================================================

def select_molformer_features(
    embedding
):

    selected = {}

    for feature in (
        MOLFORMER_FEATURES
    ):

        index = int(
            feature.split(
                "_"
            )[1]
        )

        selected[
            feature
        ] = float(
            embedding[
                index
            ]
        )

    return selected


# =============================================================================
# ENTROPY
# =============================================================================

def binary_entropy(p):

    p = float(p)

    eps = 1e-15

    p = min(
        max(
            p,
            eps
        ),
        1.0 - eps
    )

    return -(
        p * math.log2(p)
        +
        (1.0 - p)
        *
        math.log2(
            1.0 - p
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smiles",
        required=True,
        help="Input phytochemical SMILES"
    )

    parser.add_argument(
        "--name",
        default="New_Phytochemical"
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_DIR
    )

    args = parser.parse_args()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    header(
        "SCRIPT 09 — TRIAL PHYTOCHEMICAL INFERENCE"
    )

    print(
        "Started:",
        datetime.now().isoformat()
    )

    print(
        "Compound:",
        args.name
    )

    print(
        "Input SMILES:",
        args.smiles
    )


    # =========================================================================
    # LOAD DEPLOYMENT MODEL
    # =========================================================================

    section(
        "1. LOADING DEPLOYMENT MODEL"
    )

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            MODEL_FILE
        )

    artifact = joblib.load(
        MODEL_FILE
    )

    if not isinstance(
        artifact,
        dict
    ):

        raise RuntimeError(
            "Unexpected deployment artifact format."
        )

    required_keys = [

        "model",
        "model_name",
        "feature_order",
        "threshold",
        "calibration",
        "representation"
    ]

    for key in required_keys:

        if key not in artifact:

            raise RuntimeError(
                f"Missing model artifact key: {key}"
            )

    model = artifact[
        "model"
    ]

    feature_order = list(
        artifact[
            "feature_order"
        ]
    )

    threshold = float(
        artifact[
            "threshold"
        ]
    )

    print(
        "Model:",
        artifact[
            "model_name"
        ]
    )

    print(
        "Calibration:",
        artifact[
            "calibration"
        ]
    )

    print(
        "Feature count:",
        len(
            feature_order
        )
    )

    print(
        "Threshold:",
        threshold
    )

    if len(
        feature_order
    ) != 40:

        raise RuntimeError(
            "Deployment model does not contain 40 predictors."
        )


    # =========================================================================
    # TRAINING MEDIANS
    # =========================================================================

    training_medians = (
        load_training_medians()
    )


    # =========================================================================
    # SMILES
    # =========================================================================

    section(
        "2. SMILES VALIDATION"
    )

    mol, canonical = (
        prepare_molecule(
            args.smiles
        )
    )

    print(
        "Canonical SMILES:",
        canonical
    )

    print(
        "Formula:",
        rdMolDescriptors.CalcMolFormula(
            mol
        )
    )

    print(
        "Atoms:",
        mol.GetNumAtoms()
    )

    print(
        "Heavy atoms:",
        mol.GetNumHeavyAtoms()
    )


    # =========================================================================
    # MOLECULAR FEATURES
    # =========================================================================

    section(
        "3. MOLECULAR FEATURES"
    )

    molecular = (
        calculate_molecular_features(
            mol
        )
    )

    molecular, molecular_imputed = (
        clean_handcrafted_features(
            molecular,
            training_medians
        )
    )

    for feature in (
        MOLECULAR_FEATURES
    ):

        print(
            f"{feature:28s} "
            f"{molecular[feature]:.10g}"
        )


    # =========================================================================
    # 3D
    # =========================================================================

    section(
        "4. 3D CONFORMER GENERATION"
    )

    conformer_info = (
        generate_best_conformer(
            mol
        )
    )

    print(
        "Conformers generated:",
        conformer_info[
            "conformers_generated"
        ]
    )

    print(
        "Optimization method:",
        conformer_info[
            "method"
        ]
    )

    print(
        "Best energy:",
        conformer_info[
            "energy"
        ]
    )


    # =========================================================================
    # 3D FEATURES
    # =========================================================================

    section(
        "5. 3D-QSAR FEATURES"
    )

    three_d = (
        calculate_3d_features(
            conformer_info[
                "mol"
            ]
        )
    )

    three_d, threed_imputed = (
        clean_handcrafted_features(
            three_d,
            training_medians
        )
    )

    for feature in (
        THREED_FEATURES
    ):

        print(
            f"{feature:28s} "
            f"{three_d[feature]:.10g}"
        )


    # =========================================================================
    # MOLFORMER
    # =========================================================================

    tokenizer, molformer_model, device = (
        load_molformer()
    )

    section(
        "6. GENERATING MOLFORMER EMBEDDING"
    )

    embedding = (
        generate_molformer_embedding(
            canonical,
            tokenizer,
            molformer_model,
            device
        )
    )

    print(
        "Embedding dimension:",
        len(
            embedding
        )
    )

    print(
        "Embedding NaN:",
        int(
            np.isnan(
                embedding
            ).sum()
        )
    )

    print(
        "Embedding Inf:",
        int(
            np.isinf(
                embedding
            ).sum()
        )
    )


    # =========================================================================
    # SELECT 21
    # =========================================================================

    section(
        "7. SELECTED MOLFORMER FEATURES"
    )

    molformer_selected = (
        select_molformer_features(
            embedding
        )
    )

    for feature in (
        MOLFORMER_FEATURES
    ):

        print(
            f"{feature:28s} "
            f"{molformer_selected[feature]:.10g}"
        )


    # =========================================================================
    # FINAL FEATURE DICTIONARY
    # =========================================================================

    complete_features = {}

    complete_features.update(
        molecular
    )

    complete_features.update(
        three_d
    )

    complete_features.update(
        molformer_selected
    )


    # =========================================================================
    # VALIDATE FEATURE ORDER
    # =========================================================================

    section(
        "8. CONSTRUCTING FINAL 40-FEATURE MATRIX"
    )

    missing = [

        feature

        for feature
        in feature_order

        if feature
        not in complete_features
    ]

    if missing:

        raise RuntimeError(

            "Missing deployment features:\n"
            +
            "\n".join(
                missing
            )
        )

    X = pd.DataFrame(

        [[
            complete_features[
                feature
            ]

            for feature
            in feature_order
        ]],

        columns=feature_order
    )

    X = X.astype(
        float
    )

    print(
        "Feature matrix shape:",
        X.shape
    )

    print(
        "NaN:",
        int(
            X.isna()
            .sum()
            .sum()
        )
    )

    print(
        "Inf:",
        int(
            np.isinf(
                X.to_numpy()
            ).sum()
        )
    )

    if (
        X.shape
        !=
        (1, 40)
    ):

        raise RuntimeError(
            "Expected final feature matrix shape (1,40)."
        )

    if (
        list(
            X.columns
        )
        !=
        feature_order
    ):

        raise RuntimeError(
            "Feature ordering mismatch."
        )

    if not np.isfinite(
        X.to_numpy()
    ).all():

        raise RuntimeError(
            "Final feature matrix contains NaN or Inf."
        )


    # =========================================================================
    # PREDICTION
    # =========================================================================

    section(
        "9. RUNNING CALIBRATED MODEL"
    )

    probability_matrix = (
        model.predict_proba(
            X
        )
    )

    classes = list(
        model.classes_
    )

    print(
        "Classes:",
        classes
    )

    print(
        "Probability matrix:"
    )

    print(
        probability_matrix
    )

    active_index = (
        classes.index(
            1
        )
    )

    inactive_index = (
        classes.index(
            0
        )
    )

    p_active = float(
        probability_matrix[
            0,
            active_index
        ]
    )

    p_inactive = float(
        probability_matrix[
            0,
            inactive_index
        ]
    )


    # =========================================================================
    # DERIVED SCORES
    # =========================================================================

    entropy = binary_entropy(
        p_active
    )

    certainty = (
        1.0
        -
        entropy
    )

    probability_margin = (
        abs(
            p_active
            -
            p_inactive
        )
    )

    distance_from_threshold = (
        p_active
        -
        threshold
    )

    threshold_margin_absolute = (
        abs(
            distance_from_threshold
        )
    )

    if (
        p_active
        >=
        threshold
    ):

        prediction = (
            "SCREEN_POSITIVE"
        )

        predicted_class = 1

    else:

        prediction = (
            "SCREEN_NEGATIVE"
        )

        predicted_class = 0


    # -------------------------------------------------------------------------
    # ENTROPY-BASED CONFIDENCE CATEGORY
    # -------------------------------------------------------------------------

    if certainty >= 0.70:

        confidence = (
            "HIGH"
        )

    elif certainty >= 0.40:

        confidence = (
            "MEDIUM"
        )

    else:

        confidence = (
            "LOW"
        )


    # =========================================================================
    # OUTPUT
    # =========================================================================

    header(
        "FINAL INFERENCE RESULT"
    )

    print(
        f"Compound                 : {args.name}"
    )

    print(
        f"Canonical SMILES         : {canonical}"
    )

    print()

    print(
        f"P(Inactive)              : {p_inactive:.10f}"
    )

    print(
        f"P(Active)                : {p_active:.10f}"
    )

    print(
        f"Locked threshold         : {threshold:.10f}"
    )

    print(
        f"Distance from threshold  : {distance_from_threshold:.10f}"
    )

    print(
        f"Probability margin       : {probability_margin:.10f}"
    )

    print()

    print(
        f"Entropy uncertainty      : {entropy:.10f}"
    )

    print(
        f"Certainty                : {certainty:.10f}"
    )

    print(
        f"Confidence level         : {confidence}"
    )

    print()

    print(
        f"Screening prediction     : {prediction}"
    )

    print(
        f"Predicted class          : {predicted_class}"
    )

    print()

    print(
        "Interpretation:"
    )

    if predicted_class == 1:

        print(
            "The compound crosses the locked screening threshold "
            "and is prioritized as an active-like bioactivity hypothesis."
        )

    else:

        print(
            "The compound does not cross the locked screening threshold "
            "and receives a lower-priority bioactivity hypothesis."
        )


    # =========================================================================
    # SAVE MOLECULAR
    # =========================================================================

    pd.DataFrame(
        [{
            "Compound":
                args.name,

            "Input_SMILES":
                args.smiles,

            "Canonical_SMILES":
                canonical,

            **molecular
        }]
    ).to_csv(

        os.path.join(
            args.output,
            "molecular_features.tsv"
        ),

        sep="\t",

        index=False
    )


    # =========================================================================
    # SAVE 3D
    # =========================================================================

    pd.DataFrame(
        [{
            "Compound":
                args.name,

            "Optimization_Method":
                conformer_info[
                    "method"
                ],

            "Best_Conformer_Energy":
                conformer_info[
                    "energy"
                ],

            "Conformers_Generated":
                conformer_info[
                    "conformers_generated"
                ],

            **three_d
        }]
    ).to_csv(

        os.path.join(
            args.output,
            "qsar_3d_features.tsv"
        ),

        sep="\t",

        index=False
    )


    # =========================================================================
    # SAVE FULL MOLFORMER
    # =========================================================================

    embedding_dict = {

        f"molformer_{i:04d}":
            float(value)

        for i, value
        in enumerate(
            embedding
        )
    }

    pd.DataFrame(
        [{
            "Compound":
                args.name,

            **embedding_dict
        }]
    ).to_csv(

        os.path.join(
            args.output,
            "molformer_768_embedding.tsv"
        ),

        sep="\t",

        index=False
    )


    # =========================================================================
    # SAVE 21 MOLFORMER
    # =========================================================================

    pd.DataFrame(
        [{
            "Compound":
                args.name,

            **molformer_selected
        }]
    ).to_csv(

        os.path.join(
            args.output,
            "molformer_selected_21.tsv"
        ),

        sep="\t",

        index=False
    )


    # =========================================================================
    # SAVE FINAL 40
    # =========================================================================

    final_matrix = X.copy()

    final_matrix.insert(
        0,
        "Compound",
        args.name
    )

    final_matrix.insert(
        1,
        "Canonical_SMILES",
        canonical
    )

    final_matrix.to_csv(

        os.path.join(
            args.output,
            "final_40_feature_matrix.tsv"
        ),

        sep="\t",

        index=False
    )


    # =========================================================================
    # SAVE PREDICTION
    # =========================================================================

    prediction_record = {

        "Compound":
            args.name,

        "Input_SMILES":
            args.smiles,

        "Canonical_SMILES":
            canonical,

        "Model":
            artifact[
                "model_name"
            ],

        "Calibration":
            artifact[
                "calibration"
            ],

        "P_Inactive":
            p_inactive,

        "P_Active":
            p_active,

        "Locked_Threshold":
            threshold,

        "Distance_From_Threshold":
            distance_from_threshold,

        "Absolute_Threshold_Margin":
            threshold_margin_absolute,

        "Probability_Margin":
            probability_margin,

        "Entropy_Uncertainty":
            entropy,

        "Certainty":
            certainty,

        "Confidence_Level":
            confidence,

        "Predicted_Class":
            predicted_class,

        "Screening_Prediction":
            prediction,

        "Conformer_Method":
            conformer_info[
                "method"
            ],

        "Conformer_Energy":
            conformer_info[
                "energy"
            ],

        "Conformers_Generated":
            conformer_info[
                "conformers_generated"
            ]
    }

    pd.DataFrame(
        [prediction_record]
    ).to_csv(

        os.path.join(
            args.output,
            "prediction_summary.tsv"
        ),

        sep="\t",

        index=False
    )


    # =========================================================================
    # JSON REPORT
    # =========================================================================

    report = {

        "script":
            "09_inference_trial.py",

        "generated":
            datetime.now().isoformat(),

        "input": {

            "compound":
                args.name,

            "input_smiles":
                args.smiles,

            "canonical_smiles":
                canonical
        },

        "model": {

            "name":
                artifact[
                    "model_name"
                ],

            "calibration":
                artifact[
                    "calibration"
                ],

            "threshold":
                threshold,

            "feature_count":
                len(
                    feature_order
                ),

            "training_n":
                artifact.get(
                    "training_n"
                ),

            "training_active":
                artifact.get(
                    "training_active"
                ),

            "training_inactive":
                artifact.get(
                    "training_inactive"
                )
        },

        "representation": {

            "molecular_features":
                MOLECULAR_FEATURES,

            "3d_features":
                THREED_FEATURES,

            "molformer_features":
                MOLFORMER_FEATURES
        },

        "3d_qc": {

            "optimization_method":
                conformer_info[
                    "method"
                ],

            "best_energy":
                conformer_info[
                    "energy"
                ],

            "conformers_generated":
                conformer_info[
                    "conformers_generated"
                ],

            "imputed_features":
                threed_imputed
        },

        "molecular_qc": {

            "imputed_features":
                molecular_imputed
        },

        "molformer_qc": {

            "model":
                MOLFORMER_MODEL,

            "dimension":
                768,

            "pooling":
                "attention_masked_mean_pooling",

            "nan":
                int(
                    np.isnan(
                        embedding
                    ).sum()
                ),

            "inf":
                int(
                    np.isinf(
                        embedding
                    ).sum()
                )
        },

        "prediction": {

            "probability_matrix":
                probability_matrix.tolist(),

            "p_inactive":
                p_inactive,

            "p_active":
                p_active,

            "locked_threshold":
                threshold,

            "distance_from_threshold":
                distance_from_threshold,

            "probability_margin":
                probability_margin,

            "entropy_uncertainty":
                entropy,

            "certainty":
                certainty,

            "confidence_level":
                confidence,

            "predicted_class":
                predicted_class,

            "screening_prediction":
                prediction
        }
    }

    with open(

        os.path.join(
            args.output,
            "inference_report.json"
        ),

        "w"

    ) as fh:

        json.dump(
            report,
            fh,
            indent=2
        )


    # =========================================================================
    # FINAL FILE SUMMARY
    # =========================================================================

    section(
        "OUTPUT FILES"
    )

    for filename in [

        "prediction_summary.tsv",

        "molecular_features.tsv",

        "qsar_3d_features.tsv",

        "molformer_768_embedding.tsv",

        "molformer_selected_21.tsv",

        "final_40_feature_matrix.tsv",

        "inference_report.json"
    ]:

        print(
            os.path.join(
                args.output,
                filename
            )
        )

    print()
    print(
        "FINAL STATUS: PASS"
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":

    main()
