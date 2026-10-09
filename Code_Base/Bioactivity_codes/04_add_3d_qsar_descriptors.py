#!/usr/bin/env python3

"""
Script 04 v2
============================================================

3D QSAR / Conformation-Dependent Molecular Descriptor
Generation for CMAUP phytochemicals.

Designed for:
    Slurm compute node: service3
    CPUs: 64

Input:
    feature_engineering/split/train_80.tsv
    feature_engineering/split/test_20.tsv

Output:
    feature_engineering/3d_qsar_descriptors/

Features:
    - ETKDGv3 conformer generation
    - Multiple conformers
    - MMFF94s optimization
    - UFF fallback
    - Metal detection
    - 3D descriptors
    - PBF
    - Checkpointing
    - Resume capability
    - Parallel processing
    - Train/test handled separately

NO MODEL TRAINING IS PERFORMED.
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import time
import math
import traceback

from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit import rdBase

from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_INPUT = (
    "feature_engineering/split/train_80.tsv"
)

TEST_INPUT = (
    "feature_engineering/split/test_20.tsv"
)

OUTPUT_DIR = Path(
    "feature_engineering/3d_qsar_descriptors"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Final outputs
# ------------------------------------------------------------

TRAIN_OUTPUT = (
    OUTPUT_DIR /
    "train_80_3d_qsar_descriptors.tsv"
)

TEST_OUTPUT = (
    OUTPUT_DIR /
    "test_20_3d_qsar_descriptors.tsv"
)


# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

TRAIN_CHECKPOINT = (
    OUTPUT_DIR /
    "train_80_checkpoint.tsv"
)

TEST_CHECKPOINT = (
    OUTPUT_DIR /
    "test_20_checkpoint.tsv"
)


# ------------------------------------------------------------
# QC
# ------------------------------------------------------------

TRAIN_QC = (
    OUTPUT_DIR /
    "train_80_3d_qc.tsv"
)

TEST_QC = (
    OUTPUT_DIR /
    "test_20_3d_qc.tsv"
)

COMBINED_QC = (
    OUTPUT_DIR /
    "3d_conformer_qc.tsv"
)

SUMMARY_FILE = (
    OUTPUT_DIR /
    "3d_qsar_summary.txt"
)


# ============================================================
# HPC SETTINGS
# ============================================================

# service3 has 64 CPUs.
# Use all 64 CPUs.
N_WORKERS = 64

# Process this many molecules before writing a checkpoint.
CHECKPOINT_INTERVAL = 500

# Number of conformers per molecule.
NUM_CONFORMERS = 3

# RMS threshold for conformer pruning.
RMS_PRUNE = 0.5

# Reproducibility.
RANDOM_SEED = 42

# Force-field optimization limits.
MMFF_MAX_ITERS = 300
UFF_MAX_ITERS = 300


# ============================================================
# RDKit LOGGING
# ============================================================

# UFF can generate enormous numbers of warnings for
# metal-containing structures.
#
# We record these cases in QC instead of flooding the
# Slurm .err file.

RDLogger.DisableLog("rdApp.*")


# ============================================================
# RDKit VERSION
# ============================================================

print("=" * 80)
print("SCRIPT 04 v2 - 3D QSAR DESCRIPTOR GENERATION")
print("=" * 80)

print(
    f"RDKit version: {rdBase.rdkitVersion}"
)

print(
    f"Workers: {N_WORKERS}"
)

print(
    f"Conformers per molecule: {NUM_CONFORMERS}"
)

print(
    f"Checkpoint interval: {CHECKPOINT_INTERVAL}"
)


# ============================================================
# 3D DESCRIPTORS
# ============================================================

DESCRIPTOR_FUNCTIONS = list(
    Descriptors3D.descList
)

DESCRIPTOR_NAMES = [
    name
    for name, func in DESCRIPTOR_FUNCTIONS
]


# Add PBF separately.
DESCRIPTOR_NAMES.append("PBF")


print(
    f"3D descriptors: {len(DESCRIPTOR_NAMES)}"
)


# ============================================================
# METAL DEFINITIONS
# ============================================================

# Common metallic elements that may occur in CMAUP.
#
# Atomic numbers:
#
# Li 3
# Be 4
# Na 11
# Mg 12
# Al 13
# K 19
# Ca 20
# Sc 21
# Ti 22
# V 23
# Cr 24
# Mn 25
# Fe 26
# Co 27
# Ni 28
# Cu 29
# Zn 30
# Ga 31
# Rb 37
# Sr 38
# Y 39
# Zr 40
# Nb 41
# Mo 42
# Tc 43
# Ru 44
# Rh 45
# Pd 46
# Ag 47
# Cd 48
# In 49
# Sn 50
# Cs 55
# Ba 56
# La 57
# Ce 58
# Pr 59
# Nd 60
# Pm 61
# Sm 62
# Eu 63
# Gd 64
# Tb 65
# Dy 66
# Ho 67
# Er 68
# Tm 69
# Yb 70
# Lu 71
# Hf 72
# Ta 73
# W 74
# Re 75
# Os 76
# Ir 77
# Pt 78
# Au 79
# Hg 80
# Tl 81
# Pb 82
# Bi 83
#
# We intentionally do not classify B, Si, P, S, etc.
# as metals here.

METAL_ATOMIC_NUMBERS = {
    3, 4,
    11, 12, 13,
    19, 20,
    21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
    37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50,
    55, 56,
    57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83
}


# ============================================================
# METAL DETECTION
# ============================================================

def contains_metal(mol):

    """
    Return True if molecule contains a defined metal.
    """

    if mol is None:
        return False

    for atom in mol.GetAtoms():

        if atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS:

            return True

    return False


# ============================================================
# GENERATE CONFORMERS
# ============================================================

def generate_conformers(mol):

    """
    Generate multiple 3D conformers using ETKDGv3.
    """

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()

    params.randomSeed = RANDOM_SEED

    params.pruneRmsThresh = RMS_PRUNE

    # VERY IMPORTANT:
    #
    # We have 64 Python processes.
    #
    # Each process must use one RDKit thread.
    #
    # Otherwise 64 processes × many RDKit threads
    # would cause severe CPU oversubscription.

    params.numThreads = 1

    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=NUM_CONFORMERS,
            params=params
        )
    )

    # Fallback embedding using random coordinates.
    if len(conformer_ids) == 0:

        params.useRandomCoords = True

        conformer_ids = list(
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=NUM_CONFORMERS,
                params=params
            )
        )

    return mol, conformer_ids


# ============================================================
# MMFF OPTIMIZATION
# ============================================================

def optimize_mmff(mol, conf_id):

    """
    Try MMFF94s optimization.
    """

    try:

        if not AllChem.MMFFHasAllMoleculeParams(mol):

            return None

        properties = (
            AllChem.MMFFGetMoleculeProperties(
                mol,
                mmffVariant="MMFF94s"
            )
        )

        if properties is None:

            return None

        force_field = (
            AllChem.MMFFGetMoleculeForceField(
                mol,
                properties,
                confId=conf_id
            )
        )

        if force_field is None:

            return None

        force_field.Initialize()

        status = force_field.Minimize(
            maxIts=MMFF_MAX_ITERS
        )

        energy = force_field.CalcEnergy()

        if not np.isfinite(energy):

            return None

        return (
            float(energy),
            int(status),
            "MMFF94s"
        )

    except Exception:

        return None


# ============================================================
# UFF OPTIMIZATION
# ============================================================

def optimize_uff(mol, conf_id):

    """
    UFF fallback.
    """

    try:

        if not AllChem.UFFHasAllMoleculeParams(mol):

            return None

        force_field = (
            AllChem.UFFGetMoleculeForceField(
                mol,
                confId=conf_id
            )
        )

        if force_field is None:

            return None

        force_field.Initialize()

        status = force_field.Minimize(
            maxIts=UFF_MAX_ITERS
        )

        energy = force_field.CalcEnergy()

        if not np.isfinite(energy):

            return None

        return (
            float(energy),
            int(status),
            "UFF"
        )

    except Exception:

        return None


# ============================================================
# SELECT BEST CONFORMER
# ============================================================

def generate_best_conformer(mol):

    """
    Generate multiple conformers.

    Try MMFF94s for each conformer.
    If MMFF is unavailable, try UFF.

    Return the lowest-energy successful conformer.
    """

    metal_flag = contains_metal(mol)

    try:

        mol3d, conformer_ids = (
            generate_conformers(mol)
        )

    except Exception:

        return {
            "mol": None,
            "conf_id": None,
            "energy": np.nan,
            "method": "NONE",
            "status": "EMBED_FAILED",
            "n_conformers": 0,
            "metal": metal_flag
        }


    if len(conformer_ids) == 0:

        return {
            "mol": None,
            "conf_id": None,
            "energy": np.nan,
            "method": "NONE",
            "status": "NO_CONFORMER",
            "n_conformers": 0,
            "metal": metal_flag
        }


    candidates = []


    # --------------------------------------------------------
    # Try MMFF first
    # --------------------------------------------------------

    for conf_id in conformer_ids:

        result = optimize_mmff(
            mol3d,
            conf_id
        )

        if result is not None:

            energy, status, method = result

            candidates.append(
                (
                    conf_id,
                    energy,
                    status,
                    method
                )
            )


    # --------------------------------------------------------
    # If no MMFF conformer succeeded,
    # use UFF.
    # --------------------------------------------------------

    if len(candidates) == 0:

        for conf_id in conformer_ids:

            result = optimize_uff(
                mol3d,
                conf_id
            )

            if result is not None:

                energy, status, method = result

                candidates.append(
                    (
                        conf_id,
                        energy,
                        status,
                        method
                    )
                )


    # --------------------------------------------------------
    # Nothing worked
    # --------------------------------------------------------

    if len(candidates) == 0:

        return {
            "mol": None,
            "conf_id": None,
            "energy": np.nan,
            "method": "NONE",
            "status": "OPTIMIZATION_FAILED",
            "n_conformers": len(conformer_ids),
            "metal": metal_flag
        }


    # --------------------------------------------------------
    # Select lowest-energy conformer
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[1]
    )

    best_conf_id = candidates[0][0]
    best_energy = candidates[0][1]
    best_status = candidates[0][2]
    best_method = candidates[0][3]


    return {
        "mol": mol3d,
        "conf_id": best_conf_id,
        "energy": best_energy,
        "method": best_method,
        "status": "SUCCESS",
        "n_conformers": len(conformer_ids),
        "metal": metal_flag
    }


# ============================================================
# CALCULATE 3D DESCRIPTORS
# ============================================================

def calculate_descriptors(
    mol,
    conf_id
):

    """
    Calculate all RDKit 3D descriptors plus PBF.
    """

    values = {}

    for name, function in DESCRIPTOR_FUNCTIONS:

        try:

            value = function(
                mol,
                confId=conf_id
            )

            if value is None:

                value = np.nan

            elif not np.isfinite(value):

                value = np.nan

            values[name] = value

        except Exception:

            values[name] = np.nan


    # PBF
    try:

        value = rdMolDescriptors.CalcPBF(
            mol,
            confId=conf_id
        )

        if np.isfinite(value):

            values["PBF"] = value

        else:

            values["PBF"] = np.nan

    except Exception:

        values["PBF"] = np.nan


    return values


# ============================================================
# WORKER FUNCTION
# ============================================================

def process_one(record):

    """
    Process one molecule.

    This function runs inside a separate worker process.
    """

    index, np_id, smiles = record

    # Initialize result
    result = {
        "np_id": np_id,
        "3D_Status": "FAILED",
        "Optimization_Method": "NONE",
        "Best_Conformer_Energy": np.nan,
        "Conformers_Generated": 0,
        "Metal_Containing": 0
    }

    # Initialize descriptors
    for descriptor in DESCRIPTOR_NAMES:

        result[descriptor] = np.nan


    try:

        if (
            smiles is None
            or
            str(smiles).strip() == ""
        ):

            result["3D_Status"] = (
                "MISSING_SMILES"
            )

            return index, result


        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        if mol is None:

            result["3D_Status"] = (
                "INVALID_SMILES"
            )

            return index, result


        # ----------------------------------------------------
        # Generate / optimize conformer
        # ----------------------------------------------------

        best = generate_best_conformer(
            mol
        )


        result["Metal_Containing"] = (
            1 if best["metal"] else 0
        )

        result["Conformers_Generated"] = (
            best["n_conformers"]
        )

        result["Best_Conformer_Energy"] = (
            best["energy"]
        )

        result["Optimization_Method"] = (
            best["method"]
        )

        result["3D_Status"] = (
            best["status"]
        )


        # ----------------------------------------------------
        # Calculate descriptors
        # ----------------------------------------------------

        if (
            best["mol"] is not None
            and
            best["conf_id"] is not None
            and
            best["status"] == "SUCCESS"
        ):

            descriptor_values = (
                calculate_descriptors(
                    best["mol"],
                    best["conf_id"]
                )
            )

            for descriptor, value in (
                descriptor_values.items()
            ):

                result[descriptor] = value


    except Exception as error:

        result["3D_Status"] = (
            "WORKER_ERROR"
        )


    return index, result


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(input_file):

    print(
        f"\nLoading: {input_file}"
    )

    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype=str,
        low_memory=False
    )

    required_columns = [
        "np_id",
        "Canonical_SMILES"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    print(
        f"Molecules: {len(df):,}"
    )

    return df


# ============================================================
# LOAD EXISTING CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_file
):

    if not checkpoint_file.exists():

        return None

    print(
        f"\nCheckpoint found:"
        f"\n{checkpoint_file}"
    )

    checkpoint = pd.read_csv(
        checkpoint_file,
        sep="\t",
        dtype=str,
        low_memory=False
    )

    if "np_id" not in checkpoint.columns:

        print(
            "Checkpoint invalid; "
            "starting from beginning."
        )

        return None

    print(
        f"Checkpoint records: "
        f"{len(checkpoint):,}"
    )

    return checkpoint


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    results,
    checkpoint_file
):

    checkpoint_df = pd.DataFrame(
        results
    )

    # Atomic-ish write:
    # write temporary file first.
    temp_file = (
        str(checkpoint_file) +
        ".tmp"
    )

    checkpoint_df.to_csv(
        temp_file,
        sep="\t",
        index=False
    )

    os.replace(
        temp_file,
        checkpoint_file
    )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    input_file,
    checkpoint_file,
    final_output,
    qc_output,
    dataset_name
):

    print("\n")
    print("=" * 80)
    print(
        f"PROCESSING {dataset_name}"
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    df = load_dataset(
        input_file
    )


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        checkpoint_file
    )


    completed_ids = set()

    previous_results = []


    if checkpoint is not None:

        completed_ids = set(
            checkpoint["np_id"].astype(str)
        )

        previous_results = (
            checkpoint.to_dict(
                orient="records"
            )
        )

        print(
            f"Already completed: "
            f"{len(completed_ids):,}"
        )


    # --------------------------------------------------------
    # Create records to process
    # --------------------------------------------------------

    records = []

    for index, row in df.iterrows():

        np_id = str(
            row["np_id"]
        )

        if np_id in completed_ids:

            continue

        smiles = row[
            "Canonical_SMILES"
        ]

        records.append(
            (
                index,
                np_id,
                smiles
            )
        )


    print(
        f"Remaining molecules: "
        f"{len(records):,}"
    )


    # --------------------------------------------------------
    # Nothing left to process
    # --------------------------------------------------------

    if len(records) == 0:

        print(
            "\nAll molecules already "
            "processed."
        )

        result_df = pd.DataFrame(
            previous_results
        )

    else:

        results = list(
            previous_results
        )

        completed_count = (
            len(previous_results)
        )

        total_remaining = len(records)

        start_time = time.time()


        # ----------------------------------------------------
        # 64 parallel workers
        # ----------------------------------------------------

        print(
            f"\nStarting "
            f"{N_WORKERS} worker processes..."
        )

        with Pool(
            processes=N_WORKERS
        ) as pool:

            # chunksize controls how many records are
            # distributed at a time.
            #
            # 1 is safer for molecules with highly variable
            # processing time.

            iterator = pool.imap_unordered(
                process_one,
                records,
                chunksize=1
            )


            for result_index, result in iterator:

                results.append(
                    result
                )

                completed_count += 1


                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    completed_count % 100 == 0
                    or
                    completed_count == (
                        len(previous_results)
                        +
                        total_remaining
                    )
                ):

                    elapsed = (
                        time.time()
                        -
                        start_time
                    )

                    processed_now = (
                        completed_count
                        -
                        len(previous_results)
                    )

                    if elapsed > 0:

                        rate = (
                            processed_now /
                            elapsed
                        )

                    else:

                        rate = 0


                    remaining = (
                        total_remaining
                        -
                        processed_now
                    )

                    if rate > 0:

                        eta_seconds = (
                            remaining /
                            rate
                        )

                    else:

                        eta_seconds = 0


                    print(
                        f"\r"
                        f"{dataset_name}: "
                        f"{completed_count:,}/"
                        f"{len(df):,} | "
                        f"Current batch: "
                        f"{processed_now:,}/"
                        f"{total_remaining:,} | "
                        f"Rate: "
                        f"{rate:.2f} mol/s | "
                        f"ETA: "
                        f"{eta_seconds/60:.1f} min",
                        end="",
                        flush=True
                    )


                # ------------------------------------------------
                # Checkpoint
                # ------------------------------------------------

                if (
                    completed_count %
                    CHECKPOINT_INTERVAL
                    == 0
                ):

                    save_checkpoint(
                        results,
                        checkpoint_file
                    )

                    print(
                        f"\n"
                        f"Checkpoint saved: "
                        f"{completed_count:,}"
                    )


        print("\n")

        # ----------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            results,
            checkpoint_file
        )

        result_df = pd.DataFrame(
            results
        )


    # ========================================================
    # MERGE WITH ORIGINAL DATA
    # ========================================================

    # Make sure np_id is string.
    df["np_id"] = (
        df["np_id"].astype(str)
    )

    result_df["np_id"] = (
        result_df["np_id"].astype(str)
    )


    # Remove accidental duplicate result rows.
    result_df = (
        result_df
        .drop_duplicates(
            subset=["np_id"],
            keep="last"
        )
    )


    # Merge preserves the original dataset order.
    final_df = df.merge(
        result_df,
        on="np_id",
        how="left",
        sort=False
    )


    # ========================================================
    # SAVE FINAL OUTPUT
    # ========================================================

    temp_output = (
        str(final_output) +
        ".tmp"
    )

    final_df.to_csv(
        temp_output,
        sep="\t",
        index=False
    )

    os.replace(
        temp_output,
        final_output
    )


    # ========================================================
    # QC
    # ========================================================

    qc_columns = [
        "np_id",
        "3D_Status",
        "Optimization_Method",
        "Best_Conformer_Energy",
        "Conformers_Generated",
        "Metal_Containing"
    ]

    qc_df = final_df[
        qc_columns
    ].copy()

    qc_df[
        "Dataset"
    ] = dataset_name

    qc_df.to_csv(
        qc_output,
        sep="\t",
        index=False
    )


    # ========================================================
    # REPORT
    # ========================================================

    print(
        f"\n{dataset_name} completed."
    )

    print(
        f"Input molecules: "
        f"{len(df):,}"
    )

    print(
        f"Output molecules: "
        f"{len(final_df):,}"
    )

    print(
        "\n3D status:"
    )

    print(
        final_df[
            "3D_Status"
        ].value_counts(
            dropna=False
        ).to_string()
    )

    print(
        "\nOptimization method:"
    )

    print(
        final_df[
            "Optimization_Method"
        ].value_counts(
            dropna=False
        ).to_string()
    )

    print(
        "\nMetal-containing:"
    )

    print(
        final_df[
            "Metal_Containing"
        ].value_counts(
            dropna=False
        ).to_string()
    )

    return final_df


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_df = process_dataset(
        TRAIN_INPUT,
        TRAIN_CHECKPOINT,
        TRAIN_OUTPUT,
        TRAIN_QC,
        "TRAIN"
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_df = process_dataset(
        TEST_INPUT,
        TEST_CHECKPOINT,
        TEST_OUTPUT,
        TEST_QC,
        "TEST"
    )


    # ========================================================
    # COMBINED QC
    # ========================================================

    train_qc = pd.read_csv(
        TRAIN_QC,
        sep="\t",
        dtype=str
    )

    test_qc = pd.read_csv(
        TEST_QC,
        sep="\t",
        dtype=str
    )

    combined_qc = pd.concat(
        [
            train_qc,
            test_qc
        ],
        ignore_index=True
    )

    combined_qc.to_csv(
        COMBINED_QC,
        sep="\t",
        index=False
    )


    # ========================================================
    # DESCRIPTOR QUALITY
    # ========================================================

    descriptor_columns = (
        DESCRIPTOR_NAMES
    )

    combined_3d = pd.concat(
        [
            train_df[
                descriptor_columns
            ],
            test_df[
                descriptor_columns
            ]
        ],
        ignore_index=True
    )

    combined_3d = combined_3d.replace(
        [np.inf, -np.inf],
        np.nan
    )


    usable_descriptors = []

    completely_missing = []

    for descriptor in descriptor_columns:

        if combined_3d[
            descriptor
        ].notna().any():

            usable_descriptors.append(
                descriptor
            )

        else:

            completely_missing.append(
                descriptor
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w"
    ) as f:

        f.write(
            "3D QSAR DESCRIPTOR GENERATION SUMMARY\n"
        )

        f.write(
            "=" * 80 +
            "\n\n"
        )

        f.write(
            f"RDKit version: "
            f"{rdBase.rdkitVersion}\n"
        )

        f.write(
            f"Workers: "
            f"{N_WORKERS}\n"
        )

        f.write(
            f"Conformers requested: "
            f"{NUM_CONFORMERS}\n"
        )

        f.write(
            f"RMS pruning: "
            f"{RMS_PRUNE}\n"
        )

        f.write(
            f"Random seed: "
            f"{RANDOM_SEED}\n\n"
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        f.write(
            "TRAINING SET\n"
        )

        f.write(
            "-" * 80 +
            "\n"
        )

        f.write(
            f"Molecules: "
            f"{len(train_df):,}\n"
        )

        for status, count in (
            train_df[
                "3D_Status"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        ):

            f.write(
                f"{status}: "
                f"{count:,}\n"
            )


        f.write("\n")

        f.write(
            "Optimization methods:\n"
        )

        for method, count in (
            train_df[
                "Optimization_Method"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        ):

            f.write(
                f"{method}: "
                f"{count:,}\n"
            )


        f.write("\n")

        f.write(
            "Metal-containing:\n"
        )

        for value, count in (
            train_df[
                "Metal_Containing"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        ):

            f.write(
                f"{value}: "
                f"{count:,}\n"
            )


        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        f.write(
            "\n\nTESTING SET\n"
        )

        f.write(
            "-" * 80 +
            "\n"
        )

        f.write(
            f"Molecules: "
            f"{len(test_df):,}\n"
        )

        for status, count in (
            test_df[
                "3D_Status"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        ):

            f.write(
                f"{status}: "
                f"{count:,}\n"
            )


        f.write("\n")

        f.write(
            "Optimization methods:\n"
        )

        for method, count in (
            test_df[
                "Optimization_Method"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        ):

            f.write(
                f"{method}: "
                f"{count:,}\n"
            )


        f.write("\n")

        f.write(
            "Metal-containing:\n"
        )

        for value, count in (
            test_df[
                "Metal_Containing"
            ]
            .value_counts(
                dropna=False
            )
            .items()
        ):

            f.write(
                f"{value}: "
                f"{count:,}\n"
            )


        # ----------------------------------------------------
        # DESCRIPTORS
        # ----------------------------------------------------

        f.write(
            "\n\n3D DESCRIPTORS\n"
        )

        f.write(
            "-" * 80 +
            "\n"
        )

        f.write(
            f"Total descriptors: "
            f"{len(descriptor_columns)}\n"
        )

        f.write(
            f"Usable descriptors: "
            f"{len(usable_descriptors)}\n"
        )

        f.write(
            f"Completely missing: "
            f"{len(completely_missing)}\n"
        )


        if completely_missing:

            f.write(
                "\nCompletely missing descriptors:\n"
            )

            for descriptor in (
                completely_missing
            ):

                f.write(
                    descriptor +
                    "\n"
                )


        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        f.write(
            "\nMissing values per descriptor:\n"
        )

        for descriptor in (
            descriptor_columns
        ):

            missing = int(
                combined_3d[
                    descriptor
                ].isna().sum()
            )

            if missing > 0:

                f.write(
                    f"{descriptor}\t"
                    f"{missing}\n"
                )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    elapsed = (
        time.time()
        -
        overall_start
    )

    print("\n")
    print("=" * 80)
    print("SCRIPT 04 v2 COMPLETED")
    print("=" * 80)

    print(
        f"\nTraining output:"
        f"\n  {TRAIN_OUTPUT}"
    )

    print(
        f"\nTesting output:"
        f"\n  {TEST_OUTPUT}"
    )

    print(
        f"\nCombined QC:"
        f"\n  {COMBINED_QC}"
    )

    print(
        f"\nSummary:"
        f"\n  {SUMMARY_FILE}"
    )

    print(
        f"\n3D descriptors:"
        f" {len(descriptor_columns)}"
    )

    print(
        f"Usable descriptors:"
        f" {len(usable_descriptors)}"
    )

    print(
        f"\nTotal runtime:"
        f" {elapsed / 3600:.2f} hours"
    )

    print(
        "\nNo model was trained."
    )

    print(
        "No feature selection was performed."
    )

    print(
        "No scaling was performed."
    )

    print(
        "\nDONE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
