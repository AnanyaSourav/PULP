#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 06 — MoLFormer EMBEDDING GENERATION v1
==============================================================================

Purpose
-------
Generate frozen molecular embeddings from SMILES using:

    ibm-research/MoLFormer-XL-both-10pct

Input
-----
feature_engineering/final_228_features/train_80_final_228_features.tsv
feature_engineering/final_228_features/test_20_final_228_features.tsv

Output
------
feature_engineering/molformer_embeddings/

    train_molformer_embeddings.tsv
    test_molformer_embeddings.tsv

    train_invalid_smiles.tsv
    test_invalid_smiles.tsv

    embedding_metadata.json
    embedding_generation_report.txt

    train_embedding_matrix.npy
    test_embedding_matrix.npy

    train_embedding_stats.tsv
    test_embedding_stats.tsv

    checksums.sha256

Important
---------
NO original feature files are modified.
NO features are removed.
NO model fine-tuning is performed.

The transformer is used as a frozen feature extractor.
==============================================================================


Author: Phytochemical Bioactivity Prediction Project
Version: v1
"""


# ============================================================================
# 0. IMPORTS
# ============================================================================

import os
import sys
import json
import time
import hashlib
import platform
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from rdkit import Chem

import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

TRAIN_FILE = (
    "feature_engineering/final_228_features/"
    "train_80_final_228_features.tsv"
)

TEST_FILE = (
    "feature_engineering/final_228_features/"
    "test_20_final_228_features.tsv"
)

OUTPUT_DIR = "feature_engineering/molformer_embeddings"

MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"

ID_COLUMN = "np_id"

# The script will automatically identify a likely SMILES column.
SMILES_CANDIDATES = [
    "SMILES",
    "smiles",
    "Smiles",
    "canonical_smiles",
    "Canonical_SMILES",
    "canonical_SMILES",
    "structure",
    "Structure",
]

# Batch size:
# Start conservatively.
# Increase if GPU memory permits.
BATCH_SIZE = 32

# Maximum token length.
# MoLFormer training used molecules up to approximately 202 tokens.
MAX_LENGTH = 202

# Number of DataLoader-style worker processes is not used here.
# We perform straightforward batched inference for reproducibility.

RANDOM_SEED = 42


# ============================================================================
# 2. REPRODUCIBILITY
# ============================================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================

def sha256_file(path, chunk_size=1024 * 1024):
    """Calculate SHA256 checksum."""

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def find_smiles_column(df):
    """Identify the SMILES column."""

    # First check exact candidates.
    for col in SMILES_CANDIDATES:
        if col in df.columns:
            return col

    # Then case-insensitive matching.
    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in SMILES_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    # Last-resort heuristic.
    possible = []

    for col in df.columns:

        name = str(col).lower()

        if "smiles" in name:
            possible.append(col)

    if len(possible) == 1:
        return possible[0]

    raise RuntimeError(
        "Could not identify SMILES column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def canonicalize_smiles(smiles):
    """
    Validate and canonicalize SMILES with RDKit.

    Returns:
        canonical_smiles or None
    """

    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if smiles == "":
        return None

    try:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        # Keep stereochemical information.
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )

        return canonical

    except Exception:
        return None


def detect_device():
    """Select GPU if available."""

    if torch.cuda.is_available():

        device = torch.device("cuda")

        gpu_name = torch.cuda.get_device_name(0)

        print(f"DEVICE: CUDA")
        print(f"GPU   : {gpu_name}")

        return device

    print("DEVICE: CPU")

    return torch.device("cpu")


def get_model_dtype(device):
    """
    Use float16 on CUDA to reduce memory consumption.
    CPU remains float32.
    """

    if device.type == "cuda":
        return torch.float16

    return torch.float32


def write_text(path, text):

    with open(path, "w") as f:
        f.write(text)


# ============================================================================
# 5. LOAD MODEL
# ============================================================================

print()
print("=" * 78)
print("SCRIPT 06 — MoLFormer EMBEDDING GENERATION v1")
print("=" * 78)
print()

start_time = time.time()

print("MODEL")
print("-" * 78)
print(f"Model: {MODEL_NAME}")
print()

device = detect_device()

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Tokenizer loaded.")

print()
print("Loading MoLFormer model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    deterministic_eval=True,
    trust_remote_code=True
)

model.eval()

model = model.to(device)

print("Model loaded.")
print()


# ============================================================================
# 6. MODEL INFORMATION
# ============================================================================

print("=" * 78)
print("MODEL INFORMATION")
print("=" * 78)

try:
    hidden_size = model.config.hidden_size
except Exception:
    hidden_size = None

print(f"Hidden size: {hidden_size}")

print(f"Model type : {model.__class__.__name__}")

print(f"Device     : {device}")

print()


# ============================================================================
# 7. PROCESSING FUNCTION
# ============================================================================

def process_dataset(input_file, split_name):

    print()
    print("=" * 78)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # File checksum
    # ------------------------------------------------------------------------

    input_sha256 = sha256_file(input_file)

    print()
    print("Input file:")
    print(f"  {input_file}")

    print()
    print("Input SHA256:")
    print(f"  {input_sha256}")

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    df = pd.read_csv(
        input_file,
        sep="\t"
    )

    print()
    print(f"Shape: {df.shape}")

    # ------------------------------------------------------------------------
    # Required ID
    # ------------------------------------------------------------------------

    if ID_COLUMN not in df.columns:

        raise RuntimeError(
            f"Required ID column '{ID_COLUMN}' not found."
        )

    # ------------------------------------------------------------------------
    # SMILES
    # ------------------------------------------------------------------------

    smiles_column = find_smiles_column(df)

    print(f"SMILES column: {smiles_column}")

    # ------------------------------------------------------------------------
    # Duplicate IDs
    # ------------------------------------------------------------------------

    duplicate_ids = int(
        df[ID_COLUMN].duplicated().sum()
    )

    print(f"Duplicate IDs: {duplicate_ids}")

    if duplicate_ids > 0:

        raise RuntimeError(
            f"{split_name}: duplicate np_id values detected."
        )

    # ------------------------------------------------------------------------
    # Canonicalize and validate SMILES
    # ------------------------------------------------------------------------

    print()
    print("Validating SMILES...")

    original_smiles = df[smiles_column].astype(str).tolist()

    canonical_smiles = []
    invalid_records = []

    for i, smi in enumerate(original_smiles):

        canonical = canonicalize_smiles(smi)

        canonical_smiles.append(canonical)

        if canonical is None:

            invalid_records.append({
                "row_index": i,
                "np_id": df.iloc[i][ID_COLUMN],
                "original_smiles": smi,
                "reason": "RDKit_SMILES_validation_failed"
            })

    valid_mask = [
        x is not None
        for x in canonical_smiles
    ]

    n_total = len(df)
    n_invalid = len(invalid_records)
    n_valid = n_total - n_invalid

    print(f"Total molecules : {n_total}")
    print(f"Valid SMILES    : {n_valid}")
    print(f"Invalid SMILES  : {n_invalid}")

    # ------------------------------------------------------------------------
    # Save invalid SMILES
    # ------------------------------------------------------------------------

    invalid_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_invalid_smiles.tsv"
    )

    if invalid_records:

        invalid_df = pd.DataFrame(
            invalid_records
        )

        invalid_df.to_csv(
            invalid_file,
            sep="\t",
            index=False
        )

    else:

        pd.DataFrame(
            columns=[
                "row_index",
                "np_id",
                "original_smiles",
                "reason"
            ]
        ).to_csv(
            invalid_file,
            sep="\t",
            index=False
        )

    # ------------------------------------------------------------------------
    # Prepare valid data
    # ------------------------------------------------------------------------

    valid_indices = [
        i for i, valid in enumerate(valid_mask)
        if valid
    ]

    valid_smiles = [
        canonical_smiles[i]
        for i in valid_indices
    ]

    valid_ids = [
        df.iloc[i][ID_COLUMN]
        for i in valid_indices
    ]

    # ------------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------------

    print()
    print("Generating embeddings...")

    all_embeddings = []

    n_batches = int(
        np.ceil(len(valid_smiles) / BATCH_SIZE)
    )

    inference_start = time.time()

    for batch_number in range(n_batches):

        start = batch_number * BATCH_SIZE

        end = min(
            start + BATCH_SIZE,
            len(valid_smiles)
        )

        batch_smiles = valid_smiles[start:end]

        # ------------------------------------------------------------
        # Tokenization
        # ------------------------------------------------------------

        inputs = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ------------------------------------------------------------
        # Inference
        # ------------------------------------------------------------

        with torch.no_grad():

            outputs = model(**inputs)

            # Official MoLFormer feature extraction interface.
            embeddings = outputs.pooler_output

        # ------------------------------------------------------------
        # Convert to CPU float32
        # ------------------------------------------------------------

        embeddings = (
            embeddings
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        all_embeddings.append(embeddings)

        # ------------------------------------------------------------
        # Progress
        # ------------------------------------------------------------

        if (
            batch_number == 0
            or (batch_number + 1) % 50 == 0
            or batch_number == n_batches - 1
        ):

            elapsed = time.time() - inference_start

            processed = end

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"  Batch {batch_number + 1}/{n_batches} | "
                f"Molecules {processed}/{len(valid_smiles)} | "
                f"{rate:.2f} mol/s"
            )

    # ------------------------------------------------------------------------
    # Combine embeddings
    # ------------------------------------------------------------------------

    if all_embeddings:

        embedding_matrix = np.vstack(
            all_embeddings
        )

    else:

        embedding_matrix = np.empty(
            (0, hidden_size or 0),
            dtype=np.float32
        )

    print()
    print(
        f"Embedding matrix shape: "
        f"{embedding_matrix.shape}"
    )

    # ------------------------------------------------------------------------
    # Verify dimensions
    # ------------------------------------------------------------------------

    if embedding_matrix.shape[0] != len(valid_ids):

        raise RuntimeError(
            "Number of embeddings does not match "
            "number of valid molecules."
        )

    embedding_dim = embedding_matrix.shape[1]

    print(f"Embedding dimensions: {embedding_dim}")

    # ------------------------------------------------------------------------
    # NaN / Inf QC
    # ------------------------------------------------------------------------

    nan_count = int(
        np.isnan(embedding_matrix).sum()
    )

    inf_count = int(
        np.isinf(embedding_matrix).sum()
    )

    print()
    print("Embedding numeric QC:")
    print(f"  NaN values : {nan_count}")
    print(f"  Inf values : {inf_count}")

    if nan_count > 0 or inf_count > 0:

        raise RuntimeError(
            "Embedding matrix contains NaN or Inf values."
        )

    # ------------------------------------------------------------------------
    # Save matrix
    # ------------------------------------------------------------------------

    matrix_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_embedding_matrix.npy"
    )

    np.save(
        matrix_file,
        embedding_matrix
    )

    # ------------------------------------------------------------------------
    # Build TSV
    # ------------------------------------------------------------------------

    embedding_columns = [
        f"molformer_{i:03d}"
        for i in range(embedding_dim)
    ]

    output_df = pd.DataFrame(
        embedding_matrix,
        columns=embedding_columns
    )

    output_df.insert(
        0,
        ID_COLUMN,
        valid_ids
    )

    # Keep original SMILES.
    output_df.insert(
        1,
        "SMILES_original",
        [
            original_smiles[i]
            for i in valid_indices
        ]
    )

    # Store canonical SMILES used by transformer.
    output_df.insert(
        2,
        "SMILES_canonical",
        valid_smiles
    )

    embedding_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_molformer_embeddings.tsv"
    )

    output_df.to_csv(
        embedding_file,
        sep="\t",
        index=False,
        float_format="%.8g"
    )

    # ------------------------------------------------------------------------
    # Embedding statistics
    # ------------------------------------------------------------------------

    stats = []

    for j, column in enumerate(embedding_columns):

        values = embedding_matrix[:, j]

        stats.append({
            "embedding": column,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "zero_fraction": float(
                np.mean(values == 0)
            ),
            "nan_count": int(
                np.isnan(values).sum()
            ),
            "inf_count": int(
                np.isinf(values).sum()
            )
        })

    stats_df = pd.DataFrame(stats)

    stats_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_embedding_stats.tsv"
    )

    stats_df.to_csv(
        stats_file,
        sep="\t",
        index=False
    )

    # ------------------------------------------------------------------------
    # Norm statistics
    # ------------------------------------------------------------------------

    norms = np.linalg.norm(
        embedding_matrix,
        axis=1
    )

    norm_summary = {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "median": float(np.median(norms)),
        "max": float(np.max(norms)),
    }

    # ------------------------------------------------------------------------
    # Duplicate embedding vectors
    # ------------------------------------------------------------------------

    if len(embedding_matrix) > 0:

        unique_embeddings = np.unique(
            embedding_matrix,
            axis=0
        )

        duplicate_embedding_vectors = (
            len(embedding_matrix)
            - len(unique_embeddings)
        )

    else:

        duplicate_embedding_vectors = 0

    # ------------------------------------------------------------------------
    # Return metadata
    # ------------------------------------------------------------------------

    return {
        "split": split_name,
        "input_file": input_file,
        "input_sha256": input_sha256,
        "n_total": n_total,
        "n_valid_smiles": n_valid,
        "n_invalid_smiles": n_invalid,
        "invalid_smiles_file": invalid_file,
        "smiles_column": str(smiles_column),
        "id_column": ID_COLUMN,
        "embedding_file": embedding_file,
        "matrix_file": matrix_file,
        "statistics_file": stats_file,
        "embedding_dimension": embedding_dim,
        "embedding_rows": int(embedding_matrix.shape[0]),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_embedding_vectors": int(
            duplicate_embedding_vectors
        ),
        "embedding_norm_summary": norm_summary,
    }


# ============================================================================
# 8. RUN TRAIN
# ============================================================================

train_metadata = process_dataset(
    TRAIN_FILE,
    "train"
)


# ============================================================================
# 9. RUN TEST
# ============================================================================

test_metadata = process_dataset(
    TEST_FILE,
    "test"
)


# ============================================================================
# 10. TRAIN / TEST EMBEDDING COMPATIBILITY
# ============================================================================

print()
print("=" * 78)
print("TRAIN / TEST EMBEDDING COMPATIBILITY")
print("=" * 78)

train_dim = train_metadata["embedding_dimension"]
test_dim = test_metadata["embedding_dimension"]

print(f"TRAIN embedding dimension: {train_dim}")
print(f"TEST embedding dimension : {test_dim}")

if train_dim != test_dim:

    raise RuntimeError(
        "TRAIN and TEST embedding dimensions differ."
    )

print("Embedding dimension match: TRUE")


# ============================================================================
# 11. METADATA
# ============================================================================

metadata = {

    "script": "06_generate_molformer_embeddings_v1.py",

    "generated": datetime.now().isoformat(),

    "model": {
        "name": MODEL_NAME,
        "type": "frozen_feature_extractor",
        "fine_tuned": False,
        "trust_remote_code": True,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "pooling": "pooler_output"
    },

    "software": {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": __import__(
            "transformers"
        ).__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },

    "hardware": {
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": bool(
            torch.cuda.is_available()
        ),
    },

    "random_seed": RANDOM_SEED,

    "train": train_metadata,

    "test": test_metadata,

    "compatibility": {
        "embedding_dimensions_match": (
            train_dim == test_dim
        )
    }
}


metadata_file = os.path.join(
    OUTPUT_DIR,
    "embedding_metadata.json"
)

with open(
    metadata_file,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================================
# 12. CHECKSUMS
# ============================================================================

files_to_hash = [

    TRAIN_FILE,
    TEST_FILE,

    train_metadata["embedding_file"],
    test_metadata["embedding_file"],

    train_metadata["matrix_file"],
    test_metadata["matrix_file"],

    train_metadata["statistics_file"],
    test_metadata["statistics_file"],

    metadata_file,
]

checksum_file = os.path.join(
    OUTPUT_DIR,
    "checksums.sha256"
)

with open(
    checksum_file,
    "w"
) as f:

    for path in files_to_hash:

        if os.path.exists(path):

            checksum = sha256_file(path)

            f.write(
                f"{checksum}  {path}\n"
            )


# ============================================================================
# 13. FINAL REPORT
# ============================================================================

elapsed_total = time.time() - start_time

report = []

report.append(
    "=" * 78
)

report.append(
    "SCRIPT 06 — MoLFormer EMBEDDING GENERATION REPORT"
)

report.append(
    "=" * 78
)

report.append("")

report.append(
    f"Generated: {datetime.now().isoformat()}"
)

report.append("")

report.append(
    "MODEL"
)

report.append(
    "-" * 78
)

report.append(
    f"Model: {MODEL_NAME}"
)

report.append(
    "Mode: Frozen feature extraction"
)

report.append(
    "Fine-tuning: NO"
)

report.append(
    f"Pooling: pooler_output"
)

report.append(
    f"Maximum sequence length: {MAX_LENGTH}"
)

report.append(
    f"Embedding dimension: {train_dim}"
)

report.append("")

report.append(
    "TRAIN"
)

report.append(
    "-" * 78
)

report.append(
    f"Input: {TRAIN_FILE}"
)

report.append(
    f"Rows: {train_metadata['n_total']}"
)

report.append(
    f"Valid SMILES: {train_metadata['n_valid_smiles']}"
)

report.append(
    f"Invalid SMILES: {train_metadata['n_invalid_smiles']}"
)

report.append(
    f"Embedding matrix: "
    f"{train_metadata['embedding_rows']} x "
    f"{train_metadata['embedding_dimension']}"
)

report.append(
    f"NaN: {train_metadata['nan_count']}"
)

report.append(
    f"Inf: {train_metadata['inf_count']}"
)

report.append(
    f"Duplicate embedding vectors: "
    f"{train_metadata['duplicate_embedding_vectors']}"
)

report.append("")

report.append(
    "TEST"
)

report.append(
    "-" * 78
)

report.append(
    f"Input: {TEST_FILE}"
)

report.append(
    f"Rows: {test_metadata['n_total']}"
)

report.append(
    f"Valid SMILES: {test_metadata['n_valid_smiles']}"
)

report.append(
    f"Invalid SMILES: {test_metadata['n_invalid_smiles']}"
)

report.append(
    f"Embedding matrix: "
    f"{test_metadata['embedding_rows']} x "
    f"{test_metadata['embedding_dimension']}"
)

report.append(
    f"NaN: {test_metadata['nan_count']}"
)

report.append(
    f"Inf: {test_metadata['inf_count']}"
)

report.append(
    f"Duplicate embedding vectors: "
    f"{test_metadata['duplicate_embedding_vectors']}"
)

report.append("")

report.append(
    "COMPATIBILITY"
)

report.append(
    "-" * 78
)

report.append(
    f"TRAIN/TEST embedding dimension match: "
    f"{train_dim == test_dim}"
)

report.append("")

report.append(
    "OUTPUT FILES"
)

report.append(
    "-" * 78
)

report.append(
    f"{train_metadata['embedding_file']}"
)

report.append(
    f"{test_metadata['embedding_file']}"
)

report.append(
    f"{train_metadata['matrix_file']}"
)

report.append(
    f"{test_metadata['matrix_file']}"
)

report.append(
    f"{train_metadata['statistics_file']}"
)

report.append(
    f"{test_metadata['statistics_file']}"
)

report.append(
    f"{metadata_file}"
)

report.append(
    f"{checksum_file}"
)

report.append("")

report.append(
    f"TOTAL RUNTIME: {elapsed_total:.2f} seconds"
)

report.append("")

report.append(
    "=" * 78
)

report.append(
    "SCRIPT 06 COMPLETE"
)

report.append(
    "=" * 78
)


report_text = "\n".join(report)

report_file = os.path.join(
    OUTPUT_DIR,
    "embedding_generation_report.txt"
)

write_text(
    report_file,
    report_text
)

print()
print(report_text)

print()
print(
    f"Report written: {report_file}"
)

print(
    f"Metadata written: {metadata_file}"
)

print()
print("=" * 78)
print("MOlFORMER EMBEDDING GENERATION COMPLETE")
print("=" * 78)
