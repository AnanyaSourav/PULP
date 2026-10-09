#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 06 — MoLFormer EMBEDDING GENERATION v3
==============================================================================

Purpose
-------
Generate 768-dimensional MoLFormer embeddings directly from SMILES.

INPUT
-----
feature_engineering/split/train_80.tsv
feature_engineering/split/test_20.tsv

SMILES column
-------------
SMILES

ID column
---------
np_id

LABEL
-----
Activity_Label

MODEL
-----
ibm-research/MoLFormer-XL-both-10pct

OUTPUT
------
feature_engineering/molformer_embeddings/
    train_80_molformer_embeddings.tsv
    test_20_molformer_embeddings.tsv
    train_80_molformer_embeddings.npy
    test_20_molformer_embeddings.npy
    molformer_embedding_metadata.json
    molformer_embedding_qc_report.json

IMPORTANT
---------
- No features are removed.
- No rows are removed.
- Existing train/test split is preserved exactly.
- np_id is retained for alignment.
- Embeddings are generated from SMILES, NOT from the 228 descriptors.
- The activity label is NOT used by MoLFormer.
- Mean pooling over the last hidden state is used.
- Output dimensionality is 768.
==============================================================================

"""

import os
import sys
import json
import hashlib
import time
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"

TRAIN_FILE = "feature_engineering/split/train_80.tsv"
TEST_FILE  = "feature_engineering/split/test_20.tsv"

OUTPUT_DIR = "feature_engineering/molformer_embeddings"

ID_COLUMN = "np_id"
SMILES_COLUMN = "SMILES"
LABEL_COLUMN = "Activity_Label"

BATCH_SIZE = 32
MAX_LENGTH = 512

# Number of dimensions expected from MoLFormer
EMBEDDING_DIM = 768


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def print_section(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def clean_smiles(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


# =============================================================================
# START
# =============================================================================

start_time = datetime.now()

print("=" * 78)
print("SCRIPT 06 — MoLFormer EMBEDDING GENERATION v3")
print("=" * 78)

print(f"Started: {start_time.isoformat()}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {__import__('transformers').__version__}")
print(f"Model: {MODEL_NAME}")
print(f"CPU threads: {os.cpu_count()}")

cuda_available = torch.cuda.is_available()

print(f"CUDA available: {cuda_available}")

if cuda_available:
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Device: cpu")


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. INPUT FILE CHECK
# =============================================================================

print_section("1. INPUT FILE CHECK")

for path in [TRAIN_FILE, TEST_FILE]:

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )

    print(f"FOUND: {path}")


# =============================================================================
# 2. INPUT SHA256
# =============================================================================

print_section("2. INPUT SHA256")

train_sha256 = sha256_file(TRAIN_FILE)
test_sha256 = sha256_file(TEST_FILE)

print(f"TRAIN SHA256:\n  {train_sha256}")
print(f"TEST SHA256:\n  {test_sha256}")


# =============================================================================
# 3. LOAD DATA
# =============================================================================

print_section("3. LOADING INPUT DATA")

train = pd.read_csv(
    TRAIN_FILE,
    sep="\t",
    dtype={ID_COLUMN: str},
    low_memory=False
)

test = pd.read_csv(
    TEST_FILE,
    sep="\t",
    dtype={ID_COLUMN: str},
    low_memory=False
)

print(f"TRAIN shape: {train.shape}")
print(f"TEST shape : {test.shape}")


# =============================================================================
# 4. COLUMN VALIDATION
# =============================================================================

print_section("4. COLUMN VALIDATION")

required_columns = [
    ID_COLUMN,
    SMILES_COLUMN,
    LABEL_COLUMN
]

for column in required_columns:

    if column not in train.columns:
        raise RuntimeError(
            f"TRAIN missing required column: {column}"
        )

    if column not in test.columns:
        raise RuntimeError(
            f"TEST missing required column: {column}"
        )

print(f"ID column       : {ID_COLUMN}")
print(f"SMILES column   : {SMILES_COLUMN}")
print(f"Activity column : {LABEL_COLUMN}")

print("Column validation: PASS")


# =============================================================================
# 5. ID QC
# =============================================================================

print_section("5. ID QC")

train_ids = train[ID_COLUMN].astype(str)
test_ids = test[ID_COLUMN].astype(str)

train_duplicate_ids = int(train_ids.duplicated().sum())
test_duplicate_ids = int(test_ids.duplicated().sum())

overlap = set(train_ids) & set(test_ids)

print(f"TRAIN duplicate IDs: {train_duplicate_ids}")
print(f"TEST duplicate IDs : {test_duplicate_ids}")
print(f"TRAIN/TEST overlap : {len(overlap)}")

if train_duplicate_ids != 0:
    raise RuntimeError("TRAIN contains duplicate np_id values.")

if test_duplicate_ids != 0:
    raise RuntimeError("TEST contains duplicate np_id values.")

if len(overlap) != 0:
    raise RuntimeError(
        "TRAIN/TEST ID overlap detected."
    )

print("ID QC: PASS")


# =============================================================================
# 6. SMILES QC
# =============================================================================

print_section("6. SMILES QC")

train_smiles = train[SMILES_COLUMN].map(clean_smiles)
test_smiles = test[SMILES_COLUMN].map(clean_smiles)

train_missing_smiles = int(train_smiles.isna().sum())
test_missing_smiles = int(test_smiles.isna().sum())

print(f"TRAIN missing SMILES: {train_missing_smiles}")
print(f"TEST missing SMILES : {test_missing_smiles}")

if train_missing_smiles > 0:
    raise RuntimeError(
        f"TRAIN contains {train_missing_smiles} missing SMILES."
    )

if test_missing_smiles > 0:
    raise RuntimeError(
        f"TEST contains {test_missing_smiles} missing SMILES."
    )

print("SMILES QC: PASS")


# =============================================================================
# 7. LOAD MoLFormer
# =============================================================================

print_section("7. LOADING MoLFormer")

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Tokenizer: OK")

print()
print("Loading model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)
model.eval()

print("Model: OK")


# =============================================================================
# 8. MODEL VALIDATION
# =============================================================================

print_section("8. MODEL VALIDATION")

test_smiles_examples = [
    "CCO",
    "CC(=O)OC1=CC=CC=C1C(=O)O"
]

with torch.no_grad():

    encoded = tokenizer(
        test_smiles_examples,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    outputs = model(**encoded)

    hidden = outputs.last_hidden_state

print(f"Test hidden-state shape: {tuple(hidden.shape)}")

if hidden.shape[-1] != EMBEDDING_DIM:

    raise RuntimeError(
        f"Unexpected embedding dimension: "
        f"{hidden.shape[-1]} "
        f"(expected {EMBEDDING_DIM})"
    )

print(f"Embedding dimension: {EMBEDDING_DIM}")
print("Model validation: PASS")


# =============================================================================
# 9. EMBEDDING FUNCTION
# =============================================================================

def generate_embeddings(smiles_list):

    all_embeddings = []

    total = len(smiles_list)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch_smiles = smiles_list[start:end]

        encoded = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model(**encoded)

            hidden = outputs.last_hidden_state

            attention_mask = encoded["attention_mask"]

            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()

            masked_hidden = hidden * mask

            summed = masked_hidden.sum(dim=1)

            counts = mask.sum(dim=1).clamp(min=1e-9)

            embeddings = summed / counts

        embeddings = embeddings.detach().cpu().numpy()

        all_embeddings.append(
            embeddings.astype(np.float32)
        )

        completed = end

        percent = 100.0 * completed / total

        print(
            f"  Processed {completed:,}/{total:,} "
            f"({percent:6.2f}%)",
            flush=True
        )

    return np.vstack(all_embeddings)


# =============================================================================
# 10. TRAIN EMBEDDINGS
# =============================================================================

print_section("10. GENERATING TRAIN EMBEDDINGS")

train_start = time.time()

train_embeddings = generate_embeddings(
    train_smiles.tolist()
)

train_time = time.time() - train_start

print()
print(f"TRAIN embedding matrix: {train_embeddings.shape}")
print(f"TRAIN generation time: {train_time / 60:.2f} minutes")


# =============================================================================
# 11. TEST EMBEDDINGS
# =============================================================================

print_section("11. GENERATING TEST EMBEDDINGS")

test_start = time.time()

test_embeddings = generate_embeddings(
    test_smiles.tolist()
)

test_time = time.time() - test_start

print()
print(f"TEST embedding matrix: {test_embeddings.shape}")
print(f"TEST generation time: {test_time / 60:.2f} minutes")


# =============================================================================
# 12. EMBEDDING MATRIX QC
# =============================================================================

print_section("12. EMBEDDING MATRIX QC")

expected_train_shape = (
    len(train),
    EMBEDDING_DIM
)

expected_test_shape = (
    len(test),
    EMBEDDING_DIM
)

if train_embeddings.shape != expected_train_shape:

    raise RuntimeError(
        f"Unexpected TRAIN embedding shape: "
        f"{train_embeddings.shape}; "
        f"expected {expected_train_shape}"
    )

if test_embeddings.shape != expected_test_shape:

    raise RuntimeError(
        f"Unexpected TEST embedding shape: "
        f"{test_embeddings.shape}; "
        f"expected {expected_test_shape}"
    )

train_nan = int(np.isnan(train_embeddings).sum())
test_nan = int(np.isnan(test_embeddings).sum())

train_inf = int(np.isinf(train_embeddings).sum())
test_inf = int(np.isinf(test_embeddings).sum())

print(f"TRAIN NaN values : {train_nan}")
print(f"TEST NaN values  : {test_nan}")
print(f"TRAIN Inf values : {train_inf}")
print(f"TEST Inf values  : {test_inf}")

if train_nan or test_nan or train_inf or test_inf:

    raise RuntimeError(
        "NaN or infinite values detected in embeddings."
    )

print("Embedding matrix QC: PASS")


# =============================================================================
# 13. SAVE NUMPY MATRICES
# =============================================================================

print_section("13. WRITING NUMPY OUTPUT")

train_npy = os.path.join(
    OUTPUT_DIR,
    "train_80_molformer_embeddings.npy"
)

test_npy = os.path.join(
    OUTPUT_DIR,
    "test_20_molformer_embeddings.npy"
)

np.save(train_npy, train_embeddings)
np.save(test_npy, test_embeddings)

print(f"Written: {train_npy}")
print(f"Written: {test_npy}")


# =============================================================================
# 14. SAVE TSV FILES
# =============================================================================

print_section("14. WRITING TSV OUTPUT")

embedding_columns = [
    f"molformer_{i:03d}"
    for i in range(1, EMBEDDING_DIM + 1)
]

train_output = pd.DataFrame(
    train_embeddings,
    columns=embedding_columns
)

test_output = pd.DataFrame(
    test_embeddings,
    columns=embedding_columns
)

# Retain identity and label information
train_output.insert(
    0,
    LABEL_COLUMN,
    train[LABEL_COLUMN].values
)

train_output.insert(
    0,
    SMILES_COLUMN,
    train[SMILES_COLUMN].values
)

train_output.insert(
    0,
    ID_COLUMN,
    train[ID_COLUMN].values
)

test_output.insert(
    0,
    LABEL_COLUMN,
    test[LABEL_COLUMN].values
)

test_output.insert(
    0,
    SMILES_COLUMN,
    test[SMILES_COLUMN].values
)

test_output.insert(
    0,
    ID_COLUMN,
    test[ID_COLUMN].values
)

train_tsv = os.path.join(
    OUTPUT_DIR,
    "train_80_molformer_embeddings.tsv"
)

test_tsv = os.path.join(
    OUTPUT_DIR,
    "test_20_molformer_embeddings.tsv"
)

train_output.to_csv(
    train_tsv,
    sep="\t",
    index=False
)

test_output.to_csv(
    test_tsv,
    sep="\t",
    index=False
)

print(f"Written: {train_tsv}")
print(f"Written: {test_tsv}")


# =============================================================================
# 15. OUTPUT ALIGNMENT QC
# =============================================================================

print_section("15. OUTPUT ALIGNMENT QC")

if not np.array_equal(
    train_output[ID_COLUMN].astype(str).values,
    train_ids.values
):
    raise RuntimeError(
        "TRAIN ID ordering changed during embedding generation."
    )

if not np.array_equal(
    test_output[ID_COLUMN].astype(str).values,
    test_ids.values
):
    raise RuntimeError(
        "TEST ID ordering changed during embedding generation."
    )

if not np.array_equal(
    train_output[LABEL_COLUMN].values,
    train[LABEL_COLUMN].values
):
    raise RuntimeError(
        "TRAIN activity labels changed."
    )

if not np.array_equal(
    test_output[LABEL_COLUMN].values,
    test[LABEL_COLUMN].values
):
    raise RuntimeError(
        "TEST activity labels changed."
    )

print("TRAIN row alignment: PASS")
print("TEST row alignment : PASS")
print("Activity labels     : PASS")


# =============================================================================
# 16. OUTPUT SHA256
# =============================================================================

print_section("16. OUTPUT SHA256")

output_sha256 = {

    "train_npy": sha256_file(train_npy),

    "test_npy": sha256_file(test_npy),

    "train_tsv": sha256_file(train_tsv),

    "test_tsv": sha256_file(test_tsv)

}

for key, value in output_sha256.items():

    print(f"{key}:")
    print(f"  {value}")


# =============================================================================
# 17. MACHINE-READABLE QC REPORT
# =============================================================================

print_section("17. WRITING MACHINE-READABLE QC REPORT")

end_time = datetime.now()

qc_report = {

    "script": "06_generate_molformer_embeddings_v3.py",

    "status": "PASS",

    "generated": end_time.isoformat(),

    "model": {
        "name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "pooling": "attention_masked_mean_pooling",
        "max_length": MAX_LENGTH
    },

    "environment": {
        "python": sys.version,
        "pytorch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": str(device),
        "cuda_available": bool(cuda_available),
        "cpu_count": os.cpu_count()
    },

    "inputs": {

        "train": {
            "file": TRAIN_FILE,
            "rows": int(len(train)),
            "sha256": train_sha256
        },

        "test": {
            "file": TEST_FILE,
            "rows": int(len(test)),
            "sha256": test_sha256
        }

    },

    "columns": {
        "id": ID_COLUMN,
        "smiles": SMILES_COLUMN,
        "activity_label": LABEL_COLUMN
    },

    "id_qc": {

        "train_duplicate_ids": train_duplicate_ids,

        "test_duplicate_ids": test_duplicate_ids,

        "train_test_overlap": len(overlap)

    },

    "smiles_qc": {

        "train_missing": train_missing_smiles,

        "test_missing": test_missing_smiles

    },

    "embeddings": {

        "train_shape": list(train_embeddings.shape),

        "test_shape": list(test_embeddings.shape),

        "train_dtype": str(train_embeddings.dtype),

        "test_dtype": str(test_embeddings.dtype),

        "train_nan": train_nan,

        "test_nan": test_nan,

        "train_inf": train_inf,

        "test_inf": test_inf

    },

    "outputs": {

        "train_npy": train_npy,

        "test_npy": test_npy,

        "train_tsv": train_tsv,

        "test_tsv": test_tsv,

        "sha256": output_sha256

    },

    "runtime": {

        "train_minutes": round(train_time / 60, 3),

        "test_minutes": round(test_time / 60, 3),

        "total_minutes": round(
            (time.time() - train_start) / 60,
            3
        )

    }

}

qc_path = os.path.join(
    OUTPUT_DIR,
    "molformer_embedding_qc_report.json"
)

with open(
    qc_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        qc_report,
        f,
        indent=2
    )

print(f"Written: {qc_path}")


# =============================================================================
# 18. METADATA
# =============================================================================

metadata = {

    "model": MODEL_NAME,

    "embedding_dimension": EMBEDDING_DIM,

    "pooling": "attention_masked_mean_pooling",

    "input_smiles_column": SMILES_COLUMN,

    "id_column": ID_COLUMN,

    "activity_label_column": LABEL_COLUMN,

    "train_input": TRAIN_FILE,

    "test_input": TEST_FILE,

    "train_input_sha256": train_sha256,

    "test_input_sha256": test_sha256,

    "train_rows": int(len(train)),

    "test_rows": int(len(test)),

    "batch_size": BATCH_SIZE,

    "max_length": MAX_LENGTH,

    "device": str(device),

    "generated": end_time.isoformat()

}

metadata_path = os.path.join(
    OUTPUT_DIR,
    "molformer_embedding_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

print(f"Written: {metadata_path}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 78)
print("SCRIPT 06 MoLFormer EMBEDDING GENERATION COMPLETE")
print("=" * 78)

print()
print("FINAL STATUS: PASS")

print()
print("INPUT")
print(f"  TRAIN rows: {len(train):,}")
print(f"  TEST rows : {len(test):,}")

print()
print("EMBEDDINGS")
print(f"  TRAIN: {train_embeddings.shape}")
print(f"  TEST : {test_embeddings.shape}")

print()
print("MODEL")
print(f"  {MODEL_NAME}")

print()
print("OUTPUT DIRECTORY")
print(f"  {OUTPUT_DIR}")

print()
print("OUTPUT FILES")
print(f"  {train_tsv}")
print(f"  {test_tsv}")
print(f"  {train_npy}")
print(f"  {test_npy}")
print(f"  {qc_path}")
print(f"  {metadata_path}")

print()
print(f"Finished: {end_time.isoformat()}")

print("=" * 78)
