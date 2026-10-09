#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 06 — MoLFormer EMBEDDING GENERATION v2
==============================================================================

Purpose
-------
Generate MoLFormer molecular embeddings from SMILES for the finalized
TRAIN/TEST datasets.

INPUT
-----
feature_engineering/final_228_features/train_80_final_228_features.tsv
feature_engineering/final_228_features/test_20_final_228_features.tsv

MODEL
-----
ibm-research/MoLFormer-XL-both-10pct

OUTPUT
------
feature_engineering/molformer_embeddings_v2/

    train_molformer_embeddings.npy
    test_molformer_embeddings.npy

    train_molformer_metadata.tsv
    test_molformer_metadata.tsv

    train_failed_smiles.tsv
    test_failed_smiles.tsv

    train_checkpoint.npz
    test_checkpoint.npz

    molformer_embedding_qc_report.json
    molformer_embedding_qc_report.txt

NOTES
-----
- NO feature removal
- NO dataset modification
- NO activity-label modification
- Embedding dimension = 768
- CPU compatible
- Checkpoint/resume supported
- Input SHA256 recorded
- Failed SMILES are logged
- Original row order is preserved
==============================================================================

"""

import os
import sys
import json
import time
import hashlib
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"

TRAIN_FILE = (
    "feature_engineering/final_228_features/"
    "train_80_final_228_features.tsv"
)

TEST_FILE = (
    "feature_engineering/final_228_features/"
    "test_20_final_228_features.tsv"
)

OUTPUT_DIR = "feature_engineering/molformer_embeddings_v2"

ID_COLUMN = "np_id"

# We will automatically detect the SMILES column.
SMILES_CANDIDATES = [
    "SMILES",
    "smiles",
    "Smiles",
    "canonical_smiles",
    "Canonical_SMILES",
    "canonical_SMILES",
    "mol_smiles",
]

ACTIVITY_CANDIDATES = [
    "Activity",
    "activity",
    "label",
    "Label",
    "active",
    "Active",
    "class",
    "Class",
]

# CPU-safe batch size.
# Start conservatively because MoLFormer is being run without CUDA.
BATCH_SIZE = 16

# Save a checkpoint after this many processed rows.
CHECKPOINT_INTERVAL = 256

# Maximum SMILES sequence length.
MAX_LENGTH = 512

# Number of CPU threads.
CPU_THREADS = max(1, min(8, os.cpu_count() or 1))

# Use pooler_output from the verified model test.
USE_POOLER_OUTPUT = True


# =============================================================================
# 1. UTILITY FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    """Calculate SHA256 checksum of a file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_section(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def find_column(df, candidates):
    """Return the first matching column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def atomic_save_npy(path, array):
    """Safely write numpy array using a temporary file."""
    temp = path + ".tmp"

    with open(temp, "wb") as f:
        np.save(f, array)

    os.replace(temp, path)


def atomic_save_npz(path, **arrays):
    """Safely write numpy checkpoint."""
    temp = path + ".tmp"

    np.savez_compressed(temp, **arrays)

    # np.savez_compressed adds .npz if not present.
    actual_temp = temp
    if not actual_temp.endswith(".npz"):
        actual_temp = temp + ".npz"

    os.replace(actual_temp, path)


def safe_json_dump(data, path):
    temp = path + ".tmp"

    with open(temp, "w") as f:
        json.dump(data, f, indent=2, default=str)

    os.replace(temp, path)


# =============================================================================
# 2. ENVIRONMENT
# =============================================================================

print_header("SCRIPT 06 — MoLFormer EMBEDDING GENERATION v2")

print("Started:", timestamp())
print("Python:", sys.version.replace("\n", " "))
print("PyTorch:", torch.__version__)

try:
    import transformers

    print("Transformers:", transformers.__version__)
except Exception:
    print("Transformers: unavailable")

print("Model:", MODEL_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_num_threads(CPU_THREADS)

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("CPU threads:", CPU_THREADS)
print("CUDA available:", CUDA_AVAILABLE)
print("Device:", DEVICE)


# =============================================================================
# 3. INPUT FILE CHECK
# =============================================================================

print_section("1. INPUT FILE CHECK")

if not os.path.isfile(TRAIN_FILE):
    raise FileNotFoundError(
        f"TRAIN file not found:\n{TRAIN_FILE}"
    )

if not os.path.isfile(TEST_FILE):
    raise FileNotFoundError(
        f"TEST file not found:\n{TEST_FILE}"
    )

print("TRAIN:")
print(" ", TRAIN_FILE)

print("TEST:")
print(" ", TEST_FILE)


# =============================================================================
# 4. SHA256
# =============================================================================

print_section("2. INPUT SHA256")

train_sha256 = sha256_file(TRAIN_FILE)
test_sha256 = sha256_file(TEST_FILE)

print("TRAIN SHA256:")
print(" ", train_sha256)

print("TEST SHA256:")
print(" ", test_sha256)


# =============================================================================
# 5. LOAD DATA
# =============================================================================

print_section("3. LOADING INPUT DATA")

train_df = pd.read_csv(
    TRAIN_FILE,
    sep="\t",
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    sep="\t",
    low_memory=False
)

print("TRAIN shape:", train_df.shape)
print("TEST shape :", test_df.shape)


# =============================================================================
# 6. IDENTIFY COLUMNS
# =============================================================================

print_section("4. IDENTIFYING COLUMNS")

if ID_COLUMN not in train_df.columns:
    raise RuntimeError(
        f"Required ID column '{ID_COLUMN}' not found in TRAIN."
    )

if ID_COLUMN not in test_df.columns:
    raise RuntimeError(
        f"Required ID column '{ID_COLUMN}' not found in TEST."
    )

train_smiles_col = find_column(train_df, SMILES_CANDIDATES)
test_smiles_col = find_column(test_df, SMILES_CANDIDATES)

if train_smiles_col is None:
    raise RuntimeError(
        "Could not identify SMILES column in TRAIN.\n"
        f"Available columns:\n{list(train_df.columns)}"
    )

if test_smiles_col is None:
    raise RuntimeError(
        "Could not identify SMILES column in TEST.\n"
        f"Available columns:\n{list(test_df.columns)}"
    )

if train_smiles_col != test_smiles_col:
    print(
        "WARNING: TRAIN and TEST use different SMILES column names."
    )

train_activity_col = find_column(train_df, ACTIVITY_CANDIDATES)
test_activity_col = find_column(test_df, ACTIVITY_CANDIDATES)

print("ID column:")
print(" ", ID_COLUMN)

print("TRAIN SMILES column:")
print(" ", train_smiles_col)

print("TEST SMILES column:")
print(" ", test_smiles_col)

print("TRAIN activity column:")
print(" ", train_activity_col)

print("TEST activity column:")
print(" ", test_activity_col)


# =============================================================================
# 7. BASIC INPUT QC
# =============================================================================

print_section("5. INPUT DATA QC")

input_qc = {}

input_qc["train_rows"] = int(len(train_df))
input_qc["test_rows"] = int(len(test_df))

input_qc["train_duplicate_ids"] = int(
    train_df[ID_COLUMN].duplicated().sum()
)

input_qc["test_duplicate_ids"] = int(
    test_df[ID_COLUMN].duplicated().sum()
)

train_ids = set(train_df[ID_COLUMN].astype(str))
test_ids = set(test_df[ID_COLUMN].astype(str))

input_qc["train_test_id_overlap"] = int(
    len(train_ids.intersection(test_ids))
)

print("TRAIN rows:", len(train_df))
print("TEST rows :", len(test_df))

print("TRAIN duplicate IDs:", input_qc["train_duplicate_ids"])
print("TEST duplicate IDs :", input_qc["test_duplicate_ids"])
print(
    "TRAIN/TEST ID overlap:",
    input_qc["train_test_id_overlap"]
)


# =============================================================================
# 8. LOAD MODEL
# =============================================================================

print_section("6. LOADING MoLFormer")

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Tokenizer: OK")

print("\nLoading model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(DEVICE)
model.eval()

print("Model: OK")
print("Device:", DEVICE)


# =============================================================================
# 9. MODEL DIMENSION TEST
# =============================================================================

print_section("7. MODEL EMBEDDING TEST")

test_smiles = ["CCO"]

test_inputs = tokenizer(
    test_smiles,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

test_inputs = {
    k: v.to(DEVICE)
    for k, v in test_inputs.items()
}

with torch.no_grad():
    test_outputs = model(**test_inputs)

if USE_POOLER_OUTPUT and hasattr(
    test_outputs,
    "pooler_output"
) and test_outputs.pooler_output is not None:

    test_embedding = test_outputs.pooler_output

else:

    # Fallback: mean pooling over valid tokens.
    hidden = test_outputs.last_hidden_state
    attention_mask = test_inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()

    summed = torch.sum(hidden * mask, dim=1)

    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    test_embedding = summed / counts

EMBEDDING_DIM = int(test_embedding.shape[-1])

print("Embedding dimension:", EMBEDDING_DIM)

if EMBEDDING_DIM != 768:
    raise RuntimeError(
        f"Unexpected MoLFormer embedding dimension: "
        f"{EMBEDDING_DIM}. Expected 768."
    )

print("Embedding dimension check: PASS")


# =============================================================================
# 10. EMBEDDING FUNCTION
# =============================================================================

def generate_embeddings(
    df,
    smiles_column,
    split_name,
    output_prefix
):
    """
    Generate embeddings for one dataset.

    Resume behavior:
    - If checkpoint exists, continue from checkpoint.
    - Previously processed rows are not recomputed.
    """

    print_header(
        f"{split_name.upper()} — MoLFormer EMBEDDING GENERATION"
    )

    n = len(df)

    final_embedding_file = os.path.join(
        OUTPUT_DIR,
        f"{output_prefix}_molformer_embeddings.npy"
    )

    metadata_file = os.path.join(
        OUTPUT_DIR,
        f"{output_prefix}_molformer_metadata.tsv"
    )

    failed_file = os.path.join(
        OUTPUT_DIR,
        f"{output_prefix}_failed_smiles.tsv"
    )

    checkpoint_file = os.path.join(
        OUTPUT_DIR,
        f"{output_prefix}_checkpoint.npz"
    )

    # -------------------------------------------------------------------------
    # Initialize storage
    # -------------------------------------------------------------------------

    embeddings = np.zeros(
        (n, EMBEDDING_DIM),
        dtype=np.float32
    )

    processed = np.zeros(
        n,
        dtype=np.bool_
    )

    failed = np.zeros(
        n,
        dtype=np.bool_
    )

    failed_records = []

    # -------------------------------------------------------------------------
    # Resume checkpoint if present
    # -------------------------------------------------------------------------

    if os.path.exists(checkpoint_file):

        print("Existing checkpoint found:")
        print(" ", checkpoint_file)

        checkpoint = np.load(
            checkpoint_file,
            allow_pickle=True
        )

        old_embeddings = checkpoint["embeddings"]
        old_processed = checkpoint["processed"]
        old_failed = checkpoint["failed"]

        if old_embeddings.shape != embeddings.shape:
            raise RuntimeError(
                "Checkpoint shape does not match current input dataset."
            )

        embeddings[:] = old_embeddings
        processed[:] = old_processed
        failed[:] = old_failed

        if "failed_records" in checkpoint:
            raw_failed = checkpoint["failed_records"]

            for item in raw_failed:
                failed_records.append(
                    item.item()
                    if hasattr(item, "item")
                    else item
                )

        print(
            "Previously processed:",
            int(processed.sum())
        )

        print(
            "Previously failed:",
            int(failed.sum())
        )

        print("Resuming...")

    else:

        print("No checkpoint found.")
        print("Starting from row 0.")

    # -------------------------------------------------------------------------
    # Process batches
    # -------------------------------------------------------------------------

    start_time = time.time()

    total_batches = (
        (n + BATCH_SIZE - 1) // BATCH_SIZE
    )

    for batch_number, start in enumerate(
        range(0, n, BATCH_SIZE),
        start=1
    ):

        end = min(start + BATCH_SIZE, n)

        batch_indices = [
            i for i in range(start, end)
            if not processed[i]
        ]

        if not batch_indices:
            continue

        smiles_batch = []

        valid_indices = []

        for i in batch_indices:

            value = df.iloc[i][smiles_column]

            if pd.isna(value):
                failed[i] = True

                failed_records.append({
                    "row_index": int(i),
                    "np_id": str(df.iloc[i][ID_COLUMN]),
                    "smiles": "",
                    "error": "missing_smiles"
                })

                continue

            smiles = str(value).strip()

            if not smiles:
                failed[i] = True

                failed_records.append({
                    "row_index": int(i),
                    "np_id": str(df.iloc[i][ID_COLUMN]),
                    "smiles": "",
                    "error": "empty_smiles"
                })

                continue

            smiles_batch.append(smiles)
            valid_indices.append(i)

        # ---------------------------------------------------------------------
        # Tokenization
        # ---------------------------------------------------------------------

        if smiles_batch:

            try:

                inputs = tokenizer(
                    smiles_batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )

                inputs = {
                    k: v.to(DEVICE)
                    for k, v in inputs.items()
                }

                # -------------------------------------------------------------
                # Forward pass
                # -------------------------------------------------------------

                with torch.no_grad():

                    outputs = model(**inputs)

                    if (
                        USE_POOLER_OUTPUT
                        and hasattr(outputs, "pooler_output")
                        and outputs.pooler_output is not None
                    ):

                        batch_embeddings = (
                            outputs.pooler_output
                        )

                    else:

                        hidden = outputs.last_hidden_state

                        attention_mask = (
                            inputs["attention_mask"]
                        )

                        mask = (
                            attention_mask
                            .unsqueeze(-1)
                            .expand(hidden.size())
                            .float()
                        )

                        summed = torch.sum(
                            hidden * mask,
                            dim=1
                        )

                        counts = torch.clamp(
                            mask.sum(dim=1),
                            min=1e-9
                        )

                        batch_embeddings = (
                            summed / counts
                        )

                batch_embeddings = (
                    batch_embeddings
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                if batch_embeddings.shape != (
                    len(valid_indices),
                    EMBEDDING_DIM
                ):
                    raise RuntimeError(
                        "Unexpected batch embedding shape: "
                        f"{batch_embeddings.shape}"
                    )

                for local_idx, global_idx in enumerate(
                    valid_indices
                ):

                    embeddings[global_idx] = (
                        batch_embeddings[local_idx]
                    )

                    processed[global_idx] = True

            except Exception as exc:

                error_text = (
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"\nERROR in batch "
                    f"{batch_number}/{total_batches}: "
                    f"{error_text}"
                )

                traceback.print_exc()

                # Mark every molecule in this batch as failed.
                # This allows the run to continue and preserves provenance.

                for i, smiles in zip(
                    valid_indices,
                    smiles_batch
                ):

                    failed[i] = True

                    failed_records.append({
                        "row_index": int(i),
                        "np_id": str(df.iloc[i][ID_COLUMN]),
                        "smiles": smiles,
                        "error": error_text
                    })

        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        completed = int(processed.sum())
        failed_count = int(failed.sum())

        elapsed = time.time() - start_time

        if elapsed > 0:
            rate = completed / elapsed
        else:
            rate = 0.0

        remaining = n - completed - failed_count

        if rate > 0:
            eta_seconds = remaining / rate
        else:
            eta_seconds = 0

        print(
            f"\r{split_name}: "
            f"batch {batch_number}/{total_batches} | "
            f"processed={completed:,} | "
            f"failed={failed_count:,} | "
            f"remaining={remaining:,} | "
            f"rate={rate:.2f}/s | "
            f"ETA={eta_seconds/60:.1f} min",
            end="",
            flush=True
        )

        # ---------------------------------------------------------------------
        # Checkpoint
        # ---------------------------------------------------------------------

        if (
            completed + failed_count
        ) % CHECKPOINT_INTERVAL < BATCH_SIZE:

            failed_array = np.array(
                failed_records,
                dtype=object
            )

            atomic_save_npz(
                checkpoint_file,
                embeddings=embeddings,
                processed=processed,
                failed=failed,
                failed_records=failed_array
            )

    print("\n")

    # =========================================================================
    # 11. FINAL CHECKPOINT
    # =========================================================================

    failed_array = np.array(
        failed_records,
        dtype=object
    )

    atomic_save_npz(
        checkpoint_file,
        embeddings=embeddings,
        processed=processed,
        failed=failed,
        failed_records=failed_array
    )

    # =========================================================================
    # 12. SAVE EMBEDDINGS
    # =========================================================================

    print_section(
        f"{split_name.upper()} — SAVING EMBEDDINGS"
    )

    atomic_save_npy(
        final_embedding_file,
        embeddings
    )

    print("Written:")
    print(" ", final_embedding_file)

    # =========================================================================
    # 13. SAVE METADATA
    # =========================================================================

    metadata = pd.DataFrame()

    metadata["row_index"] = np.arange(n)

    metadata[ID_COLUMN] = (
        df[ID_COLUMN].astype(str).values
    )

    metadata["SMILES"] = (
        df[smiles_column].astype(str).values
    )

    if (
        split_name.lower() == "train"
        and train_activity_col is not None
    ):

        metadata["activity"] = (
            df[train_activity_col].astype(str).values
        )

    elif (
        split_name.lower() == "test"
        and test_activity_col is not None
    ):

        metadata["activity"] = (
            df[test_activity_col].astype(str).values
        )

    metadata["embedding_status"] = np.where(
        processed,
        "success",
        np.where(
            failed,
            "failed",
            "not_processed"
        )
    )

    metadata.to_csv(
        metadata_file,
        sep="\t",
        index=False
    )

    print("Written:")
    print(" ", metadata_file)

    # =========================================================================
    # 14. FAILED SMILES
    # =========================================================================

    if failed_records:

        failed_df = pd.DataFrame(
            failed_records
        )

        failed_df.to_csv(
            failed_file,
            sep="\t",
            index=False
        )

        print("Failed SMILES:")
        print(" ", failed_file)

    else:

        # Write an empty machine-readable file with headers.
        failed_df = pd.DataFrame(
            columns=[
                "row_index",
                "np_id",
                "smiles",
                "error"
            ]
        )

        failed_df.to_csv(
            failed_file,
            sep="\t",
            index=False
        )

        print("No failed SMILES.")

    # =========================================================================
    # 15. FINAL VALIDATION
    # =========================================================================

    success_count = int(processed.sum())
    failed_count = int(failed.sum())
    unresolved_count = n - success_count - failed_count

    finite_rows = np.isfinite(
        embeddings
    ).all(axis=1)

    embedding_nonfinite_rows = int(
        (~finite_rows).sum()
    )

    successful_nonfinite_rows = int(
        (
            processed
            & (~finite_rows)
        ).sum()
    )

    # Check that successful embeddings are not zero vectors.
    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    successful_zero_vectors = int(
        (
            processed
            & (norms == 0)
        ).sum()
    )

    # =========================================================================
    # 16. RESULT
    # =========================================================================

    print_section(
        f"{split_name.upper()} — FINAL RESULT"
    )

    print("Input rows:", n)
    print("Successful embeddings:", success_count)
    print("Failed rows:", failed_count)
    print("Unresolved rows:", unresolved_count)

    print(
        "Embedding dimension:",
        EMBEDDING_DIM
    )

    print(
        "Non-finite embedding rows:",
        embedding_nonfinite_rows
    )

    print(
        "Successful non-finite rows:",
        successful_nonfinite_rows
    )

    print(
        "Successful zero vectors:",
        successful_zero_vectors
    )

    status = "PASS"

    if unresolved_count > 0:
        status = "FAIL"

    if successful_nonfinite_rows > 0:
        status = "FAIL"

    if successful_zero_vectors > 0:
        status = "FAIL"

    if status == "PASS":
        print("STATUS: PASS")
    else:
        print("STATUS:", status)

    return {
        "split": split_name,
        "input_rows": n,
        "successful_embeddings": success_count,
        "failed_rows": failed_count,
        "unresolved_rows": unresolved_count,
        "embedding_dimension": EMBEDDING_DIM,
        "embedding_nonfinite_rows": embedding_nonfinite_rows,
        "successful_nonfinite_rows": successful_nonfinite_rows,
        "successful_zero_vectors": successful_zero_vectors,
        "status": status,
        "embedding_file": final_embedding_file,
        "metadata_file": metadata_file,
        "failed_file": failed_file,
        "checkpoint_file": checkpoint_file
    }


# =============================================================================
# 17. TRAIN EMBEDDINGS
# =============================================================================

train_result = generate_embeddings(
    train_df,
    train_smiles_col,
    "train",
    "train"
)


# =============================================================================
# 18. TEST EMBEDDINGS
# =============================================================================

test_result = generate_embeddings(
    test_df,
    test_smiles_col,
    "test",
    "test"
)


# =============================================================================
# 19. GLOBAL QC
# =============================================================================

print_header("SCRIPT 06 — GLOBAL EMBEDDING QC")

global_status = "PASS"

if train_result["status"] != "PASS":
    global_status = "FAIL"

if test_result["status"] != "PASS":
    global_status = "FAIL"

# Verify saved arrays can be loaded.
print_section("SAVED EMBEDDING VALIDATION")

train_embeddings = np.load(
    train_result["embedding_file"],
    mmap_mode="r"
)

test_embeddings = np.load(
    test_result["embedding_file"],
    mmap_mode="r"
)

print(
    "TRAIN embedding shape:",
    train_embeddings.shape
)

print(
    "TEST embedding shape:",
    test_embeddings.shape
)

expected_train_shape = (
    len(train_df),
    EMBEDDING_DIM
)

expected_test_shape = (
    len(test_df),
    EMBEDDING_DIM
)

if train_embeddings.shape != expected_train_shape:
    global_status = "FAIL"
    print("TRAIN shape check: FAIL")
else:
    print("TRAIN shape check: PASS")

if test_embeddings.shape != expected_test_shape:
    global_status = "FAIL"
    print("TEST shape check: FAIL")
else:
    print("TEST shape check: PASS")


# =============================================================================
# 20. MACHINE-READABLE REPORT
# =============================================================================

print_section(
    "WRITING MACHINE-READABLE REPORT"
)

report = {
    "script": "06_generate_molformer_embeddings_v2.py",
    "script_version": "v2",
    "generated": timestamp(),

    "model": {
        "name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "pooler_output_used": USE_POOLER_OUTPUT,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE
    },

    "environment": {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_available": CUDA_AVAILABLE,
        "device": str(DEVICE),
        "cpu_threads": CPU_THREADS
    },

    "input_files": {
        "train": {
            "path": TRAIN_FILE,
            "sha256": train_sha256,
            "rows": len(train_df)
        },
        "test": {
            "path": TEST_FILE,
            "sha256": test_sha256,
            "rows": len(test_df)
        }
    },

    "columns": {
        "id_column": ID_COLUMN,
        "train_smiles_column": train_smiles_col,
        "test_smiles_column": test_smiles_col,
        "train_activity_column": train_activity_col,
        "test_activity_column": test_activity_col
    },

    "input_qc": input_qc,

    "train_result": train_result,
    "test_result": test_result,

    "global_status": global_status,

    "output_directory": OUTPUT_DIR
}

json_report = os.path.join(
    OUTPUT_DIR,
    "molformer_embedding_qc_report.json"
)

safe_json_dump(
    report,
    json_report
)

print("Written:")
print(" ", json_report)


# =============================================================================
# 21. HUMAN-READABLE REPORT
# =============================================================================

txt_report = os.path.join(
    OUTPUT_DIR,
    "molformer_embedding_qc_report.txt"
)

with open(txt_report, "w") as f:

    f.write("=" * 78 + "\n")
    f.write("SCRIPT 06 — MoLFormer EMBEDDING QC REPORT\n")
    f.write("=" * 78 + "\n\n")

    f.write(
        f"Generated: {timestamp()}\n\n"
    )

    f.write("MODEL\n")
    f.write("-" * 78 + "\n")
    f.write(
        f"Model: {MODEL_NAME}\n"
    )
    f.write(
        f"Embedding dimension: {EMBEDDING_DIM}\n"
    )
    f.write(
        f"Pooler output used: {USE_POOLER_OUTPUT}\n"
    )
    f.write(
        f"Batch size: {BATCH_SIZE}\n"
    )
    f.write(
        f"Max length: {MAX_LENGTH}\n\n"
    )

    f.write("INPUT FILES\n")
    f.write("-" * 78 + "\n")
    f.write(
        f"TRAIN: {TRAIN_FILE}\n"
    )
    f.write(
        f"TRAIN SHA256: {train_sha256}\n"
    )
    f.write(
        f"TEST: {TEST_FILE}\n"
    )
    f.write(
        f"TEST SHA256: {test_sha256}\n\n"
    )

    f.write("DATASET\n")
    f.write("-" * 78 + "\n")
    f.write(
        f"TRAIN rows: {len(train_df)}\n"
    )
    f.write(
        f"TEST rows: {len(test_df)}\n\n"
    )

    f.write("IDENTIFIERS\n")
    f.write("-" * 78 + "\n")
    f.write(
        f"TRAIN duplicate IDs: "
        f"{input_qc['train_duplicate_ids']}\n"
    )
    f.write(
        f"TEST duplicate IDs: "
        f"{input_qc['test_duplicate_ids']}\n"
    )
    f.write(
        f"TRAIN/TEST ID overlap: "
        f"{input_qc['train_test_id_overlap']}\n\n"
    )

    f.write("TRAIN EMBEDDINGS\n")
    f.write("-" * 78 + "\n")

    for key, value in train_result.items():
        f.write(
            f"{key}: {value}\n"
        )

    f.write("\nTEST EMBEDDINGS\n")
    f.write("-" * 78 + "\n")

    for key, value in test_result.items():
        f.write(
            f"{key}: {value}\n"
        )

    f.write("\nGLOBAL STATUS\n")
    f.write("-" * 78 + "\n")
    f.write(
        f"{global_status}\n"
    )

    f.write("\nOUTPUT DIRECTORY\n")
    f.write("-" * 78 + "\n")
    f.write(
        f"{OUTPUT_DIR}\n"
    )

print("Written:")
print(" ", txt_report)


# =============================================================================
# 22. FINAL SUMMARY
# =============================================================================

print_header(
    "SCRIPT 06 MoLFormer EMBEDDING GENERATION COMPLETE"
)

print("MODEL:")
print(" ", MODEL_NAME)

print("\nDEVICE:")
print(" ", DEVICE)

print("\nEMBEDDING DIMENSION:")
print(" ", EMBEDDING_DIM)

print("\nTRAIN:")
print(
    f"  {train_result['successful_embeddings']:,} successful"
)
print(
    f"  {train_result['failed_rows']:,} failed"
)

print("\nTEST:")
print(
    f"  {test_result['successful_embeddings']:,} successful"
)
print(
    f"  {test_result['failed_rows']:,} failed"
)

print("\nGLOBAL STATUS:")
print(" ", global_status)

print("\nOUTPUT:")
print(" ", OUTPUT_DIR)

print("\nReports:")
print(
    " ",
    json_report
)
print(
    " ",
    txt_report
)

print("\nFinished:", timestamp())

print("=" * 78)
