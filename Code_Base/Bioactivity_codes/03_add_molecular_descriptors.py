#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_FILE = (
    "feature_engineering/split/train_80.tsv"
)

TEST_FILE = (
    "feature_engineering/split/test_20.tsv"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path(
    "feature_engineering/molecular_descriptors"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

TRAIN_OUTPUT = (
    OUTPUT_DIR /
    "train_80_molecular_descriptors.tsv"
)

TEST_OUTPUT = (
    OUTPUT_DIR /
    "test_20_molecular_descriptors.tsv"
)

DESCRIPTOR_LIST_OUTPUT = (
    OUTPUT_DIR /
    "descriptor_list.txt"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR /
    "descriptor_summary.txt"
)


# ============================================================
# RDKit DESCRIPTOR LIST
# ============================================================

print("=" * 70)
print("RDKit MOLECULAR DESCRIPTOR GENERATION")
print("=" * 70)

descriptor_names = [
    name
    for name, func in Descriptors.descList
]

descriptor_functions = MoleculeDescriptors.MolecularDescriptorCalculator(
    descriptor_names
)

print(
    f"\nTotal RDKit descriptors available: "
    f"{len(descriptor_names):,}"
)


# ============================================================
# LOAD TRAINING AND TEST DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(
    TRAIN_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)

test_df = pd.read_csv(
    TEST_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)

print(
    f"Training molecules: {len(train_df):,}"
)

print(
    f"Testing molecules:  {len(test_df):,}"
)


# ============================================================
# CHECK CANONICAL SMILES
# ============================================================

for name, dataframe in [
    ("training", train_df),
    ("testing", test_df)
]:

    if "Canonical_SMILES" not in dataframe.columns:

        raise ValueError(
            f"Canonical_SMILES column is missing "
            f"from {name} dataset."
        )


# ============================================================
# FUNCTION TO GENERATE DESCRIPTORS
# ============================================================

def calculate_descriptors(dataframe, dataset_name):

    print("\n" + "-" * 70)
    print(
        f"Generating descriptors for {dataset_name}"
    )
    print("-" * 70)

    molecules = []
    valid_indices = []
    invalid_indices = []

    for index, smiles in enumerate(
        dataframe["Canonical_SMILES"]
    ):

        try:

            mol = Chem.MolFromSmiles(
                smiles
            )

        except Exception:

            mol = None

        if mol is None:

            invalid_indices.append(index)
            molecules.append(None)

        else:

            valid_indices.append(index)
            molecules.append(mol)

    print(
        f"Valid molecules:   {len(valid_indices):,}"
    )

    print(
        f"Invalid molecules: {len(invalid_indices):,}"
    )

    if invalid_indices:

        raise ValueError(
            f"{dataset_name} contains molecules that "
            f"cannot be parsed from Canonical_SMILES."
        )

    # --------------------------------------------------------
    # Calculate descriptors
    # --------------------------------------------------------

    descriptor_values = []

    total = len(molecules)

    for i, mol in enumerate(molecules):

        if (i + 1) % 5000 == 0 or i + 1 == total:

            print(
                f"Processed {i + 1:,}/{total:,}",
                end="\r"
            )

        try:

            values = descriptor_functions.CalcDescriptors(
                mol
            )

        except Exception as e:

            print(
                f"\nDescriptor calculation failed "
                f"for row {i}: {e}"
            )

            values = [
                np.nan
                for _ in descriptor_names
            ]

        descriptor_values.append(values)

    print()

    descriptor_df = pd.DataFrame(
        descriptor_values,
        columns=descriptor_names,
        index=dataframe.index
    )

    return descriptor_df


# ============================================================
# GENERATE TRAIN DESCRIPTORS
# ============================================================

train_descriptors = calculate_descriptors(
    train_df,
    "TRAINING SET"
)


# ============================================================
# GENERATE TEST DESCRIPTORS
# ============================================================

test_descriptors = calculate_descriptors(
    test_df,
    "TESTING SET"
)


# ============================================================
# COMBINE TRAIN + TEST FOR COLUMN QC ONLY
# ============================================================

# IMPORTANT:
# This is NOT fitting or selecting features.
# We only use both matrices here to identify descriptor
# columns that are completely unusable in either dataset.

combined_descriptors = pd.concat(
    [
        train_descriptors,
        test_descriptors
    ],
    axis=0
)


# ============================================================
# CONVERT NON-FINITE VALUES TO NaN
# ============================================================

combined_descriptors = combined_descriptors.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# REMOVE COMPLETELY EMPTY DESCRIPTORS
# ============================================================

valid_descriptor_columns = [
    col
    for col in combined_descriptors.columns
    if not combined_descriptors[col].isna().all()
]


removed_descriptors = [
    col
    for col in combined_descriptors.columns
    if col not in valid_descriptor_columns
]


print("\n" + "=" * 70)
print("DESCRIPTOR QUALITY CONTROL")
print("=" * 70)

print(
    f"Initial descriptors: "
    f"{len(descriptor_names):,}"
)

print(
    f"Usable descriptors: "
    f"{len(valid_descriptor_columns):,}"
)

print(
    f"Completely invalid descriptors removed: "
    f"{len(removed_descriptors):,}"
)


# ============================================================
# KEEP ONLY VALID DESCRIPTORS
# ============================================================

train_descriptors = (
    train_descriptors[
        valid_descriptor_columns
    ]
    .replace([np.inf, -np.inf], np.nan)
)

test_descriptors = (
    test_descriptors[
        valid_descriptor_columns
    ]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# DESCRIPTOR MISSING-VALUE REPORT
# ============================================================

train_missing = (
    train_descriptors.isna().sum()
)

test_missing = (
    test_descriptors.isna().sum()
)

missing_descriptor_report = pd.DataFrame({
    "Descriptor": valid_descriptor_columns,
    "Train_Missing": [
        train_missing[col]
        for col in valid_descriptor_columns
    ],
    "Test_Missing": [
        test_missing[col]
        for col in valid_descriptor_columns
    ]
})

missing_descriptor_report[
    "Total_Missing"
] = (
    missing_descriptor_report["Train_Missing"] +
    missing_descriptor_report["Test_Missing"]
)


# ============================================================
# COMBINE ORIGINAL DATA + DESCRIPTORS
# ============================================================

train_final = pd.concat(
    [
        train_df.reset_index(drop=True),
        train_descriptors.reset_index(drop=True)
    ],
    axis=1
)

test_final = pd.concat(
    [
        test_df.reset_index(drop=True),
        test_descriptors.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# SAVE TRAINING DATASET
# ============================================================

train_final.to_csv(
    TRAIN_OUTPUT,
    sep="\t",
    index=False
)


# ============================================================
# SAVE TESTING DATASET
# ============================================================

test_final.to_csv(
    TEST_OUTPUT,
    sep="\t",
    index=False
)


# ============================================================
# SAVE DESCRIPTOR LIST
# ============================================================

with open(
    DESCRIPTOR_LIST_OUTPUT,
    "w"
) as f:

    for descriptor in valid_descriptor_columns:

        f.write(
            descriptor + "\n"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_OUTPUT,
    "w"
) as f:

    f.write(
        "RDKit MOLECULAR DESCRIPTOR GENERATION SUMMARY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Training input: {TRAIN_FILE}\n"
    )

    f.write(
        f"Testing input: {TEST_FILE}\n\n"
    )

    f.write(
        f"Training molecules: {len(train_final):,}\n"
    )

    f.write(
        f"Testing molecules: {len(test_final):,}\n\n"
    )

    f.write(
        f"Initial RDKit descriptors: "
        f"{len(descriptor_names):,}\n"
    )

    f.write(
        f"Usable descriptors: "
        f"{len(valid_descriptor_columns):,}\n"
    )

    f.write(
        f"Completely invalid descriptors removed: "
        f"{len(removed_descriptors):,}\n\n"
    )

    f.write(
        "Descriptors with missing values:\n"
    )

    for _, row in missing_descriptor_report[
        missing_descriptor_report["Total_Missing"] > 0
    ].iterrows():

        f.write(
            f"{row['Descriptor']}\t"
            f"train={row['Train_Missing']}\t"
            f"test={row['Test_Missing']}\n"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("MOLECULAR DESCRIPTOR GENERATION COMPLETED")
print("=" * 70)

print(
    f"\nTraining output:\n{TRAIN_OUTPUT}"
)

print(
    f"\nTesting output:\n{TEST_OUTPUT}"
)

print(
    f"\nNumber of descriptor features: "
    f"{len(valid_descriptor_columns):,}"
)

print(
    f"Training rows: {len(train_final):,}"
)

print(
    f"Testing rows:  {len(test_final):,}"
)

print("\nNo model was trained.")
print("No feature selection was performed.")
print("No scaling was performed.")

print("\nDONE.")
