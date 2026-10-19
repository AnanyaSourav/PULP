#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 08 v2 — PUBLICATION STATISTICAL ANALYSIS PACKAGE
==============================================================================

Purpose
-------
Convert the outputs of:

    SCRIPT 07 — Statistical Feature Analysis
    SCRIPT 08 v1 — Statistical Determination / Replication Analysis

into publication-ready:

    1. Figures
    2. Main manuscript tables
    3. Supplementary tables
    4. Feature-level evidence files
    5. Dataset/class-distribution summaries
    6. Train/test replication summaries
    7. Representation comparison summaries
    8. Model-development strategy evidence
    9. Machine-readable metadata
   10. SHA256 checksums

IMPORTANT
---------
This script DOES NOT:

    - remove features
    - select final predictive features
    - train a predictive model
    - tune a predictive model
    - use TEST data for model fitting
    - alter the original statistical results

The TEST set is treated as an independent replication set.

Primary statistical framework inherited from SCRIPT 07:

    - Mann-Whitney U
    - ROC-AUC
    - Cliff's delta = 2*AUC - 1
    - Kolmogorov-Smirnov
    - Benjamini-Hochberg FDR
    - FDR alpha = 0.05

The purpose is publication packaging and interpretation.

==============================================================================

EXPECTED DIRECTORY
------------------

feature_engineering/
└── statistical_analysis/

    train_molecular_statistics.tsv
    test_molecular_replication.tsv

    train_3d_statistics.tsv
    test_3d_replication.tsv

    train_molformer_statistics.tsv
    test_molformer_replication.tsv

    representation_statistical_summary.tsv
    statistical_feature_summary.json
    statistical_feature_analysis_report.txt

    [08 v1 outputs, if present]

OUTPUT
------

feature_engineering/
└── statistical_analysis/
    └── publication_v2/

        figures/
        tables/
        supplementary/
        metadata/
        README_publication_statistics.txt

==============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import shutil
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("feature_engineering/statistical_analysis")

OUTPUT_DIR = BASE_DIR / "publication_v2"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
SUPP_DIR = OUTPUT_DIR / "supplementary"
META_DIR = OUTPUT_DIR / "metadata"

FDR_ALPHA = 0.05

# Practical effect thresholds.
#
# AUC:
#     0.50 = no discrimination
#     deviation from 0.50 is used as the practical magnitude.
#
# The original Script 07 used a meaningful-AUC criterion.
# We preserve the already-computed classifications rather than inventing
# a new selection rule here.
#
# These are only fallback thresholds if an input file does not contain
# the classification columns.
AUC_EFFECT_THRESHOLD = 0.10
CLIFFS_DELTA_THRESHOLD = 0.20

RANDOM_STATE = 42

warnings.filterwarnings("ignore")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def ensure_dirs():
    for d in [
        OUTPUT_DIR,
        FIG_DIR,
        TABLE_DIR,
        SUPP_DIR,
        META_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def safe_read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    if df.empty:
        raise RuntimeError(f"Input file is empty: {path}")

    return df


def write_tsv(df: pd.DataFrame, path: Path):
    df.to_csv(path, sep="\t", index=False)


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def save_figure(fig, basename: str):
    """
    Save each figure as:
        PNG
        PDF
        SVG
    """

    png = FIG_DIR / f"{basename}.png"
    pdf = FIG_DIR / f"{basename}.pdf"
    svg = FIG_DIR / f"{basename}.svg"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
    )

    plt.close(fig)

    return [png, pdf, svg]


def find_column(df, candidates):
    """
    Case-insensitive column finder.
    """

    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p

    return None


def normalise_representation(name):
    name = str(name).lower()

    if "molecular" in name:
        return "Molecular descriptors"

    if "3d" in name:
        return "3D-QSAR descriptors"

    if "molformer" in name:
        return "MoLFormer embeddings"

    return name


def detect_representation_from_filename(path):
    name = path.name.lower()

    if "molecular" in name:
        return "molecular"

    if "3d" in name:
        return "3d"

    if "molformer" in name:
        return "molformer"

    return "unknown"


# =============================================================================
# LOAD INPUTS
# =============================================================================

print("=" * 78)
print("SCRIPT 08 v2 — PUBLICATION STATISTICAL ANALYSIS PACKAGE")
print("=" * 78)
print(f"Started: {timestamp()}")

ensure_dirs()

print("\n" + "-" * 78)
print("1. INPUT DISCOVERY")
print("-" * 78)

INPUT_FILES = {
    "train_molecular": BASE_DIR / "train_molecular_statistics.tsv",
    "test_molecular": BASE_DIR / "test_molecular_replication.tsv",

    "train_3d": BASE_DIR / "train_3d_statistics.tsv",
    "test_3d": BASE_DIR / "test_3d_replication.tsv",

    "train_molformer": BASE_DIR / "train_molformer_statistics.tsv",
    "test_molformer": BASE_DIR / "test_molformer_replication.tsv",

    "representation_summary":
        BASE_DIR / "representation_statistical_summary.tsv",

    "summary_json":
        BASE_DIR / "statistical_feature_summary.json",

    "summary_report":
        BASE_DIR / "statistical_feature_analysis_report.txt",
}

loaded = {}

for key, path in INPUT_FILES.items():

    if path.exists():
        print(f"FOUND: {path}")
        loaded[key] = path

    else:
        print(f"NOT FOUND: {path}")


# Require Script 07 core outputs.
required_keys = [
    "train_molecular",
    "test_molecular",
    "train_3d",
    "test_3d",
    "train_molformer",
    "test_molformer",
    "representation_summary",
    "summary_json",
]

missing = [k for k in required_keys if k not in loaded]

if missing:
    raise RuntimeError(
        "Missing required Script 07 outputs:\n"
        + "\n".join(missing)
    )


# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "-" * 78)
print("2. LOADING STATISTICAL RESULTS")
print("-" * 78)

train_molecular = safe_read_tsv(loaded["train_molecular"])
test_molecular = safe_read_tsv(loaded["test_molecular"])

train_3d = safe_read_tsv(loaded["train_3d"])
test_3d = safe_read_tsv(loaded["test_3d"])

train_molformer = safe_read_tsv(loaded["train_molformer"])
test_molformer = safe_read_tsv(loaded["test_molformer"])

representation_summary = safe_read_tsv(
    loaded["representation_summary"]
)

with open(loaded["summary_json"]) as f:
    statistical_summary = json.load(f)

print(f"TRAIN molecular:  {train_molecular.shape}")
print(f"TEST molecular:   {test_molecular.shape}")
print(f"TRAIN 3D:         {train_3d.shape}")
print(f"TEST 3D:          {test_3d.shape}")
print(f"TRAIN MoLFormer:  {train_molformer.shape}")
print(f"TEST MoLFormer:   {test_molformer.shape}")


# =============================================================================
# 3. DATASET SUMMARY
# =============================================================================

print("\n" + "-" * 78)
print("3. DATASET SUMMARY")
print("-" * 78)

class_summary = pd.DataFrame([
    {
        "dataset": "TRAIN",
        "total": 48152,
        "active": 2383,
        "inactive": 45769,
        "active_percent": 2383 / 48152 * 100,
        "inactive_percent": 45769 / 48152 * 100,
    },
    {
        "dataset": "TEST",
        "total": 12038,
        "active": 596,
        "inactive": 11442,
        "active_percent": 596 / 12038 * 100,
        "inactive_percent": 11442 / 12038 * 100,
    },
])

write_tsv(
    class_summary,
    TABLE_DIR / "Table_1_dataset_class_distribution.tsv"
)

write_csv(
    class_summary,
    TABLE_DIR / "Table_1_dataset_class_distribution.csv"
)


# =============================================================================
# FIGURE 1 — CLASS DISTRIBUTION
# =============================================================================

print("\nGenerating Figure 1...")

fig, ax = plt.subplots(figsize=(7, 5))

x = np.arange(len(class_summary))
width = 0.36

ax.bar(
    x - width / 2,
    class_summary["active"],
    width,
    label="Active",
)

ax.bar(
    x + width / 2,
    class_summary["inactive"],
    width,
    label="Inactive",
)

ax.set_xticks(x)
ax.set_xticklabels(class_summary["dataset"])
ax.set_ylabel("Number of molecules")
ax.set_title("Dataset class distribution")
ax.legend(frameon=False)

for i, row in class_summary.iterrows():

    ax.text(
        i - width / 2,
        row["active"] + max(class_summary["total"]) * 0.01,
        f'{row["active"]:,}\n({row["active_percent"]:.2f}%)',
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.text(
        i + width / 2,
        row["inactive"] + max(class_summary["total"]) * 0.01,
        f'{row["inactive"]:,}\n({row["inactive_percent"]:.2f}%)',
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()

save_figure(fig, "Figure_1_class_distribution")


# =============================================================================
# 4. STANDARDISE FEATURE STATISTICS
# =============================================================================

print("\n" + "-" * 78)
print("4. STANDARDISING FEATURE-LEVEL RESULTS")
print("-" * 78)


def standardise_stats(df, representation, dataset):

    out = df.copy()

    out["representation"] = representation
    out["dataset"] = dataset

    feature_col = find_column(
        out,
        [
            "feature",
            "feature_name",
            "Feature",
            "Feature_Name",
            "name",
        ],
    )

    auc_col = find_column(
        out,
        [
            "auc",
            "roc_auc",
            "AUC",
            "ROC_AUC",
        ],
    )

    p_col = find_column(
        out,
        [
            "p_value",
            "p",
            "pvalue",
            "P_value",
        ],
    )

    fdr_col = find_column(
        out,
        [
            "fdr",
            "fdr_p",
            "fdr_p_value",
            "adjusted_p_value",
            "q_value",
        ],
    )

    delta_col = find_column(
        out,
        [
            "cliffs_delta",
            "cliffs_delta",
            "Cliffs_delta",
            "effect_size",
            "delta",
        ],
    )

    ks_p_col = find_column(
        out,
        [
            "ks_p",
            "ks_p_value",
            "ks_pvalue",
            "KS_p",
        ],
    )

    if feature_col is not None:
        out["feature_name_standard"] = out[feature_col].astype(str)

    if auc_col is not None:
        out["auc_standard"] = pd.to_numeric(
            out[auc_col],
            errors="coerce"
        )

        out["abs_auc_effect"] = (
            out["auc_standard"] - 0.5
        ).abs()

        out["auc_direction"] = np.where(
            out["auc_standard"] >= 0.5,
            "positive",
            "negative",
        )

    if delta_col is not None:
        out["cliffs_delta_standard"] = pd.to_numeric(
            out[delta_col],
            errors="coerce"
        )

    if fdr_col is not None:
        out["fdr_standard"] = pd.to_numeric(
            out[fdr_col],
            errors="coerce"
        )

        out["fdr_significant_standard"] = (
            out["fdr_standard"] < FDR_ALPHA
        )

    if auc_col is not None:

        out["meaningful_auc_standard"] = (
            out["abs_auc_effect"] >= AUC_EFFECT_THRESHOLD
        )

    if delta_col is not None:

        out["meaningful_effect_standard"] = (
            out["cliffs_delta_standard"].abs()
            >= CLIFFS_DELTA_THRESHOLD
        )

    return out


datasets = {
    "molecular": {
        "train": train_molecular,
        "test": test_molecular,
    },
    "3d": {
        "train": train_3d,
        "test": test_3d,
    },
    "molformer": {
        "train": train_molformer,
        "test": test_molformer,
    },
}

standardised = {}

for rep, vals in datasets.items():

    standardised[f"{rep}_train"] = standardise_stats(
        vals["train"],
        rep,
        "TRAIN",
    )

    standardised[f"{rep}_test"] = standardise_stats(
        vals["test"],
        rep,
        "TEST",
    )


# =============================================================================
# 5. COMPLETE FEATURE-LEVEL SUPPLEMENTARY FILES
# =============================================================================

print("\n" + "-" * 78)
print("5. WRITING FEATURE-LEVEL SUPPLEMENTARY TABLES")
print("-" * 78)

for key, df in standardised.items():

    out_file = SUPP_DIR / f"Supplementary_{key}_feature_statistics.tsv"

    write_tsv(df, out_file)

    print(f"Written: {out_file}")


# =============================================================================
# 6. TRAIN FEATURE SUMMARY
# =============================================================================

print("\n" + "-" * 78)
print("6. TRAIN FEATURE EVIDENCE")
print("-" * 78)


train_feature_records = []

for rep in ["molecular", "3d", "molformer"]:

    df = standardised[f"{rep}_train"]

    auc_col = "auc_standard"
    fdr_col = "fdr_standard"
    delta_col = "cliffs_delta_standard"

    record = {
        "representation": normalise_representation(rep),
        "representation_code": rep,
        "total_features": len(df),
        "testable_features":
            int(df[auc_col].notna().sum())
            if auc_col in df.columns else np.nan,
    }

    if fdr_col in df.columns:
        record["fdr_significant"] = int(
            (df[fdr_col] < FDR_ALPHA).sum()
        )
    else:
        record["fdr_significant"] = np.nan

    if auc_col in df.columns:

        record["meaningful_auc"] = int(
            (
                df[auc_col].sub(0.5).abs()
                >= AUC_EFFECT_THRESHOLD
            ).sum()
        )

        record["median_auc"] = df[auc_col].median()
        record["median_abs_auc_effect"] = (
            df[auc_col].sub(0.5).abs().median()
        )

        record["maximum_abs_auc_effect"] = (
            df[auc_col].sub(0.5).abs().max()
        )

    if delta_col in df.columns:

        record["meaningful_effect"] = int(
            (
                df[delta_col].abs()
                >= CLIFFS_DELTA_THRESHOLD
            ).sum()
        )

        record["maximum_abs_cliffs_delta"] = (
            df[delta_col].abs().max()
        )

    train_feature_records.append(record)


train_evidence = pd.DataFrame(train_feature_records)

write_tsv(
    train_evidence,
    TABLE_DIR / "Table_2_train_representation_statistics.tsv"
)

write_csv(
    train_evidence,
    TABLE_DIR / "Table_2_train_representation_statistics.csv"
)


# =============================================================================
# FIGURE 2 — NUMBER OF SIGNIFICANT FEATURES
# =============================================================================

print("\nGenerating Figure 2...")

plot_df = train_evidence.copy()

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(plot_df))
width = 0.36

ax.bar(
    x - width / 2,
    plot_df["fdr_significant"],
    width,
    label="FDR significant",
)

ax.bar(
    x + width / 2,
    plot_df["meaningful_auc"],
    width,
    label="Meaningful AUC effect",
)

ax.set_xticks(x)
ax.set_xticklabels(plot_df["representation"])
ax.set_ylabel("Number of features")
ax.set_title("Feature-level statistical evidence in the training set")
ax.legend(frameon=False)

fig.tight_layout()

save_figure(
    fig,
    "Figure_2_feature_statistical_evidence"
)


# =============================================================================
# FIGURE 3 — AUC DISTRIBUTIONS
# =============================================================================

print("\nGenerating Figure 3...")

fig, ax = plt.subplots(figsize=(8, 5))

for rep in ["molecular", "3d", "molformer"]:

    df = standardised[f"{rep}_train"]

    if "auc_standard" not in df.columns:
        continue

    vals = df["auc_standard"].dropna()

    if len(vals) == 0:
        continue

    ax.hist(
        vals,
        bins=50,
        alpha=0.45,
        density=True,
        label=normalise_representation(rep),
    )

ax.axvline(
    0.5,
    linestyle="--",
    linewidth=1.2,
)

ax.set_xlabel("ROC-AUC")
ax.set_ylabel("Density")
ax.set_title("Distribution of univariate feature discrimination")
ax.legend(frameon=False)

fig.tight_layout()

save_figure(
    fig,
    "Figure_3_auc_distributions"
)


# =============================================================================
# 7. TRAIN → TEST REPLICATION
# =============================================================================

print("\n" + "-" * 78)
print("7. TRAIN → TEST REPLICATION")
print("-" * 78)


def build_replication_table(rep):

    train = standardised[f"{rep}_train"].copy()
    test = standardised[f"{rep}_test"].copy()

    if "feature_name_standard" not in train.columns:
        raise RuntimeError(
            f"Feature column could not be identified for {rep}."
        )

    if "feature_name_standard" not in test.columns:
        raise RuntimeError(
            f"Feature column could not be identified for test {rep}."
        )

    train_keep = [
        c for c in [
            "feature_name_standard",
            "auc_standard",
            "abs_auc_effect",
            "fdr_standard",
            "cliffs_delta_standard",
        ]
        if c in train.columns
    ]

    test_keep = [
        c for c in [
            "feature_name_standard",
            "auc_standard",
            "abs_auc_effect",
            "fdr_standard",
            "cliffs_delta_standard",
        ]
        if c in test.columns
    ]

    train2 = train[train_keep].copy()
    test2 = test[test_keep].copy()

    train2 = train2.rename(
        columns={
            "auc_standard": "train_auc",
            "abs_auc_effect": "train_abs_auc_effect",
            "fdr_standard": "train_fdr",
            "cliffs_delta_standard": "train_cliffs_delta",
        }
    )

    test2 = test2.rename(
        columns={
            "auc_standard": "test_auc",
            "abs_auc_effect": "test_abs_auc_effect",
            "fdr_standard": "test_fdr",
            "cliffs_delta_standard": "test_cliffs_delta",
        }
    )

    merged = train2.merge(
        test2,
        on="feature_name_standard",
        how="inner",
        validate="one_to_one",
    )

    merged["representation"] = rep

    if "train_auc" in merged.columns and "test_auc" in merged.columns:

        merged["auc_replication_difference"] = (
            merged["test_auc"] - merged["train_auc"]
        )

        merged["abs_auc_replication_difference"] = (
            merged["auc_replication_difference"].abs()
        )

        merged["train_test_auc_direction_consistent"] = (
            (
                merged["train_auc"] >= 0.5
            )
            ==
            (
                merged["test_auc"] >= 0.5
            )
        )

    if (
        "train_fdr" in merged.columns
        and "test_fdr" in merged.columns
    ):

        merged["train_fdr_significant"] = (
            merged["train_fdr"] < FDR_ALPHA
        )

        merged["test_fdr_significant"] = (
            merged["test_fdr"] < FDR_ALPHA
        )

    return merged


replication_tables = {}

for rep in ["molecular", "3d", "molformer"]:

    rep_df = build_replication_table(rep)

    replication_tables[rep] = rep_df

    out = SUPP_DIR / f"Supplementary_{rep}_train_test_replication.tsv"

    write_tsv(rep_df, out)

    print(
        f"{rep}: {len(rep_df):,} matched features"
    )


# =============================================================================
# TABLE 3 — REPLICATION SUMMARY
# =============================================================================

replication_summary = []

for rep, df in replication_tables.items():

    record = {
        "representation": normalise_representation(rep),
        "features_compared": len(df),
    }

    if "abs_auc_replication_difference" in df.columns:

        record["median_abs_auc_difference"] = (
            df["abs_auc_replication_difference"].median()
        )

        record["mean_abs_auc_difference"] = (
            df["abs_auc_replication_difference"].mean()
        )

    if "train_test_auc_direction_consistent" in df.columns:

        record["direction_consistent_features"] = int(
            df["train_test_auc_direction_consistent"].sum()
        )

        record["direction_consistency_percent"] = (
            df["train_test_auc_direction_consistent"].mean()
            * 100
        )

    if "train_fdr_significant" in df.columns:

        record["train_fdr_significant"] = int(
            df["train_fdr_significant"].sum()
        )

    if "test_fdr_significant" in df.columns:

        record["test_fdr_significant"] = int(
            df["test_fdr_significant"].sum()
        )

    replication_summary.append(record)


replication_summary = pd.DataFrame(replication_summary)

write_tsv(
    replication_summary,
    TABLE_DIR / "Table_3_train_test_replication.tsv"
)

write_csv(
    replication_summary,
    TABLE_DIR / "Table_3_train_test_replication.csv"
)


# =============================================================================
# FIGURE 4 — TRAIN VS TEST AUC
# =============================================================================

print("\nGenerating Figure 4...")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.5),
)

for ax, rep in zip(
    axes,
    ["molecular", "3d", "molformer"],
):

    df = replication_tables[rep]

    if not {
        "train_auc",
        "test_auc"
    }.issubset(df.columns):

        ax.set_visible(False)
        continue

    x = df["train_auc"].to_numpy()
    y = df["test_auc"].to_numpy()

    ax.scatter(
        x,
        y,
        s=12,
        alpha=0.45,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        0.5,
        linestyle=":",
        linewidth=1,
    )

    ax.axhline(
        0.5,
        linestyle=":",
        linewidth=1,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("TRAIN ROC-AUC")
    ax.set_ylabel("TEST ROC-AUC")

    ax.set_title(
        normalise_representation(rep)
    )

fig.suptitle(
    "Independent replication of feature-level discrimination",
    y=1.02,
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_4_train_test_auc_replication"
)


# =============================================================================
# 8. CROSS-REPRESENTATION SUMMARY
# =============================================================================

print("\n" + "-" * 78)
print("8. CROSS-REPRESENTATION SUMMARY")
print("-" * 78)

summary_rows = []

for rep in ["molecular", "3d", "molformer"]:

    df = standardised[f"{rep}_train"]

    row = {
        "representation":
            normalise_representation(rep),

        "representation_code":
            rep,

        "features":
            len(df),
    }

    if "auc_standard" in df.columns:

        auc = df["auc_standard"].dropna()

        row["testable_features"] = len(auc)
        row["median_auc"] = auc.median()
        row["median_abs_auc_effect"] = (
            (auc - 0.5).abs().median()
        )
        row["maximum_abs_auc_effect"] = (
            (auc - 0.5).abs().max()
        )

        row["meaningful_auc_features"] = int(
            ((auc - 0.5).abs() >= AUC_EFFECT_THRESHOLD).sum()
        )

    if "fdr_standard" in df.columns:

        fdr = df["fdr_standard"].dropna()

        row["fdr_significant_features"] = int(
            (fdr < FDR_ALPHA).sum()
        )

    if "cliffs_delta_standard" in df.columns:

        delta = df["cliffs_delta_standard"].dropna()

        row["maximum_abs_cliffs_delta"] = delta.abs().max()

        row["meaningful_effect_features"] = int(
            (delta.abs() >= CLIFFS_DELTA_THRESHOLD).sum()
        )

    summary_rows.append(row)


cross_representation = pd.DataFrame(summary_rows)

write_tsv(
    cross_representation,
    TABLE_DIR / "Table_4_cross_representation_comparison.tsv"
)

write_csv(
    cross_representation,
    TABLE_DIR / "Table_4_cross_representation_comparison.csv"
)


# =============================================================================
# FIGURE 5 — MEDIAN EFFECT SIZE BY REPRESENTATION
# =============================================================================

print("\nGenerating Figure 5...")

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(cross_representation))

ax.bar(
    x,
    cross_representation["median_abs_auc_effect"],
)

ax.set_xticks(x)
ax.set_xticklabels(
    cross_representation["representation"]
)

ax.set_ylabel("Median |AUC − 0.50|")
ax.set_title(
    "Typical univariate discrimination by molecular representation"
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_5_representation_effect_comparison"
)


# =============================================================================
# 9. TOP FEATURES — TRAIN
# =============================================================================

print("\n" + "-" * 78)
print("9. TOP FEATURE EVIDENCE")
print("-" * 78)


top_feature_tables = []

for rep in ["molecular", "3d", "molformer"]:

    df = standardised[f"{rep}_train"].copy()

    if "auc_standard" not in df.columns:
        continue

    df = df.dropna(subset=["auc_standard"])

    df["abs_auc_effect"] = (
        df["auc_standard"] - 0.5
    ).abs()

    sort_columns = ["abs_auc_effect"]

    if "fdr_standard" in df.columns:
        sort_columns.append("fdr_standard")

    df = df.sort_values(
        sort_columns,
        ascending=[False] + [True] * (len(sort_columns) - 1),
    )

    top = df.head(25).copy()

    top["representation"] = normalise_representation(rep)

    top_feature_tables.append(
        top
    )


top_features = pd.concat(
    top_feature_tables,
    ignore_index=True,
)

write_tsv(
    top_features,
    SUPP_DIR / "Supplementary_top_25_features_per_representation.tsv"
)


# =============================================================================
# 10. TRAIN/TEST FEATURE REPLICATION OF TOP TRAIN FEATURES
# =============================================================================

print("\n" + "-" * 78)
print("10. TOP TRAIN FEATURES — INDEPENDENT TEST REPLICATION")
print("-" * 78)

top_replication = []

for rep in ["molecular", "3d", "molformer"]:

    train_df = standardised[f"{rep}_train"]
    test_df = standardised[f"{rep}_test"]

    if "auc_standard" not in train_df.columns:
        continue

    train_df = train_df.dropna(
        subset=["auc_standard"]
    ).copy()

    train_df["abs_auc_effect"] = (
        train_df["auc_standard"] - 0.5
    ).abs()

    train_df = train_df.sort_values(
        "abs_auc_effect",
        ascending=False,
    )

    top_names = train_df.head(25)[
        "feature_name_standard"
    ].tolist()

    test_map = test_df.set_index(
        "feature_name_standard"
    )

    for feature in top_names:

        if feature not in test_map.index:
            continue

        tr = train_df[
            train_df["feature_name_standard"] == feature
        ].iloc[0]

        te = test_map.loc[feature]

        row = {
            "representation":
                normalise_representation(rep),

            "feature":
                feature,

            "train_auc":
                tr.get("auc_standard", np.nan),

            "test_auc":
                te.get("auc_standard", np.nan),

            "train_fdr":
                tr.get("fdr_standard", np.nan),

            "test_fdr":
                te.get("fdr_standard", np.nan),

            "train_cliffs_delta":
                tr.get("cliffs_delta_standard", np.nan),

            "test_cliffs_delta":
                te.get("cliffs_delta_standard", np.nan),
        }

        row["auc_direction_consistent"] = (
            (
                row["train_auc"] >= 0.5
            )
            ==
            (
                row["test_auc"] >= 0.5
            )
        )

        row["abs_auc_difference"] = abs(
            row["test_auc"] - row["train_auc"]
        )

        top_replication.append(row)


top_replication = pd.DataFrame(
    top_replication
)

write_tsv(
    top_replication,
    SUPP_DIR / "Supplementary_top_feature_test_replication.tsv"
)


# =============================================================================
# FIGURE 6 — TOP FEATURES TRAIN VS TEST
# =============================================================================

print("\nGenerating Figure 6...")

if not top_replication.empty:

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        top_replication["train_auc"],
        top_replication["test_auc"],
        s=22,
        alpha=0.65,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        0.5,
        linestyle=":",
        linewidth=1,
    )

    ax.axhline(
        0.5,
        linestyle=":",
        linewidth=1,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(
        "TRAIN ROC-AUC"
    )

    ax.set_ylabel(
        "Independent TEST ROC-AUC"
    )

    ax.set_title(
        "Independent replication of strongest univariate features"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_6_top_feature_replication"
    )


# =============================================================================
# 11. STATISTICAL STRATEGY TABLE
# =============================================================================

print("\n" + "-" * 78)
print("11. MODEL-DEVELOPMENT STATISTICAL STRATEGY")
print("-" * 78)

strategy = pd.DataFrame([
    {
        "stage": 1,
        "component": "Dataset definition",
        "decision": "Use predefined TRAIN/TEST split",
        "statistical_basis":
            "TRAIN n=48,152; TEST n=12,038; independent ID sets",
        "role_in_model_development":
            "Preserve independent final evaluation set",
    },
    {
        "stage": 2,
        "component": "Class distribution",
        "decision": "Treat activity as strongly imbalanced binary classification",
        "statistical_basis":
            "TRAIN active fraction ≈4.95%; TEST active fraction ≈4.95%",
        "role_in_model_development":
            "Accuracy alone should not be the primary evaluation metric",
    },
    {
        "stage": 3,
        "component": "Univariate feature testing",
        "decision": "Mann-Whitney U",
        "statistical_basis":
            "Non-parametric comparison of active vs inactive feature distributions",
        "role_in_model_development":
            "Characterise class-associated feature signal",
    },
    {
        "stage": 4,
        "component": "Discrimination effect",
        "decision": "ROC-AUC and |AUC−0.50|",
        "statistical_basis":
            "Quantifies univariate discrimination independently of prevalence",
        "role_in_model_development":
            "Distinguish statistically detectable effects from practically useful effects",
    },
    {
        "stage": 5,
        "component": "Effect size",
        "decision": "Cliff's delta",
        "statistical_basis":
            "Delta = 2×AUC−1",
        "role_in_model_development":
            "Quantify magnitude of active/inactive separation",
    },
    {
        "stage": 6,
        "component": "Multiple testing",
        "decision": "Benjamini-Hochberg FDR",
        "statistical_basis":
            "Controls false discovery rate across high-dimensional feature sets",
        "role_in_model_development":
            "Prevent interpretation based on raw p-values alone",
    },
    {
        "stage": 7,
        "component": "Independent replication",
        "decision": "Repeat feature-level analysis on TEST",
        "statistical_basis":
            "TEST remains independent of feature screening/model fitting",
        "role_in_model_development":
            "Identify signal that generalises beyond TRAIN",
    },
    {
        "stage": 8,
        "component": "Representation comparison",
        "decision":
            "Compare molecular descriptors, 3D-QSAR descriptors and MoLFormer embeddings",
        "statistical_basis":
            "Compare distributions of feature-level discrimination and replication",
        "role_in_model_development":
            "Inform candidate representation strategy",
    },
    {
        "stage": 9,
        "component": "Predictive modelling",
        "decision":
            "Proceed to model development only after statistical characterisation",
        "statistical_basis":
            "Statistical screening is descriptive/evidence-generating rather than final model selection",
        "role_in_model_development":
            "Build and compare predictive models using TRAIN only",
    },
])

write_tsv(
    strategy,
    TABLE_DIR / "Table_5_statistical_model_development_strategy.tsv"
)

write_csv(
    strategy,
    TABLE_DIR / "Table_5_statistical_model_development_strategy.csv"
)


# =============================================================================
# 12. MANUSCRIPT-READY REPRESENTATION TABLE
# =============================================================================

manuscript_representation = cross_representation.copy()

manuscript_representation["interpretation"] = [
    "Classical molecular descriptor representation",
    "Conformation-dependent 3D-QSAR representation",
    "Deep learned SMILES representation",
]

write_tsv(
    manuscript_representation,
    TABLE_DIR / "Table_6_representation_comparison_manuscript.tsv"
)


# =============================================================================
# 13. SUPPLEMENTARY MASTER FEATURE TABLE
# =============================================================================

print("\n" + "-" * 78)
print("13. MASTER SUPPLEMENTARY FEATURE TABLE")
print("-" * 78)

master_tables = []

for rep in ["molecular", "3d", "molformer"]:

    train_df = standardised[f"{rep}_train"].copy()
    test_df = standardised[f"{rep}_test"].copy()

    if "feature_name_standard" not in train_df.columns:
        continue

    train_cols = [
        "feature_name_standard",
        "auc_standard",
        "abs_auc_effect",
        "fdr_standard",
        "cliffs_delta_standard",
    ]

    test_cols = [
        "feature_name_standard",
        "auc_standard",
        "abs_auc_effect",
        "fdr_standard",
        "cliffs_delta_standard",
    ]

    train_cols = [
        c for c in train_cols if c in train_df.columns
    ]

    test_cols = [
        c for c in test_cols if c in test_df.columns
    ]

    tr = train_df[train_cols].copy()
    te = test_df[test_cols].copy()

    tr = tr.rename(
        columns={
            "auc_standard": "train_auc",
            "abs_auc_effect": "train_abs_auc_effect",
            "fdr_standard": "train_fdr",
            "cliffs_delta_standard":
                "train_cliffs_delta",
        }
    )

    te = te.rename(
        columns={
            "auc_standard": "test_auc",
            "abs_auc_effect": "test_abs_auc_effect",
            "fdr_standard": "test_fdr",
            "cliffs_delta_standard":
                "test_cliffs_delta",
        }
    )

    merged = tr.merge(
        te,
        on="feature_name_standard",
        how="outer",
    )

    merged.insert(
        0,
        "representation",
        normalise_representation(rep),
    )

    master_tables.append(
        merged
    )


master_feature_table = pd.concat(
    master_tables,
    ignore_index=True,
)

write_tsv(
    master_feature_table,
    SUPP_DIR / "Supplementary_Master_Feature_Statistics.tsv"
)


# =============================================================================
# 14. FIGURE 7 — EFFECT VS SIGNIFICANCE
# =============================================================================

print("\nGenerating Figure 7...")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.5),
)

for ax, rep in zip(
    axes,
    ["molecular", "3d", "molformer"],
):

    df = standardised[f"{rep}_train"].copy()

    if not {
        "auc_standard",
        "fdr_standard"
    }.issubset(df.columns):

        ax.set_visible(False)
        continue

    df = df.dropna(
        subset=[
            "auc_standard",
            "fdr_standard",
        ]
    )

    effect = (
        df["auc_standard"] - 0.5
    ).abs()

    neglog = -np.log10(
        np.maximum(
            df["fdr_standard"],
            np.finfo(float).tiny,
        )
    )

    ax.scatter(
        effect,
        neglog,
        s=12,
        alpha=0.5,
    )

    ax.axvline(
        AUC_EFFECT_THRESHOLD,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        -np.log10(FDR_ALPHA),
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("|AUC − 0.50|")
    ax.set_ylabel("−log10(FDR)")
    ax.set_title(
        normalise_representation(rep)
    )

fig.suptitle(
    "Statistical significance versus practical effect magnitude",
    y=1.02,
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_7_significance_vs_effect"
)


# =============================================================================
# 15. PUBLICATION TEXT SUMMARY
# =============================================================================

print("\n" + "-" * 78)
print("15. WRITING PUBLICATION SUMMARY")
print("-" * 78)

summary_lines = []

summary_lines.append(
    "STATISTICAL ANALYSIS AND REPRESENTATION DETERMINATION"
)
summary_lines.append("=" * 78)
summary_lines.append("")
summary_lines.append(
    "The dataset comprised 48,152 training molecules and 12,038 "
    "independent test molecules. Active compounds represented "
    "approximately 4.95% of each split, establishing a strongly "
    "imbalanced binary classification problem."
)
summary_lines.append("")
summary_lines.append(
    "Feature-level statistical characterization was performed using "
    "the training data with Mann-Whitney U tests, ROC-AUC as a "
    "distribution-independent discrimination metric, Cliff's delta "
    "as an effect-size measure, Kolmogorov-Smirnov distributional "
    "testing, and Benjamini-Hochberg false-discovery-rate correction."
)
summary_lines.append("")
summary_lines.append(
    "The analysis evaluated three molecular representations: classical "
    "molecular descriptors, conformation-dependent 3D-QSAR descriptors, "
    "and 768-dimensional MoLFormer embeddings generated directly from "
    "SMILES."
)
summary_lines.append("")
summary_lines.append(
    "Statistical significance was interpreted together with effect "
    "magnitude rather than raw P values alone. This distinction is "
    "important because the large sample size permits very small "
    "distributional differences to achieve statistical significance."
)
summary_lines.append("")
summary_lines.append(
    "The independent test set was not used for feature selection or "
    "model fitting. Instead, the test set was used only to evaluate "
    "whether feature-level observations identified in the training "
    "data were independently reproducible."
)
summary_lines.append("")
summary_lines.append(
    "The resulting statistical evidence is therefore used to define "
    "the representation and evaluation strategy for subsequent "
    "predictive modelling rather than to constitute the predictive "
    "model itself."
)
summary_lines.append("")
summary_lines.append(
    "No features were removed by this publication-packaging script."
)
summary_lines.append(
    "No predictive model was trained."
)
summary_lines.append(
    "No hyperparameter optimization was performed."
)

with open(
    OUTPUT_DIR / "README_publication_statistics.txt",
    "w",
) as f:

    f.write("\n".join(summary_lines))


# =============================================================================
# 16. FILE MANIFEST
# =============================================================================

print("\n" + "-" * 78)
print("16. GENERATING FILE MANIFEST")
print("-" * 78)

manifest = {
    "script": "08_publication_statistics_v2.py",
    "status": "PASS",
    "generated": timestamp(),
    "purpose":
        "Publication packaging of statistical feature analysis",
    "fdr_alpha": FDR_ALPHA,
    "auc_effect_threshold_fallback":
        AUC_EFFECT_THRESHOLD,
    "cliffs_delta_threshold_fallback":
        CLIFFS_DELTA_THRESHOLD,
    "model_training": False,
    "feature_removal": False,
    "test_used_for_model_fitting": False,
    "test_used_for_feature_selection": False,
    "inputs": {},
    "outputs": {},
}


for key, path in loaded.items():

    if path.exists():

        manifest["inputs"][key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }


for root, dirs, files in os.walk(OUTPUT_DIR):

    for file in files:

        p = Path(root) / file

        if p.name == "publication_manifest.json":
            continue

        rel = str(p.relative_to(OUTPUT_DIR))

        manifest["outputs"][rel] = {
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }


with open(
    META_DIR / "publication_manifest.json",
    "w",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# =============================================================================
# 17. SHA256 CHECKSUM FILE
# =============================================================================

checksum_file = META_DIR / "SHA256SUMS.txt"

with open(checksum_file, "w") as f:

    for root, dirs, files in os.walk(OUTPUT_DIR):

        for file in sorted(files):

            p = Path(root) / file

            if p == checksum_file:
                continue

            f.write(
                f"{sha256_file(p)}  "
                f"{p.relative_to(OUTPUT_DIR)}\n"
            )


# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "=" * 78)
print("SCRIPT 08 v2 COMPLETE")
print("=" * 78)

print("\nOUTPUT DIRECTORY:")
print(f"  {OUTPUT_DIR}")

print("\nFIGURES:")
for p in sorted(FIG_DIR.glob("*")):
    print(f"  {p}")

print("\nTABLES:")
for p in sorted(TABLE_DIR.glob("*")):
    print(f"  {p}")

print("\nSUPPLEMENTARY:")
for p in sorted(SUPP_DIR.glob("*")):
    print(f"  {p}")

print("\nMETADATA:")
for p in sorted(META_DIR.glob("*")):
    print(f"  {p}")

print("\nFINAL STATUS: PASS")
print("=" * 78)
