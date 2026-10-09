#!/usr/bin/env python3
"""
==============================================================================
SCRIPT 08 — TARGET / POSSIBLE OFF-TARGET PUBLICATION FIGURE GENERATOR
==============================================================================

Purpose
-------
Generate publication-quality plots and graphical summaries for the
CMAUP 40D similarity-supported target / possible-off-target inference engine.

This script DOES NOT:
    - train a model
    - fit a new scaler
    - perform feature selection
    - calculate new molecular descriptors
    - calculate new molecular embeddings
    - modify Script-05 similarity
    - modify Script-07 target inference
    - invent target annotations
    - invent validation statistics

It only reads existing Script-06 / Script-07 outputs and generates figures.

Primary outputs
---------------
01_inference_workflow.png
02_neighbor_similarity_distribution.png
03_top20_neighbor_similarity.png
04_target_vs_offtarget_similarity.png
05_target_candidate_support.png
06_target_hierarchy.png
07_target_evidence_heatmap.png
08_target_similarity_vs_support.png
09_known_target_recovery.png
10_similarity_threshold_validation.png
11_target_coverage_validation.png
12_offtarget_candidate_distribution.png
13_example_target_network.png
14_inference_summary.png

All figures are also written as vector PDF/SVG where appropriate.

==============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

PULP_DIR = Path("/lustre/home/sklab202/sourav/PULP")

SCRIPT07_DIR = (
    PULP_DIR
    / "target_model"
    / "inference"
    / "07_query_40d_target_offtarget"
)

SIMILARITY_DIR = (
    PULP_DIR
    / "target_model"
    / "similarity"
    / "05_cmaup_40d_similarity"
)

PLOTS_DIR = (
    PULP_DIR
    / "target_model"
    / "plots"
    / "08_target_offtarget"
)

FIGURE_DATA_DIR = PLOTS_DIR / "figure_data"


# Candidate directories for Script-06 outputs.
SCRIPT06_CANDIDATE_DIRS = [
    PULP_DIR / "target_model" / "validation" / "06_target_validation",
    PULP_DIR / "target_model" / "validation" / "06_cmaup_target_validation",
    PULP_DIR / "target_model" / "similarity" / "06_target_validation",
    PULP_DIR / "target_model" / "inference" / "06_target_validation",
    PULP_DIR / "target_model",
]


# =============================================================================
# 1. GLOBAL PLOT SETTINGS
# =============================================================================

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.autolayout": False,
    }
)


# =============================================================================
# 2. LOGGING
# =============================================================================

START_TIME = datetime.now(timezone.utc).astimezone()


def log(message: str) -> None:
    print(message, flush=True)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# 3. FILE UTILITIES
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def save_figure(
    fig: plt.Figure,
    filename: str,
    *,
    svg: bool = True,
    pdf: bool = True,
) -> List[str]:

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    png_path = PLOTS_DIR / f"{filename}.png"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    outputs = [str(png_path)]

    if pdf:
        pdf_path = PLOTS_DIR / f"{filename}.pdf"

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        outputs.append(str(pdf_path))

    if svg:
        svg_path = PLOTS_DIR / f"{filename}.svg"

        fig.savefig(
            svg_path,
            bbox_inches="tight",
            facecolor="white",
        )

        outputs.append(str(svg_path))

    plt.close(fig)

    return outputs


# =============================================================================
# 4. DATASET DISCOVERY
# =============================================================================

def discover_tsv_files() -> List[Path]:

    roots = [
        SCRIPT07_DIR,
        SIMILARITY_DIR,
    ]

    for directory in SCRIPT06_CANDIDATE_DIRS:
        if directory.exists():
            roots.append(directory)

    files = []

    seen = set()

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob("*.tsv"):

            try:
                resolved = path.resolve()
            except Exception:
                resolved = path

            if resolved in seen:
                continue

            seen.add(resolved)
            files.append(path)

    return sorted(files)


def classify_file(path: Path) -> str:

    name = path.name.lower()

    if "neighbor" in name:
        return "neighbor"

    if "similarity" in name:
        return "similarity"

    if "target" in name and "candidate" in name:
        return "target_candidate"

    if "offtarget" in name or "off_target" in name:
        return "offtarget"

    if "support" in name:
        return "support"

    if "validation" in name:
        return "validation"

    if "qc" in name:
        return "qc"

    return "other"


def print_discovered_files(files: Sequence[Path]) -> None:

    section("DISCOVERED TSV FILES")

    if not files:
        log("No TSV files discovered.")
        return

    for path in files:

        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except Exception:
            size_mb = float("nan")

        log(
            f"{classify_file(path):18s} | "
            f"{size_mb:10.2f} MB | "
            f"{path}"
        )


# =============================================================================
# 5. COLUMN NORMALIZATION
# =============================================================================

def normalized_column_name(value: str) -> str:

    value = str(value)

    value = value.strip().lower()

    value = re.sub(r"[^a-z0-9]+", "_", value)

    value = re.sub(r"_+", "_", value)

    return value.strip("_")


def column_map(df: pd.DataFrame) -> Dict[str, str]:

    return {
        normalized_column_name(column): column
        for column in df.columns
    }


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:

    cmap = column_map(df)

    for candidate in candidates:

        key = normalized_column_name(candidate)

        if key in cmap:
            return cmap[key]

    return None


def find_columns(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> List[str]:

    cmap = column_map(df)

    result = []

    for candidate in candidates:

        key = normalized_column_name(candidate)

        if key in cmap:
            result.append(cmap[key])

    return result


# =============================================================================
# 6. SAFE NUMERIC CONVERSION
# =============================================================================

def numeric_series(
    df: pd.DataFrame,
    column: Optional[str],
) -> Optional[pd.Series]:

    if column is None:
        return None

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# =============================================================================
# 7. LOAD SCRIPT-07 OUTPUTS
# =============================================================================

def load_script07_tables() -> Dict[str, pd.DataFrame]:

    section("LOADING SCRIPT-07 TABLES")

    tables: Dict[str, pd.DataFrame] = {}

    if not SCRIPT07_DIR.exists():

        log(
            f"WARNING: Script-07 output directory not found:\n"
            f"{SCRIPT07_DIR}"
        )

        return tables

    for path in sorted(SCRIPT07_DIR.glob("*.tsv")):

        try:

            df = pd.read_csv(
                path,
                sep="\t",
                low_memory=False,
            )

            tables[path.stem] = df

            log(
                f"LOADED | {path.name} | "
                f"rows={len(df)} | cols={len(df.columns)}"
            )

        except Exception as exc:

            log(
                f"SKIPPED | {path.name} | "
                f"ERROR={repr(exc)}"
            )

    return tables


# =============================================================================
# 8. FIND RELEVANT DATASET
# =============================================================================

def choose_table(
    tables: Dict[str, pd.DataFrame],
    keywords: Sequence[str],
    *,
    exclude: Sequence[str] = (),
) -> Optional[pd.DataFrame]:

    candidates = []

    for name, df in tables.items():

        lower = name.lower()

        if any(word.lower() in lower for word in keywords):

            if not any(
                word.lower() in lower
                for word in exclude
            ):
                candidates.append((name, df))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    selected_name, selected_df = candidates[0]

    log(
        f"Selected table: {selected_name} "
        f"(rows={len(selected_df)}, cols={len(selected_df.columns)})"
    )

    return selected_df


# =============================================================================
# 9. TARGET CANDIDATE PREPARATION
# =============================================================================

def prepare_target_candidates(
    tables: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:

    candidates = choose_table(
        tables,
        keywords=[
            "target_candidate",
            "target_candidates",
            "inference",
        ],
        exclude=[
            "metadata",
            "feature",
        ],
    )

    if candidates is None:
        return None

    df = candidates.copy()

    similarity_col = find_column(
        df,
        [
            "Similarity",
            "Target_Similarity",
            "Best_Similarity",
            "Max_Similarity",
            "Cosine_Similarity",
        ],
    )

    role_col = find_column(
        df,
        [
            "Inference_Role",
            "Role",
        ],
    )

    rank_col = find_column(
        df,
        [
            "Inference_Rank",
            "Rank",
            "Target_Rank",
        ],
    )

    protein_col = find_column(
        df,
        [
            "Protein",
            "Protein_Name",
            "Target_Protein",
        ],
    )

    gene_col = find_column(
        df,
        [
            "Gene",
            "Gene_Symbol",
            "Target_Gene",
        ],
    )

    support_col = find_column(
        df,
        [
            "Supporting_Molecule_Count",
            "Support_Count",
            "Number_of_Supporting_Molecules",
            "Supporting_Molecules",
            "Evidence_Count",
        ],
    )

    if similarity_col is not None:

        df["_plot_similarity"] = numeric_series(
            df,
            similarity_col,
        )

    else:

        df["_plot_similarity"] = np.nan

    if rank_col is not None:

        df["_plot_rank"] = numeric_series(
            df,
            rank_col,
        )

    else:

        df["_plot_rank"] = np.arange(
            1,
            len(df) + 1,
        )

    if support_col is not None:

        df["_plot_support"] = numeric_series(
            df,
            support_col,
        )

    else:

        df["_plot_support"] = np.nan

    df["_plot_role"] = (
        df[role_col].astype(str)
        if role_col is not None
        else "UNKNOWN"
    )

    df["_plot_protein"] = (
        df[protein_col].astype(str)
        if protein_col is not None
        else "n.a."
    )

    df["_plot_gene"] = (
        df[gene_col].astype(str)
        if gene_col is not None
        else "n.a."
    )

    return df


# =============================================================================
# 10. FIGURE 1 — WORKFLOW
# =============================================================================

def plot_inference_workflow() -> List[str]:

    fig, ax = plt.subplots(
        figsize=(14, 7),
    )

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        (
            0.4,
            4.8,
            2.2,
            1.1,
            "Query SMILES",
        ),
        (
            3.1,
            4.8,
            2.6,
            1.1,
            "Locked 40D\nrepresentation",
        ),
        (
            6.2,
            4.8,
            2.6,
            1.1,
            "Train-only\nstandardization",
        ),
        (
            9.3,
            4.8,
            2.6,
            1.1,
            "Cosine similarity\n60,190 molecules",
        ),
        (
            12.0,
            4.8,
            1.6,
            1.1,
            "Nearest\nneighbors",
        ),
        (
            2.0,
            2.0,
            2.8,
            1.1,
            "CMAUP target\nannotations",
        ),
        (
            5.7,
            2.0,
            2.8,
            1.1,
            "Target-level\nevidence aggregation",
        ),
        (
            9.4,
            2.8,
            2.8,
            1.1,
            "Probable target\nhypothesis",
        ),
        (
            9.4,
            1.1,
            2.8,
            1.1,
            "Possible off-target\nhypotheses",
        ),
    ]

    for x, y, w, h, text in boxes:

        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            fill=False,
        )

        ax.add_patch(patch)

        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    arrows = [
        ((2.6, 5.35), (3.1, 5.35)),
        ((5.7, 5.35), (6.2, 5.35)),
        ((8.8, 5.35), (9.3, 5.35)),
        ((11.9, 5.35), (12.0, 5.35)),
        ((12.7, 4.8), (7.1, 3.1)),
        ((4.8, 2.55), (5.7, 2.55)),
        ((8.5, 2.55), (9.4, 3.35)),
        ((8.5, 2.45), (9.4, 1.65)),
    ]

    for start, end in arrows:

        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.2,
        )

        ax.add_patch(arrow)

    ax.text(
        7,
        6.45,
        "Similarity-supported target / possible off-target inference engine",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    ax.text(
        7,
        0.25,
        "10 molecular descriptors + 9 3D-QSAR descriptors + "
        "21 MoLFormer dimensions",
        ha="center",
        va="center",
        fontsize=9,
    )

    return save_figure(
        fig,
        "01_inference_workflow",
    )


# =============================================================================
# 11. FIGURE 2 — NEIGHBOR SIMILARITY DISTRIBUTION
# =============================================================================

def plot_neighbor_similarity_distribution(
    files: Sequence[Path],
) -> Optional[List[str]]:

    candidate_files = [
        path
        for path in files
        if "neighbor" in path.name.lower()
        or "similarity" in path.name.lower()
    ]

    if not candidate_files:
        log("SKIP Figure 02: no neighbor/similarity TSV found.")
        return None

    frames = []

    for path in candidate_files:

        try:

            df = pd.read_csv(
                path,
                sep="\t",
                low_memory=False,
            )

        except Exception:
            continue

        sim_col = find_column(
            df,
            [
                "Similarity",
                "Cosine_Similarity",
                "Best_Similarity",
                "Target_Similarity",
            ],
        )

        if sim_col is None:
            continue

        values = numeric_series(
            df,
            sim_col,
        ).dropna()

        if len(values) == 0:
            continue

        frames.append(values)

    if not frames:
        log("SKIP Figure 02: similarity column not found.")
        return None

    similarities = pd.concat(
        frames,
        ignore_index=True,
    )

    similarities = similarities[
        np.isfinite(similarities)
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5.5),
    )

    ax.hist(
        similarities,
        bins=40,
        density=True,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_xlabel(
        "Cosine similarity"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Similarity distribution of retrieved molecular neighbors"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return save_figure(
        fig,
        "02_neighbor_similarity_distribution",
    )


# =============================================================================
# 12. FIGURE 3 — TOP 20 NEIGHBORS
# =============================================================================

def plot_top20_neighbors(
    tables: Dict[str, pd.DataFrame],
) -> Optional[List[str]]:

    df = choose_table(
        tables,
        keywords=[
            "neighbor",
            "similarity",
        ],
    )

    if df is None:
        log("SKIP Figure 03: neighbor table unavailable.")
        return None

    similarity_col = find_column(
        df,
        [
            "Similarity",
            "Cosine_Similarity",
            "Best_Similarity",
        ],
    )

    rank_col = find_column(
        df,
        [
            "Rank",
            "Neighbor_Rank",
            "Inference_Rank",
        ],
    )

    if similarity_col is None:
        log("SKIP Figure 03: similarity column unavailable.")
        return None

    work = df.copy()

    work["_sim"] = numeric_series(
        work,
        similarity_col,
    )

    if rank_col is not None:
        work["_rank"] = numeric_series(
            work,
            rank_col,
        )
    else:
        work["_rank"] = np.arange(
            1,
            len(work) + 1,
        )

    work = work.dropna(
        subset=["_sim", "_rank"]
    )

    work = work.sort_values(
        "_rank"
    ).head(20)

    if work.empty:
        return None

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    ax.barh(
        work["_rank"].astype(int)[::-1],
        work["_sim"][::-1],
    )

    ax.set_xlabel(
        "Cosine similarity"
    )

    ax.set_ylabel(
        "Neighbor rank"
    )

    ax.set_title(
        "Top 20 CMAUP molecular neighbors"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    return save_figure(
        fig,
        "03_top20_neighbor_similarity",
    )


# =============================================================================
# 13. FIGURE 4 — TARGET VS OFF-TARGET
# =============================================================================

def plot_target_vs_offtarget(
    candidates: pd.DataFrame,
) -> Optional[List[str]]:

    if candidates is None or candidates.empty:
        return None

    work = candidates.copy()

    work = work[
        np.isfinite(
            work["_plot_similarity"]
        )
    ]

    if work.empty:
        log(
            "SKIP Figure 04: candidate similarity values unavailable."
        )
        return None

    target_mask = (
        work["_plot_role"]
        .str.upper()
        .str.contains("PROBABLE")
    )

    target_values = work.loc[
        target_mask,
        "_plot_similarity",
    ]

    off_values = work.loc[
        ~target_mask,
        "_plot_similarity",
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )

    data = []

    labels = []

    if len(target_values):
        data.append(target_values.values)
        labels.append("Probable target")

    if len(off_values):
        data.append(off_values.values)
        labels.append("Possible off-target")

    if not data:
        return None

    ax.boxplot(
        data,
        labels=labels,
        showfliers=True,
    )

    ax.set_ylabel(
        "Similarity"
    )

    ax.set_title(
        "Similarity of inferred target candidates"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return save_figure(
        fig,
        "04_target_vs_offtarget_similarity",
    )


# =============================================================================
# 14. FIGURE 5 — TARGET SUPPORT
# =============================================================================

def plot_target_support(
    candidates: pd.DataFrame,
) -> Optional[List[str]]:

    if candidates is None or candidates.empty:
        return None

    if candidates["_plot_support"].isna().all():
        log(
            "SKIP Figure 05: supporting-molecule count unavailable."
        )
        return None

    work = candidates.copy()

    work = work[
        np.isfinite(
            work["_plot_support"]
        )
    ]

    work = work.sort_values(
        "_plot_support",
        ascending=False,
    ).head(20)

    labels = []

    for _, row in work.iterrows():

        gene = str(row["_plot_gene"])

        if gene.lower() != "nan" and gene.lower() != "n.a.":
            labels.append(gene)
        else:
            labels.append(
                str(row["_plot_protein"])
            )

    fig, ax = plt.subplots(
        figsize=(9, 7),
    )

    ax.barh(
        labels[::-1],
        work["_plot_support"].values[::-1],
    )

    ax.set_xlabel(
        "Number of supporting molecules"
    )

    ax.set_ylabel(
        "Target"
    )

    ax.set_title(
        "Target-level supporting molecular evidence"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    return save_figure(
        fig,
        "05_target_candidate_support",
    )


# =============================================================================
# 15. FIGURE 6 — TARGET HIERARCHY
# =============================================================================

def plot_target_hierarchy(
    candidates: pd.DataFrame,
) -> Optional[List[str]]:

    if candidates is None or candidates.empty:
        return None

    columns = [
        find_column(candidates, ["Class_1", "Class1"]),
        find_column(candidates, ["Class_2", "Class2"]),
        find_column(candidates, ["Class_3", "Class3"]),
        find_column(candidates, ["Protein", "Protein_Name"]),
        find_column(candidates, ["Gene", "Gene_Symbol"]),
    ]

    labels = [
        "Class 1",
        "Class 2",
        "Class 3",
        "Protein",
        "Gene",
    ]

    counts = []

    for col in columns:

        if col is None:

            counts.append(0)

            continue

        values = (
            candidates[col]
            .astype(str)
            .str.strip()
        )

        valid = (
            values.notna()
            & (values != "")
            & (values.str.lower() != "n.a.")
            & (values.str.lower() != "nan")
            & (values.str.lower() != "none")
        )

        counts.append(
            int(valid.sum())
        )

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    ax.bar(
        labels,
        counts,
    )

    ax.set_ylabel(
        "Candidates with annotation"
    )

    ax.set_title(
        "Target annotation hierarchy completeness"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return save_figure(
        fig,
        "06_target_hierarchy",
    )


# =============================================================================
# 16. FIGURE 7 — EVIDENCE HEATMAP
# =============================================================================

def plot_target_evidence_heatmap(
    candidates: pd.DataFrame,
) -> Optional[List[str]]:

    if candidates is None or candidates.empty:
        return None

    if candidates["_plot_support"].isna().all():
        log(
            "SKIP Figure 07: support counts unavailable."
        )
        return None

    work = candidates.copy()

    work = work[
        np.isfinite(
            work["_plot_similarity"]
        )
        & np.isfinite(
            work["_plot_support"]
        )
    ]

    if work.empty:
        return None

    work = work.sort_values(
        "_plot_similarity",
        ascending=False,
    ).head(20)

    labels = []

    for _, row in work.iterrows():

        gene = str(row["_plot_gene"])

        if gene.lower() not in {
            "nan",
            "n.a.",
            "none",
        }:

            labels.append(gene)

        else:

            labels.append(
                str(row["_plot_protein"])
            )

    matrix = np.column_stack(
        [
            work["_plot_similarity"].values,
            work["_plot_support"].values,
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 8),
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Similarity",
            "Support count",
        ]
    )

    ax.set_yticks(
        np.arange(len(labels))
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_title(
        "Target-level evidence landscape"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Scaled value",
    )

    return save_figure(
        fig,
        "07_target_evidence_heatmap",
    )


# =============================================================================
# 17. FIGURE 8 — SIMILARITY VS SUPPORT
# =============================================================================

def plot_similarity_vs_support(
    candidates: pd.DataFrame,
) -> Optional[List[str]]:

    if candidates is None or candidates.empty:
        return None

    work = candidates.copy()

    work = work[
        np.isfinite(
            work["_plot_similarity"]
        )
        & np.isfinite(
            work["_plot_support"]
        )
    ]

    if work.empty:
        log(
            "SKIP Figure 08: similarity/support data unavailable."
        )
        return None

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    target_mask = (
        work["_plot_role"]
        .str.upper()
        .str.contains("PROBABLE")
    )

    ax.scatter(
        work.loc[
            ~target_mask,
            "_plot_support",
        ],
        work.loc[
            ~target_mask,
            "_plot_similarity",
        ],
        s=45,
        alpha=0.7,
        label="Possible off-target",
    )

    if target_mask.any():

        ax.scatter(
            work.loc[
                target_mask,
                "_plot_support",
            ],
            work.loc[
                target_mask,
                "_plot_similarity",
            ],
            s=70,
            marker="*",
            label="Probable target",
        )

    ax.set_xlabel(
        "Supporting molecule count"
    )

    ax.set_ylabel(
        "Similarity"
    )

    ax.set_title(
        "Target similarity versus supporting evidence"
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        alpha=0.2,
    )

    return save_figure(
        fig,
        "08_target_similarity_vs_support",
    )


# =============================================================================
# 18. VALIDATION DATA DISCOVERY
# =============================================================================

def load_validation_tables() -> Dict[str, pd.DataFrame]:

    section("LOADING VALIDATION TABLES")

    tables = {}

    for directory in SCRIPT06_CANDIDATE_DIRS:

        if not directory.exists():
            continue

        for path in directory.rglob("*.tsv"):

            name = path.name.lower()

            if not any(
                token in name
                for token in [
                    "validation",
                    "recall",
                    "neighbor",
                    "coverage",
                    "similarity",
                    "target",
                ]
            ):
                continue

            try:

                df = pd.read_csv(
                    path,
                    sep="\t",
                    low_memory=False,
                )

                key = (
                    f"{path.parent.name}__{path.stem}"
                )

                tables[key] = df

                log(
                    f"VALIDATION | {path} | "
                    f"rows={len(df)} | cols={len(df.columns)}"
                )

            except Exception:
                continue

    return tables


# =============================================================================
# 19. FIGURE 9 — KNOWN TARGET RECOVERY
# =============================================================================

def plot_known_target_recovery(
    validation_tables: Dict[str, pd.DataFrame],
) -> Optional[List[str]]:

    selected = None

    for name, df in validation_tables.items():

        lower = name.lower()

        if (
            "recall" in lower
            or "recovery" in lower
            or "validation" in lower
        ):

            selected = df
            break

    if selected is None:
        log(
            "SKIP Figure 09: no known-target recovery table found."
        )
        return None

    df = selected.copy()

    recall_col = find_column(
        df,
        [
            "Recall",
            "Known_Target_Recall",
            "Target_Recall",
            "Recovery",
        ],
    )

    if recall_col is None:
        log(
            "SKIP Figure 09: recall column unavailable."
        )
        return None

    values = numeric_series(
        df,
        recall_col,
    ).dropna()

    if values.empty:
        return None

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )

    ax.hist(
        values,
        bins=30,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.75,
    )

    ax.axvline(
        values.mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {values.mean():.3f}",
    )

    ax.axvline(
        values.median(),
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {values.median():.3f}",
    )

    ax.set_xlabel(
        "Known-target recall"
    )

    ax.set_ylabel(
        "Number of compounds"
    )

    ax.set_title(
        "Known-target recovery across validation compounds"
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return save_figure(
        fig,
        "09_known_target_recovery",
    )


# =============================================================================
# 20. FIGURE 10 — SIMILARITY THRESHOLD VALIDATION
# =============================================================================

def plot_similarity_threshold_validation(
    validation_tables: Dict[str, pd.DataFrame],
) -> Optional[List[str]]:

    selected = None

    for name, df in validation_tables.items():

        lower = name.lower()

        if (
            "threshold" in lower
            or "validation" in lower
            or "recall" in lower
        ):

            similarity_col = find_column(
                df,
                [
                    "Similarity_Threshold",
                    "Threshold",
                    "Min_Similarity",
                ],
            )

            recall_col = find_column(
                df,
                [
                    "Recall",
                    "Known_Target_Recall",
                    "Target_Recall",
                ],
            )

            if (
                similarity_col is not None
                and recall_col is not None
            ):

                selected = df

                break

    if selected is None:

        log(
            "SKIP Figure 10: threshold-validation table unavailable."
        )

        return None

    threshold_col = find_column(
        selected,
        [
            "Similarity_Threshold",
            "Threshold",
            "Min_Similarity",
        ],
    )

    recall_col = find_column(
        selected,
        [
            "Recall",
            "Known_Target_Recall",
            "Target_Recall",
        ],
    )

    work = selected.copy()

    work["_threshold"] = numeric_series(
        work,
        threshold_col,
    )

    work["_recall"] = numeric_series(
        work,
        recall_col,
    )

    work = work.dropna(
        subset=[
            "_threshold",
            "_recall",
        ]
    )

    if work.empty:
        return None

    work = work.sort_values(
        "_threshold"
    )

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )

    ax.plot(
        work["_threshold"],
        work["_recall"],
        marker="o",
        linewidth=1.8,
        markersize=5,
    )

    ax.set_xlabel(
        "Similarity threshold"
    )

    ax.set_ylabel(
        "Known-target recall"
    )

    ax.set_title(
        "Known-target recovery across similarity thresholds"
    )

    ax.grid(
        alpha=0.2,
    )

    return save_figure(
        fig,
        "10_similarity_threshold_validation",
    )


# =============================================================================
# 21. FIGURE 11 — TARGET COVERAGE
# =============================================================================

def plot_target_coverage(
    validation_tables: Dict[str, pd.DataFrame],
) -> Optional[List[str]]:

    selected = None

    for name, df in validation_tables.items():

        lower = name.lower()

        coverage_col = find_column(
            df,
            [
                "Target_Bearing_Neighbor_Coverage",
                "Target_Coverage",
                "Coverage",
            ],
        )

        if coverage_col is not None:

            selected = df

            break

    if selected is None:

        log(
            "SKIP Figure 11: target coverage table unavailable."
        )

        return None

    coverage_col = find_column(
        selected,
        [
            "Target_Bearing_Neighbor_Coverage",
            "Target_Coverage",
            "Coverage",
        ],
    )

    values = numeric_series(
        selected,
        coverage_col,
    ).dropna()

    if values.empty:
        return None

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )

    ax.hist(
        values,
        bins=30,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.75,
    )

    ax.axvline(
        values.mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {values.mean():.3f}",
    )

    ax.set_xlabel(
        "Target-bearing-neighbor coverage"
    )

    ax.set_ylabel(
        "Number of compounds"
    )

    ax.set_title(
        "Target-bearing neighbor coverage"
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return save_figure(
        fig,
        "11_target_coverage_validation",
    )


# =============================================================================
# 22. FIGURE 12 — OFF-TARGET CANDIDATE DISTRIBUTION
# =============================================================================

def plot_offtarget_distribution(
    candidates: pd.DataFrame,
) -> Optional[List[str]]:

    if candidates is None or candidates.empty:
        return None

    roles = (
        candidates["_plot_role"]
        .astype(str)
        .str.upper()
    )

    off = candidates[
        roles.str.contains(
            "OFF"
        )
    ]

    if off.empty:
        log(
            "SKIP Figure 12: no possible off-target candidates."
        )
        return None

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )

    ax.hist(
        off["_plot_similarity"].dropna(),
        bins=20,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.75,
    )

    ax.set_xlabel(
        "Similarity"
    )

    ax.set_ylabel(
        "Number of possible off-target candidates"
    )

    ax.set_title(
        "Similarity distribution of possible off-target hypotheses"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return save_figure(
        fig,
        "12_offtarget_candidate_distribution",
    )


# =============================================================================
# 23. FIGURE 13 — TARGET EVIDENCE NETWORK
# =============================================================================

def plot_example_target_network(
    candidates: pd.DataFrame,
) -> Optional[List[str]]:

    if candidates is None or candidates.empty:
        return None

    work = candidates.copy()

    work = work.sort_values(
        "_plot_similarity",
        ascending=False,
    ).head(10)

    if work.empty:
        return None

    fig, ax = plt.subplots(
        figsize=(10, 7),
    )

    ax.set_xlim(
        -1,
        1,
    )

    ax.set_ylim(
        -1,
        1,
    )

    ax.axis("off")

    center = np.array(
        [0.0, 0.0]
    )

    ax.scatter(
        [center[0]],
        [center[1]],
        s=900,
        marker="o",
        edgecolor="black",
        linewidth=1.5,
    )

    ax.text(
        center[0],
        center[1],
        "QUERY",
        ha="center",
        va="center",
        fontweight="bold",
    )

    n = len(work)

    angles = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False,
    )

    radius = 0.72

    for angle, (_, row) in zip(
        angles,
        work.iterrows(),
    ):

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        role = str(
            row["_plot_role"]
        ).upper()

        marker = "*"

        if "OFF" in role:
            marker = "o"

        ax.plot(
            [0, x],
            [0, y],
            linewidth=0.8,
            alpha=0.5,
        )

        ax.scatter(
            [x],
            [y],
            s=350,
            marker=marker,
            edgecolor="black",
            linewidth=1,
        )

        gene = str(
            row["_plot_gene"]
        )

        if gene.lower() in {
            "nan",
            "n.a.",
            "none",
        }:

            gene = str(
                row["_plot_protein"]
            )

        if len(gene) > 22:
            gene = gene[:19] + "..."

        ax.text(
            x,
            y + 0.09,
            gene,
            ha="center",
            va="center",
            fontsize=8,
        )

        sim = row["_plot_similarity"]

        if np.isfinite(sim):

            ax.text(
                x,
                y - 0.10,
                f"{sim:.3f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title(
        "Example target / possible off-target hypothesis network"
    )

    return save_figure(
        fig,
        "13_example_target_network",
    )


# =============================================================================
# 24. FIGURE 14 — SUMMARY
# =============================================================================

def plot_inference_summary(
    candidates: pd.DataFrame,
) -> Optional[List[str]]:

    if candidates is None or candidates.empty:
        return None

    roles = (
        candidates["_plot_role"]
        .astype(str)
        .str.upper()
    )

    probable = int(
        roles.str.contains(
            "PROBABLE"
        ).sum()
    )

    off = int(
        roles.str.contains(
            "OFF"
        ).sum()
    )

    unknown = len(
        candidates
    ) - probable - off

    values = [
        probable,
        off,
    ]

    labels = [
        "Probable target",
        "Possible off-target",
    ]

    if unknown > 0:

        values.append(
            unknown
        )

        labels.append(
            "Other candidate"
        )

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Number of inferred candidates"
    )

    ax.set_title(
        "Target / possible off-target inference summary"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    for i, value in enumerate(values):

        ax.text(
            i,
            value,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    return save_figure(
        fig,
        "14_inference_summary",
    )


# =============================================================================
# 25. EXPORT FIGURE DATA
# =============================================================================

def export_candidate_plot_data(
    candidates: Optional[pd.DataFrame],
) -> Optional[str]:

    if candidates is None:
        return None

    FIGURE_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "_plot_rank",
        "_plot_similarity",
        "_plot_support",
        "_plot_role",
        "_plot_protein",
        "_plot_gene",
    ]

    available = [
        column
        for column in columns
        if column in candidates.columns
    ]

    output = candidates[
        available
    ].copy()

    path = (
        FIGURE_DATA_DIR
        / "target_candidate_plot_data.tsv"
    )

    output.to_csv(
        path,
        sep="\t",
        index=False,
    )

    return str(path)


# =============================================================================
# 26. MANIFEST
# =============================================================================

def write_manifest(
    discovered_files: Sequence[Path],
    generated_files: Sequence[str],
    skipped_figures: Sequence[str],
) -> None:

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "script": "08_target_offtarget_plots.py",
        "purpose": (
            "Publication-quality visualization of the CMAUP "
            "40D similarity-supported target / possible-off-target "
            "inference engine."
        ),
        "start_time": START_TIME.isoformat(),
        "end_time": datetime.now(
            timezone.utc
        ).astimezone().isoformat(),
        "pulp_directory": str(PULP_DIR),
        "script07_directory": str(SCRIPT07_DIR),
        "similarity_directory": str(SIMILARITY_DIR),
        "plot_directory": str(PLOTS_DIR),
        "generated_files": generated_files,
        "skipped_figures": skipped_figures,
        "input_files": [],
    }

    for path in discovered_files:

        if not path.exists():
            continue

        manifest["input_files"].append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    path = (
        PLOTS_DIR
        / "08_target_offtarget_plot_manifest.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )


# =============================================================================
# 27. MAIN
# =============================================================================

def main() -> int:

    section(
        "SCRIPT 08 — TARGET / POSSIBLE OFF-TARGET "
        "PUBLICATION FIGURE GENERATOR"
    )

    log(
        f"PULP directory : {PULP_DIR}"
    )

    log(
        f"Script-07 directory : {SCRIPT07_DIR}"
    )

    log(
        f"Output directory : {PLOTS_DIR}"
    )

    log(
        f"Start time : {START_TIME.isoformat()}"
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Discover files
    # -------------------------------------------------------------------------

    discovered_files = discover_tsv_files()

    print_discovered_files(
        discovered_files
    )

    # -------------------------------------------------------------------------
    # Load Script-07 tables
    # -------------------------------------------------------------------------

    script07_tables = load_script07_tables()

    candidates = prepare_target_candidates(
        script07_tables
    )

    if candidates is not None:

        log(
            f"Target candidate table prepared: "
            f"{len(candidates)} candidates."
        )

    else:

        log(
            "WARNING: target candidate table could not be identified."
        )

    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------

    generated_files = []

    skipped_figures = []

    def register(
        figure_name: str,
        result,
    ) -> None:

        if result:

            generated_files.extend(
                result
            )

        else:

            skipped_figures.append(
                figure_name
            )

    # Figure 1
    register(
        "01_inference_workflow",
        plot_inference_workflow(),
    )

    # Figure 2
    register(
        "02_neighbor_similarity_distribution",
        plot_neighbor_similarity_distribution(
            discovered_files
        ),
    )

    # Figure 3
    register(
        "03_top20_neighbor_similarity",
        plot_top20_neighbors(
            script07_tables
        ),
    )

    # Figure 4
    register(
        "04_target_vs_offtarget_similarity",
        (
            plot_target_vs_offtarget(
                candidates
            )
            if candidates is not None
            else None
        ),
    )

    # Figure 5
    register(
        "05_target_candidate_support",
        (
            plot_target_support(
                candidates
            )
            if candidates is not None
            else None
        ),
    )

    # Figure 6
    register(
        "06_target_hierarchy",
        (
            plot_target_hierarchy(
                candidates
            )
            if candidates is not None
            else None
        ),
    )

    # Figure 7
    register(
        "07_target_evidence_heatmap",
        (
            plot_target_evidence_heatmap(
                candidates
            )
            if candidates is not None
            else None
        ),
    )

    # Figure 8
    register(
        "08_target_similarity_vs_support",
        (
            plot_similarity_vs_support(
                candidates
            )
            if candidates is not None
            else None
        ),
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    validation_tables = load_validation_tables()

    register(
        "09_known_target_recovery",
        plot_known_target_recovery(
            validation_tables
        ),
    )

    register(
        "10_similarity_threshold_validation",
        plot_similarity_threshold_validation(
            validation_tables
        ),
    )

    register(
        "11_target_coverage_validation",
        plot_target_coverage(
            validation_tables
        ),
    )

    # -------------------------------------------------------------------------
    # Remaining candidate plots
    # -------------------------------------------------------------------------

    register(
        "12_offtarget_candidate_distribution",
        (
            plot_offtarget_distribution(
                candidates
            )
            if candidates is not None
            else None
        ),
    )

    register(
        "13_example_target_network",
        (
            plot_example_target_network(
                candidates
            )
            if candidates is not None
            else None
        ),
    )

    register(
        "14_inference_summary",
        (
            plot_inference_summary(
                candidates
            )
            if candidates is not None
            else None
        ),
    )

    # -------------------------------------------------------------------------
    # Export data used for plotting
    # -------------------------------------------------------------------------

    candidate_data_path = export_candidate_plot_data(
        candidates
    )

    if candidate_data_path:
        generated_files.append(
            candidate_data_path
        )

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------

    write_manifest(
        discovered_files,
        generated_files,
        skipped_figures,
    )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    section(
        "SCRIPT 08 COMPLETE"
    )

    log(
        f"Generated files : {len(generated_files)}"
    )

    log(
        f"Skipped figures : {len(skipped_figures)}"
    )

    if skipped_figures:

        log(
            "Skipped:"
        )

        for item in skipped_figures:
            log(
                f"  - {item}"
            )

        log(
            "Skipped figures indicate that the corresponding "
            "input dataset was not found or lacked the required columns."
        )

    log(
        f"Output directory:\n{PLOTS_DIR}"
    )

    log(
        "Manifest:"
    )

    log(
        str(
            PLOTS_DIR
            / "08_target_offtarget_plot_manifest.json"
        )
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
