#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 09 — PUBLICATION-READY TARGET / POSSIBLE OFF-TARGET RESULT FIGURES
==============================================================================

Purpose
-------
Generate six publication-quality result figures from the completed
40D target/off-target validation outputs.

IMPORTANT
---------
This script:
    - uses ONLY the validation-level TSV files
    - does NOT use any individual test-molecule/case-study result
    - does NOT retrain or modify the inference system
    - does NOT combine the figures into one image
    - generates six independent figures for later manual assembly

Figures
-------
A. Best-neighbor cosine similarity distribution
B. Similarity-band composition
C. Known-target recall distribution
D. Target evidence vs possible-off-target similarity
E. Similarity threshold vs neighborhood/evidence coverage
F. Target/off-target candidate burden

Output
------
target_model/plots/08_target_offtarget/09_publication_figures/

Each figure is saved as:
    PNG  -- high-resolution raster
    PDF  -- vector publication format
    SVG  -- editable vector format

Also produces:
    09_figure_statistics.tsv
    09_figure_manifest.json
    09_plot_qc.tsv
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D


# =============================================================================
# CONFIGURATION
# =============================================================================

PULP_DIR = Path("/lustre/home/sklab202/sourav/PULP")

VALIDATION_DIR = (
    PULP_DIR
    / "target_model"
    / "validation"
    / "06_40d_target_offtarget_validation"
)

OUTPUT_DIR = (
    PULP_DIR
    / "target_model"
    / "plots"
    / "08_target_offtarget"
    / "09_publication_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# INPUT FILES
# =============================================================================

FILES = {
    "validation_summary":
        VALIDATION_DIR / "06_validation_summary.tsv",

    "target_recovery":
        VALIDATION_DIR / "06_target_recovery.tsv",

    "neighbor_recovery":
        VALIDATION_DIR / "06_neighbor_recovery.tsv",

    "offtarget_recovery":
        VALIDATION_DIR / "06_offtarget_recovery.tsv",

    "threshold_summary":
        VALIDATION_DIR / "06_similarity_threshold_summary.tsv",
}


# =============================================================================
# GLOBAL FIGURE SETTINGS
# =============================================================================

# Publication dimensions
FIG_WIDTH = 7.2
FIG_HEIGHT = 5.0

DPI = 600

FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,

    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    "xtick.major.size": 4,
    "ytick.major.size": 4,

    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "axes.grid": False,
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_tsv(path, label):
    print(f"Loading {label}")
    print(f"  {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        sep="\t"
    )

    print(
        f"  rows={len(df):,}, "
        f"columns={len(df.columns):,}"
    )

    return df


validation = load_tsv(
    FILES["validation_summary"],
    "06_validation_summary.tsv"
)

target_recovery = load_tsv(
    FILES["target_recovery"],
    "06_target_recovery.tsv"
)

neighbor_recovery = load_tsv(
    FILES["neighbor_recovery"],
    "06_neighbor_recovery.tsv"
)

offtarget_recovery = load_tsv(
    FILES["offtarget_recovery"],
    "06_offtarget_recovery.tsv"
)

threshold = load_tsv(
    FILES["threshold_summary"],
    "06_similarity_threshold_summary.tsv"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

figure_statistics = []
qc_records = []


def save_figure(fig, stem):
    """
    Save PNG, PDF and SVG.
    """

    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    svg = OUTPUT_DIR / f"{stem}.svg"

    fig.savefig(
        png,
        dpi=DPI,
        facecolor="white"
    )

    fig.savefig(
        pdf,
        facecolor="white"
    )

    fig.savefig(
        svg,
        facecolor="white"
    )

    plt.close(fig)

    print(f"  PNG: {png}")
    print(f"  PDF: {pdf}")
    print(f"  SVG: {svg}")


def add_panel_label(ax, label):
    """
    Add publication-style panel label.
    """

    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )


def clean_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()


def record_stat(
    figure,
    metric,
    value,
    unit=None
):
    figure_statistics.append({
        "figure": figure,
        "metric": metric,
        "value": value,
        "unit": unit,
    })


def check_finite(values, label):
    values = np.asarray(values, dtype=float)

    finite = np.isfinite(values)

    qc_records.append({
        "dataset": label,
        "n_total": int(len(values)),
        "n_finite": int(finite.sum()),
        "n_nonfinite": int((~finite).sum()),
    })

    return values[finite]


# =============================================================================
# FIGURE A
# =============================================================================

print()
print("=" * 78)
print("FIGURE A — BEST-NEIGHBOR COSINE SIMILARITY")
print("=" * 78)

best_similarity = clean_numeric(
    validation["best_neighbor_similarity"]
)

best_similarity = check_finite(
    best_similarity,
    "best_neighbor_similarity"
)

mean_similarity = best_similarity.mean()
median_similarity = np.median(best_similarity)

q25 = np.quantile(
    best_similarity,
    0.25
)

q75 = np.quantile(
    best_similarity,
    0.75
)

q95 = np.quantile(
    best_similarity,
    0.95
)

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)

# Histogram
ax.hist(
    best_similarity,
    bins=45,
    density=True,
    alpha=0.78,
    edgecolor="white",
    linewidth=0.35
)

# Median
ax.axvline(
    median_similarity,
    linestyle="--",
    linewidth=1.4,
    label=f"Median = {median_similarity:.3f}"
)

# Mean
ax.axvline(
    mean_similarity,
    linestyle=":",
    linewidth=1.4,
    label=f"Mean = {mean_similarity:.3f}"
)

ax.set_xlabel(
    "Best-neighbor cosine similarity"
)

ax.set_ylabel(
    "Density"
)

ax.set_title(
    "Best-neighbor similarity across validation queries"
)

ax.set_xlim(
    max(0.60, best_similarity.min() - 0.01),
    min(1.00, best_similarity.max() + 0.005)
)

ax.legend(
    frameon=False,
    loc="upper left"
)

ax.text(
    0.985,
    0.95,
    (
        f"N = {len(best_similarity):,}\n"
        f"IQR = {q25:.3f}–{q75:.3f}\n"
        f"95th percentile = {q95:.3f}"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9
)

add_panel_label(
    ax,
    "A"
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_A_best_neighbor_similarity"
)

record_stat(
    "A",
    "N",
    len(best_similarity),
    "queries"
)

record_stat(
    "A",
    "mean_best_similarity",
    mean_similarity,
    "cosine"
)

record_stat(
    "A",
    "median_best_similarity",
    median_similarity,
    "cosine"
)

record_stat(
    "A",
    "Q25",
    q25,
    "cosine"
)

record_stat(
    "A",
    "Q75",
    q75,
    "cosine"
)

record_stat(
    "A",
    "Q95",
    q95,
    "cosine"
)


# =============================================================================
# FIGURE B
# =============================================================================

print()
print("=" * 78)
print("FIGURE B — SIMILARITY BAND COMPOSITION")
print("=" * 78)

if "similarity_band" not in neighbor_recovery.columns:
    raise KeyError(
        "similarity_band column missing from "
        "06_neighbor_recovery.tsv"
    )

band_counts = (
    neighbor_recovery[
        "similarity_band"
    ]
    .fillna("Missing")
    .value_counts()
)

# Preserve scientifically meaningful order
preferred_order = [
    "Low",
    "Moderate",
    "High",
    "Missing",
]

ordered_bands = [
    x for x in preferred_order
    if x in band_counts.index
]

ordered_bands += [
    x for x in band_counts.index
    if x not in ordered_bands
]

band_counts = band_counts.loc[
    ordered_bands
]

band_percent = (
    band_counts
    / band_counts.sum()
    * 100
)

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, 4.3)
)

bottom = 0

for band in band_counts.index:

    value = band_percent.loc[band]

    ax.bar(
        ["Retrieved neighbors"],
        [value],
        bottom=bottom,
        width=0.48,
        label=f"{band} ({value:.2f}%)"
    )

    if value >= 5:

        ax.text(
            0,
            bottom + value / 2,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=10
        )

    bottom += value

ax.set_ylim(
    0,
    100
)

ax.set_ylabel(
    "Neighbor observations (%)"
)

ax.set_title(
    "Similarity composition of retrieved chemical neighborhoods"
)

ax.yaxis.set_major_formatter(
    PercentFormatter(100)
)

ax.legend(
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

add_panel_label(
    ax,
    "B"
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_B_similarity_band_composition"
)

for band in band_counts.index:

    record_stat(
        "B",
        f"{band}_neighbor_fraction_percent",
        float(band_percent.loc[band]),
        "%"
    )


# =============================================================================
# FIGURE C
# =============================================================================

print()
print("=" * 78)
print("FIGURE C — KNOWN-TARGET RECALL")
print("=" * 78)

recall = clean_numeric(
    validation[
        "known_target_recall_at_top_k"
    ]
)

recall = check_finite(
    recall,
    "known_target_recall_at_top_k"
)

# Convert NumPy array explicitly
recall = np.asarray(
    recall,
    dtype=float
)

# If recall is represented as fraction 0–1,
# convert to percentage.
if recall.max() <= 1.0:
    recall_percent = recall * 100.0
else:
    recall_percent = recall.copy()

# NumPy-compatible statistics
mean_recall = np.mean(
    recall_percent
)

median_recall = np.median(
    recall_percent
)

q25_recall = np.quantile(
    recall_percent,
    0.25
)

q75_recall = np.quantile(
    recall_percent,
    0.75
)

q95_recall = np.quantile(
    recall_percent,
    0.95
)

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)

# Histogram with percentage bins
bins = np.linspace(
    0,
    100,
    21
)

ax.hist(
    recall_percent,
    bins=bins,
    edgecolor="white",
    linewidth=0.35,
    alpha=0.82
)

ax.axvline(
    mean_recall,
    linestyle=":",
    linewidth=1.4,
    label=f"Mean = {mean_recall:.2f}%"
)

ax.axvline(
    median_recall,
    linestyle="--",
    linewidth=1.4,
    label=f"Median = {median_recall:.2f}%"
)

ax.set_xlim(
    0,
    100
)

ax.set_xlabel(
    "Known-target recall at evaluated retrieval depth (%)"
)

ax.set_ylabel(
    "Number of validation queries"
)

ax.set_title(
    "Distribution of known-target recovery across validation queries"
)

ax.legend(
    frameon=False,
    loc="upper right"
)

ax.text(
    0.98,
    0.95,
    (
        f"N = {len(recall_percent):,}\n"
        f"IQR = {q25_recall:.1f}–{q75_recall:.1f}%\n"
        f"95th percentile = {q95_recall:.1f}%"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9
)

add_panel_label(
    ax,
    "C"
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_C_known_target_recall"
)

record_stat(
    "C",
    "N",
    len(recall_percent),
    "queries"
)

record_stat(
    "C",
    "mean_recall",
    mean_recall,
    "%"
)

record_stat(
    "C",
    "median_recall",
    median_recall,
    "%"
)

record_stat(
    "C",
    "Q25",
    q25_recall,
    "%"
)

record_stat(
    "C",
    "Q75",
    q75_recall,
    "%"
)

record_stat(
    "C",
    "Q95",
    q95_recall,
    "%"
)

# =============================================================================
# FIGURE D
# =============================================================================

print()
print("=" * 78)
print("FIGURE D — TARGET VS POSSIBLE OFF-TARGET SIMILARITY")
print("=" * 78)

target_similarity = clean_numeric(
    target_recovery[
        "max_similarity"
    ]
)

offtarget_similarity = clean_numeric(
    offtarget_recovery[
        "max_similarity"
    ]
)

target_similarity = check_finite(
    target_similarity,
    "target_max_similarity"
)

offtarget_similarity = check_finite(
    offtarget_similarity,
    "offtarget_max_similarity"
)

# Explicit NumPy conversion
target_similarity = np.asarray(
    target_similarity,
    dtype=float
)

offtarget_similarity = np.asarray(
    offtarget_similarity,
    dtype=float
)

# NumPy-compatible statistics
target_median = np.median(
    target_similarity
)

offtarget_median = np.median(
    offtarget_similarity
)

target_q25 = np.quantile(
    target_similarity,
    0.25
)

target_q75 = np.quantile(
    target_similarity,
    0.75
)

offtarget_q25 = np.quantile(
    offtarget_similarity,
    0.25
)

offtarget_q75 = np.quantile(
    offtarget_similarity,
    0.75
)

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)

# -------------------------------------------------------------------------
# Violin distributions
# -------------------------------------------------------------------------

parts = ax.violinplot(
    [
        target_similarity,
        offtarget_similarity
    ],
    positions=[1, 2],
    widths=0.72,
    showmeans=False,
    showmedians=True,
    showextrema=True
)

# -------------------------------------------------------------------------
# Compact boxplots over the violin distributions
# -------------------------------------------------------------------------

ax.boxplot(
    [
        target_similarity,
        offtarget_similarity
    ],
    positions=[1, 2],
    widths=0.16,
    showfliers=False,
    patch_artist=False,
    medianprops={
        "linewidth": 1.8
    }
)

ax.set_xticks(
    [1, 2]
)

ax.set_xticklabels(
    [
        "Target evidence",
        "Possible off-target"
    ]
)

ax.set_ylabel(
    "Maximum cosine similarity"
)

ax.set_title(
    "Similarity strength of target and possible off-target evidence"
)

ax.set_ylim(
    0.60,
    1.00
)

# -------------------------------------------------------------------------
# Median annotations
# -------------------------------------------------------------------------

ax.text(
    1,
    target_median + 0.008,
    f"Median = {target_median:.3f}",
    ha="center",
    va="bottom",
    fontsize=9
)

ax.text(
    2,
    offtarget_median + 0.008,
    f"Median = {offtarget_median:.3f}",
    ha="center",
    va="bottom",
    fontsize=9
)

# -------------------------------------------------------------------------
# IQR annotation
# -------------------------------------------------------------------------

ax.text(
    0.98,
    0.04,
    (
        f"Target IQR: {target_q25:.3f}–{target_q75:.3f}\n"
        f"Off-target IQR: {offtarget_q25:.3f}–{offtarget_q75:.3f}"
    ),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.5
)

add_panel_label(
    ax,
    "D"
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_D_target_vs_offtarget_similarity"
)

record_stat(
    "D",
    "target_N",
    len(target_similarity),
    "observations"
)

record_stat(
    "D",
    "target_median_max_similarity",
    target_median,
    "cosine"
)

record_stat(
    "D",
    "offtarget_N",
    len(offtarget_similarity),
    "observations"
)

record_stat(
    "D",
    "offtarget_median_max_similarity",
    offtarget_median,
    "cosine"
)

# =============================================================================
# FIGURE E
# =============================================================================

print()
print("=" * 78)
print("FIGURE E — SIMILARITY THRESHOLD BEHAVIOR")
print("=" * 78)

threshold_df = threshold.copy()

threshold_df[
    "similarity_threshold"
] = pd.to_numeric(
    threshold_df[
        "similarity_threshold"
    ],
    errors="coerce"
)

threshold_df[
    "neighbor_rows"
] = pd.to_numeric(
    threshold_df[
        "neighbor_rows"
    ],
    errors="coerce"
)

threshold_df[
    "target_bearing_neighbor_rows"
] = pd.to_numeric(
    threshold_df[
        "target_bearing_neighbor_rows"
    ],
    errors="coerce"
)

threshold_df = threshold_df.dropna(
    subset=[
        "similarity_threshold",
        "neighbor_rows",
        "target_bearing_neighbor_rows"
    ]
)

threshold_df = threshold_df.sort_values(
    "similarity_threshold"
)

threshold_df[
    "target_bearing_fraction_percent"
] = (
    threshold_df[
        "target_bearing_neighbor_rows"
    ]
    / threshold_df[
        "neighbor_rows"
    ]
    * 100
)

fig, ax1 = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)

# Number of retrieved neighbors
ax1.plot(
    threshold_df[
        "similarity_threshold"
    ],
    threshold_df[
        "neighbor_rows"
    ],
    marker="o",
    markersize=5,
    linewidth=1.8,
    label="Retrieved neighbor observations"
)

ax1.set_xlabel(
    "Cosine-similarity threshold"
)

ax1.set_ylabel(
    "Retrieved neighbor observations"
)

ax1.set_xlim(
    threshold_df[
        "similarity_threshold"
    ].min() - 0.01,
    threshold_df[
        "similarity_threshold"
    ].max() + 0.01
)

# Second axis
ax2 = ax1.twinx()

ax2.plot(
    threshold_df[
        "similarity_threshold"
    ],
    threshold_df[
        "target_bearing_fraction_percent"
    ],
    marker="s",
    markersize=5,
    linewidth=1.8,
    linestyle="--",
    label="Target-bearing neighbor fraction"
)

ax2.set_ylabel(
    "Target-bearing neighbor observations (%)"
)

ax2.yaxis.set_major_formatter(
    PercentFormatter()
)

ax1.set_title(
    "Effect of similarity stringency on neighborhood evidence"
)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    frameon=False,
    loc="center right"
)

# Annotate endpoints
first = threshold_df.iloc[0]
last = threshold_df.iloc[-1]

ax2.annotate(
    f"{first['target_bearing_fraction_percent']:.2f}%",
    (
        first["similarity_threshold"],
        first["target_bearing_fraction_percent"]
    ),
    xytext=(7, 7),
    textcoords="offset points",
    fontsize=8
)

ax2.annotate(
    f"{last['target_bearing_fraction_percent']:.2f}%",
    (
        last["similarity_threshold"],
        last["target_bearing_fraction_percent"]
    ),
    xytext=(-35, 8),
    textcoords="offset points",
    fontsize=8
)

add_panel_label(
    ax1,
    "E"
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_E_similarity_threshold_behavior"
)

for _, row in threshold_df.iterrows():

    record_stat(
        "E",
        f"neighbor_rows_at_threshold_{row['similarity_threshold']:.2f}",
        float(row["neighbor_rows"]),
        "observations"
    )

    record_stat(
        "E",
        f"target_bearing_fraction_at_threshold_{row['similarity_threshold']:.2f}",
        float(row["target_bearing_fraction_percent"]),
        "%"
    )


# =============================================================================
# FIGURE F
# =============================================================================

print()
print("=" * 78)
print("FIGURE F — CANDIDATE BURDEN")
print("=" * 78)

target_candidate_count = clean_numeric(
    validation[
        "candidate_high_similarity_target_count"
    ]
)

offtarget_candidate_count = clean_numeric(
    validation[
        "possible_offtarget_count"
    ]
)

target_candidate_count = check_finite(
    target_candidate_count,
    "high_similarity_target_candidate_count"
)

offtarget_candidate_count = check_finite(
    offtarget_candidate_count,
    "possible_offtarget_candidate_count"
)

# ECDF helper
def ecdf(values):

    values = np.sort(
        np.asarray(values)
    )

    y = np.arange(
        1,
        len(values) + 1
    ) / len(values)

    return values, y


target_x, target_y = ecdf(
    target_candidate_count
)

offtarget_x, offtarget_y = ecdf(
    offtarget_candidate_count
)

fig, ax = plt.subplots(
    figsize=(FIG_WIDTH, FIG_HEIGHT)
)

ax.step(
    target_x,
    target_y * 100,
    where="post",
    linewidth=2.0,
    label="High-similarity target candidates"
)

ax.step(
    offtarget_x,
    offtarget_y * 100,
    where="post",
    linewidth=2.0,
    linestyle="--",
    label="Possible off-target candidates"
)

ax.set_xlabel(
    "Candidate count per validation query"
)

ax.set_ylabel(
    "Queries with ≤ candidate count (%)"
)

ax.set_title(
    "Distribution of target and possible-off-target candidate burden"
)

ax.yaxis.set_major_formatter(
    PercentFormatter()
)

# Use a reasonable x range based on data
max_x = max(
    target_candidate_count.max(),
    offtarget_candidate_count.max()
)

ax.set_xlim(
    0,
    max_x
)

ax.set_ylim(
    0,
    100
)

ax.legend(
    frameon=False,
    loc="lower right"
)

# Add median lines
target_median_count = np.median(
    target_candidate_count
)

offtarget_median_count = np.median(
    offtarget_candidate_count
)

ax.axvline(
    target_median_count,
    linestyle=":",
    linewidth=1.1
)

ax.axvline(
    offtarget_median_count,
    linestyle=":",
    linewidth=1.1
)

ax.text(
    target_median_count,
    5,
    f"Target median = {target_median_count:.0f}",
    rotation=90,
    va="bottom",
    ha="right",
    fontsize=8
)

ax.text(
    offtarget_median_count,
    5,
    f"Off-target median = {offtarget_median_count:.0f}",
    rotation=90,
    va="bottom",
    ha="left",
    fontsize=8
)

add_panel_label(
    ax,
    "F"
)

fig.tight_layout()

save_figure(
    fig,
    "Figure_F_candidate_burden"
)

record_stat(
    "F",
    "target_candidate_N",
    len(target_candidate_count),
    "queries"
)

record_stat(
    "F",
    "target_candidate_mean",
    target_candidate_count.mean(),
    "candidates/query"
)

record_stat(
    "F",
    "target_candidate_median",
    target_median_count,
    "candidates/query"
)

record_stat(
    "F",
    "target_candidate_max",
    target_candidate_count.max(),
    "candidates/query"
)

record_stat(
    "F",
    "offtarget_candidate_N",
    len(offtarget_candidate_count),
    "queries"
)

record_stat(
    "F",
    "offtarget_candidate_mean",
    offtarget_candidate_count.mean(),
    "candidates/query"
)

record_stat(
    "F",
    "offtarget_candidate_median",
    offtarget_median_count,
    "candidates/query"
)

record_stat(
    "F",
    "offtarget_candidate_max",
    offtarget_candidate_count.max(),
    "candidates/query"
)


# =============================================================================
# SAVE STATISTICS
# =============================================================================

print()
print("=" * 78)
print("SAVING FIGURE STATISTICS")
print("=" * 78)

statistics_df = pd.DataFrame(
    figure_statistics
)

statistics_path = (
    OUTPUT_DIR
    / "09_figure_statistics.tsv"
)

statistics_df.to_csv(
    statistics_path,
    sep="\t",
    index=False
)

print(
    f"Written: {statistics_path}"
)


# =============================================================================
# SAVE QC
# =============================================================================

qc_df = pd.DataFrame(
    qc_records
)

qc_path = (
    OUTPUT_DIR
    / "09_plot_qc.tsv"
)

qc_df.to_csv(
    qc_path,
    sep="\t",
    index=False
)

print(
    f"Written: {qc_path}"
)


# =============================================================================
# MANIFEST
# =============================================================================

figure_names = [
    "Figure_A_best_neighbor_similarity",
    "Figure_B_similarity_band_composition",
    "Figure_C_known_target_recall",
    "Figure_D_target_vs_offtarget_similarity",
    "Figure_E_similarity_threshold_behavior",
    "Figure_F_candidate_burden",
]

manifest = {
    "script": "09_generate_target_offtarget_result_plots.py",

    "purpose": (
        "Generate six publication-ready quantitative figures "
        "for similarity-supported target and possible off-target inference."
    ),

    "case_study_results_used": False,

    "figures": {
        "A": "Best-neighbor cosine similarity distribution",
        "B": "Similarity-band composition",
        "C": "Known-target recall distribution",
        "D": "Target evidence versus possible off-target similarity",
        "E": "Similarity threshold versus neighborhood/evidence coverage",
        "F": "Target and possible-off-target candidate burden",
    },

    "input_files": {
        key: str(value)
        for key, value in FILES.items()
    },

    "output_directory": str(
        OUTPUT_DIR
    ),

    "figure_formats": [
        "PNG",
        "PDF",
        "SVG",
    ],

    "figure_names": figure_names,
}

manifest_path = (
    OUTPUT_DIR
    / "09_figure_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)

print(
    f"Written: {manifest_path}"
)


# =============================================================================
# COMPLETE
# =============================================================================

print()
print("=" * 78)
print("ALL SIX FIGURES GENERATED")
print("=" * 78)
print()
print(
    f"Output directory:\n{OUTPUT_DIR}"
)
print()

for name in figure_names:

    print(f"  {name}.png")
    print(f"  {name}.pdf")
    print(f"  {name}.svg")

print()
print("No case-study molecule results were used.")
print("Figures are intentionally separate for manual composite assembly.")
print()
