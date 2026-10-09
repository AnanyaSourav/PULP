#!/usr/bin/env python3

# =============================================================================
# SCRIPT 14 — PUBLICATION FIGURE GENERATION
# =============================================================================
#
# PURPOSE
# -------
# Generate publication-ready MAIN and SUPPLEMENTARY figures from the completed
# phytochemical bioactivity model-development outputs.
#
# NO MODEL TRAINING IS PERFORMED.
# NO FEATURE SELECTION IS PERFORMED.
# NO TEST SET INFORMATION IS USED TO ALTER THE MODEL.
#
# INPUT DIRECTORY
# ---------------
# feature_engineering/model_development/
#
# EXPECTED INPUTS
# ----------------
#
# figure_data/
#   figure_01_roc_curve_data.tsv
#   figure_02_pr_curve_data.tsv
#   figure_03_calibration_curve_data.tsv
#
# tables/
#   TABLE_MASTER_PUBLICATION_MODEL_RESULTS.tsv
#   extra_trees_cv_fold_metrics.tsv
#   extra_trees_oof_threshold_analysis.tsv
#   logistic_regression_cv_fold_metrics.tsv
#   logistic_regression_oof_threshold_analysis.tsv
#   random_forest_cv_fold_metrics.tsv
#   random_forest_oof_threshold_analysis.tsv
#   table_01_script08_feature_shortlist.tsv
#   table_02_feature_family_counts.tsv
#   table_03_final_40_feature_manifest.tsv
#   table_04_model_comparison_cv.tsv
#   table_05_model_selection_ranking.tsv
#   table_06_independent_test_metrics.tsv
#   table_07_confusion_matrix.tsv
#   table_08_prediction_distribution_summary.tsv
#   table_09_borderline_prediction_summary.tsv
#   table_10_screening_nnt.tsv
#   table_11_permutation_feature_importance.tsv
#   table_12_feature_family_contribution.tsv
#   table_13_top20_feature_importance.tsv
#
# Optional:
# predictions/test_compound_predictions.tsv
#
# OUTPUT
# ------
# feature_engineering/model_development/publication_figures/
#
#   main/
#   supplementary/
#   standalone/
#   source_data/
#   publication_figure_manifest.tsv
#   publication_figure_generation_report.txt
#
# FIGURES
# -------
#
# MAIN FIGURE 1
# Model development and model selection
#   A. Model architecture / feature composition
#   B. CV ROC-AUC comparison
#   C. CV PR-AUC comparison
#   D. Model ranking / balanced classification
#
# MAIN FIGURE 2
# Independent test discrimination and calibration
#   A. ROC curve
#   B. Precision–Recall curve
#   C. Calibration curve
#   D. Raw vs calibrated probability-quality metrics
#
# MAIN FIGURE 3
# Threshold-dependent screening behaviour
#   A. Sensitivity/specificity vs threshold
#   B. F1/MCC/Youden J vs threshold
#   C. Confusion matrix at locked threshold
#   D. Locked threshold vs conventional 0.5 threshold
#
# MAIN FIGURE 4
# Feature interpretation
#   A. Top-20 permutation importance
#   B. Feature-family importance contribution
#   C. Predictor counts by representation
#   D. Feature importance distribution by family
#
# MAIN FIGURE 5
# Prediction behaviour and screening utility
#   A. Active vs inactive probability statistics
#   B. Probability-region distribution
#   C. Uncertainty/certainty comparison
#   D. Screening summary / NNT
#
# SUPPLEMENTARY
#   S1  Five-fold CV ROC-AUC
#   S2  Five-fold CV PR-AUC
#   S3  Five-fold Brier score
#   S4  Five-fold log loss
#   S5  Complete threshold analysis
#   S6  MCC vs threshold
#   S7  F1 vs threshold
#   S8  Sensitivity / specificity vs threshold
#   S9  Full 40-feature permutation importance
#   S10 Feature-family contribution
#   S11 Feature composition
#   S12 Prediction-distribution summary
#   S13 Borderline-prediction summary
#   S14 Raw vs calibrated metrics
#   S15 Model-comparison heatmap
#   S16 Model-performance radar plot
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import json
import math
import shutil
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

try:
    from matplotlib.patches import FancyBboxPatch
except Exception:
    FancyBboxPatch = None


warnings.filterwarnings("ignore")


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    "feature_engineering/model_development"
)

FIGURE_DATA_DIR = BASE_DIR / "figure_data"
TABLE_DIR = BASE_DIR / "tables"
PREDICTION_DIR = BASE_DIR / "predictions"

OUTPUT_DIR = BASE_DIR / "publication_figures"

MAIN_DIR = OUTPUT_DIR / "main"
SUPP_DIR = OUTPUT_DIR / "supplementary"
STANDALONE_DIR = OUTPUT_DIR / "standalone"
SOURCE_DATA_DIR = OUTPUT_DIR / "source_data"


# -----------------------------------------------------------------------------
# Figure export
# -----------------------------------------------------------------------------

PNG_DPI = 600

MAIN_WIDTH = 14
MAIN_HEIGHT = 11

SINGLE_WIDTH = 7.2
SINGLE_HEIGHT = 5.5


# -----------------------------------------------------------------------------
# Typography
# -----------------------------------------------------------------------------

FONT_FAMILY = "DejaVu Sans"

TITLE_SIZE = 14
PANEL_LABEL_SIZE = 14
AXIS_LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 9


# -----------------------------------------------------------------------------
# Model values already fixed during model development
# -----------------------------------------------------------------------------

LOCKED_THRESHOLD = 0.046126477738876355

TEST_PREVALENCE = 596 / 12038


# -----------------------------------------------------------------------------
# Display names
# -----------------------------------------------------------------------------

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
}

REPRESENTATION_DISPLAY = {
    "molecular": "Molecular",
    "3d": "3D-QSAR",
    "molformer": "MoLFormer",
}


# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": FONT_FAMILY,

    "font.size": 10,

    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,

    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,

    "legend.fontsize": LEGEND_SIZE,

    "axes.linewidth": 1.0,

    "figure.dpi": 100,

    "savefig.dpi": PNG_DPI,

    "savefig.bbox": "tight",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "svg.fonttype": "none",
})


# =============================================================================
# HELPERS
# =============================================================================

def print_header(text):

    print()
    print("=" * 90)
    print(text)
    print("=" * 90)


def print_section(text):

    print()
    print("-" * 90)
    print(text)
    print("-" * 90)


def ensure_dirs():

    for directory in [
        OUTPUT_DIR,
        MAIN_DIR,
        SUPP_DIR,
        STANDALONE_DIR,
        SOURCE_DATA_DIR,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


def read_tsv(path, required=True):

    path = Path(path)

    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        return None

    return pd.read_csv(
        path,
        sep="\t"
    )


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def finite_xy(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    return (
        x[mask],
        y[mask]
    )


def save_figure(
    fig,
    outdir,
    basename
):

    outdir = Path(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = []

    png = outdir / f"{basename}.png"
    pdf = outdir / f"{basename}.pdf"
    svg = outdir / f"{basename}.svg"

    fig.savefig(
        png,
        dpi=PNG_DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    fig.savefig(
        svg,
        bbox_inches="tight"
    )

    outputs.extend([
        str(png),
        str(pdf),
        str(svg)
    ])

    plt.close(fig)

    return outputs


def panel_label(
    ax,
    label
):

    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left"
    )


def clean_axes(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def format_feature_name(name):

    replacements = {

        "molformer_": "MF-",

        "RadiusOfGyration":
            "Radius of gyration",

        "InertialShapeFactor":
            "Inertial shape factor",

        "SpherocityIndex":
            "Spherocity index",

        "NumAtomStereoCenters":
            "Atom stereocenters",

        "NumAromaticRings":
            "Aromatic rings",

        "FractionCSP3":
            "Fraction Csp3",

        "BCUT2D_CHGLO":
            "BCUT2D CHGLO",

        "BCUT2D_MRLOW":
            "BCUT2D MRLOW",

        "VSA_EState6":
            "VSA EState6",

        "fr_Al_OH":
            "Aliphatic OH",
    }

    for old, new in replacements.items():

        if name.startswith(old):

            if old == "molformer_":

                return name.replace(
                    old,
                    new
                )

            if name == old:

                return new

    return name


def model_name(name):

    return MODEL_DISPLAY.get(
        str(name),
        str(name).replace("_", " ").title()
    )


def representation_name(name):

    return REPRESENTATION_DISPLAY.get(
        str(name),
        str(name)
    )


# =============================================================================
# LOAD ALL DATA
# =============================================================================

def load_all_data():

    print_section(
        "LOADING MODEL-DEVELOPMENT OUTPUTS"
    )

    files = {

        "roc":
            FIGURE_DATA_DIR /
            "figure_01_roc_curve_data.tsv",

        "pr":
            FIGURE_DATA_DIR /
            "figure_02_pr_curve_data.tsv",

        "calibration":
            FIGURE_DATA_DIR /
            "figure_03_calibration_curve_data.tsv",

        "master":
            TABLE_DIR /
            "TABLE_MASTER_PUBLICATION_MODEL_RESULTS.tsv",

        "extra_cv":
            TABLE_DIR /
            "extra_trees_cv_fold_metrics.tsv",

        "extra_threshold":
            TABLE_DIR /
            "extra_trees_oof_threshold_analysis.tsv",

        "lr_cv":
            TABLE_DIR /
            "logistic_regression_cv_fold_metrics.tsv",

        "lr_threshold":
            TABLE_DIR /
            "logistic_regression_oof_threshold_analysis.tsv",

        "rf_cv":
            TABLE_DIR /
            "random_forest_cv_fold_metrics.tsv",

        "rf_threshold":
            TABLE_DIR /
            "random_forest_oof_threshold_analysis.tsv",

        "feature_shortlist":
            TABLE_DIR /
            "table_01_script08_feature_shortlist.tsv",

        "family_counts":
            TABLE_DIR /
            "table_02_feature_family_counts.tsv",

        "manifest":
            TABLE_DIR /
            "table_03_final_40_feature_manifest.tsv",

        "model_comparison":
            TABLE_DIR /
            "table_04_model_comparison_cv.tsv",

        "model_ranking":
            TABLE_DIR /
            "table_05_model_selection_ranking.tsv",

        "test_metrics":
            TABLE_DIR /
            "table_06_independent_test_metrics.tsv",

        "confusion":
            TABLE_DIR /
            "table_07_confusion_matrix.tsv",

        "prediction_distribution":
            TABLE_DIR /
            "table_08_prediction_distribution_summary.tsv",

        "borderline":
            TABLE_DIR /
            "table_09_borderline_prediction_summary.tsv",

        "nnt":
            TABLE_DIR /
            "table_10_screening_nnt.tsv",

        "importance":
            TABLE_DIR /
            "table_11_permutation_feature_importance.tsv",

        "family_importance":
            TABLE_DIR /
            "table_12_feature_family_contribution.tsv",

        "top20_importance":
            TABLE_DIR /
            "table_13_top20_feature_importance.tsv",

        "test_predictions":
            PREDICTION_DIR /
            "test_compound_predictions.tsv",
    }

    data = {}

    for key, path in files.items():

        required = (
            key != "test_predictions"
        )

        try:

            data[key] = read_tsv(
                path,
                required=required
            )

            if data[key] is not None:

                print(
                    f"FOUND {key:24s} "
                    f"{data[key].shape}  {path}"
                )

        except Exception as exc:

            print(
                f"ERROR reading {path}: {exc}"
            )

            raise

    return data, files


# =============================================================================
# EXTRACT TEST METRIC
# =============================================================================

def metric_from_test_table(
    table,
    metric,
    column="calibrated"
):

    sub = table[
        table["metric"]
        .astype(str)
        .str.lower()
        ==
        metric.lower()
    ]

    if len(sub) == 0:

        return np.nan

    try:

        return float(
            sub.iloc[0][column]
        )

    except Exception:

        return np.nan


# =============================================================================
# FIGURE 1A — FEATURE COMPOSITION
# =============================================================================

def plot_feature_composition(
    ax,
    family_counts
):

    df = family_counts.copy()

    df["label"] = (
        df["representation"]
        .map(representation_name)
    )

    df = df.sort_values(
        "n_features",
        ascending=False
    )

    bars = ax.bar(
        df["label"],
        df["n_features"],
        edgecolor="black",
        linewidth=0.7
    )

    for bar, value in zip(
        bars,
        df["n_features"]
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            bar.get_height()
            +
            0.4,

            str(int(value)),

            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_ylabel(
        "Number of selected predictors"
    )

    ax.set_title(
        "Final 40-feature representation"
    )

    clean_axes(ax)


# =============================================================================
# FIGURE 1B — CV ROC-AUC
# =============================================================================

def plot_cv_auc(
    ax,
    model_comparison
):

    df = model_comparison.copy()

    df["display"] = (
        df["model"]
        .map(model_name)
    )

    order = (
        df.sort_values(
            "cv_auc_mean"
        )
    )

    ax.errorbar(

        order["cv_auc_mean"],

        np.arange(
            len(order)
        ),

        xerr=
            order["cv_auc_sd"],

        fmt="o",

        capsize=4,

        linewidth=1.5
    )

    ax.set_yticks(
        np.arange(
            len(order)
        )
    )

    ax.set_yticklabels(
        order["display"]
    )

    ax.set_xlabel(
        "5-fold ROC-AUC"
    )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    ax.set_title(
        "Cross-validation discrimination"
    )

    clean_axes(ax)


# =============================================================================
# FIGURE 1C — CV PR-AUC
# =============================================================================

def plot_cv_pr_auc(
    ax,
    model_comparison
):

    df = model_comparison.copy()

    df["display"] = (
        df["model"]
        .map(model_name)
    )

    order = (
        df.sort_values(
            "cv_pr_auc_mean"
        )
    )

    ax.errorbar(

        order["cv_pr_auc_mean"],

        np.arange(
            len(order)
        ),

        xerr=
            order["cv_pr_auc_sd"],

        fmt="o",

        capsize=4,

        linewidth=1.5
    )

    ax.axvline(
        TEST_PREVALENCE,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Prevalence = {TEST_PREVALENCE:.3f}"
    )

    ax.set_yticks(
        np.arange(
            len(order)
        )
    )

    ax.set_yticklabels(
        order["display"]
    )

    ax.set_xlabel(
        "5-fold PR-AUC"
    )

    ax.set_title(
        "Cross-validation precision–recall"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# FIGURE 1D — BALANCED CLASSIFICATION
# =============================================================================

def plot_model_balanced_metrics(
    ax,
    model_comparison
):

    metrics = [
        "youden_sensitivity",
        "youden_specificity",
        "youden_f1",
        "youden_mcc",
        "youden_balanced_accuracy"
    ]

    display = [
        "Sensitivity",
        "Specificity",
        "F1",
        "MCC",
        "Balanced accuracy"
    ]

    df = model_comparison.copy()

    x = np.arange(
        len(metrics)
    )

    width = (
        0.8 /
        len(df)
    )

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        values = [
            float(
                row[m]
            )
            for m in metrics
        ]

        ax.bar(

            x
            +
            (i - (len(df)-1)/2)
            *
            width,

            values,

            width=width,

            label=model_name(
                row["model"]
            )
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        display,
        rotation=25,
        ha="right"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_title(
        "OOF threshold-dependent performance"
    )

    ax.legend(
        frameon=False,
        fontsize=8
    )

    clean_axes(ax)


# =============================================================================
# MAIN FIGURE 1
# =============================================================================

def make_main_figure_1(data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            MAIN_WIDTH,
            MAIN_HEIGHT
        )
    )

    plot_feature_composition(
        axes[0, 0],
        data["family_counts"]
    )

    plot_cv_auc(
        axes[0, 1],
        data["model_comparison"]
    )

    plot_cv_pr_auc(
        axes[1, 0],
        data["model_comparison"]
    )

    plot_model_balanced_metrics(
        axes[1, 1],
        data["model_comparison"]
    )

    for ax, label in zip(
        axes.flat,
        ["A", "B", "C", "D"]
    ):

        panel_label(
            ax,
            label
        )

    fig.suptitle(
        "Model development, selection and final predictor representation",
        fontsize=16,
        y=1.01
    )

    fig.tight_layout()

    return save_figure(
        fig,
        MAIN_DIR,
        "Figure_01_Model_Development_and_Selection"
    )


# =============================================================================
# ROC CURVE
# =============================================================================

def plot_roc(
    ax,
    roc,
    test_metrics
):

    groups = (
        roc["model_probability"]
        .dropna()
        .unique()
    )

    for group in groups:

        sub = roc[
            roc["model_probability"]
            ==
            group
        ]

        x, y = finite_xy(
            sub["fpr"],
            sub["tpr"]
        )

        label = (
            "Raw"
            if group == "raw"
            else "Calibrated"
        )

        auc_metric = (
            metric_from_test_table(
                test_metrics,
                "ROC_AUC",
                "raw"
                if group == "raw"
                else "calibrated"
            )
        )

        if np.isfinite(
            auc_metric
        ):

            label += (
                f" (AUC={auc_metric:.3f})"
            )

        ax.plot(
            x,
            y,
            linewidth=2,
            label=label
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1.02
    )

    ax.set_xlabel(
        "False-positive rate"
    )

    ax.set_ylabel(
        "True-positive rate"
    )

    ax.set_title(
        "Independent-test ROC"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# PR CURVE
# =============================================================================

def plot_pr(
    ax,
    pr,
    test_metrics
):

    groups = (
        pr["model_probability"]
        .dropna()
        .unique()
    )

    for group in groups:

        sub = pr[
            pr["model_probability"]
            ==
            group
        ]

        x, y = finite_xy(
            sub["recall"],
            sub["precision"]
        )

        label = (
            "Raw"
            if group == "raw"
            else "Calibrated"
        )

        ap_metric = (
            metric_from_test_table(
                test_metrics,
                "PR_AUC_average_precision",
                "raw"
                if group == "raw"
                else "calibrated"
            )
        )

        if np.isfinite(
            ap_metric
        ):

            label += (
                f" (AP={ap_metric:.3f})"
            )

        ax.plot(
            x,
            y,
            linewidth=2,
            label=label
        )

    ax.axhline(
        TEST_PREVALENCE,
        linestyle="--",
        linewidth=1,
        label=f"Prevalence={TEST_PREVALENCE:.3f}"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Independent-test precision–recall"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# CALIBRATION
# =============================================================================

def plot_calibration(
    ax,
    calibration
):

    groups = (
        calibration[
            "model_probability"
        ]
        .dropna()
        .unique()
    )

    for group in groups:

        sub = calibration[
            calibration[
                "model_probability"
            ]
            ==
            group
        ]

        sub = sub[
            sub["n"] > 0
        ]

        x, y = finite_xy(
            sub["mean_predicted"],
            sub["observed_fraction"]
        )

        label = (
            "Raw"
            if group == "raw"
            else "Calibrated"
        )

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.7,
            label=label
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Perfect calibration"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_xlabel(
        "Mean predicted probability"
    )

    ax.set_ylabel(
        "Observed active fraction"
    )

    ax.set_title(
        "Reliability / calibration"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# RAW VS CALIBRATED
# =============================================================================

def plot_raw_calibrated_metrics(
    ax,
    test_metrics
):

    wanted = [
        ("Brier_score", "Brier"),
        ("Log_loss", "Log loss"),
        ("ECE", "ECE"),
        ("MCE", "MCE"),
    ]

    available = []

    for key, display in wanted:

        sub = test_metrics[
            test_metrics["metric"]
            .astype(str)
            .str.lower()
            ==
            key.lower()
        ]

        if len(sub):

            available.append(
                (
                    display,
                    float(
                        sub.iloc[0][
                            "raw"
                        ]
                    ),
                    float(
                        sub.iloc[0][
                            "calibrated"
                        ]
                    )
                )
            )

    if len(available) == 0:

        ax.text(
            0.5,
            0.5,
            "Calibration metrics not found",
            ha="center",
            va="center"
        )

        ax.axis(
            "off"
        )

        return

    names = [
        x[0]
        for x in available
    ]

    raw = [
        x[1]
        for x in available
    ]

    calibrated = [
        x[2]
        for x in available
    ]

    x = np.arange(
        len(names)
    )

    width = 0.36

    ax.bar(
        x - width/2,
        raw,
        width,
        label="Raw"
    )

    ax.bar(
        x + width/2,
        calibrated,
        width,
        label="Calibrated"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        names
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_title(
        "Probability-quality metrics"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# MAIN FIGURE 2
# =============================================================================

def make_main_figure_2(data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            MAIN_WIDTH,
            MAIN_HEIGHT
        )
    )

    plot_roc(
        axes[0, 0],
        data["roc"],
        data["test_metrics"]
    )

    plot_pr(
        axes[0, 1],
        data["pr"],
        data["test_metrics"]
    )

    plot_calibration(
        axes[1, 0],
        data["calibration"]
    )

    plot_raw_calibrated_metrics(
        axes[1, 1],
        data["test_metrics"]
    )

    for ax, label in zip(
        axes.flat,
        ["A", "B", "C", "D"]
    ):

        panel_label(
            ax,
            label
        )

    fig.suptitle(
        "Independent-test discrimination and probability calibration",
        fontsize=16,
        y=1.01
    )

    fig.tight_layout()

    return save_figure(
        fig,
        MAIN_DIR,
        "Figure_02_Independent_Test_Discrimination_and_Calibration"
    )


# =============================================================================
# THRESHOLD CURVES
# =============================================================================

def filtered_threshold_table(df):

    x = df.copy()

    x["threshold"] = (
        pd.to_numeric(
            x["threshold"],
            errors="coerce"
        )
    )

    x = x[
        np.isfinite(
            x["threshold"]
        )
    ]

    x = x[
        (x["threshold"] >= 0)
        &
        (x["threshold"] <= 1)
    ]

    return x.sort_values(
        "threshold"
    )


def plot_sensitivity_specificity(
    ax,
    threshold_df
):

    df = filtered_threshold_table(
        threshold_df
    )

    ax.plot(
        df["threshold"],
        df["sensitivity"],
        linewidth=1.7,
        label="Sensitivity"
    )

    ax.plot(
        df["threshold"],
        df["specificity"],
        linewidth=1.7,
        label="Specificity"
    )

    ax.axvline(
        LOCKED_THRESHOLD,
        linestyle="--",
        linewidth=1.2,
        label=f"Locked threshold={LOCKED_THRESHOLD:.3f}"
    )

    ax.set_xlim(
        0,
        min(
            1,
            max(
                0.25,
                df["threshold"]
                .quantile(0.98)
            )
        )
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_xlabel(
        "Probability threshold"
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_title(
        "Sensitivity–specificity trade-off"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


def plot_threshold_composite(
    ax,
    threshold_df
):

    df = filtered_threshold_table(
        threshold_df
    )

    for metric, label in [
        ("f1", "F1"),
        ("mcc", "MCC"),
        ("youden_j", "Youden J"),
    ]:

        ax.plot(
            df["threshold"],
            df[metric],
            linewidth=1.6,
            label=label
        )

    ax.axvline(
        LOCKED_THRESHOLD,
        linestyle="--",
        linewidth=1.2
    )

    ax.set_xlim(
        0,
        min(
            1,
            max(
                0.25,
                df["threshold"]
                .quantile(0.98)
            )
        )
    )

    ax.set_xlabel(
        "Probability threshold"
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_title(
        "Threshold-dependent objective metrics"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def confusion_array(row):

    return np.array([
        [
            int(
                row[
                    "true_inactive_pred_inactive"
                ]
            ),
            int(
                row[
                    "true_inactive_pred_active"
                ]
            )
        ],
        [
            int(
                row[
                    "true_active_pred_inactive"
                ]
            ),
            int(
                row[
                    "true_active_pred_active"
                ]
            )
        ],
    ])


def plot_confusion_matrix(
    ax,
    confusion
):

    row = confusion[
        confusion[
            "threshold_policy"
        ]
        ==
        "locked_youden"
    ]

    if len(row) == 0:

        row = confusion.iloc[
            [0]
        ]

    row = row.iloc[0]

    matrix = confusion_array(
        row
    )

    im = ax.imshow(
        matrix,
        aspect="auto"
    )

    for i in range(2):

        for j in range(2):

            value = matrix[
                i,
                j
            ]

            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold"
            )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Predicted\ninactive",
            "Predicted\nactive"
        ]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        [
            "True inactive",
            "True active"
        ]
    )

    ax.set_title(
        f"Locked threshold confusion matrix\n"
        f"threshold={float(row['threshold']):.4f}"
    )

    plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )


# =============================================================================
# CONFUSION COMPARISON
# =============================================================================

def plot_threshold_policy_comparison(
    ax,
    confusion
):

    rows = []

    for _, row in confusion.iterrows():

        matrix = confusion_array(
            row
        )

        tn, fp = matrix[0]
        fn, tp = matrix[1]

        sensitivity = (
            tp /
            (tp + fn)
            if tp + fn > 0
            else np.nan
        )

        specificity = (
            tn /
            (tn + fp)
            if tn + fp > 0
            else np.nan
        )

        precision = (
            tp /
            (tp + fp)
            if tp + fp > 0
            else np.nan
        )

        f1 = (
            2 *
            precision *
            sensitivity /
            (
                precision
                +
                sensitivity
            )
            if (
                precision
                +
                sensitivity
            ) > 0
            else 0
        )

        rows.append(
            {
                "policy":
                    str(
                        row[
                            "threshold_policy"
                        ]
                    ),

                "sensitivity":
                    sensitivity,

                "specificity":
                    specificity,

                "precision":
                    precision,

                "f1":
                    f1
            }
        )

    df = pd.DataFrame(
        rows
    )

    metrics = [
        "sensitivity",
        "specificity",
        "precision",
        "f1"
    ]

    labels = [
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1"
    ]

    x = np.arange(
        len(metrics)
    )

    width = (
        0.8 /
        max(
            1,
            len(df)
        )
    )

    for i, row in df.iterrows():

        values = [
            row[m]
            for m in metrics
        ]

        ax.bar(

            x +
            (
                i
                -
                (len(df)-1)/2
            )
            *
            width,

            values,

            width=width,

            label=row["policy"]
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=20
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_title(
        "Locked threshold vs 0.5 threshold"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# MAIN FIGURE 3
# =============================================================================

def make_main_figure_3(data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            MAIN_WIDTH,
            MAIN_HEIGHT
        )
    )

    plot_sensitivity_specificity(
        axes[0, 0],
        data["extra_threshold"]
    )

    plot_threshold_composite(
        axes[0, 1],
        data["extra_threshold"]
    )

    plot_confusion_matrix(
        axes[1, 0],
        data["confusion"]
    )

    plot_threshold_policy_comparison(
        axes[1, 1],
        data["confusion"]
    )

    for ax, label in zip(
        axes.flat,
        ["A", "B", "C", "D"]
    ):

        panel_label(
            ax,
            label
        )

    fig.suptitle(
        "Locked-threshold screening behaviour of the final Extra Trees model",
        fontsize=16,
        y=1.01
    )

    fig.tight_layout()

    return save_figure(
        fig,
        MAIN_DIR,
        "Figure_03_Threshold_and_Classification_Performance"
    )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def plot_top20_importance(
    ax,
    importance
):

    df = importance.copy()

    df = df.sort_values(
        "importance_rank",
        ascending=False
    )

    labels = [
        format_feature_name(x)
        for x in df["feature"]
    ]

    ax.barh(

        labels,

        df[
            "permutation_importance_mean"
        ],

        xerr=
            df[
                "permutation_importance_sd"
            ],

        capsize=2
    )

    ax.set_xlabel(
        "Permutation importance\n"
        "(decrease in ROC-AUC)"
    )

    ax.set_title(
        "Top-20 predictors"
    )

    clean_axes(ax)


# =============================================================================
# FAMILY CONTRIBUTION
# =============================================================================

def plot_family_contribution(
    ax,
    family
):

    df = family.copy()

    df["display"] = (
        df["representation"]
        .map(representation_name)
    )

    df = df.sort_values(
        "relative_importance_percent",
        ascending=False
    )

    bars = ax.bar(

        df["display"],

        df[
            "relative_importance_percent"
        ]
    )

    for bar, value in zip(
        bars,
        df[
            "relative_importance_percent"
        ]
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width()/2,

            bar.get_height()
            +
            1,

            f"{value:.1f}%",

            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_ylabel(
        "Relative permutation importance (%)"
    )

    ax.set_title(
        "Contribution by molecular representation"
    )

    ax.set_ylim(
        0,
        max(
            df[
                "relative_importance_percent"
            ]
        )
        *
        1.20
    )

    clean_axes(ax)


# =============================================================================
# IMPORTANCE DISTRIBUTION
# =============================================================================

def plot_family_importance_distribution(
    ax,
    importance
):

    families = [
        "molecular",
        "3d",
        "molformer"
    ]

    arrays = []

    labels = []

    for family in families:

        values = (
            importance[
                importance[
                    "representation"
                ]
                ==
                family
            ][
                "permutation_importance_mean"
            ]
            .astype(float)
            .to_numpy()
        )

        arrays.append(
            values
        )

        labels.append(
            representation_name(
                family
            )
        )

    ax.boxplot(
        arrays,
        labels=labels,
        showfliers=True
    )

    ax.set_ylabel(
        "Permutation importance"
    )

    ax.set_title(
        "Importance distribution by representation"
    )

    clean_axes(ax)


# =============================================================================
# MAIN FIGURE 4
# =============================================================================

def make_main_figure_4(data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            MAIN_WIDTH,
            MAIN_HEIGHT
        )
    )

    plot_top20_importance(
        axes[0, 0],
        data["top20_importance"]
    )

    plot_family_contribution(
        axes[0, 1],
        data["family_importance"]
    )

    plot_feature_composition(
        axes[1, 0],
        data["family_counts"]
    )

    plot_family_importance_distribution(
        axes[1, 1],
        data["importance"]
    )

    for ax, label in zip(
        axes.flat,
        ["A", "B", "C", "D"]
    ):

        panel_label(
            ax,
            label
        )

    fig.suptitle(
        "Feature-level interpretation of the final phytochemical screening model",
        fontsize=16,
        y=1.01
    )

    fig.tight_layout()

    return save_figure(
        fig,
        MAIN_DIR,
        "Figure_04_Feature_Importance_and_Representation_Contribution"
    )


# =============================================================================
# PREDICTION SUMMARY
# =============================================================================

def plot_probability_summary(
    ax,
    summary
):

    df = summary.copy()

    x = np.arange(
        len(df)
    )

    means = (
        df[
            "mean_probability"
        ]
        .astype(float)
    )

    stds = (
        df[
            "std_probability"
        ]
        .astype(float)
    )

    ax.errorbar(

        x,

        means,

        yerr=stds,

        fmt="o",

        capsize=5,

        markersize=8
    )

    ax.axhline(
        LOCKED_THRESHOLD,
        linestyle="--",
        linewidth=1,
        label=f"Threshold={LOCKED_THRESHOLD:.3f}"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df[
            "true_activity"
        ]
    )

    ax.set_ylabel(
        "Predicted active probability"
    )

    ax.set_title(
        "Probability distribution by true class"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# PROBABILITY REGIONS
# =============================================================================

def plot_probability_regions(
    ax,
    summary
):

    df = summary.copy()

    categories = [
        "fraction_probability_lt_0.3",
        "fraction_probability_0.3_0.7",
        "fraction_probability_gt_0.7",
    ]

    labels = [
        "<0.3",
        "0.3–0.7",
        ">0.7"
    ]

    x = np.arange(
        len(df)
    )

    bottom = np.zeros(
        len(df)
    )

    for col, label in zip(
        categories,
        labels
    ):

        values = (
            df[col]
            .astype(float)
            .to_numpy()
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=label
        )

        bottom += values

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df[
            "true_activity"
        ]
    )

    ax.set_ylim(
        0,
        1
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            1.0
        )
    )

    ax.set_ylabel(
        "Fraction of predictions"
    )

    ax.set_title(
        "Probability-region distribution"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# UNCERTAINTY
# =============================================================================

def plot_uncertainty_summary(
    ax,
    summary
):

    df = summary.copy()

    x = np.arange(
        len(df)
    )

    width = 0.36

    uncertainty = (
        df[
            "mean_uncertainty"
        ]
        .astype(float)
    )

    certainty = (
        df[
            "mean_certainty"
        ]
        .astype(float)
    )

    ax.bar(
        x - width/2,
        uncertainty,
        width,
        label="Uncertainty"
    )

    ax.bar(
        x + width/2,
        certainty,
        width,
        label="Certainty"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        df[
            "true_activity"
        ]
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Mean score"
    )

    ax.set_title(
        "Prediction certainty"
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)


# =============================================================================
# SCREENING UTILITY
# =============================================================================

def plot_screening_utility(
    ax,
    nnt,
    confusion
):

    nnt_value = float(
        nnt.iloc[0][
            "value"
        ]
    )

    row = confusion[
        confusion[
            "threshold_policy"
        ]
        ==
        "locked_youden"
    ].iloc[0]

    matrix = confusion_array(
        row
    )

    tn, fp = matrix[0]
    fn, tp = matrix[1]

    precision = (
        tp /
        (tp + fp)
    )

    sensitivity = (
        tp /
        (tp + fn)
    )

    specificity = (
        tn /
        (tn + fp)
    )

    values = [
        precision,
        sensitivity,
        specificity
    ]

    labels = [
        "Precision",
        "Sensitivity",
        "Specificity"
    ]

    bars = ax.bar(
        labels,
        values
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width()/2,

            bar.get_height()
            +
            0.025,

            f"{value:.3f}",

            ha="center",
            fontsize=9
        )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Performance"
    )

    ax.set_title(
        f"Screening utility\n"
        f"NNT = {nnt_value:.2f}"
    )

    clean_axes(ax)


# =============================================================================
# MAIN FIGURE 5
# =============================================================================

def make_main_figure_5(data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            MAIN_WIDTH,
            MAIN_HEIGHT
        )
    )

    plot_probability_summary(
        axes[0, 0],
        data[
            "prediction_distribution"
        ]
    )

    plot_probability_regions(
        axes[0, 1],
        data[
            "prediction_distribution"
        ]
    )

    plot_uncertainty_summary(
        axes[1, 0],
        data[
            "prediction_distribution"
        ]
    )

    plot_screening_utility(
        axes[1, 1],
        data["nnt"],
        data["confusion"]
    )

    for ax, label in zip(
        axes.flat,
        ["A", "B", "C", "D"]
    ):

        panel_label(
            ax,
            label
        )

    fig.suptitle(
        "Independent-test prediction behaviour and screening utility",
        fontsize=16,
        y=1.01
    )

    fig.tight_layout()

    return save_figure(
        fig,
        MAIN_DIR,
        "Figure_05_Prediction_Behaviour_and_Screening_Utility"
    )


# =============================================================================
# SUPPLEMENTARY CV PLOTS
# =============================================================================

def combine_cv_tables(data):

    return pd.concat(
        [
            data["lr_cv"],
            data["rf_cv"],
            data["extra_cv"]
        ],
        ignore_index=True
    )


def cv_metric_plot(
    cv,
    metric,
    ylabel,
    title
):

    fig, ax = plt.subplots(
        figsize=(
            SINGLE_WIDTH,
            SINGLE_HEIGHT
        )
    )

    models = [
        "logistic_regression",
        "random_forest",
        "extra_trees"
    ]

    for model in models:

        sub = cv[
            cv["model"]
            ==
            model
        ].sort_values(
            "fold"
        )

        ax.plot(
            sub["fold"],
            sub[metric],
            marker="o",
            linewidth=1.7,
            label=model_name(
                model
            )
        )

    ax.set_xticks(
        sorted(
            cv[
                "fold"
            ]
            .unique()
        )
    )

    ax.set_xlabel(
        "Cross-validation fold"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)

    fig.tight_layout()

    return fig


# =============================================================================
# S5 THRESHOLD ANALYSIS
# =============================================================================

def make_full_threshold_plot(
    threshold_df
):

    df = filtered_threshold_table(
        threshold_df
    )

    fig, ax = plt.subplots(
        figsize=(
            SINGLE_WIDTH,
            SINGLE_HEIGHT
        )
    )

    metrics = [
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("f1", "F1"),
        ("mcc", "MCC"),
        ("youden_j", "Youden J"),
    ]

    for col, label in metrics:

        ax.plot(
            df["threshold"],
            df[col],
            linewidth=1.4,
            label=label
        )

    ax.axvline(
        LOCKED_THRESHOLD,
        linestyle="--",
        linewidth=1.2,
        label=f"Locked threshold={LOCKED_THRESHOLD:.4f}"
    )

    ax.set_xlim(
        0,
        min(
            1,
            max(
                0.3,
                df[
                    "threshold"
                ]
                .quantile(
                    0.99
                )
            )
        )
    )

    ax.set_xlabel(
        "Probability threshold"
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_title(
        "Complete Extra Trees OOF threshold analysis"
    )

    ax.legend(
        frameon=False,
        ncol=2
    )

    clean_axes(ax)

    fig.tight_layout()

    return fig


# =============================================================================
# SINGLE THRESHOLD METRIC
# =============================================================================

def make_threshold_metric_plot(
    threshold_df,
    metric,
    ylabel,
    title
):

    df = filtered_threshold_table(
        threshold_df
    )

    fig, ax = plt.subplots(
        figsize=(
            SINGLE_WIDTH,
            SINGLE_HEIGHT
        )
    )

    ax.plot(
        df["threshold"],
        df[metric],
        linewidth=1.8
    )

    ax.axvline(
        LOCKED_THRESHOLD,
        linestyle="--",
        linewidth=1.2,
        label=f"Locked={LOCKED_THRESHOLD:.4f}"
    )

    ax.set_xlim(
        0,
        min(
            1,
            max(
                0.3,
                df[
                    "threshold"
                ]
                .quantile(
                    0.99
                )
            )
        )
    )

    ax.set_xlabel(
        "Probability threshold"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.legend(
        frameon=False
    )

    clean_axes(ax)

    fig.tight_layout()

    return fig


# =============================================================================
# FULL 40 FEATURE IMPORTANCE
# =============================================================================

def full_importance_plot(
    importance
):

    df = importance.sort_values(
        "importance_rank",
        ascending=False
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            11
        )
    )

    labels = [
        format_feature_name(x)
        for x in df["feature"]
    ]

    ax.barh(

        labels,

        df[
            "permutation_importance_mean"
        ],

        xerr=
            df[
                "permutation_importance_sd"
            ],

        capsize=1.5
    )

    ax.set_xlabel(
        "Permutation importance"
    )

    ax.set_title(
        "Permutation importance of all 40 final predictors"
    )

    clean_axes(ax)

    fig.tight_layout()

    return fig


# =============================================================================
# BORDERLINE PREDICTIONS
# =============================================================================

def borderline_plot(
    borderline
):

    fig, ax = plt.subplots(
        figsize=(
            SINGLE_WIDTH,
            SINGLE_HEIGHT
        )
    )

    df = borderline.copy()

    labels = (
        df[
            "definition"
        ]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False
        )
    )

    bars = ax.bar(
        labels,
        df["fraction"]
    )

    for bar, fraction, n in zip(
        bars,
        df["fraction"],
        df["n"]
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width()/2,

            bar.get_height()
            +
            max(
                df["fraction"]
            ) * 0.04,

            f"{fraction*100:.2f}%\n(n={int(n):,})",

            ha="center",
            va="bottom"
        )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            1.0
        )
    )

    ax.set_ylabel(
        "Fraction of independent test set"
    )

    ax.set_title(
        "Borderline prediction frequency"
    )

    clean_axes(ax)

    fig.tight_layout()

    return fig


# =============================================================================
# MODEL COMPARISON HEATMAP
# =============================================================================

def model_heatmap(
    model_comparison
):

    df = (
        model_comparison
        .set_index(
            "model"
        )
    )

    metrics = [

        "cv_auc_mean",
        "cv_pr_auc_mean",
        "youden_sensitivity",
        "youden_specificity",
        "youden_precision",
        "youden_f1",
        "youden_mcc",
        "youden_balanced_accuracy",
    ]

    labels = [
        "CV ROC-AUC",
        "CV PR-AUC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
        "MCC",
        "Balanced accuracy"
    ]

    matrix = (
        df[metrics]
        .astype(float)
        .to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            4.8
        )
    )

    im = ax.imshow(
        matrix,
        aspect="auto"
    )

    ax.set_yticks(
        np.arange(
            len(df)
        )
    )

    ax.set_yticklabels(
        [
            model_name(x)
            for x in df.index
        ]
    )

    ax.set_xticks(
        np.arange(
            len(metrics)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=40,
        ha="right"
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            ax.text(
                j,
                i,
                f"{matrix[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.colorbar(
        im,
        ax=ax,
        label="Metric value"
    )

    ax.set_title(
        "Model-performance comparison"
    )

    fig.tight_layout()

    return fig


# =============================================================================
# RADAR PLOT
# =============================================================================

def radar_plot(
    model_comparison
):

    metrics = [
        "cv_auc_mean",
        "cv_pr_auc_mean",
        "youden_sensitivity",
        "youden_specificity",
        "youden_f1",
        "youden_mcc",
        "youden_balanced_accuracy"
    ]

    labels = [
        "ROC-AUC",
        "PR-AUC",
        "Sensitivity",
        "Specificity",
        "F1",
        "MCC",
        "Balanced\naccuracy"
    ]

    n = len(
        metrics
    )

    angles = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig = plt.figure(
        figsize=(
            8,
            8
        )
    )

    ax = fig.add_subplot(
        111,
        polar=True
    )

    for _, row in (
        model_comparison
        .iterrows()
    ):

        values = [
            float(
                row[m]
            )
            for m in metrics
        ]

        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            label=model_name(
                row["model"]
            )
        )

        ax.fill(
            angles,
            values,
            alpha=0.08
        )

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        "Multimetric model comparison",
        pad=25
    )

    ax.legend(
        bbox_to_anchor=(
            1.25,
            1.10
        ),
        frameon=False
    )

    fig.tight_layout()

    return fig


# =============================================================================
# SUPPLEMENTARY FIGURES
# =============================================================================

def make_supplementary_figures(data):

    outputs = []

    cv = combine_cv_tables(
        data
    )

    # S1
    fig = cv_metric_plot(
        cv,
        "roc_auc",
        "ROC-AUC",
        "Five-fold ROC-AUC stability"
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S01_CV_ROC_AUC"
    )

    # S2
    fig = cv_metric_plot(
        cv,
        "pr_auc_average_precision",
        "PR-AUC",
        "Five-fold PR-AUC stability"
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S02_CV_PR_AUC"
    )

    # S3
    fig = cv_metric_plot(
        cv,
        "brier",
        "Brier score",
        "Five-fold Brier-score stability"
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S03_CV_Brier"
    )

    # S4
    fig = cv_metric_plot(
        cv,
        "log_loss",
        "Log loss",
        "Five-fold log-loss stability"
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S04_CV_LogLoss"
    )

    # S5
    fig = make_full_threshold_plot(
        data[
            "extra_threshold"
        ]
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S05_Complete_Threshold_Analysis"
    )

    # S6
    fig = make_threshold_metric_plot(
        data[
            "extra_threshold"
        ],
        "mcc",
        "MCC",
        "Matthews correlation coefficient vs threshold"
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S06_MCC_vs_Threshold"
    )

    # S7
    fig = make_threshold_metric_plot(
        data[
            "extra_threshold"
        ],
        "f1",
        "F1 score",
        "F1 score vs threshold"
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S07_F1_vs_Threshold"
    )

    # S8
    fig, ax = plt.subplots(
        figsize=(
            SINGLE_WIDTH,
            SINGLE_HEIGHT
        )
    )

    plot_sensitivity_specificity(
        ax,
        data[
            "extra_threshold"
        ]
    )

    fig.tight_layout()

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S08_Sensitivity_Specificity_vs_Threshold"
    )

    # S9
    fig = full_importance_plot(
        data[
            "importance"
        ]
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S09_All_40_Feature_Importance"
    )

    # S10
    fig, ax = plt.subplots(
        figsize=(
            SINGLE_WIDTH,
            SINGLE_HEIGHT
        )
    )

    plot_family_contribution(
        ax,
        data[
            "family_importance"
        ]
    )

    fig.tight_layout()

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S10_Feature_Family_Contribution"
    )

    # S11
    fig, ax = plt.subplots(
        figsize=(
            SINGLE_WIDTH,
            SINGLE_HEIGHT
        )
    )

    plot_feature_composition(
        ax,
        data[
            "family_counts"
        ]
    )

    fig.tight_layout()

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S11_Feature_Composition"
    )

    # S12
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            16,
            5
        )
    )

    plot_probability_summary(
        axes[0],
        data[
            "prediction_distribution"
        ]
    )

    plot_probability_regions(
        axes[1],
        data[
            "prediction_distribution"
        ]
    )

    plot_uncertainty_summary(
        axes[2],
        data[
            "prediction_distribution"
        ]
    )

    fig.tight_layout()

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S12_Prediction_Distribution_Summary"
    )

    # S13
    fig = borderline_plot(
        data[
            "borderline"
        ]
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S13_Borderline_Predictions"
    )

    # S14
    fig, ax = plt.subplots(
        figsize=(
            SINGLE_WIDTH,
            SINGLE_HEIGHT
        )
    )

    plot_raw_calibrated_metrics(
        ax,
        data[
            "test_metrics"
        ]
    )

    fig.tight_layout()

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S14_Raw_vs_Calibrated_Metrics"
    )

    # S15
    fig = model_heatmap(
        data[
            "model_comparison"
        ]
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S15_Model_Comparison_Heatmap"
    )

    # S16
    fig = radar_plot(
        data[
            "model_comparison"
        ]
    )

    outputs += save_figure(
        fig,
        SUPP_DIR,
        "Figure_S16_Model_Performance_Radar"
    )

    return outputs


# =============================================================================
# STANDALONE CORE PLOTS
# =============================================================================

def make_standalone_figures(data):

    outputs = []

    items = [

        (
            "ROC_Curve",
            lambda ax:
                plot_roc(
                    ax,
                    data["roc"],
                    data[
                        "test_metrics"
                    ]
                )
        ),

        (
            "Precision_Recall_Curve",
            lambda ax:
                plot_pr(
                    ax,
                    data["pr"],
                    data[
                        "test_metrics"
                    ]
                )
        ),

        (
            "Calibration_Curve",
            lambda ax:
                plot_calibration(
                    ax,
                    data[
                        "calibration"
                    ]
                )
        ),

        (
            "Confusion_Matrix",
            lambda ax:
                plot_confusion_matrix(
                    ax,
                    data[
                        "confusion"
                    ]
                )
        ),

        (
            "Top20_Feature_Importance",
            lambda ax:
                plot_top20_importance(
                    ax,
                    data[
                        "top20_importance"
                    ]
                )
        ),

        (
            "Feature_Family_Contribution",
            lambda ax:
                plot_family_contribution(
                    ax,
                    data[
                        "family_importance"
                    ]
                )
        ),

        (
            "Model_CV_ROC_AUC",
            lambda ax:
                plot_cv_auc(
                    ax,
                    data[
                        "model_comparison"
                    ]
                )
        ),

        (
            "Model_CV_PR_AUC",
            lambda ax:
                plot_cv_pr_auc(
                    ax,
                    data[
                        "model_comparison"
                    ]
                )
        ),
    ]

    for name, func in items:

        fig, ax = plt.subplots(
            figsize=(
                SINGLE_WIDTH,
                SINGLE_HEIGHT
            )
        )

        func(
            ax
        )

        fig.tight_layout()

        outputs += save_figure(
            fig,
            STANDALONE_DIR,
            name
        )

    return outputs


# =============================================================================
# COPY SOURCE DATA USED FOR FIGURES
# =============================================================================

def copy_source_data(files):

    copied = []

    for key, path in files.items():

        path = Path(
            path
        )

        if not path.exists():

            continue

        destination = (
            SOURCE_DATA_DIR /
            path.name
        )

        shutil.copy2(
            path,
            destination
        )

        copied.append(
            str(
                destination
            )
        )

    return copied


# =============================================================================
# MANIFEST
# =============================================================================

def build_manifest(
    generated_files
):

    rows = []

    for file in generated_files:

        path = Path(
            file
        )

        if not path.exists():

            continue

        parent = (
            path.parent.name
        )

        rows.append(
            {
                "category":
                    parent,

                "file":
                    path.name,

                "format":
                    path.suffix
                    .replace(
                        ".",
                        ""
                    )
                    .upper(),

                "relative_path":
                    str(
                        path
                    ),

                "size_bytes":
                    path.stat()
                    .st_size,
            }
        )

    manifest = pd.DataFrame(
        rows
    )

    manifest_file = (
        OUTPUT_DIR /
        "publication_figure_manifest.tsv"
    )

    manifest.to_csv(
        manifest_file,
        sep="\t",
        index=False
    )

    return manifest_file


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    generated_files,
    source_files
):

    report = (
        OUTPUT_DIR /
        "publication_figure_generation_report.txt"
    )

    lines = []

    lines.append(
        "=" * 90
    )

    lines.append(
        "SCRIPT 14 — PUBLICATION FIGURE GENERATION REPORT"
    )

    lines.append(
        "=" * 90
    )

    lines.append("")

    lines.append(
        f"Generated: {datetime.now().isoformat()}"
    )

    lines.append("")

    lines.append(
        "MODEL CONTEXT"
    )

    lines.append(
        "-" * 90
    )

    lines.append(
        "Selected model: Extra Trees"
    )

    lines.append(
        "Calibration: sigmoid / Platt"
    )

    lines.append(
        "Final predictors: 40"
    )

    lines.append(
        "Molecular predictors: 10"
    )

    lines.append(
        "3D-QSAR predictors: 9"
    )

    lines.append(
        "MoLFormer predictors: 21"
    )

    lines.append(
        f"Locked threshold: {LOCKED_THRESHOLD}"
    )

    lines.append(
        f"Independent-test prevalence: {TEST_PREVALENCE:.8f}"
    )

    lines.append("")

    lines.append(
        "OUTPUT FORMAT"
    )

    lines.append(
        "-" * 90
    )

    lines.append(
        f"PNG resolution: {PNG_DPI} dpi"
    )

    lines.append(
        "Vector outputs: PDF and SVG"
    )

    lines.append("")

    lines.append(
        "GENERATED FIGURE FILES"
    )

    lines.append(
        "-" * 90
    )

    for file in generated_files:

        lines.append(
            str(
                file
            )
        )

    lines.append("")

    lines.append(
        "COPIED SOURCE DATA"
    )

    lines.append(
        "-" * 90
    )

    for file in source_files:

        lines.append(
            str(
                file
            )
        )

    lines.append("")

    lines.append(
        "IMPORTANT"
    )

    lines.append(
        "-" * 90
    )

    lines.append(
        "No model was trained or modified by Script 14."
    )

    lines.append(
        "No feature selection was performed by Script 14."
    )

    lines.append(
        "All plots were generated from existing finalized model-development outputs."
    )

    lines.append(
        "Test-set information was used only for visualization of the previously "
        "completed independent evaluation."
    )

    lines.append("")

    lines.append(
        "=" * 90
    )

    lines.append(
        "END OF REPORT"
    )

    lines.append(
        "=" * 90
    )

    with open(
        report,
        "w"
    ) as handle:

        handle.write(
            "\n".join(
                lines
            )
        )

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "SCRIPT 14 — PUBLICATION FIGURE GENERATION"
    )

    print(
        "Started:",
        datetime.now().isoformat()
    )

    print(
        "Base directory:",
        BASE_DIR
    )

    ensure_dirs()

    data, files = load_all_data()

    generated = []

    # =========================================================================
    # MAIN FIGURES
    # =========================================================================

    print_section(
        "GENERATING MAIN FIGURES"
    )

    print(
        "Main Figure 1..."
    )

    generated += make_main_figure_1(
        data
    )

    print(
        "Main Figure 2..."
    )

    generated += make_main_figure_2(
        data
    )

    print(
        "Main Figure 3..."
    )

    generated += make_main_figure_3(
        data
    )

    print(
        "Main Figure 4..."
    )

    generated += make_main_figure_4(
        data
    )

    print(
        "Main Figure 5..."
    )

    generated += make_main_figure_5(
        data
    )

    # =========================================================================
    # SUPPLEMENTARY
    # =========================================================================

    print_section(
        "GENERATING SUPPLEMENTARY FIGURES"
    )

    generated += make_supplementary_figures(
        data
    )

    # =========================================================================
    # STANDALONE
    # =========================================================================

    print_section(
        "GENERATING STANDALONE CORE FIGURES"
    )

    generated += make_standalone_figures(
        data
    )

    # =========================================================================
    # SOURCE DATA
    # =========================================================================

    print_section(
        "COPYING FIGURE SOURCE DATA"
    )

    copied_source = copy_source_data(
        files
    )

    # =========================================================================
    # MANIFEST
    # =========================================================================

    manifest = build_manifest(
        generated
    )

    report = write_report(
        generated,
        copied_source
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_header(
        "SCRIPT 14 COMPLETE"
    )

    print(
        "Main figures:"
    )

    print(
        " ",
        MAIN_DIR
    )

    print()
    print(
        "Supplementary figures:"
    )

    print(
        " ",
        SUPP_DIR
    )

    print()
    print(
        "Standalone figures:"
    )

    print(
        " ",
        STANDALONE_DIR
    )

    print()
    print(
        "Source data:"
    )

    print(
        " ",
        SOURCE_DATA_DIR
    )

    print()
    print(
        "Manifest:"
    )

    print(
        " ",
        manifest
    )

    print()
    print(
        "Report:"
    )

    print(
        " ",
        report
    )

    print()

    print(
        f"Total exported figure files: "
        f"{len(generated):,}"
    )

    print(
        "FINAL STATUS: PASS"
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":

    main()
