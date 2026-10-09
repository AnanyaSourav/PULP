#!/usr/bin/env python3

"""
===============================================================================
SCRIPT 09 v1
DETERMINISTIC PREDICTION MODEL DEVELOPMENT
===============================================================================

Purpose
-------
Develop, compare, calibrate, validate, interpret, and serialize the final
phytochemical bioactivity prediction model using the statistically supported
feature representation established in Scripts 01-08.

INPUTS
------
1. Master TRAIN/TEST splits
2. Script 05 final handcrafted features
3. Script 08 TRAIN-derived 40-feature shortlist
4. MoLFormer TRAIN/TEST embeddings

DESIGN PRINCIPLES
-----------------
* TRAIN/TEST split remains untouched.
* TEST is used exactly once for final independent evaluation.
* No SMOTE.
* No random oversampling.
* Class imbalance handled by deterministic class weighting.
* Feature shortlist comes from Script 08.
* Model comparison is performed using stratified 5-fold CV.
* Probability calibration is performed using training data only.
* Operating threshold is determined from training OOF predictions.
* Final threshold is locked before TEST evaluation.
* All random processes use explicit seeds.
* All publication tables are machine-readable TSV files.
* All prediction-level results are saved.
* Final calibrated model is serialized for deployment.

OUTPUT ROOT
-----------
feature_engineering/model_development/

IMPORTANT
---------
This script does NOT perform feature generation from raw SMILES.
It develops the predictive model from the representations already generated
by Scripts 01-08.

The eventual deployment pipeline will separately reproduce:
SMILES -> molecular descriptors + 3D-QSAR + MoLFormer -> selected 40 features
-> calibrated model -> prediction/confidence/explanation.
===============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import math
import hashlib
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
    auc,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SEED = 42
N_SPLITS = 5

BASE_DIR = Path("feature_engineering")

TRAIN_MASTER = BASE_DIR / "split" / "train_80.tsv"
TEST_MASTER = BASE_DIR / "split" / "test_20.tsv"

HANDCRAFTED_TRAIN = (
    BASE_DIR / "final_228_features" / "train_80_final_228_features.tsv"
)
HANDCRAFTED_TEST = (
    BASE_DIR / "final_228_features" / "test_20_final_228_features.tsv"
)

MOLFORMER_TRAIN = (
    BASE_DIR / "molformer_embeddings" / "train_80_molformer_embeddings.npy"
)
MOLFORMER_TEST = (
    BASE_DIR / "molformer_embeddings" / "test_20_molformer_embeddings.npy"
)

SHORTLIST = (
    BASE_DIR / "publication_statistics" / "shortlisted_features.tsv"
)

OUTDIR = BASE_DIR / "model_development"

TABLE_DIR = OUTDIR / "tables"
FIGURE_DIR = OUTDIR / "figure_data"
PRED_DIR = OUTDIR / "predictions"
MODEL_DIR = OUTDIR / "models"
LOG_DIR = OUTDIR / "logs"

for d in [
    OUTDIR,
    TABLE_DIR,
    FIGURE_DIR,
    PRED_DIR,
    MODEL_DIR,
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def log(msg):
    print(msg, flush=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def normalized_entropy(p):
    """
    Shannon binary entropy normalized to [0,1].
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-15, 1 - 1e-15)

    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return h / np.log(2.0)


def expected_calibration_error(y, p, n_bins=10):
    """
    Standard equal-width ECE.
    """
    y = np.asarray(y)
    p = np.asarray(p)

    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []

    n = len(y)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin": i + 1,
                "bin_lower": edges[i],
                "bin_upper": edges[i + 1],
                "n": 0,
                "mean_predicted": np.nan,
                "observed_fraction": np.nan,
                "absolute_error": np.nan,
            })
            continue

        mean_pred = float(np.mean(p[mask]))
        observed = float(np.mean(y[mask]))
        error = abs(mean_pred - observed)

        ece += (count / n) * error

        rows.append({
            "bin": i + 1,
            "bin_lower": edges[i],
            "bin_upper": edges[i + 1],
            "n": count,
            "mean_predicted": mean_pred,
            "observed_fraction": observed,
            "absolute_error": error,
        })

    return float(ece), pd.DataFrame(rows)


def maximum_calibration_error(calibration_df):
    vals = calibration_df["absolute_error"].dropna()

    if len(vals) == 0:
        return np.nan

    return float(vals.max())


def ks_statistic(y, p):
    """
    KS statistic between predicted-probability distributions for
    active and inactive observations.
    """
    y = np.asarray(y)
    p = np.asarray(p)

    active = np.sort(p[y == 1])
    inactive = np.sort(p[y == 0])

    if len(active) == 0 or len(inactive) == 0:
        return np.nan, np.nan

    values = np.sort(np.unique(np.concatenate([active, inactive])))

    # empirical CDF
    active_cdf = np.searchsorted(active, values, side="right") / len(active)
    inactive_cdf = np.searchsorted(
        inactive, values, side="right"
    ) / len(inactive)

    differences = np.abs(active_cdf - inactive_cdf)
    idx = np.argmax(differences)

    return float(differences[idx]), float(values[idx])


def classification_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision,
        "negative_predictive_value": npv,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def threshold_analysis(y, p):
    """
    Evaluate all ROC thresholds and identify:
      * Youden J optimum
      * F1 optimum
      * MCC optimum
      * threshold closest to requested sensitivity levels
    """

    fpr, tpr, thresholds = roc_curve(y, p)

    rows = []

    for fpr_i, tpr_i, threshold in zip(
        fpr, tpr, thresholds
    ):
        specificity = 1.0 - fpr_i
        youden_j = tpr_i + specificity - 1.0

        pred = (p >= threshold).astype(int)

        precision = precision_score(
            y,
            pred,
            zero_division=0,
        )

        f1 = f1_score(
            y,
            pred,
            zero_division=0,
        )

        mcc = matthews_corrcoef(y, pred)

        rows.append({
            "threshold": float(threshold),
            "sensitivity": float(tpr_i),
            "specificity": float(specificity),
            "fpr": float(fpr_i),
            "precision": float(precision),
            "f1": float(f1),
            "mcc": float(mcc),
            "youden_j": float(youden_j),
        })

    df = pd.DataFrame(rows)

    best_youden = df.loc[df["youden_j"].idxmax()]
    best_f1 = df.loc[df["f1"].idxmax()]
    best_mcc = df.loc[df["mcc"].idxmax()]

    summary = pd.DataFrame([
        {
            "criterion": "Youden_J",
            **best_youden.to_dict(),
        },
        {
            "criterion": "F1",
            **best_f1.to_dict(),
        },
        {
            "criterion": "MCC",
            **best_mcc.to_dict(),
        },
    ])

    return df, summary


def log_loss_per_sample(y, p):
    p = np.clip(np.asarray(p), 1e-15, 1 - 1e-15)
    y = np.asarray(y)

    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def make_model_definitions():

    models = {}

    # -------------------------------------------------------------------------
    # Logistic regression
    # -------------------------------------------------------------------------
    models["logistic_regression"] = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=SEED,
            )
        ),
    ])

    # -------------------------------------------------------------------------
    # Random forest
    # -------------------------------------------------------------------------
    models["random_forest"] = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    # -------------------------------------------------------------------------
    # Extra Trees
    # -------------------------------------------------------------------------
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    return models


def fit_calibrated_model(model, X, y):
    """
    Sigmoid/Platt calibration using training data only.

    cv=5 means the calibrator obtains cross-validated predictions internally
    rather than fitting calibration directly on the same observations used to
    fit each base estimator.
    """

    calibrated = CalibratedClassifierCV(
        estimator=clone(model),
        method="sigmoid",
        cv=5,
        n_jobs=-1,
    )

    calibrated.fit(X, y)

    return calibrated


# =============================================================================
# INPUT VALIDATION
# =============================================================================

log("=" * 80)
log("SCRIPT 09 v1 — PREDICTION MODEL DEVELOPMENT")
log("=" * 80)
log(f"Started: {timestamp()}")
log(f"Random seed: {SEED}")
log("")

for path in [
    TRAIN_MASTER,
    TEST_MASTER,
    HANDCRAFTED_TRAIN,
    HANDCRAFTED_TEST,
    MOLFORMER_TRAIN,
    MOLFORMER_TEST,
    SHORTLIST,
]:
    require_file(path)
    log(f"FOUND: {path}")

# =============================================================================
# LOAD MASTER LABELS
# =============================================================================

log("")
log("=" * 80)
log("1. LOADING MASTER DATASETS")
log("=" * 80)

train_master = pd.read_csv(
    TRAIN_MASTER,
    sep="\t",
)

test_master = pd.read_csv(
    TEST_MASTER,
    sep="\t",
)

log(f"TRAIN shape: {train_master.shape}")
log(f"TEST shape : {test_master.shape}")

# -------------------------------------------------------------------------
# identify ID
# -------------------------------------------------------------------------

ID_CANDIDATES = [
    "np_id",
    "NP_ID",
    "id",
    "ID",
]

id_col = None

for c in ID_CANDIDATES:
    if c in train_master.columns:
        id_col = c
        break

if id_col is None:
    raise RuntimeError(
        "Could not identify compound ID column."
    )

if "Activity_Label" not in train_master.columns:
    raise RuntimeError(
        "Activity_Label not found in TRAIN."
    )

if "Activity_Label" not in test_master.columns:
    raise RuntimeError(
        "Activity_Label not found in TEST."
    )

log(f"ID column: {id_col}")

# convert label
label_map = {
    "Active": 1,
    "Inactive": 0,
    1: 1,
    0: 0,
}

train_master["target"] = (
    train_master["Activity_Label"]
    .map(label_map)
)

test_master["target"] = (
    test_master["Activity_Label"]
    .map(label_map)
)

if train_master["target"].isna().any():
    raise RuntimeError("Unexpected TRAIN activity labels.")

if test_master["target"].isna().any():
    raise RuntimeError("Unexpected TEST activity labels.")

y_train = train_master["target"].astype(int).to_numpy()
y_test = test_master["target"].astype(int).to_numpy()

log("")
log("TRAIN class distribution:")
log(str(pd.Series(y_train).value_counts().sort_index()))

log("")
log("TEST class distribution:")
log(str(pd.Series(y_test).value_counts().sort_index()))

# =============================================================================
# LOAD HANDCRAFTED FEATURES
# =============================================================================

log("")
log("=" * 80)
log("2. LOADING FINAL HANDCRAFTED FEATURES")
log("=" * 80)

hc_train = pd.read_csv(
    HANDCRAFTED_TRAIN,
    sep="\t",
)

hc_test = pd.read_csv(
    HANDCRAFTED_TEST,
    sep="\t",
)

log(f"Handcrafted TRAIN: {hc_train.shape}")
log(f"Handcrafted TEST : {hc_test.shape}")

if id_col not in hc_train.columns:
    raise RuntimeError(
        f"{id_col} missing from handcrafted TRAIN."
    )

if id_col not in hc_test.columns:
    raise RuntimeError(
        f"{id_col} missing from handcrafted TEST."
    )

# =============================================================================
# LOAD MOLFORMER
# =============================================================================

log("")
log("=" * 80)
log("3. LOADING MOLFORMER EMBEDDINGS")
log("=" * 80)

emb_train = np.load(
    MOLFORMER_TRAIN,
    allow_pickle=False,
)

emb_test = np.load(
    MOLFORMER_TEST,
    allow_pickle=False,
)

log(f"MoLFormer TRAIN shape: {emb_train.shape}")
log(f"MoLFormer TEST shape : {emb_test.shape}")

if emb_train.ndim != 2:
    raise RuntimeError(
        "MoLFormer TRAIN embedding array is not 2D."
    )

if emb_test.ndim != 2:
    raise RuntimeError(
        "MoLFormer TEST embedding array is not 2D."
    )

if emb_train.shape[1] != 768:
    raise RuntimeError(
        f"Expected 768 MoLFormer dimensions, got {emb_train.shape[1]}"
    )

if emb_test.shape[1] != 768:
    raise RuntimeError(
        f"Expected 768 MoLFormer dimensions, got {emb_test.shape[1]}"
    )

if len(emb_train) != len(train_master):
    raise RuntimeError(
        "MoLFormer TRAIN row count does not match master TRAIN."
    )

if len(emb_test) != len(test_master):
    raise RuntimeError(
        "MoLFormer TEST row count does not match master TEST."
    )

molformer_names = [
    f"molformer_{i:04d}"
    for i in range(768)
]

emb_train_df = pd.DataFrame(
    emb_train,
    columns=molformer_names,
)

emb_test_df = pd.DataFrame(
    emb_test,
    columns=molformer_names,
)

# =============================================================================
# LOAD SCRIPT 08 SHORTLIST
# =============================================================================

log("")
log("=" * 80)
log("4. LOADING SCRIPT 08 FEATURE SHORTLIST")
log("=" * 80)

shortlist = pd.read_csv(
    SHORTLIST,
    sep="\t",
)

log(
    f"Shortlist columns: "
    f"{list(shortlist.columns)}"
)

if "feature" not in shortlist.columns:
    raise RuntimeError(
        "Script 08 shortlist does not contain 'feature'."
    )

if "representation" not in shortlist.columns:
    raise RuntimeError(
        "Script 08 shortlist does not contain 'representation'."
    )

shortlist = shortlist.drop_duplicates(
    subset=["representation", "feature"]
).copy()

feature_records = shortlist[
    ["representation", "feature"]
].to_dict("records")

if len(feature_records) != 40:
    raise RuntimeError(
        f"Expected exactly 40 Script 08 shortlisted features; "
        f"found {len(feature_records)}."
    )

# save exact shortlist
shortlist.to_csv(
    TABLE_DIR / "table_01_script08_feature_shortlist.tsv",
    sep="\t",
    index=False,
)

log(f"Script 08 shortlist size: {len(feature_records)}")

log("")
log("Feature family counts:")

family_counts = (
    shortlist["representation"]
    .value_counts()
    .rename_axis("representation")
    .reset_index(name="n_features")
)

log(str(family_counts))

family_counts.to_csv(
    TABLE_DIR / "table_02_feature_family_counts.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# BUILD COMBINED REPRESENTATION
# =============================================================================

log("")
log("=" * 80)
log("5. BUILDING 40-FEATURE MODEL MATRIX")
log("=" * 80)

molecular_features = shortlist.loc[
    shortlist["representation"].str.lower() == "molecular",
    "feature"
].tolist()

three_d_features = shortlist.loc[
    shortlist["representation"].str.lower().isin(
        ["3d", "3d_qsar", "3d-qsar"]
    ),
    "feature"
].tolist()

molformer_features = shortlist.loc[
    shortlist["representation"].str.lower() == "molformer",
    "feature"
].tolist()

log(f"Molecular features : {len(molecular_features)}")
log(f"3D-QSAR features   : {len(three_d_features)}")
log(f"MoLFormer features : {len(molformer_features)}")

# Validate expected structure
if len(molecular_features) != 10:
    raise RuntimeError(
        f"Expected 10 molecular shortlisted features, "
        f"found {len(molecular_features)}."
    )

if len(three_d_features) != 9:
    raise RuntimeError(
        f"Expected 9 3D shortlisted features, "
        f"found {len(three_d_features)}."
    )

if len(molformer_features) != 21:
    raise RuntimeError(
        f"Expected 21 MoLFormer shortlisted features, "
        f"found {len(molformer_features)}."
    )

hand_features = (
    molecular_features +
    three_d_features
)

missing_hand_train = [
    f for f in hand_features
    if f not in hc_train.columns
]

missing_hand_test = [
    f for f in hand_features
    if f not in hc_test.columns
]

if missing_hand_train:
    raise RuntimeError(
        f"Missing handcrafted TRAIN features: {missing_hand_train}"
    )

if missing_hand_test:
    raise RuntimeError(
        f"Missing handcrafted TEST features: {missing_hand_test}"
    )

missing_emb = [
    f for f in molformer_features
    if f not in emb_train_df.columns
]

if missing_emb:
    raise RuntimeError(
        f"Missing MoLFormer features: {missing_emb}"
    )

Xhc_train = hc_train[
    hand_features
].copy()

Xhc_test = hc_test[
    hand_features
].copy()

Xemb_train = emb_train_df[
    molformer_features
].copy()

Xemb_test = emb_test_df[
    molformer_features
].copy()

X_train_df = pd.concat(
    [
        Xhc_train.reset_index(drop=True),
        Xemb_train.reset_index(drop=True),
    ],
    axis=1,
)

X_test_df = pd.concat(
    [
        Xhc_test.reset_index(drop=True),
        Xemb_test.reset_index(drop=True),
    ],
    axis=1,
)

feature_order = (
    hand_features +
    molformer_features
)

X_train_df = X_train_df[
    feature_order
]

X_test_df = X_test_df[
    feature_order
]

if X_train_df.shape[1] != 40:
    raise RuntimeError(
        f"Expected 40 predictors; got {X_train_df.shape[1]}"
    )

# numerical QC
for name, df in [
    ("TRAIN", X_train_df),
    ("TEST", X_test_df),
]:

    df[:] = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df.isna().any().any():
        bad = df.columns[
            df.isna().any()
        ].tolist()

        raise RuntimeError(
            f"{name} contains NaN after feature assembly: {bad}"
        )

    if np.isinf(df.to_numpy()).any():
        raise RuntimeError(
            f"{name} contains infinite values."
        )

log(f"Final TRAIN matrix: {X_train_df.shape}")
log(f"Final TEST matrix : {X_test_df.shape}")

feature_manifest = pd.DataFrame({
    "feature_index": np.arange(1, 41),
    "feature": feature_order,
})

feature_manifest["representation"] = (
    feature_manifest["feature"]
    .map(
        {
            f: "molecular"
            for f in molecular_features
        }
    )
    .fillna(
        feature_manifest["feature"].map(
            {
                f: "3d"
                for f in three_d_features
            }
        )
    )
    .fillna("molformer")
)

feature_manifest.to_csv(
    TABLE_DIR / "table_03_final_40_feature_manifest.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# SAVE MODEL MATRICES
# =============================================================================

X_train_df.to_csv(
    OUTDIR / "model_train_40_features.tsv",
    sep="\t",
    index=False,
)

X_test_df.to_csv(
    OUTDIR / "model_test_40_features.tsv",
    sep="\t",
    index=False,
)

X_train = X_train_df.to_numpy(dtype=float)
X_test = X_test_df.to_numpy(dtype=float)

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

log("")
log("=" * 80)
log("6. MODEL DEFINITIONS")
log("=" * 80)

models = make_model_definitions()

for name in models:
    log(f"Configured: {name}")

# =============================================================================
# STRATIFIED CV MODEL COMPARISON
# =============================================================================

log("")
log("=" * 80)
log("7. 5-FOLD STRATIFIED MODEL DEVELOPMENT")
log("=" * 80)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

fold_rows = []
oof_rows = []

for model_name, model in models.items():

    log("")
    log("-" * 80)
    log(f"MODEL: {model_name}")
    log("-" * 80)

    oof_raw = np.full(
        len(y_train),
        np.nan,
        dtype=float,
    )

    oof_cal = np.full(
        len(y_train),
        np.nan,
        dtype=float,
    )

    for fold_id, (tr_idx, va_idx) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):

        log(
            f"{model_name} | fold {fold_id}/{N_SPLITS}"
        )

        Xtr = X_train[tr_idx]
        Xva = X_train[va_idx]

        ytr = y_train[tr_idx]
        yva = y_train[va_idx]

        # -------------------------------------------------------------
        # RAW MODEL
        # -------------------------------------------------------------

        fitted_raw = clone(model)

        fitted_raw.fit(
            Xtr,
            ytr,
        )

        raw_prob = fitted_raw.predict_proba(
            Xva
        )[:, 1]

        oof_raw[va_idx] = raw_prob

        # -------------------------------------------------------------
        # CALIBRATED MODEL
        # -------------------------------------------------------------

        calibrated = CalibratedClassifierCV(
            estimator=clone(model),
            method="sigmoid",
            cv=3,
            n_jobs=-1,
        )

        calibrated.fit(
            Xtr,
            ytr,
        )

        cal_prob = calibrated.predict_proba(
            Xva
        )[:, 1]

        oof_cal[va_idx] = cal_prob

        # fold metrics
        fold_auc = roc_auc_score(
            yva,
            cal_prob,
        )

        fold_ap = average_precision_score(
            yva,
            cal_prob,
        )

        fold_brier = brier_score_loss(
            yva,
            cal_prob,
        )

        fold_logloss = log_loss(
            yva,
            cal_prob,
            labels=[0, 1],
        )

        fold_rows.append({
            "model": model_name,
            "fold": fold_id,
            "n_train": len(tr_idx),
            "n_validation": len(va_idx),
            "active_train": int(ytr.sum()),
            "inactive_train": int((ytr == 0).sum()),
            "active_validation": int(yva.sum()),
            "inactive_validation": int((yva == 0).sum()),
            "roc_auc": fold_auc,
            "pr_auc_average_precision": fold_ap,
            "brier": fold_brier,
            "log_loss": fold_logloss,
        })

    # ---------------------------------------------------------------------
    # model-level OOF metrics
    # ---------------------------------------------------------------------

    raw_auc = roc_auc_score(
        y_train,
        oof_raw,
    )

    raw_ap = average_precision_score(
        y_train,
        oof_raw,
    )

    cal_auc = roc_auc_score(
        y_train,
        oof_cal,
    )

    cal_ap = average_precision_score(
        y_train,
        oof_cal,
    )

    raw_brier = brier_score_loss(
        y_train,
        oof_raw,
    )

    cal_brier = brier_score_loss(
        y_train,
        oof_cal,
    )

    raw_ll = log_loss(
        y_train,
        oof_raw,
        labels=[0, 1],
    )

    cal_ll = log_loss(
        y_train,
        oof_cal,
        labels=[0, 1],
    )

    ece_raw, _ = expected_calibration_error(
        y_train,
        oof_raw,
    )

    ece_cal, _ = expected_calibration_error(
        y_train,
        oof_cal,
    )

    mce_raw = maximum_calibration_error(
        expected_calibration_error(
            y_train,
            oof_raw
        )[1]
    )

    mce_cal = maximum_calibration_error(
        expected_calibration_error(
            y_train,
            oof_cal
        )[1]
    )

    ks, ks_threshold = ks_statistic(
        y_train,
        oof_cal,
    )

    # threshold selection
    threshold_df, threshold_summary = (
        threshold_analysis(
            y_train,
            oof_cal,
        )
    )

    youden_threshold = float(
        threshold_summary.loc[
            threshold_summary["criterion"] == "Youden_J",
            "threshold",
        ].iloc[0]
    )

    youden_metrics = classification_metrics(
        y_train,
        oof_cal,
        youden_threshold,
    )

    # store OOF predictions
    oof_df = pd.DataFrame({
        "row_index": np.arange(len(y_train)),
        "true_label": y_train,
        "raw_probability": oof_raw,
        "calibrated_probability": oof_cal,
    })

    oof_df["uncertainty"] = normalized_entropy(
        oof_df["calibrated_probability"]
    )

    oof_df["certainty"] = (
        1.0 -
        oof_df["uncertainty"]
    )

    oof_df["brier_per_sample"] = (
        oof_df["calibrated_probability"] -
        oof_df["true_label"]
    ) ** 2

    oof_df["log_loss_per_sample"] = log_loss_per_sample(
        oof_df["true_label"],
        oof_df["calibrated_probability"],
    )

    oof_df.to_csv(
        PRED_DIR /
        f"{model_name}_train_oof_predictions.tsv",
        sep="\t",
        index=False,
    )

    threshold_df.to_csv(
        TABLE_DIR /
        f"{model_name}_oof_threshold_analysis.tsv",
        sep="\t",
        index=False,
    )

    fold_df = pd.DataFrame(
        [
            r for r in fold_rows
            if r["model"] == model_name
        ]
    )

    fold_df.to_csv(
        TABLE_DIR /
        f"{model_name}_cv_fold_metrics.tsv",
        sep="\t",
        index=False,
    )

    fold_mean = fold_df.mean(
        numeric_only=True
    )

    fold_sd = fold_df.std(
        numeric_only=True,
        ddof=1,
    )

    summary = {
        "model": model_name,

        "cv_auc_mean": fold_mean["roc_auc"],
        "cv_auc_sd": fold_sd["roc_auc"],

        "cv_pr_auc_mean": fold_mean[
            "pr_auc_average_precision"
        ],
        "cv_pr_auc_sd": fold_sd[
            "pr_auc_average_precision"
        ],

        "cv_brier_mean": fold_mean["brier"],
        "cv_brier_sd": fold_sd["brier"],

        "cv_log_loss_mean": fold_mean["log_loss"],
        "cv_log_loss_sd": fold_sd["log_loss"],

        "oof_raw_auc": raw_auc,
        "oof_raw_pr_auc": raw_ap,
        "oof_raw_brier": raw_brier,
        "oof_raw_log_loss": raw_ll,
        "oof_raw_ece": ece_raw,
        "oof_raw_mce": mce_raw,

        "oof_calibrated_auc": cal_auc,
        "oof_calibrated_pr_auc": cal_ap,
        "oof_calibrated_brier": cal_brier,
        "oof_calibrated_log_loss": cal_ll,
        "oof_calibrated_ece": ece_cal,
        "oof_calibrated_mce": mce_cal,

        "brier_improvement_percent":
            100.0 * (
                raw_brier - cal_brier
            ) / raw_brier
            if raw_brier > 0 else np.nan,

        "log_loss_improvement_percent":
            100.0 * (
                raw_ll - cal_ll
            ) / raw_ll
            if raw_ll > 0 else np.nan,

        "ks_statistic": ks,
        "ks_threshold": ks_threshold,

        "youden_threshold": youden_threshold,
        "youden_j": youden_metrics[
            "sensitivity_recall"
        ] + youden_metrics[
            "specificity"
        ] - 1.0,

        "youden_sensitivity":
            youden_metrics[
                "sensitivity_recall"
            ],

        "youden_specificity":
            youden_metrics[
                "specificity"
            ],

        "youden_precision":
            youden_metrics[
                "precision_ppv"
            ],

        "youden_f1":
            youden_metrics["f1"],

        "youden_mcc":
            youden_metrics["mcc"],

        "youden_balanced_accuracy":
            youden_metrics[
                "balanced_accuracy"
            ],
    }

    oof_rows.append(summary)

# =============================================================================
# CV SUMMARY
# =============================================================================

cv_summary = pd.DataFrame(oof_rows)

cv_summary.to_csv(
    TABLE_DIR / "table_04_model_comparison_cv.tsv",
    sep="\t",
    index=False,
)

log("")
log("MODEL COMPARISON")
log(str(cv_summary))

# =============================================================================
# MODEL SELECTION
# =============================================================================

log("")
log("=" * 80)
log("8. MODEL SELECTION")
log("=" * 80)

"""
Selection hierarchy
-------------------
1. Calibrated PR-AUC
2. Calibrated ROC-AUC
3. Calibrated Brier
4. Calibrated Log Loss
5. MCC at Youden threshold

The independent TEST set is NOT used.
"""

selection_df = cv_summary.copy()

selection_df = selection_df.sort_values(
    by=[
        "oof_calibrated_pr_auc",
        "oof_calibrated_auc",
        "oof_calibrated_brier",
        "oof_calibrated_log_loss",
        "youden_mcc",
    ],
    ascending=[
        False,
        False,
        True,
        True,
        False,
    ],
).reset_index(drop=True)

selection_df["selection_rank"] = (
    np.arange(len(selection_df)) + 1
)

selection_df.to_csv(
    TABLE_DIR /
    "table_05_model_selection_ranking.tsv",
    sep="\t",
    index=False,
)

selected_model_name = selection_df.iloc[0]["model"]

log(
    f"SELECTED MODEL: {selected_model_name}"
)

selected_base_model = models[
    selected_model_name
]

selected_threshold = float(
    selection_df.iloc[0]["youden_threshold"]
)

log(
    f"LOCKED Youden threshold: "
    f"{selected_threshold:.8f}"
)

# =============================================================================
# FINAL TRAINING
# =============================================================================

log("")
log("=" * 80)
log("9. FINAL MODEL FIT")
log("=" * 80)

final_raw_model = clone(
    selected_base_model
)

final_raw_model.fit(
    X_train,
    y_train,
)

log("Final raw model fitted.")

final_calibrated_model = CalibratedClassifierCV(
    estimator=clone(selected_base_model),
    method="sigmoid",
    cv=5,
    n_jobs=-1,
)

final_calibrated_model.fit(
    X_train,
    y_train,
)

log("Final calibrated model fitted.")

# =============================================================================
# FINAL TEST PREDICTIONS
# =============================================================================

log("")
log("=" * 80)
log("10. INDEPENDENT TEST EVALUATION")
log("=" * 80)

test_raw_prob = final_raw_model.predict_proba(
    X_test
)[:, 1]

test_cal_prob = final_calibrated_model.predict_proba(
    X_test
)[:, 1]

# -------------------------------------------------------------------------
# TEST discrimination
# -------------------------------------------------------------------------

test_raw_auc = roc_auc_score(
    y_test,
    test_raw_prob,
)

test_cal_auc = roc_auc_score(
    y_test,
    test_cal_prob,
)

test_raw_ap = average_precision_score(
    y_test,
    test_raw_prob,
)

test_cal_ap = average_precision_score(
    y_test,
    test_cal_prob,
)

# -------------------------------------------------------------------------
# calibration
# -------------------------------------------------------------------------

test_raw_brier = brier_score_loss(
    y_test,
    test_raw_prob,
)

test_cal_brier = brier_score_loss(
    y_test,
    test_cal_prob,
)

test_raw_ll = log_loss(
    y_test,
    test_raw_prob,
    labels=[0, 1],
)

test_cal_ll = log_loss(
    y_test,
    test_cal_prob,
    labels=[0, 1],
)

test_raw_ece, raw_cal_df = (
    expected_calibration_error(
        y_test,
        test_raw_prob,
    )
)

test_cal_ece, cal_cal_df = (
    expected_calibration_error(
        y_test,
        test_cal_prob,
    )
)

test_raw_mce = maximum_calibration_error(
    raw_cal_df
)

test_cal_mce = maximum_calibration_error(
    cal_cal_df
)

test_ks, test_ks_threshold = ks_statistic(
    y_test,
    test_cal_prob,
)

# -------------------------------------------------------------------------
# classification at locked threshold
# -------------------------------------------------------------------------

test_metrics = classification_metrics(
    y_test,
    test_cal_prob,
    selected_threshold,
)

# -------------------------------------------------------------------------
# standard 0.5 threshold
# -------------------------------------------------------------------------

test_metrics_05 = classification_metrics(
    y_test,
    test_cal_prob,
    0.5,
)

# -------------------------------------------------------------------------
# threshold analysis — descriptive only
#
# IMPORTANT:
# We do NOT use this to change the locked threshold.
# This is only a supplementary diagnostic showing what happens across TEST.
# -------------------------------------------------------------------------

test_threshold_df, test_threshold_summary = (
    threshold_analysis(
        y_test,
        test_cal_prob,
    )
)

# =============================================================================
# TEST METRICS TABLE
# =============================================================================

test_metrics_rows = [

    {
        "metric": "ROC_AUC",
        "raw": test_raw_auc,
        "calibrated": test_cal_auc,
        "primary": True,
    },

    {
        "metric": "PR_AUC_average_precision",
        "raw": test_raw_ap,
        "calibrated": test_cal_ap,
        "primary": True,
    },

    {
        "metric": "Brier_score",
        "raw": test_raw_brier,
        "calibrated": test_cal_brier,
        "primary": True,
    },

    {
        "metric": "Log_loss",
        "raw": test_raw_ll,
        "calibrated": test_cal_ll,
        "primary": True,
    },

    {
        "metric": "ECE",
        "raw": test_raw_ece,
        "calibrated": test_cal_ece,
        "primary": True,
    },

    {
        "metric": "MCE",
        "raw": test_raw_mce,
        "calibrated": test_cal_mce,
        "primary": False,
    },

    {
        "metric": "KS_statistic",
        "raw": np.nan,
        "calibrated": test_ks,
        "primary": False,
    },

    {
        "metric": "Accuracy_locked_threshold",
        "raw": np.nan,
        "calibrated": test_metrics[
            "accuracy"
        ],
        "primary": False,
    },

    {
        "metric": "Balanced_accuracy_locked_threshold",
        "raw": np.nan,
        "calibrated": test_metrics[
            "balanced_accuracy"
        ],
        "primary": True,
    },

    {
        "metric": "Sensitivity_locked_threshold",
        "raw": np.nan,
        "calibrated": test_metrics[
            "sensitivity_recall"
        ],
        "primary": True,
    },

    {
        "metric": "Specificity_locked_threshold",
        "raw": np.nan,
        "calibrated": test_metrics[
            "specificity"
        ],
        "primary": True,
    },

    {
        "metric": "Precision_PPV_locked_threshold",
        "raw": np.nan,
        "calibrated": test_metrics[
            "precision_ppv"
        ],
        "primary": True,
    },

    {
        "metric": "NPV_locked_threshold",
        "raw": np.nan,
        "calibrated": test_metrics[
            "negative_predictive_value"
        ],
        "primary": False,
    },

    {
        "metric": "F1_locked_threshold",
        "raw": np.nan,
        "calibrated": test_metrics["f1"],
        "primary": True,
    },

    {
        "metric": "MCC_locked_threshold",
        "raw": np.nan,
        "calibrated": test_metrics["mcc"],
        "primary": True,
    },

    {
        "metric": "Accuracy_0.5",
        "raw": np.nan,
        "calibrated": test_metrics_05[
            "accuracy"
        ],
        "primary": False,
    },

    {
        "metric": "Balanced_accuracy_0.5",
        "raw": np.nan,
        "calibrated": test_metrics_05[
            "balanced_accuracy"
        ],
        "primary": False,
    },

    {
        "metric": "Sensitivity_0.5",
        "raw": np.nan,
        "calibrated": test_metrics_05[
            "sensitivity_recall"
        ],
        "primary": False,
    },

    {
        "metric": "Specificity_0.5",
        "raw": np.nan,
        "calibrated": test_metrics_05[
            "specificity"
        ],
        "primary": False,
    },

    {
        "metric": "Precision_0.5",
        "raw": np.nan,
        "calibrated": test_metrics_05[
            "precision_ppv"
        ],
        "primary": False,
    },

    {
        "metric": "F1_0.5",
        "raw": np.nan,
        "calibrated": test_metrics_05[
            "f1"
        ],
        "primary": False,
    },

    {
        "metric": "MCC_0.5",
        "raw": np.nan,
        "calibrated": test_metrics_05[
            "mcc"
        ],
        "primary": False,
    },

]

test_metrics_df = pd.DataFrame(
    test_metrics_rows
)

test_metrics_df.to_csv(
    TABLE_DIR /
    "table_06_independent_test_metrics.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

cm_locked = confusion_matrix(
    y_test,
    (
        test_cal_prob >= selected_threshold
    ).astype(int),
    labels=[0, 1],
)

cm_05 = confusion_matrix(
    y_test,
    (
        test_cal_prob >= 0.5
    ).astype(int),
    labels=[0, 1],
)

cm_df = pd.DataFrame(
    [
        {
            "threshold_policy": "locked_youden",
            "threshold": selected_threshold,
            "true_inactive_pred_inactive":
                cm_locked[0, 0],
            "true_inactive_pred_active":
                cm_locked[0, 1],
            "true_active_pred_inactive":
                cm_locked[1, 0],
            "true_active_pred_active":
                cm_locked[1, 1],
        },
        {
            "threshold_policy": "0.5",
            "threshold": 0.5,
            "true_inactive_pred_inactive":
                cm_05[0, 0],
            "true_inactive_pred_active":
                cm_05[0, 1],
            "true_active_pred_inactive":
                cm_05[1, 0],
            "true_active_pred_active":
                cm_05[1, 1],
        },
    ]
)

cm_df.to_csv(
    TABLE_DIR /
    "table_07_confusion_matrix.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# ROC CURVE DATA
# =============================================================================

fpr_raw, tpr_raw, thr_raw = roc_curve(
    y_test,
    test_raw_prob,
)

fpr_cal, tpr_cal, thr_cal = roc_curve(
    y_test,
    test_cal_prob,
)

roc_raw_df = pd.DataFrame({
    "fpr": fpr_raw,
    "tpr": tpr_raw,
    "threshold": thr_raw,
    "model_probability": "raw",
})

roc_cal_df = pd.DataFrame({
    "fpr": fpr_cal,
    "tpr": tpr_cal,
    "threshold": thr_cal,
    "model_probability": "calibrated",
})

roc_df = pd.concat(
    [
        roc_raw_df,
        roc_cal_df,
    ],
    ignore_index=True,
)

roc_df.to_csv(
    FIGURE_DIR /
    "figure_01_roc_curve_data.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# PR CURVE DATA
# =============================================================================

precision_raw, recall_raw, pr_thr_raw = (
    precision_recall_curve(
        y_test,
        test_raw_prob,
    )
)

precision_cal, recall_cal, pr_thr_cal = (
    precision_recall_curve(
        y_test,
        test_cal_prob,
    )
)

pr_raw_df = pd.DataFrame({
    "precision": precision_raw,
    "recall": recall_raw,
    "model_probability": "raw",
})

pr_cal_df = pd.DataFrame({
    "precision": precision_cal,
    "recall": recall_cal,
    "model_probability": "calibrated",
})

pr_df = pd.concat(
    [
        pr_raw_df,
        pr_cal_df,
    ],
    ignore_index=True,
)

pr_df.to_csv(
    FIGURE_DIR /
    "figure_02_pr_curve_data.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# CALIBRATION CURVE DATA
# =============================================================================

raw_cal_df["model_probability"] = "raw"
cal_cal_df["model_probability"] = "calibrated"

calibration_df = pd.concat(
    [
        raw_cal_df,
        cal_cal_df,
    ],
    ignore_index=True,
)

calibration_df.to_csv(
    FIGURE_DIR /
    "figure_03_calibration_curve_data.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# PREDICTION DISTRIBUTION
# =============================================================================

prediction_df = pd.DataFrame({
    id_col: test_master[id_col].values,
    "true_label": y_test,
    "true_activity": test_master[
        "Activity_Label"
    ].values,
    "raw_probability": test_raw_prob,
    "calibrated_probability": test_cal_prob,
})

prediction_df["active_probability"] = (
    prediction_df["calibrated_probability"]
)

prediction_df["inactive_probability"] = (
    1.0 -
    prediction_df["active_probability"]
)

prediction_df["uncertainty"] = normalized_entropy(
    prediction_df["calibrated_probability"]
)

prediction_df["certainty"] = (
    1.0 -
    prediction_df["uncertainty"]
)

prediction_df["probability_margin"] = (
    np.abs(
        prediction_df["active_probability"] -
        prediction_df["inactive_probability"]
    )
)

prediction_df["binary_prediction"] = np.where(
    prediction_df["calibrated_probability"]
    >= selected_threshold,
    "Active",
    "Inactive",
)

prediction_df["confidence_level"] = pd.cut(
    prediction_df["certainty"],
    bins=[
        -np.inf,
        0.40,
        0.70,
        np.inf,
    ],
    labels=[
        "Low",
        "Medium",
        "High",
    ],
    right=False,
)

prediction_df["borderline_0.4_0.6"] = (
    prediction_df["calibrated_probability"]
    .between(0.4, 0.6, inclusive="both")
)

prediction_df["brier_per_sample"] = (
    prediction_df["calibrated_probability"] -
    prediction_df["true_label"]
) ** 2

prediction_df["log_loss_per_sample"] = (
    log_loss_per_sample(
        prediction_df["true_label"],
        prediction_df["calibrated_probability"],
    )
)

# Rank active candidates by calibrated probability
prediction_df["prediction_rank"] = (
    prediction_df[
        "calibrated_probability"
    ]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

prediction_df.to_csv(
    PRED_DIR /
    "test_compound_predictions.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# PREDICTION DISTRIBUTION SUMMARY
# =============================================================================

dist_rows = []

for label, group in prediction_df.groupby(
    "true_activity"
):

    p = group["calibrated_probability"]

    dist_rows.append({
        "true_activity": label,
        "n": len(group),
        "mean_probability": p.mean(),
        "median_probability": p.median(),
        "std_probability": p.std(),
        "min_probability": p.min(),
        "max_probability": p.max(),
        "fraction_probability_lt_0.3":
            (p < 0.3).mean(),
        "fraction_probability_0.3_0.7":
            p.between(0.3, 0.7).mean(),
        "fraction_probability_gt_0.7":
            (p > 0.7).mean(),
        "mean_uncertainty":
            group["uncertainty"].mean(),
        "mean_certainty":
            group["certainty"].mean(),
    })

dist_summary = pd.DataFrame(
    dist_rows
)

dist_summary.to_csv(
    TABLE_DIR /
    "table_08_prediction_distribution_summary.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# BORDERLINE SUMMARY
# =============================================================================

borderline_summary = pd.DataFrame([
    {
        "definition": "probability_0.4_to_0.6",
        "n": int(
            prediction_df[
                "borderline_0.4_0.6"
            ].sum()
        ),
        "fraction":
            prediction_df[
                "borderline_0.4_0.6"
            ].mean(),
    }
])

borderline_summary.to_csv(
    TABLE_DIR /
    "table_09_borderline_prediction_summary.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# NNT
# =============================================================================

"""
For screening utility we report:
NNT_screen = 1 / precision

This is a screening interpretation rather than a clinical NNT.
"""

precision_locked = test_metrics[
    "precision_ppv"
]

nnt_screen = (
    1.0 / precision_locked
    if precision_locked > 0
    else np.inf
)

nnt_df = pd.DataFrame([
    {
        "metric": "Number_needed_to_test_screening",
        "value": nnt_screen,
        "definition":
            "1 / positive predictive value",
        "threshold":
            selected_threshold,
    }
])

nnt_df.to_csv(
    TABLE_DIR /
    "table_10_screening_nnt.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# FEATURE IMPORTANCE — PERMUTATION
# =============================================================================

log("")
log("=" * 80)
log("11. PERMUTATION FEATURE IMPORTANCE")
log("=" * 80)

"""
Permutation importance is evaluated on the independent TEST set only for
descriptive interpretation AFTER the model has been locked.

It does not modify the model.
It does not select features.
"""

perm = permutation_importance(
    final_calibrated_model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=30,
    random_state=SEED,
    n_jobs=-1,
)

perm_df = pd.DataFrame({
    "feature": feature_order,
    "representation":
        feature_manifest[
            "representation"
        ].values,
    "permutation_importance_mean":
        perm.importances_mean,
    "permutation_importance_sd":
        perm.importances_std,
})

perm_df["importance_rank"] = (
    perm_df[
        "permutation_importance_mean"
    ]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

perm_df = perm_df.sort_values(
    "importance_rank"
)

perm_df.to_csv(
    TABLE_DIR /
    "table_11_permutation_feature_importance.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# FEATURE FAMILY IMPORTANCE
# =============================================================================

family_importance = (
    perm_df
    .groupby("representation")
    .agg(
        n_features=("feature", "count"),
        mean_importance=(
            "permutation_importance_mean",
            "mean",
        ),
        sum_absolute_importance=(
            "permutation_importance_mean",
            lambda x: np.abs(x).sum(),
        ),
    )
    .reset_index()
)

total_family_importance = (
    family_importance[
        "sum_absolute_importance"
    ].sum()
)

family_importance[
    "relative_importance_fraction"
] = (
    family_importance[
        "sum_absolute_importance"
    ] / total_family_importance
    if total_family_importance > 0
    else np.nan
)

family_importance[
    "relative_importance_percent"
] = (
    100.0 *
    family_importance[
        "relative_importance_fraction"
    ]
)

family_importance.to_csv(
    TABLE_DIR /
    "table_12_feature_family_contribution.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# TOP FEATURE TABLE
# =============================================================================

top_features = perm_df.head(20).copy()

top_features.to_csv(
    TABLE_DIR /
    "table_13_top20_feature_importance.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# MODEL SUMMARY
# =============================================================================

model_summary = {
    "script": "09_prediction_model_development_v1.py",
    "status": "PASS",
    "generated": timestamp(),
    "seed": SEED,
    "n_splits": N_SPLITS,

    "selected_model": selected_model_name,

    "feature_count": 40,

    "feature_families": {
        "molecular": len(molecular_features),
        "3d": len(three_d_features),
        "molformer": len(molformer_features),
    },

    "train": {
        "n": int(len(y_train)),
        "active": int(y_train.sum()),
        "inactive": int((y_train == 0).sum()),
        "prevalence": float(y_train.mean()),
    },

    "test": {
        "n": int(len(y_test)),
        "active": int(y_test.sum()),
        "inactive": int((y_test == 0).sum()),
        "prevalence": float(y_test.mean()),
    },

    "threshold": {
        "selection_method": "Youden_J_on_training_OOF_calibrated_predictions",
        "locked_threshold": selected_threshold,
    },

    "test_metrics": {
        "raw_roc_auc": test_raw_auc,
        "calibrated_roc_auc": test_cal_auc,

        "raw_pr_auc": test_raw_ap,
        "calibrated_pr_auc": test_cal_ap,

        "raw_brier": test_raw_brier,
        "calibrated_brier": test_cal_brier,

        "raw_log_loss": test_raw_ll,
        "calibrated_log_loss": test_cal_ll,

        "raw_ece": test_raw_ece,
        "calibrated_ece": test_cal_ece,

        "raw_mce": test_raw_mce,
        "calibrated_mce": test_cal_mce,

        "ks_statistic": test_ks,

        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy":
            test_metrics["balanced_accuracy"],
        "sensitivity":
            test_metrics["sensitivity_recall"],
        "specificity":
            test_metrics["specificity"],
        "precision":
            test_metrics["precision_ppv"],
        "npv":
            test_metrics[
                "negative_predictive_value"
            ],
        "f1": test_metrics["f1"],
        "mcc": test_metrics["mcc"],

        "tp": test_metrics["true_positive"],
        "fp": test_metrics["false_positive"],
        "tn": test_metrics["true_negative"],
        "fn": test_metrics["false_negative"],

        "screening_nnt": nnt_screen,
    },

    "input_sha256": {
        str(TRAIN_MASTER):
            sha256_file(TRAIN_MASTER),

        str(TEST_MASTER):
            sha256_file(TEST_MASTER),

        str(HANDCRAFTED_TRAIN):
            sha256_file(HANDCRAFTED_TRAIN),

        str(HANDCRAFTED_TEST):
            sha256_file(HANDCRAFTED_TEST),

        str(MOLFORMER_TRAIN):
            sha256_file(MOLFORMER_TRAIN),

        str(MOLFORMER_TEST):
            sha256_file(MOLFORMER_TEST),

        str(SHORTLIST):
            sha256_file(SHORTLIST),
    },

    "output_directory": str(OUTDIR),
}

with open(
    OUTDIR / "model_development_summary.json",
    "w",
) as f:
    json.dump(
        model_summary,
        f,
        indent=2,
    )

# =============================================================================
# DEPLOYMENT ARTIFACT
# =============================================================================

log("")
log("=" * 80)
log("12. SERIALIZING DEPLOYMENT MODEL")
log("=" * 80)

try:

    import joblib

    deployment_artifact = {
        "model": final_calibrated_model,
        "model_name": selected_model_name,
        "feature_order": feature_order,
        "feature_manifest":
            feature_manifest.to_dict(
                orient="records"
            ),
        "threshold": selected_threshold,
        "random_seed": SEED,
        "calibration": "sigmoid_platt",
        "training_n": len(y_train),
        "training_active": int(y_train.sum()),
        "training_inactive":
            int((y_train == 0).sum()),
        "representation": {
            "molecular": molecular_features,
            "3d": three_d_features,
            "molformer": molformer_features,
        },
    }

    joblib.dump(
        deployment_artifact,
        MODEL_DIR /
        "final_calibrated_bioactivity_model.joblib",
        compress=3,
    )

    log(
        "Deployment artifact written successfully."
    )

except ImportError:

    log(
        "WARNING: joblib unavailable. "
        "Model artifact was not serialized."
    )

# =============================================================================
# PUBLICATION METRIC MASTER TABLE
# =============================================================================

publication_master = pd.DataFrame([
    {
        "section": "Model",
        "metric": "Selected model",
        "value": selected_model_name,
        "dataset": "development",
    },
    {
        "section": "Model",
        "metric": "Number of predictors",
        "value": 40,
        "dataset": "development",
    },
    {
        "section": "Model",
        "metric": "Molecular predictors",
        "value": len(molecular_features),
        "dataset": "development",
    },
    {
        "section": "Model",
        "metric": "3D-QSAR predictors",
        "value": len(three_d_features),
        "dataset": "development",
    },
    {
        "section": "Model",
        "metric": "MoLFormer predictors",
        "value": len(molformer_features),
        "dataset": "development",
    },
    {
        "section": "Threshold",
        "metric": "Locked threshold",
        "value": selected_threshold,
        "dataset": "training OOF",
    },
    {
        "section": "Discrimination",
        "metric": "ROC-AUC",
        "value": test_cal_auc,
        "dataset": "independent test",
    },
    {
        "section": "Discrimination",
        "metric": "PR-AUC",
        "value": test_cal_ap,
        "dataset": "independent test",
    },
    {
        "section": "Calibration",
        "metric": "Brier score",
        "value": test_cal_brier,
        "dataset": "independent test",
    },
    {
        "section": "Calibration",
        "metric": "Log loss",
        "value": test_cal_ll,
        "dataset": "independent test",
    },
    {
        "section": "Calibration",
        "metric": "ECE",
        "value": test_cal_ece,
        "dataset": "independent test",
    },
    {
        "section": "Calibration",
        "metric": "MCE",
        "value": test_cal_mce,
        "dataset": "independent test",
    },
    {
        "section": "Classification",
        "metric": "Sensitivity",
        "value":
            test_metrics[
                "sensitivity_recall"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Classification",
        "metric": "Specificity",
        "value":
            test_metrics[
                "specificity"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Classification",
        "metric": "Precision",
        "value":
            test_metrics[
                "precision_ppv"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Classification",
        "metric": "NPV",
        "value":
            test_metrics[
                "negative_predictive_value"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Classification",
        "metric": "F1",
        "value": test_metrics["f1"],
        "dataset": "independent test",
    },
    {
        "section": "Classification",
        "metric": "MCC",
        "value": test_metrics["mcc"],
        "dataset": "independent test",
    },
    {
        "section": "Classification",
        "metric": "Balanced accuracy",
        "value":
            test_metrics[
                "balanced_accuracy"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Confusion matrix",
        "metric": "TP",
        "value":
            test_metrics[
                "true_positive"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Confusion matrix",
        "metric": "FP",
        "value":
            test_metrics[
                "false_positive"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Confusion matrix",
        "metric": "TN",
        "value":
            test_metrics[
                "true_negative"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Confusion matrix",
        "metric": "FN",
        "value":
            test_metrics[
                "false_negative"
            ],
        "dataset": "independent test",
    },
    {
        "section": "Screening utility",
        "metric": "Screening NNT",
        "value": nnt_screen,
        "dataset": "independent test",
    },
])

publication_master.to_csv(
    TABLE_DIR /
    "TABLE_MASTER_PUBLICATION_MODEL_RESULTS.tsv",
    sep="\t",
    index=False,
)

# =============================================================================
# FINAL LOG
# =============================================================================

log("")
log("=" * 80)
log("SCRIPT 09 v1 COMPLETE")
log("=" * 80)

log("")
log(f"Selected model: {selected_model_name}")
log(f"Final predictors: {len(feature_order)}")
log(f"Locked threshold: {selected_threshold:.8f}")

log("")
log("INDEPENDENT TEST PERFORMANCE")
log(
    f"Calibrated ROC-AUC : {test_cal_auc:.6f}"
)
log(
    f"Calibrated PR-AUC  : {test_cal_ap:.6f}"
)
log(
    f"Calibrated Brier   : {test_cal_brier:.6f}"
)
log(
    f"Calibrated LogLoss : {test_cal_ll:.6f}"
)
log(
    f"Calibrated ECE     : {test_cal_ece:.6f}"
)
log(
    f"Calibrated MCE     : {test_cal_mce:.6f}"
)

log("")
log("LOCKED-THRESHOLD CLASSIFICATION")
log(
    f"Sensitivity        : "
    f"{test_metrics['sensitivity_recall']:.6f}"
)
log(
    f"Specificity        : "
    f"{test_metrics['specificity']:.6f}"
)
log(
    f"Precision          : "
    f"{test_metrics['precision_ppv']:.6f}"
)
log(
    f"NPV                : "
    f"{test_metrics['negative_predictive_value']:.6f}"
)
log(
    f"F1                 : "
    f"{test_metrics['f1']:.6f}"
)
log(
    f"MCC                : "
    f"{test_metrics['mcc']:.6f}"
)
log(
    f"Balanced accuracy  : "
    f"{test_metrics['balanced_accuracy']:.6f}"
)

log("")
log("CONFUSION MATRIX")
log(
    f"TP={test_metrics['true_positive']} "
    f"FP={test_metrics['false_positive']} "
    f"TN={test_metrics['true_negative']} "
    f"FN={test_metrics['false_negative']}"
)

log("")
log("OUTPUT DIRECTORY:")
log(str(OUTDIR))

log("")
log(f"Finished: {timestamp()}")
log("=" * 80)
