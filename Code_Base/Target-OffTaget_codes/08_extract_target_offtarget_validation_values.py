#!/usr/bin/env python3

"""
==============================================================================
SCRIPT 08 — TARGET / POSSIBLE OFF-TARGET VALIDATION VALUE EXTRACTION
==============================================================================

Purpose
-------
Extract publication-ready numerical summaries from the completed
target/off-target validation tables.

This script DOES NOT:
    - retrain any model
    - change the 40D feature space
    - change similarity thresholds
    - modify validation files
    - generate case-study results
    - use individual test-molecule inference results

It ONLY reads the existing validation outputs and calculates descriptive
statistics required for:
    1. Results section
    2. Publication figures
    3. Figure captions
    4. Supplementary tables

Main validation inputs
----------------------
06_validation_summary.tsv
06_target_recovery.tsv
06_neighbor_recovery.tsv
06_offtarget_recovery.tsv
06_similarity_threshold_summary.tsv
06_target_annotation_qc.tsv

Outputs
-------
08_validation_value_summary/
    01_dataset_sizes.tsv
    02_known_target_recovery_summary.tsv
    03_recall_by_retrieval_depth.tsv
    04_similarity_summary.tsv
    05_similarity_quantiles.tsv
    06_similarity_band_summary.tsv
    07_target_evidence_summary.tsv
    08_target_rank_summary.tsv
    09_offtarget_summary.tsv
    10_offtarget_rank_summary.tsv
    11_threshold_summary.tsv
    12_annotation_completeness.tsv
    13_query_candidate_burden.tsv
    14_publication_statistics.txt
    08_value_extraction_manifest.json
    08_value_extraction_qc.tsv
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

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
    / "08_validation_value_summary"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    "annotation_qc":
        VALIDATION_DIR / "06_target_annotation_qc.tsv",
}


# ============================================================================
# UTILITIES
# ============================================================================

def load_tsv(path, name):
    """Load TSV and perform basic validation."""
    print(f"Loading: {name}")
    print(f"        {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Required validation file does not exist:\n{path}"
        )

    df = pd.read_csv(path, sep="\t")

    print(f"        rows = {len(df):,}")
    print(f"        cols = {len(df.columns):,}")
    print()

    return df


def numeric_series(df, column):
    """Return numeric version of a column."""
    if column not in df.columns:
        return None

    return pd.to_numeric(df[column], errors="coerce")


def describe_numeric(df, column):
    """Generate robust descriptive statistics."""
    s = numeric_series(df, column)

    if s is None:
        return None

    s = s.dropna()

    if len(s) == 0:
        return None

    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "q01": float(s.quantile(0.01)),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def save_tsv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, sep="\t", index=False)
    print(f"Written: {path}")
    return path


def pct(x):
    """Format fraction/percentage safely."""
    if pd.isna(x):
        return "NA"
    return f"{x:.2f}%"


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 78)
print("TARGET / POSSIBLE OFF-TARGET VALIDATION VALUE EXTRACTION")
print("=" * 78)
print()

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

offtarget = load_tsv(
    FILES["offtarget_recovery"],
    "06_offtarget_recovery.tsv"
)

threshold = load_tsv(
    FILES["threshold_summary"],
    "06_similarity_threshold_summary.tsv"
)

annotation_qc = load_tsv(
    FILES["annotation_qc"],
    "06_target_annotation_qc.tsv"
)


# ============================================================================
# 1. DATASET SIZE SUMMARY
# ============================================================================

dataset_rows = [
    {
        "dataset": "Validation queries",
        "rows": len(validation),
        "source": "06_validation_summary.tsv",
    },
    {
        "dataset": "Target recovery observations",
        "rows": len(target_recovery),
        "source": "06_target_recovery.tsv",
    },
    {
        "dataset": "Neighbor recovery observations",
        "rows": len(neighbor_recovery),
        "source": "06_neighbor_recovery.tsv",
    },
    {
        "dataset": "Possible off-target observations",
        "rows": len(offtarget),
        "source": "06_offtarget_recovery.tsv",
    },
    {
        "dataset": "Similarity threshold evaluations",
        "rows": len(threshold),
        "source": "06_similarity_threshold_summary.tsv",
    },
    {
        "dataset": "Target annotation QC fields",
        "rows": len(annotation_qc),
        "source": "06_target_annotation_qc.tsv",
    },
]

dataset_sizes = pd.DataFrame(dataset_rows)

save_tsv(
    dataset_sizes,
    "01_dataset_sizes.tsv"
)


# ============================================================================
# 2. KNOWN TARGET RECOVERY
# ============================================================================

print("=" * 78)
print("KNOWN TARGET RECOVERY")
print("=" * 78)

required_recovery_columns = [
    "known_target_count",
    "recovered_known_target_count",
    "known_target_recall_at_top_k",
]

for col in required_recovery_columns:
    if col not in validation.columns:
        print(f"WARNING: Missing column: {col}")

recovery_summary_rows = []

for col in required_recovery_columns:

    if col not in validation.columns:
        continue

    stats = describe_numeric(validation, col)

    if stats is None:
        continue

    recovery_summary_rows.append(
        {
            "metric": col,
            **stats
        }
    )

recovery_summary = pd.DataFrame(recovery_summary_rows)

save_tsv(
    recovery_summary,
    "02_known_target_recovery_summary.tsv"
)


# ============================================================================
# 3. RECALL BY RETRIEVAL DEPTH / QUERY GROUP
# ============================================================================

print("=" * 78)
print("KNOWN TARGET RECALL STRUCTURE")
print("=" * 78)

if "known_target_recall_at_top_k" in validation.columns:

    recall = pd.to_numeric(
        validation["known_target_recall_at_top_k"],
        errors="coerce"
    )

    recall_table = pd.DataFrame({
        "metric": [
            "Mean recall",
            "Median recall",
            "Standard deviation",
            "Minimum recall",
            "25th percentile recall",
            "75th percentile recall",
            "95th percentile recall",
            "Maximum recall",
        ],
        "value": [
            recall.mean(),
            recall.median(),
            recall.std(),
            recall.min(),
            recall.quantile(0.25),
            recall.quantile(0.75),
            recall.quantile(0.95),
            recall.max(),
        ]
    })

    save_tsv(
        recall_table,
        "03_recall_by_retrieval_depth.tsv"
    )

else:

    print(
        "known_target_recall_at_top_k not found; "
        "recall table cannot be generated."
    )


# ============================================================================
# 4. COSINE SIMILARITY SUMMARY
# ============================================================================

print("=" * 78)
print("COSINE SIMILARITY")
print("=" * 78)

similarity_columns = [
    "best_neighbor_similarity",
    "max_similarity",
    "mean_similarity",
    "cosine_similarity",
]

similarity_rows = []

for df_name, df in [
    ("validation_summary", validation),
    ("target_recovery", target_recovery),
    ("neighbor_recovery", neighbor_recovery),
    ("offtarget_recovery", offtarget),
]:

    for col in similarity_columns:

        if col not in df.columns:
            continue

        stats = describe_numeric(df, col)

        if stats is None:
            continue

        similarity_rows.append({
            "dataset": df_name,
            "metric": col,
            **stats
        })

similarity_summary = pd.DataFrame(similarity_rows)

save_tsv(
    similarity_summary,
    "04_similarity_summary.tsv"
)


# ============================================================================
# 5. SIMILARITY QUANTILES
# ============================================================================

quantile_rows = []

for df_name, df in [
    ("validation_summary", validation),
    ("target_recovery", target_recovery),
    ("neighbor_recovery", neighbor_recovery),
    ("offtarget_recovery", offtarget),
]:

    for col in similarity_columns:

        if col not in df.columns:
            continue

        s = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(s) == 0:
            continue

        for q in [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]:

            quantile_rows.append({
                "dataset": df_name,
                "metric": col,
                "quantile": q,
                "value": float(s.quantile(q)),
            })

similarity_quantiles = pd.DataFrame(quantile_rows)

save_tsv(
    similarity_quantiles,
    "05_similarity_quantiles.tsv"
)


# ============================================================================
# 6. SIMILARITY BAND SUMMARY
# ============================================================================

print("=" * 78)
print("SIMILARITY BAND SUMMARY")
print("=" * 78)

if "similarity_band" in neighbor_recovery.columns:

    band_counts = (
        neighbor_recovery["similarity_band"]
        .fillna("MISSING")
        .value_counts(dropna=False)
        .rename_axis("similarity_band")
        .reset_index(name="neighbor_rows")
    )

    band_counts["percentage"] = (
        band_counts["neighbor_rows"]
        / band_counts["neighbor_rows"].sum()
        * 100
    )

    save_tsv(
        band_counts,
        "06_similarity_band_summary.tsv"
    )

else:

    print("similarity_band column unavailable.")


# ============================================================================
# 7. TARGET EVIDENCE SUMMARY
# ============================================================================

print("=" * 78)
print("TARGET EVIDENCE")
print("=" * 78)

evidence_rows = []

target_metrics = [
    "max_similarity",
    "mean_similarity",
    "supporting_neighbor_count",
    "target_rank",
]

for col in target_metrics:

    if col not in target_recovery.columns:
        continue

    stats = describe_numeric(
        target_recovery,
        col
    )

    if stats is None:
        continue

    evidence_rows.append({
        "metric": col,
        **stats
    })

target_evidence_summary = pd.DataFrame(evidence_rows)

save_tsv(
    target_evidence_summary,
    "07_target_evidence_summary.tsv"
)


# ============================================================================
# 8. TARGET RANK DISTRIBUTION
# ============================================================================

if "target_rank" in target_recovery.columns:

    rank = pd.to_numeric(
        target_recovery["target_rank"],
        errors="coerce"
    )

    rank_counts = (
        rank.dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("target_rank")
        .reset_index(name="target_recovery_records")
    )

    rank_counts["percentage"] = (
        rank_counts["target_recovery_records"]
        / rank_counts["target_recovery_records"].sum()
        * 100
    )

    save_tsv(
        rank_counts,
        "08_target_rank_summary.tsv"
    )


# ============================================================================
# 9. POSSIBLE OFF-TARGET SUMMARY
# ============================================================================

print("=" * 78)
print("POSSIBLE OFF-TARGET SUMMARY")
print("=" * 78)

offtarget_rows = []

for col in [
    "offtarget_rank",
    "max_similarity",
    "mean_similarity",
    "supporting_neighbor_count",
]:

    if col not in offtarget.columns:
        continue

    stats = describe_numeric(
        offtarget,
        col
    )

    if stats is None:
        continue

    offtarget_rows.append({
        "metric": col,
        **stats
    })

offtarget_summary = pd.DataFrame(offtarget_rows)

save_tsv(
    offtarget_summary,
    "09_offtarget_summary.tsv"
)


# ============================================================================
# 10. OFF-TARGET RANK DISTRIBUTION
# ============================================================================

if "offtarget_rank" in offtarget.columns:

    rank = pd.to_numeric(
        offtarget["offtarget_rank"],
        errors="coerce"
    )

    rank_counts = (
        rank.dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("offtarget_rank")
        .reset_index(name="offtarget_records")
    )

    rank_counts["percentage"] = (
        rank_counts["offtarget_records"]
        / rank_counts["offtarget_records"].sum()
        * 100
    )

    save_tsv(
        rank_counts,
        "10_offtarget_rank_summary.tsv"
    )


# ============================================================================
# 11. SIMILARITY THRESHOLD SUMMARY
# ============================================================================

print("=" * 78)
print("SIMILARITY THRESHOLD SUMMARY")
print("=" * 78)

threshold_out = threshold.copy()

for col in threshold_out.columns:

    if col == "similarity_threshold":
        continue

    threshold_out[col] = pd.to_numeric(
        threshold_out[col],
        errors="coerce"
    )

if "similarity_threshold" in threshold_out.columns:

    threshold_out = threshold_out.sort_values(
        "similarity_threshold"
    )

    if {
        "target_bearing_neighbor_rows",
        "neighbor_rows",
    }.issubset(threshold_out.columns):

        threshold_out["target_bearing_neighbor_fraction"] = (
            threshold_out["target_bearing_neighbor_rows"]
            / threshold_out["neighbor_rows"]
        )

        threshold_out["target_bearing_neighbor_percentage"] = (
            threshold_out["target_bearing_neighbor_fraction"]
            * 100
        )

save_tsv(
    threshold_out,
    "11_threshold_summary.tsv"
)


# ============================================================================
# 12. TARGET ANNOTATION COMPLETENESS
# ============================================================================

print("=" * 78)
print("TARGET ANNOTATION COMPLETENESS")
print("=" * 78)

annotation_out = annotation_qc.copy()

if "available" in annotation_out.columns:
    annotation_out["available"] = pd.to_numeric(
        annotation_out["available"],
        errors="coerce"
    )

if "missing" in annotation_out.columns:
    annotation_out["missing"] = pd.to_numeric(
        annotation_out["missing"],
        errors="coerce"
    )

if "target_rows" in annotation_out.columns:
    annotation_out["target_rows"] = pd.to_numeric(
        annotation_out["target_rows"],
        errors="coerce"
    )

if "completeness_percent" not in annotation_out.columns:

    if {
        "available",
        "target_rows"
    }.issubset(annotation_out.columns):

        annotation_out["completeness_percent"] = (
            annotation_out["available"]
            / annotation_out["target_rows"]
            * 100
        )

save_tsv(
    annotation_out,
    "12_annotation_completeness.tsv"
)


# ============================================================================
# 13. QUERY-LEVEL CANDIDATE BURDEN
# ============================================================================

print("=" * 78)
print("QUERY-LEVEL CANDIDATE BURDEN")
print("=" * 78)

candidate_columns = [
    "candidate_high_similarity_target_count",
    "possible_offtarget_count",
    "target_bearing_neighbors",
    "neighbors_sharing_at_least_one_known_target",
    "high_similarity_neighbor_count",
    "moderate_similarity_neighbor_count",
]

candidate_rows = []

for col in candidate_columns:

    if col not in validation.columns:
        continue

    stats = describe_numeric(
        validation,
        col
    )

    if stats is None:
        continue

    candidate_rows.append({
        "metric": col,
        **stats
    })

candidate_summary = pd.DataFrame(candidate_rows)

save_tsv(
    candidate_summary,
    "13_query_candidate_burden.tsv"
)


# ============================================================================
# 14. PUBLICATION STATISTICS
# ============================================================================

print("=" * 78)
print("PUBLICATION STATISTICS")
print("=" * 78)

lines = []

lines.append(
    "TARGET / POSSIBLE OFF-TARGET INFERENCE VALIDATION STATISTICS"
)
lines.append("=" * 78)
lines.append("")

lines.append(
    f"Validation queries: {len(validation):,}"
)

lines.append(
    f"Target recovery observations: {len(target_recovery):,}"
)

lines.append(
    f"Neighbor recovery observations: {len(neighbor_recovery):,}"
)

lines.append(
    f"Possible off-target recovery observations: {len(offtarget):,}"
)

lines.append(
    f"Similarity threshold evaluations: {len(threshold):,}"
)

lines.append("")

# --------------------------------------------------------------------------
# Recall
# --------------------------------------------------------------------------

if "known_target_recall_at_top_k" in validation.columns:

    recall = pd.to_numeric(
        validation["known_target_recall_at_top_k"],
        errors="coerce"
    ).dropna()

    if len(recall):

        lines.append("KNOWN-TARGET RECALL")
        lines.append("-" * 78)

        lines.append(
            f"N: {len(recall):,}"
        )

        lines.append(
            f"Mean: {recall.mean():.6f}"
        )

        lines.append(
            f"Median: {recall.median():.6f}"
        )

        lines.append(
            f"SD: {recall.std():.6f}"
        )

        lines.append(
            f"Minimum: {recall.min():.6f}"
        )

        lines.append(
            f"25th percentile: {recall.quantile(0.25):.6f}"
        )

        lines.append(
            f"75th percentile: {recall.quantile(0.75):.6f}"
        )

        lines.append(
            f"95th percentile: {recall.quantile(0.95):.6f}"
        )

        lines.append(
            f"Maximum: {recall.max():.6f}"
        )

        lines.append("")


# --------------------------------------------------------------------------
# Best-neighbor similarity
# --------------------------------------------------------------------------

if "best_neighbor_similarity" in validation.columns:

    s = pd.to_numeric(
        validation["best_neighbor_similarity"],
        errors="coerce"
    ).dropna()

    if len(s):

        lines.append("BEST-NEIGHBOR COSINE SIMILARITY")
        lines.append("-" * 78)

        lines.append(
            f"N: {len(s):,}"
        )

        lines.append(
            f"Mean: {s.mean():.6f}"
        )

        lines.append(
            f"Median: {s.median():.6f}"
        )

        lines.append(
            f"SD: {s.std():.6f}"
        )

        lines.append(
            f"Minimum: {s.min():.6f}"
        )

        lines.append(
            f"Maximum: {s.max():.6f}"
        )

        lines.append("")


# --------------------------------------------------------------------------
# Candidate burden
# --------------------------------------------------------------------------

for col in [
    "candidate_high_similarity_target_count",
    "possible_offtarget_count",
]:

    if col not in validation.columns:
        continue

    s = pd.to_numeric(
        validation[col],
        errors="coerce"
    ).dropna()

    if len(s) == 0:
        continue

    lines.append(col)
    lines.append("-" * 78)

    lines.append(
        f"N: {len(s):,}"
    )

    lines.append(
        f"Mean: {s.mean():.6f}"
    )

    lines.append(
        f"Median: {s.median():.6f}"
    )

    lines.append(
        f"SD: {s.std():.6f}"
    )

    lines.append(
        f"Minimum: {s.min():.0f}"
    )

    lines.append(
        f"Maximum: {s.max():.0f}"
    )

    lines.append("")


# --------------------------------------------------------------------------
# Target evidence
# --------------------------------------------------------------------------

if "supporting_neighbor_count" in target_recovery.columns:

    s = pd.to_numeric(
        target_recovery["supporting_neighbor_count"],
        errors="coerce"
    ).dropna()

    if len(s):

        lines.append("TARGET SUPPORTING-NEIGHBOR COUNT")
        lines.append("-" * 78)

        lines.append(
            f"N: {len(s):,}"
        )

        lines.append(
            f"Mean: {s.mean():.6f}"
        )

        lines.append(
            f"Median: {s.median():.6f}"
        )

        lines.append(
            f"Minimum: {s.min():.0f}"
        )

        lines.append(
            f"Maximum: {s.max():.0f}"
        )

        lines.append("")


# --------------------------------------------------------------------------
# Similarity threshold
# --------------------------------------------------------------------------

if "similarity_threshold" in threshold.columns:

    thresholds = pd.to_numeric(
        threshold["similarity_threshold"],
        errors="coerce"
    ).dropna()

    lines.append("SIMILARITY THRESHOLD EVALUATION")
    lines.append("-" * 78)

    lines.append(
        "Thresholds evaluated: "
        + ", ".join(
            f"{x:.6f}"
            for x in sorted(thresholds.unique())
        )
    )

    lines.append("")


publication_statistics = "\n".join(lines)

publication_path = OUTPUT_DIR / "14_publication_statistics.txt"

publication_path.write_text(
    publication_statistics,
    encoding="utf-8"
)

print(f"Written: {publication_path}")


# ============================================================================
# 15. QC
# ============================================================================

qc_rows = []

for name, df in [
    ("06_validation_summary", validation),
    ("06_target_recovery", target_recovery),
    ("06_neighbor_recovery", neighbor_recovery),
    ("06_offtarget_recovery", offtarget),
    ("06_similarity_threshold_summary", threshold),
    ("06_target_annotation_qc", annotation_qc),
]:

    qc_rows.append({
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    })


qc = pd.DataFrame(qc_rows)

save_tsv(
    qc,
    "08_value_extraction_qc.tsv"
)


# ============================================================================
# 16. MANIFEST
# ============================================================================

manifest = {
    "script": "08_extract_target_offtarget_validation_values.py",
    "purpose": (
        "Extract numerical validation statistics for publication "
        "Results and figure generation."
    ),
    "input_directory": str(VALIDATION_DIR),
    "output_directory": str(OUTPUT_DIR),
    "inputs": {
        key: str(path)
        for key, path in FILES.items()
    },
    "input_rows": {
        "validation_summary": int(len(validation)),
        "target_recovery": int(len(target_recovery)),
        "neighbor_recovery": int(len(neighbor_recovery)),
        "offtarget_recovery": int(len(offtarget)),
        "threshold_summary": int(len(threshold)),
        "annotation_qc": int(len(annotation_qc)),
    },
    "does_not_use_case_study_results": True,
    "does_not_modify_validation_data": True,
}

manifest_path = OUTPUT_DIR / "08_value_extraction_manifest.json"

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)

print(f"Written: {manifest_path}")


# ============================================================================
# COMPLETE
# ============================================================================

print()
print("=" * 78)
print("VALUE EXTRACTION COMPLETE")
print("=" * 78)
print()
print(f"Output directory:")
print(OUTPUT_DIR)
print()
print("Important files:")
print("  02_known_target_recovery_summary.tsv")
print("  03_recall_by_retrieval_depth.tsv")
print("  04_similarity_summary.tsv")
print("  05_similarity_quantiles.tsv")
print("  06_similarity_band_summary.tsv")
print("  07_target_evidence_summary.tsv")
print("  08_target_rank_summary.tsv")
print("  09_offtarget_summary.tsv")
print("  10_offtarget_rank_summary.tsv")
print("  11_threshold_summary.tsv")
print("  12_annotation_completeness.tsv")
print("  13_query_candidate_burden.tsv")
print("  14_publication_statistics.txt")
print()
print("No case-study molecule results were used.")
print()
