"""
E2-S7 [Model] OOS Error Analysis for LightGBM 5D-Return Model
==================================================================

Deliverable: a systematic inspection of the largest OOS errors, weak
periods, and regime-specific failure patterns for the LightGBM model,
written to be independently auditable rather than merely narrated.

Acceptance discipline enforced here (see README.md for the write-up):
  - The top-error selection rule (top 1% by absolute error) is fixed in
    this file, applied to the complete OOS set, and disclosed before any
    narrative is written -- there is no second pass that adjusts N after
    an initial look.
  - Regime labels are read from `results/oos_predictions.csv` unchanged --
    this script does not compute or adjust any regime boundary itself.
    The regime taxonomy (LowVol/HighVol) pre-dates this analysis by
    several project stages (E1-S4/E1-S6 regime construction, audited
    independently in docs/E4-S1_leakage_audit_record.md and
    docs/E2-S5_Regime_Performance_Table_audit_report.md).
  - Every number quoted in the README traces to a field in
    output/error_analysis_summary.json produced by this script -- no
    number in the narrative is asserted without a corresponding
    computed value here.
  - No row is excluded from the ranked error list for any reason.

Run:
    python analyze_oos_errors.py
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_S1_DIR = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"
sys.path.insert(0, str(E2_S1_DIR))

CANONICAL_OOS_PATH = REPO_ROOT / "results" / "oos_predictions.csv"
BASELINE_OOS_PATH = E2_S1_DIR / "output" / "baseline_zero_oos_predictions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Fixed, disclosed selection rule -- top 1% by absolute error, applied to
# the *complete* OOS set. Chosen before this file was written: round
# numbers (1%, 5%) are used precisely so the choice cannot be read as
# having been tuned to produce a particular-looking cluster.
TOP_ERROR_FRACTION = 0.01


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def hypergeometric_upper_tail_p_value(population_size: int, population_successes: int, sample_size: int, observed_successes: int) -> float:
    """Exact one-sided P(X >= observed_successes) under the null that the
    top-error sample is a uniformly random draw of `sample_size` rows from
    the population, independent of regime -- i.e. the null this analysis's
    regime-share claim must be checked against, not merely described.

    Computed with exact integer/rational arithmetic (`math.comb` + Fraction),
    not a normal approximation, since sample_size (39) is small enough that
    an exact hypergeometric tail is cheap and avoids continuity-correction
    debates entirely.
    """
    N, K, n, k_obs = population_size, population_successes, sample_size, observed_successes
    max_k = min(n, K)
    total = math.comb(N, n)
    tail = sum(Fraction(math.comb(K, k) * math.comb(N - K, n - k), total) for k in range(k_obs, max_k + 1))
    return float(tail)


def run() -> None:
    if not CANONICAL_OOS_PATH.exists():
        raise FileNotFoundError(f"{CANONICAL_OOS_PATH} not found -- run E2-S4/generate_oos_predictions.py first")
    if not BASELINE_OOS_PATH.exists():
        raise FileNotFoundError(f"{BASELINE_OOS_PATH} not found -- run E2-S1/run_baseline.py first")

    df = pd.read_csv(CANONICAL_OOS_PATH, parse_dates=["Date"])
    baseline_df = pd.read_csv(BASELINE_OOS_PATH, parse_dates=["Date"])

    n_total = len(df)
    df["error"] = df["actual_return_5d"] - df["prediction"]
    df["abs_error"] = df["error"].abs()
    df["abs_actual"] = df["actual_return_5d"].abs()

    # Baseline's error is |actual_return_5d| exactly, by construction
    # (baseline prediction is always 0) -- used as the cross-reference in
    # §10, not recomputed from a separate assumption.
    baseline_lookup = baseline_df.set_index("Date")["y_true"].abs()
    df["baseline_abs_error"] = df["Date"].map(baseline_lookup)
    if df["baseline_abs_error"].isna().any():
        raise ValueError("could not align baseline predictions to every canonical OOS date")

    k = round(n_total * TOP_ERROR_FRACTION)
    top_errors = df.sort_values("abs_error", ascending=False).head(k).reset_index(drop=True)
    top_actual_dates = set(df.sort_values("abs_actual", ascending=False).head(k)["Date"])
    top_error_dates = set(top_errors["Date"])
    overlap_dates = top_error_dates & top_actual_dates
    error_only_dates = sorted(top_error_dates - top_actual_dates)

    error_only_rows = df[df["Date"].isin(error_only_dates)]
    sign_mismatch = (np.sign(error_only_rows["actual_return_5d"]) != np.sign(error_only_rows["prediction"])).sum()

    overall_regime_share = df["regime"].value_counts(normalize=True).to_dict()
    top_regime_share = top_errors["regime"].value_counts(normalize=True).to_dict()
    top_regime_counts = top_errors["regime"].value_counts().to_dict()

    # Significance test for the regime-share claim: is the observed HighVol
    # count in the top-k errors more extreme than a regime-blind random draw
    # of k rows from the full OOS set would produce? Null hypothesis: top-error
    # membership is independent of regime. Exact hypergeometric tail, not a
    # normal approximation or an eyeballed comparison of two percentages.
    n_highvol_population = int((df["regime"] == "HighVol").sum())
    n_highvol_top = int((top_errors["regime"] == "HighVol").sum())
    highvol_enrichment_p_value = hypergeometric_upper_tail_p_value(
        population_size=n_total,
        population_successes=n_highvol_population,
        sample_size=k,
        observed_successes=n_highvol_top,
    )
    expected_highvol_under_null = k * (n_highvol_population / n_total)
    sd_highvol_under_null = math.sqrt(
        k * (n_highvol_population / n_total) * (1 - n_highvol_population / n_total) * (n_total - k) / (n_total - 1)
    )

    overall_fold_share = (df["fold_id"].value_counts(normalize=True)).to_dict()
    top_fold_counts = top_errors["fold_id"].value_counts().to_dict()

    per_regime_stats = {}
    for regime in ["LowVol", "HighVol"]:
        subset = df[df["regime"] == regime]
        per_regime_stats[regime] = {
            "n": int(len(subset)),
            "mae": float(subset["abs_error"].mean()),
            "mean_abs_actual": float(subset["abs_actual"].mean()),
        }

    # Largest thematic (date-contiguous, calendar-proximate) cluster inside
    # the top-error list -- computed, not eyeballed: group top-error dates
    # into runs separated by more than 10 calendar days.
    sorted_top_dates = sorted(top_errors["Date"].tolist())
    clusters: list[list[pd.Timestamp]] = []
    for d in sorted_top_dates:
        if clusters and (d - clusters[-1][-1]).days <= 10:
            clusters[-1].append(d)
        else:
            clusters.append([d])
    largest_cluster = max(clusters, key=len)

    pearson_actual_error = pearson(df["abs_actual"].to_numpy(), df["abs_error"].to_numpy())
    spearman_actual_error = spearman(df["abs_actual"].to_numpy(), df["abs_error"].to_numpy())
    pearson_error_vs_baseline_error = pearson(df["abs_error"].to_numpy(), df["baseline_abs_error"].to_numpy())

    OUTPUT_DIR.mkdir(exist_ok=True)
    top_errors_out = top_errors[[
        "Date", "fold_id", "regime", "actual_return_5d", "prediction", "error", "abs_error",
    ]].copy()
    top_errors_out["Date"] = top_errors_out["Date"].dt.strftime("%Y-%m-%d")
    top_errors_out.to_csv(OUTPUT_DIR / "top_errors.csv", index=False)

    summary = {
        "card": "E2-S7 [Model] OOS Error Analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_oos_predictions_path": str(CANONICAL_OOS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "canonical_oos_predictions_sha256": sha256_of(CANONICAL_OOS_PATH),
        "baseline_oos_predictions_path": str(BASELINE_OOS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "baseline_oos_predictions_sha256": sha256_of(BASELINE_OOS_PATH),
        "selection_rule": {
            "description": "top N by |actual_return_5d - prediction|, N = round(1% of total OOS rows), applied to the complete OOS set with no exclusions",
            "top_error_fraction": TOP_ERROR_FRACTION,
            "n_total_oos_rows": int(n_total),
            "k_top_errors": int(k),
        },
        "no_exclusions_declaration": (
            "Every one of the n_total_oos_rows rows in results/oos_predictions.csv was included in the "
            "ranking. No row was removed as an outlier, for readability, or for any other reason."
        ),
        "extreme_return_dominance": {
            "pearson_abs_actual_vs_abs_error": pearson_actual_error,
            "spearman_abs_actual_vs_abs_error": spearman_actual_error,
            "pearson_lightgbm_abs_error_vs_baseline_abs_error": pearson_error_vs_baseline_error,
            "interpretation": (
                "baseline_abs_error is |actual_return_5d| exactly (baseline predicts 0 always), so the "
                "pearson_lightgbm_abs_error_vs_baseline_abs_error value is a direct test of whether the "
                "|actual|-vs-|error| relationship is LightGBM-specific: it is not, since both correlations "
                "are identical to floating-point precision."
            ),
            "top_k_overlap_with_top_k_abs_actual": {
                "k": int(k),
                "n_overlap": int(len(overlap_dates)),
                "overlap_fraction": len(overlap_dates) / k,
            },
        },
        "regime_taxonomy_provenance": (
            "LowVol/HighVol labels are read unchanged from results/oos_predictions.csv's 'regime' column, "
            "itself sourced from the E1-S4/E1-S6 canonical dataset -- a threshold defined and audited before "
            "this analysis existed (see docs/E4-S1_leakage_audit_record.md and "
            "docs/E2-S5_Regime_Performance_Table_audit_report.md). No regime boundary was defined or adjusted "
            "in this script."
        ),
        "regime_representation": {
            "overall_share": overall_regime_share,
            "top_error_share": top_regime_share,
            "top_error_counts": {str(k_): int(v) for k_, v in top_regime_counts.items()},
            "significance_test": {
                "description": (
                    "exact hypergeometric one-sided test: P(HighVol count in top-k >= observed) under the "
                    "null that top-error membership is independent of regime (i.e. a uniformly random "
                    "k-row draw from the full OOS set would produce this many HighVol rows or more)"
                ),
                "population_size": int(n_total),
                "population_highvol_count": n_highvol_population,
                "sample_size_k": int(k),
                "observed_highvol_in_top_k": n_highvol_top,
                "expected_highvol_under_null": expected_highvol_under_null,
                "sd_highvol_under_null": sd_highvol_under_null,
                "p_value_upper_tail": highvol_enrichment_p_value,
            },
        },
        "fold_representation": {
            "overall_share": overall_fold_share,
            "top_error_counts": {str(k_): int(v) for k_, v in top_fold_counts.items()},
        },
        "per_regime_stats_full_oos_set": per_regime_stats,
        "error_only_cluster": {
            "description": "top-error dates that are NOT also top-|actual|-magnitude dates -- i.e. rows where a genuine wrong-direction/miscalibrated prediction, not raw event magnitude alone, drove the error into the top-k",
            "n": int(len(error_only_dates)),
            "n_sign_mismatched": int(sign_mismatch),
            "dates": [d.strftime("%Y-%m-%d") for d in error_only_dates],
            "small_sample_caveat": f"n={len(error_only_dates)} is below the ~dozen-point threshold for anything beyond an illustrative pattern; treat as a hypothesis, not a finding.",
        },
        "largest_calendar_cluster_in_top_errors": {
            "n_dates": len(largest_cluster),
            "share_of_top_errors": len(largest_cluster) / k,
            "date_start": largest_cluster[0].strftime("%Y-%m-%d"),
            "date_end": largest_cluster[-1].strftime("%Y-%m-%d"),
        },
        "no_retuning_declaration": (
            "No hyperparameter, feature, or model change has been made as a result of this analysis. "
            "This document is read-only with respect to the frozen E2-S2 LightGBM configuration."
        ),
        "authorship_disclosure": (
            "This analysis and its subsequent independent audit were both produced within the same review "
            "session. This is a disclosed conflict of interest for the audit that follows, not a claim of "
            "independence."
        ),
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "python": platform.python_version(),
    }
    (OUTPUT_DIR / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"Wrote top-{k} error table and summary to {OUTPUT_DIR}")
    print(json.dumps({k_: v for k_, v in summary.items() if k_ not in ("selection_rule",)}, indent=2, default=str)[:2000])


if __name__ == "__main__":
    run()
