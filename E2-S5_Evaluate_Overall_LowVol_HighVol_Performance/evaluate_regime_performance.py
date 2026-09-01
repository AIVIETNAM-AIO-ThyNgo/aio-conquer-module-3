"""
E2-S5 [P0][Model] Evaluate Overall, Low-Vol & High-Vol Performance
======================================================================

Deliverable: a canonical results table -- N, MAE, prediction correlation and
directional hit rate -- for both the zero baseline (E2-S1) and LightGBM
(E2-S2), each broken out by Overall / LowVol / HighVol. Every scope for
every model is scored with the exact same `metrics.py` functions used
everywhere else in E2, so the regime breakdown does not introduce a second
scoring standard.

Source: E2-S4's canonical `results/oos_predictions.csv` (LightGBM) and
E2-S1's `baseline_zero_oos_predictions.csv` (zero baseline). Both are the
same OOS rows (same folds, same dates) by construction -- E2-S1's
acceptance already guarantees this -- but this script re-verifies it rather
than assuming it still holds after E2-S4's reshaping, because the fairness
of the baseline/LightGBM comparison depends on it.

Depends on: E4-S2. This repository has not produced an E4-S2 artifact; the
functional dependency this script actually has is E2-S4's canonical OOS
table (`results/oos_predictions.csv`). See Scope boundary in README.md.

Run:
    python evaluate_regime_performance.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_S1_DIR = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"
sys.path.insert(0, str(E2_S1_DIR))

from metrics import (  # noqa: E402
    DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION,
    directional_hit_rate,
    mae,
    prediction_correlation,
)

CANONICAL_OOS_PATH = REPO_ROOT / "results" / "oos_predictions.csv"
BASELINE_OOS_PATH = E2_S1_DIR / "output" / "baseline_zero_oos_predictions.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

REGIMES = ["LowVol", "HighVol"]
SCOPES = ["Overall"] + REGIMES
MODELS = ["baseline_zero", "lightgbm"]

DEGENERATE_PREDICTION_STD_THRESHOLD = 1e-6


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions_are_nearly_constant(y_pred: np.ndarray) -> bool:
    return bool(np.std(y_pred) < DEGENERATE_PREDICTION_STD_THRESHOLD)


def load_lightgbm_predictions() -> pd.DataFrame:
    df = pd.read_csv(CANONICAL_OOS_PATH, parse_dates=["Date"])
    return pd.DataFrame({
        "Date": df["Date"],
        "regime": df["regime"],
        "y_true": df["actual_return_5d"],
        "y_pred": df["prediction"],
    })


def load_baseline_predictions() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_OOS_PATH, parse_dates=["Date"])
    return df[["Date", "regime", "y_true", "y_pred"]].copy()


def assert_same_oos_rows(baseline_df: pd.DataFrame, lightgbm_df: pd.DataFrame) -> None:
    """The baseline/LightGBM comparison is only fair if both are scored on
    exactly the same OOS rows -- same dates, same regime labels, same
    target values. Re-verified here rather than assumed."""
    if not baseline_df["Date"].is_unique or not lightgbm_df["Date"].is_unique:
        raise AssertionError("a source predictions file has duplicate dates -- cannot compare fairly")

    baseline_dates = np.sort(baseline_df["Date"].to_numpy())
    lightgbm_dates = np.sort(lightgbm_df["Date"].to_numpy())
    if not np.array_equal(baseline_dates, lightgbm_dates):
        raise AssertionError("baseline and LightGBM are not scored on the same OOS dates -- comparison is not fair")

    merged = baseline_df.merge(lightgbm_df, on="Date", suffixes=("_baseline", "_lightgbm"))
    if not (merged["regime_baseline"] == merged["regime_lightgbm"]).all():
        raise AssertionError("baseline and LightGBM disagree on the regime label for at least one shared date")
    if not np.allclose(merged["y_true_baseline"].to_numpy(dtype=float), merged["y_true_lightgbm"].to_numpy(dtype=float)):
        raise AssertionError("baseline and LightGBM disagree on actual_return_5d for at least one shared date")


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "Overall":
        return pd.Series(True, index=df.index)
    if scope not in REGIMES:
        raise ValueError(f"unknown scope {scope!r} -- expected one of {SCOPES}")
    return df["regime"] == scope


def evaluate_model(df: pd.DataFrame, model_name: str) -> list[dict]:
    """One row per scope, every metric always reported -- N included even
    when N is small (HighVol), so a thin sample is visible rather than
    hidden behind a metric that happens to look fine."""
    rows = []
    for scope in SCOPES:
        subset = df[scope_mask(df, scope)]
        y_true = subset["y_true"].to_numpy(dtype=float)
        y_pred = subset["y_pred"].to_numpy(dtype=float)
        n = len(subset)
        rows.append({
            "model": model_name,
            "scope": scope,
            "n": n,
            "mae": mae(y_true, y_pred) if n else float("nan"),
            "prediction_correlation": prediction_correlation(y_true, y_pred) if n else float("nan"),
            "directional_hit_rate": directional_hit_rate(y_true, y_pred) if n else float("nan"),
            "predictions_nearly_constant": predictions_are_nearly_constant(y_pred) if n else None,
            "date_start": subset["Date"].min().strftime("%Y-%m-%d") if n else None,
            "date_end": subset["Date"].max().strftime("%Y-%m-%d") if n else None,
        })
    return rows


def build_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """Wide, side-by-side view for the fairness acceptance -- baseline and
    LightGBM's numbers for the same scope, next to each other."""
    baseline = results[results["model"] == "baseline_zero"].set_index("scope")
    lightgbm = results[results["model"] == "lightgbm"].set_index("scope")
    rows = []
    for scope in SCOPES:
        rows.append({
            "scope": scope,
            "n": int(baseline.loc[scope, "n"]),
            "baseline_zero_mae": baseline.loc[scope, "mae"],
            "lightgbm_mae": lightgbm.loc[scope, "mae"],
            "mae_improvement_over_baseline": baseline.loc[scope, "mae"] - lightgbm.loc[scope, "mae"],
            "baseline_zero_prediction_correlation": baseline.loc[scope, "prediction_correlation"],
            "lightgbm_prediction_correlation": lightgbm.loc[scope, "prediction_correlation"],
            "baseline_zero_directional_hit_rate": baseline.loc[scope, "directional_hit_rate"],
            "lightgbm_directional_hit_rate": lightgbm.loc[scope, "directional_hit_rate"],
        })
    return pd.DataFrame(rows)


def run() -> None:
    if not CANONICAL_OOS_PATH.exists():
        raise FileNotFoundError(
            f"{CANONICAL_OOS_PATH} not found -- run E2-S4/generate_oos_predictions.py first"
        )
    if not BASELINE_OOS_PATH.exists():
        raise FileNotFoundError(
            f"{BASELINE_OOS_PATH} not found -- run E2-S1/run_baseline.py first"
        )

    baseline_df = load_baseline_predictions()
    lightgbm_df = load_lightgbm_predictions()
    assert_same_oos_rows(baseline_df, lightgbm_df)

    rows = evaluate_model(baseline_df, "baseline_zero") + evaluate_model(lightgbm_df, "lightgbm")
    results = pd.DataFrame(rows)
    comparison = build_comparison_table(results)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "regime_performance.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "regime_comparison.csv", index=False)

    summary = {
        "card": "E2-S5 [P0][Model] Evaluate Overall, Low-Vol & High-Vol Performance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_oos_predictions_path": str(CANONICAL_OOS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "canonical_oos_predictions_sha256": sha256_of(CANONICAL_OOS_PATH),
        "baseline_oos_predictions_path": str(BASELINE_OOS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "baseline_oos_predictions_sha256": sha256_of(BASELINE_OOS_PATH),
        "scopes": SCOPES,
        "models": MODELS,
        "directional_hit_rate_zero_prediction_convention": DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION,
        "no_cherry_picking_declaration": (
            "This table reports N, MAE, prediction_correlation and directional_hit_rate for "
            "every (model, scope) pair unconditionally -- baseline_zero and lightgbm, each "
            "across Overall/LowVol/HighVol -- regardless of which numbers look favorable. No "
            "metric or scope is omitted based on its value."
        ),
        "results": rows,
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "python": platform.python_version(),
    }
    (OUTPUT_DIR / "regime_performance_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"Wrote regime performance table ({len(results)} rows) to {OUTPUT_DIR}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    run()
