"""
E2-S5 [P0][Model] Evaluate Overall, Low-Vol & High-Vol Performance -- verification suite.

Run:
    python -m pytest E2-S5_Evaluate_Overall_LowVol_HighVol_Performance/tests/test_evaluate_regime_performance.py -v

Owner: Model. Review: QA/QC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

E2_S5_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = E2_S5_DIR.parent
sys.path.insert(0, str(E2_S5_DIR))
sys.path.insert(0, str(REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"))

from evaluate_regime_performance import (  # noqa: E402
    CANONICAL_OOS_PATH,
    MODELS,
    OUTPUT_DIR,
    REGIMES,
    SCOPES,
    assert_same_oos_rows,
    build_comparison_table,
    evaluate_model,
    load_baseline_predictions,
    load_lightgbm_predictions,
    run,
    scope_mask,
)

pytestmark = pytest.mark.skipif(
    not CANONICAL_OOS_PATH.exists(),
    reason="run E2-S4/generate_oos_predictions.py first to produce results/oos_predictions.csv",
)


@pytest.fixture(scope="module")
def baseline_df() -> pd.DataFrame:
    return load_baseline_predictions()


@pytest.fixture(scope="module")
def lightgbm_df() -> pd.DataFrame:
    return load_lightgbm_predictions()


@pytest.fixture(scope="module")
def results(baseline_df, lightgbm_df) -> pd.DataFrame:
    rows = evaluate_model(baseline_df, "baseline_zero") + evaluate_model(lightgbm_df, "lightgbm")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Acceptance: baseline and LightGBM comparison is fair
# --------------------------------------------------------------------------

def test_baseline_and_lightgbm_share_the_same_oos_rows(baseline_df, lightgbm_df):
    assert_same_oos_rows(baseline_df, lightgbm_df)  # must not raise


def test_fairness_check_catches_mismatched_dates(baseline_df, lightgbm_df):
    corrupted = lightgbm_df.copy()
    corrupted.loc[corrupted.index[0], "Date"] = pd.Timestamp("1999-01-01")
    with pytest.raises(AssertionError, match="same OOS dates"):
        assert_same_oos_rows(baseline_df, corrupted)


def test_fairness_check_catches_mismatched_targets(baseline_df, lightgbm_df):
    corrupted = lightgbm_df.copy()
    corrupted["y_true"] = corrupted["y_true"] + 1.0
    with pytest.raises(AssertionError, match="actual_return_5d"):
        assert_same_oos_rows(baseline_df, corrupted)


def test_fairness_check_catches_mismatched_regime_labels(baseline_df, lightgbm_df):
    corrupted = lightgbm_df.copy()
    corrupted.loc[corrupted.index[0], "regime"] = (
        "HighVol" if corrupted.loc[corrupted.index[0], "regime"] == "LowVol" else "LowVol"
    )
    with pytest.raises(AssertionError, match="regime label"):
        assert_same_oos_rows(baseline_df, corrupted)


def test_fairness_check_catches_duplicate_dates(baseline_df, lightgbm_df):
    corrupted = pd.concat([lightgbm_df, lightgbm_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate dates"):
        assert_same_oos_rows(baseline_df, corrupted)


# --------------------------------------------------------------------------
# Acceptance: same OOS table and metric code across regimes
# --------------------------------------------------------------------------

def test_overall_count_equals_sum_of_regime_counts(results):
    overall_n = results.loc[(results["model"] == "lightgbm") & (results["scope"] == "Overall"), "n"].iloc[0]
    regime_n_sum = results.loc[
        (results["model"] == "lightgbm") & (results["scope"].isin(REGIMES)), "n"
    ].sum()
    assert overall_n == regime_n_sum


def test_scope_mask_partitions_rows_with_no_overlap_or_gap(lightgbm_df):
    masks = [scope_mask(lightgbm_df, r) for r in REGIMES]
    combined = np.zeros(len(lightgbm_df), dtype=bool)
    for m in masks:
        assert not (combined & m.to_numpy()).any(), "regime scopes must not overlap"
        combined |= m.to_numpy()
    assert combined.all(), "every row must belong to exactly one named regime"


def test_scope_mask_rejects_unknown_scope(lightgbm_df):
    with pytest.raises(ValueError):
        scope_mask(lightgbm_df, "MediumVol")


# --------------------------------------------------------------------------
# Acceptance: sample counts always reported
# --------------------------------------------------------------------------

def test_every_row_reports_n(results):
    assert "n" in results.columns
    assert (results["n"] > 0).all()


# --------------------------------------------------------------------------
# Acceptance: undefined correlation/zero-sign conventions handled explicitly
# --------------------------------------------------------------------------

def test_baseline_correlation_and_hit_rate_are_nan_in_every_scope(results):
    baseline_rows = results[results["model"] == "baseline_zero"]
    assert baseline_rows["prediction_correlation"].isna().all()
    assert baseline_rows["directional_hit_rate"].isna().all()
    # never silently 0.0 or 0.5 -- distinguishing NaN from a real bad score
    assert not (baseline_rows["directional_hit_rate"] == 0.0).any()
    assert not (baseline_rows["directional_hit_rate"] == 0.5).any()


# --------------------------------------------------------------------------
# Edge case: HighVol sample much smaller
# --------------------------------------------------------------------------

def test_highvol_sample_is_smaller_but_still_reported(results):
    lowvol_n = results.loc[(results["model"] == "lightgbm") & (results["scope"] == "LowVol"), "n"].iloc[0]
    highvol_n = results.loc[(results["model"] == "lightgbm") & (results["scope"] == "HighVol"), "n"].iloc[0]
    assert highvol_n < lowvol_n
    assert highvol_n > 0  # small, not hidden or dropped


# --------------------------------------------------------------------------
# Edge case: crisis period dominates metric -- date ranges must be visible
# --------------------------------------------------------------------------

def test_every_scope_reports_a_date_range(results):
    assert results["date_start"].notna().all()
    assert results["date_end"].notna().all()
    assert (results["date_start"] <= results["date_end"]).all()


# --------------------------------------------------------------------------
# Edge case: near-zero or constant predictions
# --------------------------------------------------------------------------

def test_baseline_predictions_are_flagged_nearly_constant_in_every_scope(results):
    baseline_rows = results[results["model"] == "baseline_zero"]
    assert baseline_rows["predictions_nearly_constant"].all()


def test_lightgbm_predictions_are_not_flagged_nearly_constant(results):
    lightgbm_rows = results[results["model"] == "lightgbm"]
    assert not lightgbm_rows["predictions_nearly_constant"].any()


# --------------------------------------------------------------------------
# Edge case: report all metrics -- no cherry-picking
# --------------------------------------------------------------------------

def test_output_has_every_model_scope_combination(results):
    assert len(results) == len(MODELS) * len(SCOPES)
    combos = set(zip(results["model"], results["scope"]))
    expected = {(m, s) for m in MODELS for s in SCOPES}
    assert combos == expected


def test_output_carries_every_required_metric_column(results):
    required = {"model", "scope", "n", "mae", "prediction_correlation", "directional_hit_rate"}
    assert required.issubset(results.columns)
    # no metric silently dropped even where it is NaN
    assert results["mae"].notna().all()


# --------------------------------------------------------------------------
# Full run: produces the canonical table + comparison + summary artifacts
# --------------------------------------------------------------------------

def test_run_produces_output_files():
    run()

    regime_performance = pd.read_csv(OUTPUT_DIR / "regime_performance.csv")
    comparison = pd.read_csv(OUTPUT_DIR / "regime_comparison.csv")
    summary = json.loads((OUTPUT_DIR / "regime_performance_summary.json").read_text())

    assert len(regime_performance) == len(MODELS) * len(SCOPES)
    assert len(comparison) == len(SCOPES)
    assert summary["scopes"] == SCOPES
    assert summary["models"] == MODELS
    assert len(summary["results"]) == len(MODELS) * len(SCOPES)


def test_comparison_table_mae_improvement_is_arithmetically_consistent(results):
    comparison = build_comparison_table(results)
    computed = comparison["baseline_zero_mae"] - comparison["lightgbm_mae"]
    assert computed.to_numpy() == pytest.approx(comparison["mae_improvement_over_baseline"].to_numpy())
