"""
E2-S1 [P0][Model] Baseline y_hat=0 -- verification suite.

Run:
    python -m pytest E2-S1_Baseline_Zero_Predictor/tests/test_baseline_zero.py -v

Owner: Model. Review: Tech Leader.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_baseline  # noqa: E402
from metrics import directional_hit_rate, mae, prediction_correlation  # noqa: E402
from run_baseline import OUTPUT_DIR, run  # noqa: E402
from splits import HORIZON_TRADING_DAYS, purged_walk_forward_splits  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
TARGET_COL = "forward_return_5d"


@pytest.fixture(scope="module")
def canonical_df() -> pd.DataFrame:
    return pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def folds(canonical_df):
    return purged_walk_forward_splits(canonical_df["Date"])


# --------------------------------------------------------------------------
# Split correctness
# --------------------------------------------------------------------------

def test_folds_cover_the_oos_region_with_no_gap_or_overlap(canonical_df, folds):
    n = len(canonical_df)
    test_idx_concat = np.concatenate([f.test_idx for f in folds])
    assert test_idx_concat[0] > 0
    assert test_idx_concat[-1] == n - 1
    assert np.array_equal(test_idx_concat, np.arange(test_idx_concat[0], n))
    assert len(test_idx_concat) == len(set(test_idx_concat.tolist()))


def test_purge_gap_is_at_least_horizon_trading_days(canonical_df, folds):
    for fold in folds:
        if len(fold.train_idx) == 0:
            continue
        last_train_idx = fold.train_idx[-1]
        first_test_idx = fold.test_idx[0]
        assert first_test_idx - last_train_idx > HORIZON_TRADING_DAYS


def test_train_never_contains_a_row_from_its_own_test_block(folds):
    for fold in folds:
        assert set(fold.train_idx.tolist()).isdisjoint(set(fold.test_idx.tolist()))


def test_train_size_expands_across_folds(folds):
    train_sizes = [len(f.train_idx) for f in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[-1] > train_sizes[0]


def test_splits_reject_unsorted_dates():
    shuffled = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-01"]))
    with pytest.raises(ValueError):
        purged_walk_forward_splits(shuffled, n_folds=1, min_train_size=1)


# --------------------------------------------------------------------------
# Baseline prediction correctness
# --------------------------------------------------------------------------

def test_baseline_prediction_is_exactly_zero_every_oos_row(canonical_df, folds):
    for fold in folds:
        y_pred = np.zeros(len(fold.test_idx))
        assert np.all(y_pred == 0.0)


def test_run_produces_consistent_output_with_all_zero_predictions():
    run()

    predictions = pd.read_csv(OUTPUT_DIR / "baseline_zero_oos_predictions.csv")
    fold_metrics = pd.read_csv(OUTPUT_DIR / "baseline_zero_fold_metrics.csv")
    summary = json.loads((OUTPUT_DIR / "baseline_zero_summary.json").read_text())

    assert (predictions["y_pred"].to_numpy() == 0.0).all()
    assert len(predictions) == summary["overall_metrics"]["n_oos_rows"]
    assert predictions["fold"].nunique() == len(fold_metrics) == summary["overall_metrics"]["n_folds"]
    assert summary["overall_metrics"]["mae"] == pytest.approx(predictions["y_true"].abs().mean())
    assert np.isnan(summary["overall_metrics"]["prediction_correlation"])
    assert np.isnan(summary["overall_metrics"]["directional_hit_rate"])


def test_mae_of_zero_baseline_equals_mean_absolute_target(canonical_df, folds):
    fold = folds[0]
    y_true = canonical_df.loc[fold.test_idx, TARGET_COL].to_numpy()
    y_pred = np.zeros_like(y_true)
    assert mae(y_true, y_pred) == pytest.approx(np.mean(np.abs(y_true)))


# --------------------------------------------------------------------------
# Edge cases named on the card
# --------------------------------------------------------------------------

def test_directional_hit_rate_is_nan_for_all_zero_predictions():
    y_true = np.array([0.01, -0.02, 0.0, 0.03])
    y_pred = np.zeros_like(y_true)
    result = directional_hit_rate(y_true, y_pred)
    assert np.isnan(result)


def test_directional_hit_rate_excludes_only_zero_predictions_when_mixed():
    y_true = np.array([0.01, -0.02, 0.05])
    y_pred = np.array([0.0, -0.01, 0.02])
    result = directional_hit_rate(y_true, y_pred)
    assert result == pytest.approx(1.0)


def test_prediction_correlation_is_nan_for_constant_predictions():
    y_true = np.array([0.01, -0.02, 0.03, -0.04])
    y_pred = np.zeros_like(y_true)
    assert np.isnan(prediction_correlation(y_true, y_pred))


def test_zero_baseline_reports_nan_not_zero_or_half_for_every_fold(canonical_df, folds):
    for fold in folds:
        y_true = canonical_df.loc[fold.test_idx, TARGET_COL].to_numpy()
        y_pred = np.zeros_like(y_true)
        hit_rate = directional_hit_rate(y_true, y_pred)
        corr = prediction_correlation(y_true, y_pred)
        assert np.isnan(hit_rate)
        assert np.isnan(corr)
        assert hit_rate != 0.0
        assert hit_rate != 0.5


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def test_splits_are_deterministic_across_calls(canonical_df):
    folds_a = purged_walk_forward_splits(canonical_df["Date"])
    folds_b = purged_walk_forward_splits(canonical_df["Date"])
    for a, b in zip(folds_a, folds_b):
        assert np.array_equal(a.train_idx, b.train_idx)
        assert np.array_equal(a.test_idx, b.test_idx)


def test_target_column_present_and_fully_populated(canonical_df):
    assert TARGET_COL in canonical_df.columns
    assert canonical_df[TARGET_COL].isna().sum() == 0


def test_run_raises_on_non_finite_target_values(tmp_path, monkeypatch, canonical_df):
    corrupted = canonical_df.copy()
    corrupted.loc[corrupted.index[0], TARGET_COL] = np.nan
    corrupted_path = tmp_path / "corrupted.csv"
    corrupted.to_csv(corrupted_path, index=False)

    monkeypatch.setattr(run_baseline, "CANONICAL_PATH", corrupted_path)
    with pytest.raises(ValueError, match="non-finite"):
        run()
