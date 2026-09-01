"""
E2-S3 [P0][Model] Leakage-Safe Walk-Forward Validation -- verification suite.

Run:
    python -m pytest E2-S3_Leakage_Safe_Walk_Forward_Validation/tests/test_walk_forward_validation.py -v

Owner: Model. Review: Pipeline + QA/QC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

E2_S3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = E2_S3_DIR.parent
sys.path.insert(0, str(E2_S3_DIR))
sys.path.insert(0, str(REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"))
sys.path.insert(0, str(REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor"))

from validate_walk_forward import (  # noqa: E402
    MIN_ADEQUATE_FIRST_FOLD_TRAIN_SIZE,
    OUTPUT_DIR,
    assert_chronological_order,
    assert_dates_are_contiguous_trading_days,
    assert_predictions_trace_to_fold,
    assert_purge_removes_label_overlap,
    check_fold_params_unchanged_across_outputs,
    compute_verdict,
    fit_scaler_on_train_fold_only,
    fold_boundary_row,
    run,
)
from splits import (  # noqa: E402
    HORIZON_TRADING_DAYS,
    MIN_TRAIN_SIZE,
    N_FOLDS,
    Fold,
    purged_walk_forward_splits,
)
from train_lightgbm import FEATURE_COLUMNS  # noqa: E402

CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
BASELINE_PREDICTIONS_PATH = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor" / "output" / "baseline_zero_oos_predictions.csv"
LIGHTGBM_PREDICTIONS_PATH = REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor" / "output" / "lightgbm_oos_predictions.csv"


@pytest.fixture(scope="module")
def canonical_df() -> pd.DataFrame:
    return pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def dates(canonical_df) -> pd.Series:
    return canonical_df["Date"]


@pytest.fixture(scope="module")
def folds(dates):
    return purged_walk_forward_splits(dates, n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE, horizon=HORIZON_TRADING_DAYS)


# --------------------------------------------------------------------------
# Acceptance: max(train_date) < min(test_date) for every fold
# --------------------------------------------------------------------------

def test_chronological_order_holds_for_every_real_fold(dates, folds):
    for fold in folds:
        assert_chronological_order(fold, dates)  # must not raise


def test_chronological_order_catches_an_inverted_fold(dates):
    bad_fold = Fold(
        fold_id=0,
        train_idx=np.arange(100, 200),
        test_idx=np.arange(0, 50),
        test_start_date=dates.iloc[0],
        test_end_date=dates.iloc[49],
    )
    with pytest.raises(AssertionError):
        assert_chronological_order(bad_fold, dates)


# --------------------------------------------------------------------------
# Acceptance / edge case: purge/gap removes 5D label overlap at the boundary
# --------------------------------------------------------------------------

def test_purge_removes_label_overlap_for_every_real_fold(folds):
    for fold in folds:
        assert_purge_removes_label_overlap(fold, HORIZON_TRADING_DAYS)  # must not raise


def test_purge_check_catches_an_under_purged_boundary():
    """Adversarial fold: train includes a row whose label reads a price
    inside the test window (train ends 3 rows before test start, but the
    horizon is 5) -- this is exactly the leak E4-S1's audit flagged as an
    E2 follow-up."""
    under_purged = Fold(
        fold_id=0,
        train_idx=np.arange(0, 997),   # last train row = 996
        test_idx=np.arange(1000, 1100),  # gap of only 3 rows < horizon (5)
        test_start_date=pd.Timestamp("2020-01-01"),
        test_end_date=pd.Timestamp("2020-06-01"),
    )
    with pytest.raises(AssertionError):
        assert_purge_removes_label_overlap(under_purged, HORIZON_TRADING_DAYS)


def test_purge_check_accepts_an_exactly_sufficient_boundary():
    exactly_purged = Fold(
        fold_id=0,
        train_idx=np.arange(0, 995),   # last train row = 994
        test_idx=np.arange(1000, 1100),  # gap of exactly horizon (5)
        test_start_date=pd.Timestamp("2020-01-01"),
        test_end_date=pd.Timestamp("2020-06-01"),
    )
    assert_purge_removes_label_overlap(exactly_purged, HORIZON_TRADING_DAYS)  # must not raise


def test_fold_boundary_row_clamps_purge_start_instead_of_wrapping_negative(dates):
    """A train-less fold whose test block starts within `horizon` rows of the
    very first row would otherwise compute a negative purge_lo and let
    `dates.iloc[negative:purge_hi]` silently wrap around to the tail of the
    series. Not reachable with today's constants, but the code path is
    generic, so it's tested directly with a synthetic fold."""
    train_less_fold = Fold(
        fold_id=0,
        train_idx=np.array([], dtype=int),
        test_idx=np.arange(2, 50),
        test_start_date=dates.iloc[2],
        test_end_date=dates.iloc[49],
    )
    row = fold_boundary_row(train_less_fold, dates, horizon=HORIZON_TRADING_DAYS)
    assert row["purge_gap_start_date"] == dates.iloc[0].strftime("%Y-%m-%d")


def test_dates_are_contiguous_trading_days(dates):
    assert_dates_are_contiguous_trading_days(dates)  # must not raise -- real calendar has no silent gaps


def test_contiguous_check_catches_a_silent_gap():
    gappy = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-03-01"]))
    with pytest.raises(AssertionError):
        assert_dates_are_contiguous_trading_days(gappy, max_gap_days=5)


# --------------------------------------------------------------------------
# Edge case: too-small early train window
# --------------------------------------------------------------------------

def test_first_fold_train_window_shrinks_by_the_purge_but_stays_adequate(folds):
    first_fold_train_size = len(folds[0].train_idx)
    assert first_fold_train_size == MIN_TRAIN_SIZE - HORIZON_TRADING_DAYS
    assert first_fold_train_size >= MIN_ADEQUATE_FIRST_FOLD_TRAIN_SIZE


def test_splits_raise_when_min_train_size_leaves_no_room_for_any_test_block():
    tiny_dates = pd.Series(pd.bdate_range("2020-01-01", periods=10))
    with pytest.raises(ValueError):
        purged_walk_forward_splits(tiny_dates, n_folds=3, min_train_size=20)


# --------------------------------------------------------------------------
# Edge case: final partial fold absorbs the remainder, drops nothing
# --------------------------------------------------------------------------

def test_final_fold_absorbs_the_remainder_without_dropping_rows(canonical_df, folds):
    n = len(canonical_df)
    assert folds[-1].test_idx[-1] == n - 1

    test_sizes = [len(f.test_idx) for f in folds]
    # every non-final fold has the same (floor-divided) test block size;
    # the final fold gets the remainder, so it is >= the others.
    assert len(set(test_sizes[:-1])) == 1
    assert test_sizes[-1] >= test_sizes[0]

    all_test_idx = np.concatenate([f.test_idx for f in folds])
    assert len(all_test_idx) == len(set(all_test_idx.tolist())), "no test row is scored by more than one fold"


# --------------------------------------------------------------------------
# Edge case: full-sample preprocessing vs fold-scoped preprocessing
# --------------------------------------------------------------------------

def test_fold_scoped_scaler_is_fit_only_on_train_rows(canonical_df, folds):
    fold = folds[0]
    scaler = fit_scaler_on_train_fold_only(canonical_df, fold, FEATURE_COLUMNS)
    manual_mean = canonical_df.loc[fold.train_idx, FEATURE_COLUMNS].mean().to_numpy()
    assert scaler.mean_ == pytest.approx(manual_mean)


def test_fold_scoped_scaler_differs_from_a_full_sample_scaler(canonical_df, folds):
    """Demonstrates why 'preprocessing fitted before the split' is a real leak:
    a scaler fit on the whole frame sees future rows the fold-scoped scaler
    never does, so their statistics diverge."""
    fold = folds[0]
    fold_scaler = fit_scaler_on_train_fold_only(canonical_df, fold, FEATURE_COLUMNS)

    from sklearn.preprocessing import StandardScaler
    full_sample_scaler = StandardScaler().fit(canonical_df[FEATURE_COLUMNS])

    assert not np.allclose(fold_scaler.mean_, full_sample_scaler.mean_)


# --------------------------------------------------------------------------
# Edge case: fold logic changed after seeing results
# --------------------------------------------------------------------------

def test_fold_params_are_frozen_across_E2_S1_and_E2_S2_recorded_outputs():
    result = check_fold_params_unchanged_across_outputs()
    for key in ("baseline_zero", "lightgbm"):
        assert result[key] in ("MATCH",) or result[key].startswith("SKIPPED"), (
            f"{key} split_params disagree with current splits.py constants: {result[key]}"
        )


def test_compute_verdict_passes_when_every_soft_check_is_clean():
    fold_params_frozen = {"current_split_params": {}, "baseline_zero": "MATCH", "lightgbm": "SKIPPED -- output not found"}
    verdict, failures = compute_verdict(fold_params_frozen, first_fold_train_size_adequate=True)
    assert verdict == "PASS"
    assert failures == []


def test_compute_verdict_blocks_on_a_fold_params_mismatch():
    """This is exactly the scenario the verdict is supposed to catch: fold
    logic changed after E2-S1/E2-S2 already recorded OOS results under the
    old values. Before this fix, `verdict` stayed hardcoded "PASS" even here."""
    fold_params_frozen = {
        "current_split_params": {},
        "baseline_zero": "MISMATCH: recorded={'n_folds': 5}",
        "lightgbm": "MATCH",
    }
    verdict, failures = compute_verdict(fold_params_frozen, first_fold_train_size_adequate=True)
    assert verdict == "BLOCKED"
    assert failures == ["fold_params_frozen:baseline_zero"]


def test_compute_verdict_blocks_on_an_inadequate_first_fold():
    fold_params_frozen = {"current_split_params": {}, "baseline_zero": "MATCH", "lightgbm": "MATCH"}
    verdict, failures = compute_verdict(fold_params_frozen, first_fold_train_size_adequate=False)
    assert verdict == "BLOCKED"
    assert failures == ["first_fold_train_size_adequate"]


# --------------------------------------------------------------------------
# Acceptance: fold boundaries + fold_id saved, every OOS prediction traceable
# --------------------------------------------------------------------------

@pytest.mark.skipif(not BASELINE_PREDICTIONS_PATH.exists(), reason="run E2-S1/run_baseline.py first")
def test_baseline_oos_predictions_trace_to_documented_fold_boundaries(dates, folds):
    predictions = pd.read_csv(BASELINE_PREDICTIONS_PATH)
    for fold in folds:
        assert_predictions_trace_to_fold(predictions, fold, dates, "baseline_zero")  # must not raise


@pytest.mark.skipif(not LIGHTGBM_PREDICTIONS_PATH.exists(), reason="run E2-S2/train_lightgbm.py first")
def test_lightgbm_oos_predictions_trace_to_documented_fold_boundaries(dates, folds):
    predictions = pd.read_csv(LIGHTGBM_PREDICTIONS_PATH)
    for fold in folds:
        assert_predictions_trace_to_fold(predictions, fold, dates, "lightgbm")  # must not raise


def test_traceability_check_catches_a_mislabeled_prediction_row(dates, folds):
    fold = folds[0]
    predictions = pd.DataFrame({
        "fold": [fold.fold_id],
        "Date": [dates.iloc[fold.test_idx[0] - 1]],  # a purge-window date, not a test date
    })
    with pytest.raises(AssertionError):
        assert_predictions_trace_to_fold(predictions, fold, dates, "synthetic")


# --------------------------------------------------------------------------
# Full run: produces the fold boundary audit + summary artifacts
# --------------------------------------------------------------------------

def test_run_produces_fold_boundary_audit_and_summary():
    run()

    boundary_df = pd.read_csv(OUTPUT_DIR / "fold_boundary_audit.csv")
    summary = json.loads((OUTPUT_DIR / "walk_forward_validation_summary.json").read_text())

    assert len(boundary_df) == N_FOLDS
    assert list(boundary_df["fold_id"]) == list(range(N_FOLDS))
    assert (boundary_df["purge_gap_n_rows"] == HORIZON_TRADING_DAYS).all()

    assert summary["verdict"] == "PASS"
    assert summary["verdict_failures"] == []
    assert all(summary["checks"].values())
    assert summary["first_fold_train_size_adequate"] is True
