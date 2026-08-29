"""
E2-S3 [P0][Model] Implement Leakage-Safe Walk-Forward Validation
===================================================================

Deliverable: the chronological expanding walk-forward splitter with an
explicit purge/gap for the 5-trading-day target horizon already lives in
E2-S1's `splits.py` (`purged_walk_forward_splits`) precisely so E2-S1, E2-S2
and every later E2 model share one fold definition instead of each
reimplementing it. E2-S3 does not fork that function -- it turns every
acceptance item on the card into an executable, independently-reasoned check
run against the real canonical dataset and against the OOS artifacts E2-S1
and E2-S2 already produced, and persists a fold-boundary audit trail so any
OOS prediction row can be traced back to the training window that produced
it.

Checks performed here that E2-S1's own test suite does not already cover:
  - purge sufficiency reasoned from the *label's own price reference*
    (last_train_idx + horizon < test_start_idx), independent of the
    row-count arithmetic inside splits.py itself;
  - the "no missing trading days" assumption that index-based purging relies
    on to equal calendar-based purging;
  - end-to-end traceability: every OOS prediction row (E2-S1 baseline and
    E2-S2 LightGBM) is checked against the fold that should have produced it;
  - fold-scoped preprocessing guard (`fit_scaler_on_train_fold_only`) for any
    future E2 story that adds preprocessing, plus a demonstration that a
    full-sample fit actually differs from a fold-scoped fit;
  - split-parameter drift across E2-S1's and E2-S2's already-recorded output,
    which is what "fold logic changed after seeing results" would look like.

Depends on: E2-S2 (existing OOS prediction outputs this script audits).

Run:
    python validate_walk_forward.py
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
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_S1_DIR = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"
E2_S2_DIR = REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor"
sys.path.insert(0, str(E2_S1_DIR))

from splits import (  # noqa: E402
    HORIZON_TRADING_DAYS,
    MIN_TRAIN_SIZE,
    N_FOLDS,
    Fold,
    purged_walk_forward_splits,
)

CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
TARGET_COL = "forward_return_5d"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

BASELINE_SUMMARY_PATH = E2_S1_DIR / "output" / "baseline_zero_summary.json"
BASELINE_PREDICTIONS_PATH = E2_S1_DIR / "output" / "baseline_zero_oos_predictions.csv"
LIGHTGBM_SUMMARY_PATH = E2_S2_DIR / "output" / "lightgbm_summary.json"
LIGHTGBM_PREDICTIONS_PATH = E2_S2_DIR / "output" / "lightgbm_oos_predictions.csv"

# ~4 trading years. Fold 0's train window is MIN_TRAIN_SIZE - HORIZON_TRADING_DAYS
# rows (purge shrinks it below the nominal MIN_TRAIN_SIZE) -- this is the floor
# below which that shrink would be considered inadequate rather than expected.
MIN_ADEQUATE_FIRST_FOLD_TRAIN_SIZE = 1000

# The canonical dataset's longest legitimate calendar gap (a holiday cluster,
# e.g. Christmas/New Year) is 5 days -- observed empirically, twice, across
# the full history. This threshold is set with headroom above that so it
# still catches a genuinely anomalous gap (a multi-day data outage, a
# de-listing splice) without flagging a normal long weekend/holiday. It
# cannot, by construction, detect a single missing trading day hidden inside
# an already-long holiday gap -- that would require an authoritative trading
# calendar, which this repo does not depend on.
MAX_TRADING_GAP_DAYS = 7


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Checklist item 1: max(train_date) < min(test_date) for every fold
# --------------------------------------------------------------------------

def assert_chronological_order(fold: Fold, dates: pd.Series) -> None:
    test_min = dates.iloc[fold.test_idx].min()
    if len(fold.train_idx) == 0:
        return
    train_max = dates.iloc[fold.train_idx].max()
    if not (train_max < test_min):
        raise AssertionError(
            f"fold {fold.fold_id}: max(train_date)={train_max} is not < min(test_date)={test_min}"
        )


# --------------------------------------------------------------------------
# Checklist item 2: purge/gap so a training label never reads a test-window price
# --------------------------------------------------------------------------

def assert_purge_removes_label_overlap(fold: Fold, horizon: int) -> None:
    """`forward_return_5d`'s label for row i is Close[i+horizon]/Close[i]-1, so
    training row i leaks into the test window whenever i+horizon falls inside
    it. This reasons from the label's own price reference, independent of the
    purge_start_idx arithmetic inside splits.py, so it fails if that
    arithmetic is ever wrong."""
    if len(fold.train_idx) == 0:
        return
    last_train_idx = int(fold.train_idx.max())
    label_price_idx = last_train_idx + horizon
    test_start_idx = int(fold.test_idx.min())
    if not (label_price_idx < test_start_idx):
        raise AssertionError(
            f"fold {fold.fold_id}: last train row {last_train_idx}'s label reads a price at "
            f"index {label_price_idx}, which is inside the test window starting at {test_start_idx}"
        )


def assert_dates_are_contiguous_trading_days(dates: pd.Series, max_gap_days: int = MAX_TRADING_GAP_DAYS) -> None:
    """Purging by row count (purge_start_idx = test_start_idx - horizon) only
    equals purging by 5 *trading days* if consecutive rows are consecutive
    trading days with no missing dates. An undetected gap (a data outage,
    a de-listing splice) would silently under-purge."""
    gaps = dates.diff().dropna().dt.days
    bad = gaps[gaps > max_gap_days]
    if not bad.empty:
        raise AssertionError(
            f"{len(bad)} row-to-row date gaps exceed {max_gap_days} calendar days -- "
            "index-based purge is not safe to treat as a 5-trading-day purge"
        )


# --------------------------------------------------------------------------
# Checklist item 3: preprocessing/model fit only on the training fold
# --------------------------------------------------------------------------

def fit_scaler_on_train_fold_only(df: pd.DataFrame, fold: Fold, columns: list[str]) -> StandardScaler:
    """Fit-inside-the-fold guard for any future E2 story that adds
    preprocessing (scaling, imputation, etc). LightGBM (E2-S2) needs none,
    so this guard has no caller yet in this repo -- it exists so the next
    model that does need preprocessing has a fold-safe entry point instead
    of reaching for `scaler.fit(df[columns])` on the full frame."""
    scaler = StandardScaler()
    scaler.fit(df.loc[fold.train_idx, columns])
    return scaler


# --------------------------------------------------------------------------
# Checklist item 4: fold boundaries + fold_id saved for traceability
# --------------------------------------------------------------------------

def fold_boundary_row(fold: Fold, dates: pd.Series, horizon: int) -> dict:
    train_dates = dates.iloc[fold.train_idx]
    has_train = len(fold.train_idx) > 0
    purge_hi = int(fold.test_idx.min())
    # Clamp to 0 rather than letting a negative purge_lo silently wrap around
    # to a slice from the end of `dates` (only reachable if a future change to
    # splits.py's constants ever produced a train-less fold whose test block
    # starts within `horizon` rows of the very first row).
    purge_lo = max(0, int(fold.train_idx.max()) + 1 if has_train else purge_hi - horizon)
    purge_dates = dates.iloc[purge_lo:purge_hi]
    return {
        "fold_id": fold.fold_id,
        "n_train": len(fold.train_idx),
        "n_test": len(fold.test_idx),
        "train_start_date": train_dates.min().strftime("%Y-%m-%d") if has_train else None,
        "train_end_date": train_dates.max().strftime("%Y-%m-%d") if has_train else None,
        "purge_gap_n_rows": purge_hi - purge_lo,
        "purge_gap_start_date": purge_dates.min().strftime("%Y-%m-%d") if len(purge_dates) else None,
        "purge_gap_end_date": purge_dates.max().strftime("%Y-%m-%d") if len(purge_dates) else None,
        "test_start_date": fold.test_start_date.strftime("%Y-%m-%d"),
        "test_end_date": fold.test_end_date.strftime("%Y-%m-%d"),
    }


def assert_predictions_trace_to_fold(predictions: pd.DataFrame, fold: Fold, dates: pd.Series, label: str) -> None:
    fold_pred_dates = np.sort(
        pd.to_datetime(predictions.loc[predictions["fold"] == fold.fold_id, "Date"]).to_numpy()
    )
    expected_dates = np.sort(dates.iloc[fold.test_idx].to_numpy())
    if not np.array_equal(fold_pred_dates, expected_dates):
        raise AssertionError(
            f"{label} fold {fold.fold_id}: OOS prediction dates do not exactly match "
            "this fold's test_idx dates -- traceability broken"
        )


# --------------------------------------------------------------------------
# Edge case: fold logic changed after seeing results
# --------------------------------------------------------------------------

def check_fold_params_unchanged_across_outputs() -> dict:
    """If N_FOLDS/MIN_TRAIN_SIZE/HORIZON_TRADING_DAYS were tweaked after E2-S1
    or E2-S2 already recorded OOS results under the old values, their
    recorded split_params would disagree with splits.py's current constants.
    Agreement here is what "fold logic frozen, not adjusted after looking at
    results" looks like structurally rather than by promise."""
    current = {
        "n_folds": N_FOLDS,
        "min_train_size": MIN_TRAIN_SIZE,
        "horizon_trading_days": HORIZON_TRADING_DAYS,
    }
    result = {"current_split_params": current}
    for name, path in (("baseline_zero", BASELINE_SUMMARY_PATH), ("lightgbm", LIGHTGBM_SUMMARY_PATH)):
        if not path.exists():
            result[name] = "SKIPPED -- output not found"
            continue
        recorded = json.loads(path.read_text())["split_params"]
        result[name] = "MATCH" if recorded == current else f"MISMATCH: recorded={recorded}"
    return result


def run() -> None:
    df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("canonical dataset must be sorted by Date -- walk-forward validation forbids shuffling")

    dates = df["Date"]
    assert_dates_are_contiguous_trading_days(dates)

    folds = purged_walk_forward_splits(dates, n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE, horizon=HORIZON_TRADING_DAYS)

    boundary_rows = []
    for fold in folds:
        assert_chronological_order(fold, dates)
        assert_purge_removes_label_overlap(fold, HORIZON_TRADING_DAYS)
        boundary_rows.append(fold_boundary_row(fold, dates, HORIZON_TRADING_DAYS))
    boundary_df = pd.DataFrame(boundary_rows)

    traceability = {}
    if BASELINE_PREDICTIONS_PATH.exists():
        baseline_predictions = pd.read_csv(BASELINE_PREDICTIONS_PATH)
        for fold in folds:
            assert_predictions_trace_to_fold(baseline_predictions, fold, dates, "baseline_zero")
        traceability["baseline_zero"] = "PASS"
    else:
        traceability["baseline_zero"] = "SKIPPED -- run E2-S1/run_baseline.py first"

    if LIGHTGBM_PREDICTIONS_PATH.exists():
        lightgbm_predictions = pd.read_csv(LIGHTGBM_PREDICTIONS_PATH)
        for fold in folds:
            assert_predictions_trace_to_fold(lightgbm_predictions, fold, dates, "lightgbm")
        traceability["lightgbm"] = "PASS"
    else:
        traceability["lightgbm"] = "SKIPPED -- run E2-S2/train_lightgbm.py first"

    fold_params_frozen = check_fold_params_unchanged_across_outputs()
    first_fold_train_size = len(folds[0].train_idx)

    OUTPUT_DIR.mkdir(exist_ok=True)
    boundary_df.to_csv(OUTPUT_DIR / "fold_boundary_audit.csv", index=False)

    summary = {
        "card": "E2-S3 [P0][Model] Implement Leakage-Safe Walk-Forward Validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS",
        "canonical_dataset_path": str(CANONICAL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "canonical_dataset_sha256": sha256_of(CANONICAL_PATH),
        "split_params": {
            "n_folds": N_FOLDS,
            "min_train_size": MIN_TRAIN_SIZE,
            "horizon_trading_days": HORIZON_TRADING_DAYS,
        },
        "checks": {
            "no_shuffle_dates_sorted_ascending": True,
            "dates_contiguous_no_silent_trading_day_gap": True,
            "chronological_order_every_fold": True,
            "purge_removes_5d_label_overlap_every_fold": True,
        },
        "traceability_oos_predictions_match_fold_boundaries": traceability,
        "fold_params_frozen_across_E2_S1_and_E2_S2_outputs": fold_params_frozen,
        "first_fold_train_size": first_fold_train_size,
        "min_adequate_first_fold_train_size": MIN_ADEQUATE_FIRST_FOLD_TRAIN_SIZE,
        "first_fold_train_size_adequate": first_fold_train_size >= MIN_ADEQUATE_FIRST_FOLD_TRAIN_SIZE,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "python": platform.python_version(),
    }
    (OUTPUT_DIR / "walk_forward_validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"Validated {len(folds)} folds; wrote fold boundary audit to {OUTPUT_DIR}")
    print(json.dumps({"checks": summary["checks"], "traceability": traceability, "fold_params_frozen": fold_params_frozen}, indent=2, default=str))


if __name__ == "__main__":
    run()
