"""
E2-S4 [P0][Model] Generate Canonical OOS Prediction Table -- verification suite.

Run:
    python -m pytest E2-S4_Generate_Canonical_OOS_Prediction_Table/tests/test_generate_oos_predictions.py -v

Owner: Model. Review: Pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

E2_S4_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = E2_S4_DIR.parent
sys.path.insert(0, str(E2_S4_DIR))
sys.path.insert(0, str(REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"))
sys.path.insert(0, str(REPO_ROOT / "E2-S3_Leakage_Safe_Walk_Forward_Validation"))

import generate_oos_predictions  # noqa: E402
from generate_oos_predictions import (  # noqa: E402
    CANONICAL_PATH,
    OUTPUT_COLUMNS,
    OUTPUT_PATH,
    RESULTS_DIR,
    SOURCE_COLUMNS,
    SOURCE_PREDICTIONS_PATH,
    SOURCE_SUMMARY_PATH,
    build_canonical_table,
    run,
    validate_every_row_is_genuine_oos,
    validate_source_columns,
)
from splits import HORIZON_TRADING_DAYS, MIN_TRAIN_SIZE, N_FOLDS, purged_walk_forward_splits  # noqa: E402

pytestmark = pytest.mark.skipif(
    not SOURCE_PREDICTIONS_PATH.exists(),
    reason="run E2-S2/train_lightgbm.py first to produce lightgbm_oos_predictions.csv",
)


@pytest.fixture(scope="module")
def source_df() -> pd.DataFrame:
    return pd.read_csv(SOURCE_PREDICTIONS_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def canonical_dates() -> pd.Series:
    return pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])["Date"]


@pytest.fixture(scope="module")
def table(source_df) -> pd.DataFrame:
    return build_canonical_table(source_df)


# --------------------------------------------------------------------------
# Acceptance: exact canonical schema
# --------------------------------------------------------------------------

def test_output_columns_are_exactly_the_canonical_schema(table):
    assert list(table.columns) == ["Date", "prediction", "actual_return_5d", "regime", "fold_id"]


def test_column_semantics_match_source(source_df, table):
    assert table["prediction"].to_numpy() == pytest.approx(source_df.sort_values("Date")["y_pred"].to_numpy())
    assert table["actual_return_5d"].to_numpy() == pytest.approx(source_df.sort_values("Date")["y_true"].to_numpy())


# --------------------------------------------------------------------------
# Acceptance: every row is genuine OOS; no train predictions
# --------------------------------------------------------------------------

def test_every_source_row_traces_to_its_claimed_fold(source_df, canonical_dates):
    validate_every_row_is_genuine_oos(source_df, canonical_dates)  # must not raise


def test_row_count_matches_sum_of_fold_test_sizes(table, canonical_dates):
    folds = purged_walk_forward_splits(canonical_dates, n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE, horizon=HORIZON_TRADING_DAYS)
    expected_n_rows = sum(len(f.test_idx) for f in folds)
    assert len(table) == expected_n_rows


def test_genuine_oos_check_catches_a_row_relabeled_to_the_wrong_fold(source_df, canonical_dates):
    corrupted = source_df.copy()
    corrupted.loc[corrupted.index[0], "fold"] = corrupted.loc[corrupted.index[0], "fold"] + 1
    with pytest.raises(AssertionError):
        validate_every_row_is_genuine_oos(corrupted, canonical_dates)


# --------------------------------------------------------------------------
# Acceptance: one prediction per date; date order preserved
# --------------------------------------------------------------------------

def test_one_prediction_per_date(table):
    assert table["Date"].is_unique


def test_date_order_is_ascending(table):
    assert table["Date"].is_monotonic_increasing


# --------------------------------------------------------------------------
# Acceptance: no NaN in prediction/actual/regime
# --------------------------------------------------------------------------

def test_no_nan_in_required_columns(table):
    assert table[["prediction", "actual_return_5d", "regime"]].isna().sum().sum() == 0


# --------------------------------------------------------------------------
# Edge case: duplicate dates from overlapping test folds
# --------------------------------------------------------------------------

def test_duplicate_dates_are_rejected(source_df):
    corrupted = pd.concat([source_df, source_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate dates"):
        build_canonical_table(corrupted)


# --------------------------------------------------------------------------
# Edge case: concatenation/index mismatch (fold column pointing at the
# wrong dates after a bad merge/concat)
# --------------------------------------------------------------------------

def test_concatenation_mismatch_is_rejected(source_df, canonical_dates):
    corrupted = source_df.copy()
    # Simulate a concat/index bug: swap two rows' Date values so their fold
    # label no longer matches the date's true fold.
    i, j = 0, len(corrupted) - 1
    corrupted.loc[i, "Date"], corrupted.loc[j, "Date"] = corrupted.loc[j, "Date"], corrupted.loc[i, "Date"]
    with pytest.raises(AssertionError):
        validate_every_row_is_genuine_oos(corrupted, canonical_dates)


# --------------------------------------------------------------------------
# Edge case: shuffled row order
# --------------------------------------------------------------------------

def test_shuffled_source_rows_are_still_published_in_date_order(source_df):
    shuffled = source_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    table_from_shuffled = build_canonical_table(shuffled)
    assert table_from_shuffled["Date"].is_monotonic_increasing


# --------------------------------------------------------------------------
# Edge case: accidental in-sample diagnostics appended to canonical file
# --------------------------------------------------------------------------

def test_unexpected_extra_column_is_rejected(source_df):
    corrupted = source_df.copy()
    corrupted["train_mae_diagnostic_only"] = 0.01
    with pytest.raises(ValueError, match="unexpected"):
        validate_source_columns(corrupted)


def test_source_columns_constant_matches_actual_E2_S2_output(source_df):
    assert list(source_df.columns) == SOURCE_COLUMNS


# --------------------------------------------------------------------------
# Edge case: lightgbm_summary.json recorded against a stale/different dataset
# --------------------------------------------------------------------------

def test_run_raises_if_source_summary_was_generated_from_a_different_dataset(tmp_path, monkeypatch):
    stale_summary = json.loads(SOURCE_SUMMARY_PATH.read_text())
    stale_summary["canonical_dataset_sha256"] = "0" * 64
    stale_path = tmp_path / "lightgbm_summary.json"
    stale_path.write_text(json.dumps(stale_summary))

    monkeypatch.setattr(generate_oos_predictions, "SOURCE_SUMMARY_PATH", stale_path)
    with pytest.raises(ValueError, match="different canonical dataset"):
        run()


# --------------------------------------------------------------------------
# Full run: produces results/oos_predictions.csv + manifest
# --------------------------------------------------------------------------

def test_run_produces_canonical_file_and_manifest():
    run()

    published = pd.read_csv(OUTPUT_PATH, parse_dates=["Date"])
    manifest = json.loads((RESULTS_DIR / "oos_predictions_manifest.json").read_text())

    assert list(published.columns) == OUTPUT_COLUMNS
    assert published["Date"].is_monotonic_increasing
    assert published["Date"].is_unique
    assert published[["prediction", "actual_return_5d", "regime"]].isna().sum().sum() == 0

    assert manifest["n_rows"] == len(published)
    assert manifest["n_unique_dates"] == len(published)
    assert manifest["output_columns"] == OUTPUT_COLUMNS
