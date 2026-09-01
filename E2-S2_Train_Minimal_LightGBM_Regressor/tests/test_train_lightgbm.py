"""
E2-S2 [P0][Model] Train Minimal LightGBM Regressor -- verification suite.

Run:
    python -m pytest E2-S2_Train_Minimal_LightGBM_Regressor/tests/test_train_lightgbm.py -v

Owner: Model. Review: QA/QC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

E2_S2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = E2_S2_DIR.parent
sys.path.insert(0, str(E2_S2_DIR))
sys.path.insert(0, str(REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"))

import train_lightgbm  # noqa: E402
from train_lightgbm import (  # noqa: E402
    BASELINE_SUMMARY_PATH,
    FEATURE_COLUMNS,
    LIGHTGBM_PARAMS,
    OUTPUT_DIR,
    SEED,
    predictions_are_nearly_constant,
    run,
    validate_feature_columns,
    validate_no_nan_inf,
)
from splits import purged_walk_forward_splits  # noqa: E402

CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_dataset_manifest.json"
TARGET_COL = "forward_return_5d"


@pytest.fixture(scope="module")
def canonical_df() -> pd.DataFrame:
    return pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])


# --------------------------------------------------------------------------
# Acceptance: no model zoo, no broad search, seed recorded
# --------------------------------------------------------------------------

def test_single_fixed_configuration_not_a_search_space():
    for key, value in LIGHTGBM_PARAMS.items():
        assert not isinstance(value, (list, tuple, set)), (
            f"'{key}' looks like a search grid, not a fixed hyperparameter -- "
            "E2-S2 must run exactly one configuration"
        )


def test_seed_is_fixed_and_recorded_in_hyperparameters():
    assert LIGHTGBM_PARAMS["random_state"] == SEED


# --------------------------------------------------------------------------
# Edge case: deep trees overfit -- guarded by the fixed hyperparameters
# --------------------------------------------------------------------------

def test_hyperparameters_bound_tree_depth_and_leaves():
    assert LIGHTGBM_PARAMS["max_depth"] <= 6
    assert LIGHTGBM_PARAMS["num_leaves"] <= 31
    assert LIGHTGBM_PARAMS["min_child_samples"] >= 20


# --------------------------------------------------------------------------
# Edge case: NaN/inf input
# --------------------------------------------------------------------------

def test_feature_columns_match_frozen_manifest(canonical_df):
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert FEATURE_COLUMNS == manifest["feature_columns"]
    validate_feature_columns(canonical_df)  # must not raise


def test_validate_no_nan_inf_raises_on_nan(canonical_df):
    corrupted = canonical_df.copy()
    corrupted.loc[corrupted.index[0], FEATURE_COLUMNS[0]] = np.nan
    with pytest.raises(ValueError):
        validate_no_nan_inf(corrupted, FEATURE_COLUMNS, "test")


def test_validate_no_nan_inf_raises_on_inf(canonical_df):
    corrupted = canonical_df.copy()
    corrupted.loc[corrupted.index[0], FEATURE_COLUMNS[0]] = np.inf
    with pytest.raises(ValueError):
        validate_no_nan_inf(corrupted, FEATURE_COLUMNS, "test")


def test_validate_no_nan_inf_passes_on_clean_canonical_data(canonical_df):
    validate_no_nan_inf(canonical_df, FEATURE_COLUMNS + [TARGET_COL], "canonical dataset")


# --------------------------------------------------------------------------
# Edge case: predictions nearly constant
# --------------------------------------------------------------------------

def test_predictions_nearly_constant_flags_low_variance():
    assert predictions_are_nearly_constant(np.zeros(100))
    assert predictions_are_nearly_constant(np.full(100, 1e-9))


def test_predictions_nearly_constant_does_not_flag_normal_variance():
    rng = np.random.default_rng(0)
    assert not predictions_are_nearly_constant(rng.normal(0, 0.01, size=100))


# --------------------------------------------------------------------------
# Edge case: package-version behavior differences -- determinism
# --------------------------------------------------------------------------

def test_same_seed_same_fold_produces_identical_predictions(canonical_df):
    fold = purged_walk_forward_splits(canonical_df["Date"])[0]
    X_train = canonical_df.loc[fold.train_idx, FEATURE_COLUMNS]
    y_train = canonical_df.loc[fold.train_idx, TARGET_COL]
    X_test = canonical_df.loc[fold.test_idx, FEATURE_COLUMNS]

    model_a = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    model_a.fit(X_train, y_train)
    pred_a = model_a.predict(X_test)

    model_b = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    model_b.fit(X_train, y_train)
    pred_b = model_b.predict(X_test)

    assert np.array_equal(pred_a, pred_b)


# --------------------------------------------------------------------------
# Acceptance: train metrics diagnostic only, OOS output reproducible
# --------------------------------------------------------------------------

def test_run_produces_consistent_and_labeled_output():
    run()

    predictions = pd.read_csv(OUTPUT_DIR / "lightgbm_oos_predictions.csv")
    fold_metrics = pd.read_csv(OUTPUT_DIR / "lightgbm_fold_metrics.csv")
    summary = json.loads((OUTPUT_DIR / "lightgbm_summary.json").read_text())

    assert len(predictions) == summary["overall_metrics"]["n_oos_rows"]
    assert predictions["fold"].nunique() == len(fold_metrics) == summary["overall_metrics"]["n_folds"]

    assert "train_mae_diagnostic_only" in fold_metrics.columns
    assert "mae" in fold_metrics.columns
    assert summary["seed"] == SEED
    assert summary["hyperparameters"] == LIGHTGBM_PARAMS
    assert "lightgbm" in summary["package_versions"]
    assert summary["package_versions"]["lightgbm"] == lgb.__version__


def test_baseline_comparison_is_arithmetically_consistent():
    fold_metrics = pd.read_csv(OUTPUT_DIR / "lightgbm_fold_metrics.csv")
    computed = fold_metrics["baseline_zero_mae"] - fold_metrics["mae"]
    assert computed.to_numpy() == pytest.approx(fold_metrics["mae_improvement_over_baseline"].to_numpy())


# --------------------------------------------------------------------------
# Edge case: baseline_zero_summary.json recorded against a stale/different dataset
# --------------------------------------------------------------------------

def test_run_raises_if_baseline_summary_was_generated_from_a_different_dataset(tmp_path, monkeypatch):
    stale_summary = json.loads(BASELINE_SUMMARY_PATH.read_text())
    stale_summary["canonical_dataset_sha256"] = "0" * 64
    stale_path = tmp_path / "baseline_zero_summary.json"
    stale_path.write_text(json.dumps(stale_summary))

    monkeypatch.setattr(train_lightgbm, "BASELINE_SUMMARY_PATH", stale_path)
    with pytest.raises(ValueError, match="different canonical dataset"):
        run()
