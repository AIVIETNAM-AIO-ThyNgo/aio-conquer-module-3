"""
E2-S6 [Model] Train Random Forest / AdaBoost / XGBoost -- verification suite.

Run:
    python -m pytest E2-S6_Multi_Model_Comparison/tests/test_train_additional_models.py -v

Not a board card. Owner: Model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

E2_S6_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = E2_S6_DIR.parent
sys.path.insert(0, str(E2_S6_DIR))
sys.path.insert(0, str(REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"))
sys.path.insert(0, str(REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor"))

import train_additional_models  # noqa: E402
from train_additional_models import (  # noqa: E402
    ADABOOST_BASE_ESTIMATOR_PARAMS,
    ADABOOST_PARAMS,
    MODEL_SPECS,
    OUTPUT_DIR,
    RANDOM_FOREST_PARAMS,
    SEED,
    XGBOOST_PARAMS,
    run,
)
from splits import HORIZON_TRADING_DAYS, MIN_TRAIN_SIZE, N_FOLDS, purged_walk_forward_splits  # noqa: E402


# --------------------------------------------------------------------------
# Acceptance: one fixed configuration per model, not a search space
# --------------------------------------------------------------------------

@pytest.mark.parametrize("params", [RANDOM_FOREST_PARAMS, ADABOOST_BASE_ESTIMATOR_PARAMS, ADABOOST_PARAMS, XGBOOST_PARAMS])
def test_hyperparameters_are_not_a_search_space(params):
    for key, value in params.items():
        assert not isinstance(value, (list, tuple, set)), (
            f"'{key}' looks like a search grid, not a fixed hyperparameter"
        )


def test_every_model_seed_matches_the_shared_seed():
    assert RANDOM_FOREST_PARAMS["random_state"] == SEED
    assert ADABOOST_BASE_ESTIMATOR_PARAMS["random_state"] == SEED
    assert ADABOOST_PARAMS["random_state"] == SEED
    assert XGBOOST_PARAMS["random_state"] == SEED


def test_model_specs_cover_exactly_three_additional_models():
    assert {spec.name for spec in MODEL_SPECS} == {"random_forest", "adaboost", "xgboost"}


def test_model_spec_hyperparameters_are_json_serializable():
    for spec in MODEL_SPECS:
        json.dumps(spec.hyperparameters)  # must not raise


# --------------------------------------------------------------------------
# Acceptance: dataset-consistency guard against a stale baseline
# --------------------------------------------------------------------------

def test_run_raises_if_baseline_summary_was_generated_from_a_different_dataset(tmp_path, monkeypatch):
    stale_summary = json.loads(train_additional_models.BASELINE_SUMMARY_PATH.read_text())
    stale_summary["canonical_dataset_sha256"] = "0" * 64
    stale_path = tmp_path / "baseline_zero_summary.json"
    stale_path.write_text(json.dumps(stale_summary))

    monkeypatch.setattr(train_additional_models, "BASELINE_SUMMARY_PATH", stale_path)
    with pytest.raises(ValueError, match="different canonical dataset"):
        run()


# --------------------------------------------------------------------------
# Full run: produces per-model output for all three additional models
# --------------------------------------------------------------------------

def test_run_produces_output_for_every_additional_model():
    run()

    df = pd.read_csv(train_additional_models.CANONICAL_PATH, parse_dates=["Date"])
    folds = purged_walk_forward_splits(df["Date"], n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE, horizon=HORIZON_TRADING_DAYS)
    expected_n_rows = sum(len(f.test_idx) for f in folds)

    for spec in MODEL_SPECS:
        model_dir = OUTPUT_DIR / spec.name
        predictions = pd.read_csv(model_dir / f"{spec.name}_oos_predictions.csv")
        fold_metrics = pd.read_csv(model_dir / f"{spec.name}_fold_metrics.csv")
        summary = json.loads((model_dir / f"{spec.name}_summary.json").read_text())

        assert len(predictions) == expected_n_rows == summary["overall_metrics"]["n_oos_rows"]
        assert predictions["fold"].nunique() == len(fold_metrics) == N_FOLDS
        assert summary["seed"] == SEED
        assert summary["hyperparameters"] == spec.hyperparameters
        assert "train_mae_diagnostic_only" in fold_metrics.columns


def test_predictions_are_not_degenerate_for_any_additional_model():
    for spec in MODEL_SPECS:
        summary_path = OUTPUT_DIR / spec.name / f"{spec.name}_summary.json"
        summary = json.loads(summary_path.read_text())
        assert summary["overall_metrics"]["predictions_nearly_constant"] is False, (
            f"{spec.name} collapsed to a near-constant prediction"
        )
