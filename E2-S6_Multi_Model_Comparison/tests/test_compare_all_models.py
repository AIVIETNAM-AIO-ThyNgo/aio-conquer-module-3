"""
E2-S6 [Model] Compare all models -- verification suite.

Run:
    python -m pytest E2-S6_Multi_Model_Comparison/tests/test_compare_all_models.py -v

Not a board card. Owner: Model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

E2_S6_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = E2_S6_DIR.parent
sys.path.insert(0, str(E2_S6_DIR))
sys.path.insert(0, str(REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"))
sys.path.insert(0, str(REPO_ROOT / "E2-S5_Evaluate_Overall_LowVol_HighVol_Performance"))

from compare_all_models import (  # noqa: E402
    ADDITIONAL_MODELS,
    ALL_MODELS,
    OUTPUT_DIR,
    build_overall_ranking,
    load_additional_model_predictions,
    run,
)
from evaluate_regime_performance import SCOPES, assert_same_oos_rows, load_baseline_predictions  # noqa: E402

pytestmark = pytest.mark.skipif(
    not all((OUTPUT_DIR / name / f"{name}_oos_predictions.csv").exists() for name in ADDITIONAL_MODELS),
    reason="run train_additional_models.py first",
)


@pytest.fixture(scope="module")
def baseline_df():
    return load_baseline_predictions()


# --------------------------------------------------------------------------
# Acceptance: fair comparison -- every additional model checked against baseline
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["random_forest", "adaboost", "xgboost"])
def test_additional_model_shares_the_same_oos_rows_as_baseline(baseline_df, model_name):
    model_df = load_additional_model_predictions(model_name)
    assert_same_oos_rows(baseline_df, model_df)  # must not raise


# --------------------------------------------------------------------------
# Acceptance: report all models, all scopes -- no cherry-picking
# --------------------------------------------------------------------------

def test_run_produces_every_model_scope_combination():
    run()

    results = pd.read_csv(OUTPUT_DIR / "all_models_regime_performance.csv")
    assert len(results) == len(ALL_MODELS) * len(SCOPES)
    combos = set(zip(results["model"], results["scope"]))
    expected = {(m, s) for m in ALL_MODELS for s in SCOPES}
    assert combos == expected


def test_overall_ranking_includes_every_model_exactly_once():
    ranking = pd.read_csv(OUTPUT_DIR / "all_models_overall_ranking.csv")
    assert set(ranking["model"]) == set(ALL_MODELS)
    assert ranking["model"].is_unique


def test_overall_ranking_is_sorted_ascending_by_mae():
    ranking = pd.read_csv(OUTPUT_DIR / "all_models_overall_ranking.csv")
    assert (ranking["mae"].diff().dropna() >= 0).all()
    assert list(ranking["rank_by_mae"]) == list(range(1, len(ranking) + 1))


def test_mae_improvement_over_baseline_is_arithmetically_consistent():
    ranking = pd.read_csv(OUTPUT_DIR / "all_models_overall_ranking.csv").set_index("model")
    baseline_mae = ranking.loc["baseline_zero", "mae"]
    for model in ALL_MODELS:
        expected = baseline_mae - ranking.loc[model, "mae"]
        assert ranking.loc[model, "mae_improvement_over_baseline"] == pytest.approx(expected)
    assert ranking.loc["baseline_zero", "mae_improvement_over_baseline"] == pytest.approx(0.0)


def test_summary_declares_best_model_per_metric_without_a_single_forced_winner():
    summary = json.loads((OUTPUT_DIR / "all_models_comparison_summary.json").read_text())
    for key in ("best_model_by_overall_mae", "best_model_by_overall_prediction_correlation", "best_model_by_overall_directional_hit_rate"):
        assert summary[key] in ALL_MODELS
