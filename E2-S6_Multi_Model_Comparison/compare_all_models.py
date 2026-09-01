"""
E2-S6 [Model] Compare Baseline, LightGBM, Random Forest, AdaBoost & XGBoost
==============================================================================

Not a board card -- aggregates E2-S1's zero baseline, E2-S2's LightGBM and
this folder's three additional models (`train_additional_models.py`:
RandomForest, AdaBoost, XGBoost) into one ranked comparison table, both
Overall and by regime (LowVol/HighVol), to answer directly: is LightGBM
actually the best model here, or just the first one tried?

Reuses E2-S5's regime-evaluation code (`evaluate_model`, `scope_mask`,
`assert_same_oos_rows`) rather than re-deriving it, and the same
`metrics.py` functions used by every other E2 story -- one scoring standard
for all five models. Every model is checked against the baseline for the
same fairness condition E2-S5 already established: same OOS dates, same
regime labels, same target values.

Depends on: E2-S1 (baseline), E2-S4 (canonical LightGBM OOS table), E2-S5
(regime-evaluation code), and this folder's own `train_additional_models.py`.

Run:
    python compare_all_models.py
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_S1_DIR = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"
E2_S5_DIR = REPO_ROOT / "E2-S5_Evaluate_Overall_LowVol_HighVol_Performance"
sys.path.insert(0, str(E2_S1_DIR))
sys.path.insert(0, str(E2_S5_DIR))

from evaluate_regime_performance import (  # noqa: E402
    SCOPES,
    assert_same_oos_rows,
    evaluate_model,
    load_baseline_predictions,
    load_lightgbm_predictions,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
ADDITIONAL_MODELS = ["random_forest", "adaboost", "xgboost"]
ALL_MODELS = ["baseline_zero", "lightgbm"] + ADDITIONAL_MODELS


def load_additional_model_predictions(model_name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / model_name / f"{model_name}_oos_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run train_additional_models.py first")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df[["Date", "regime", "y_true", "y_pred"]].copy()


def build_overall_ranking(results: pd.DataFrame) -> pd.DataFrame:
    overall = results[results["scope"] == "Overall"].copy()
    baseline_mae = overall.loc[overall["model"] == "baseline_zero", "mae"].iloc[0]
    overall["mae_improvement_over_baseline"] = baseline_mae - overall["mae"]
    overall = overall.sort_values("mae", kind="stable").reset_index(drop=True)
    overall.insert(0, "rank_by_mae", overall.index + 1)
    return overall[[
        "rank_by_mae", "model", "n", "mae", "mae_improvement_over_baseline",
        "prediction_correlation", "directional_hit_rate", "predictions_nearly_constant",
    ]]


def run() -> None:
    baseline_df = load_baseline_predictions()
    lightgbm_df = load_lightgbm_predictions()
    model_dfs = {"baseline_zero": baseline_df, "lightgbm": lightgbm_df}
    for name in ADDITIONAL_MODELS:
        model_dfs[name] = load_additional_model_predictions(name)

    for name in ALL_MODELS:
        if name == "baseline_zero":
            continue
        assert_same_oos_rows(baseline_df, model_dfs[name])

    rows = []
    for name in ALL_MODELS:
        rows.extend(evaluate_model(model_dfs[name], name))
    results = pd.DataFrame(rows)

    overall_ranking = build_overall_ranking(results)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "all_models_regime_performance.csv", index=False)
    overall_ranking.to_csv(OUTPUT_DIR / "all_models_overall_ranking.csv", index=False)

    overall_only = results[results["scope"] == "Overall"].set_index("model")
    best_mae_model = overall_only["mae"].idxmin()
    best_corr_model = overall_only["prediction_correlation"].idxmax()
    best_hit_rate_model = overall_only["directional_hit_rate"].idxmax()

    summary = {
        "card": "E2-S6 [Model] Compare Baseline / LightGBM / RandomForest / AdaBoost / XGBoost (follow-up, not a board card)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": ALL_MODELS,
        "scopes": SCOPES,
        "no_cherry_picking_declaration": (
            "This table reports N, MAE, prediction_correlation and directional_hit_rate for "
            "every (model, scope) pair unconditionally -- all 5 models across Overall/LowVol/"
            "HighVol -- regardless of which numbers look favorable for any one model."
        ),
        "best_model_by_overall_mae": best_mae_model,
        "best_model_by_overall_prediction_correlation": best_corr_model,
        "best_model_by_overall_directional_hit_rate": best_hit_rate_model,
        "is_lightgbm_best_declaration": (
            f"By OOS MAE (Overall), the lowest-error model is '{best_mae_model}'. "
            f"By prediction correlation, the highest is '{best_corr_model}'. "
            f"By directional hit rate, the highest is '{best_hit_rate_model}'. "
            "These need not agree, and are reported separately rather than collapsed into "
            "a single 'winner' -- see README for the honest reading."
        ),
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "python": platform.python_version(),
    }
    (OUTPUT_DIR / "all_models_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"Wrote comparison for {len(ALL_MODELS)} models to {OUTPUT_DIR}")
    print(overall_ranking.to_string(index=False))


if __name__ == "__main__":
    run()
