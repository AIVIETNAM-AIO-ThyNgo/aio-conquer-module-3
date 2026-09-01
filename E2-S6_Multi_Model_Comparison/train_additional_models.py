"""
E2-S6 [Model] Train Random Forest, AdaBoost, XGBoost on the E2 walk-forward folds
====================================================================================

Not a board card -- a follow-up requested directly: is LightGBM (E2-S2) actually
the best model on this dataset, or just the only one tried? This trains three
more regressors -- RandomForest, AdaBoost, XGBoost -- on the exact same purged
walk-forward folds, the same 11 frozen features and the same target as E2-S2,
using the same acceptance discipline E2-S2 established:

  - ONE fixed hyperparameter configuration per model, chosen up front and never
    adjusted after looking at OOS numbers -- no search loop, no model zoo.
  - Seed, package versions and every hyperparameter recorded per model.
  - Train-set metrics computed but written to a `*_diagnostic_only` column and
    never fed into the OOS comparison.
  - The canonical dataset's sha256 is checked against E2-S1's recorded baseline
    sha256 before training anything, so a stale baseline can't silently make
    the comparison unfair (same guard added to E2-S2 in the last review pass).

`FEATURE_COLUMNS`, `TARGET_COL`, `validate_feature_columns`, `validate_no_nan_inf`
and `predictions_are_nearly_constant` are imported from E2-S2's `train_lightgbm.py`
unchanged, not redefined -- one frozen feature set, one NaN/inf guard, one
"nearly constant" definition for every model in E2, LightGBM included.

`compare_all_models.py` in this same folder aggregates this script's output
together with E2-S1's baseline and E2-S2's LightGBM into one ranked table.

Depends on: E2-S1 (splits.py, metrics.py), E2-S2 (feature set + validation helpers).

Run:
    python train_additional_models.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_S1_DIR = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"
E2_S2_DIR = REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor"
sys.path.insert(0, str(E2_S1_DIR))
sys.path.insert(0, str(E2_S2_DIR))

from metrics import directional_hit_rate, mae, prediction_correlation  # noqa: E402
from splits import HORIZON_TRADING_DAYS, MIN_TRAIN_SIZE, N_FOLDS, purged_walk_forward_splits  # noqa: E402
from train_lightgbm import (  # noqa: E402
    FEATURE_COLUMNS,
    predictions_are_nearly_constant,
    validate_feature_columns,
    validate_no_nan_inf,
)

CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
BASELINE_SUMMARY_PATH = E2_S1_DIR / "output" / "baseline_zero_summary.json"
TARGET_COL = "forward_return_5d"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 42

# Same rationale as E2-S2's LightGBM config: an 11-feature, noisy, low-signal
# financial regression target (E4-S1 audit: strongest feature/target
# correlation is -0.08) -- every model here is deliberately shallow/regularized
# rather than tuned for capacity, to avoid fitting noise.

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 30,
    "max_features": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
}

ADABOOST_BASE_ESTIMATOR_PARAMS = {
    "max_depth": 3,
    "min_samples_leaf": 30,
    "random_state": SEED,
}
ADABOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "loss": "linear",
    "random_state": SEED,
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "verbosity": 0,
}

DEGENERATE_PREDICTION_STD_THRESHOLD = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build_fn: Callable[[], object]
    hyperparameters: dict  # JSON-serializable -- every value a scalar, never a list/tuple/set


def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(**RANDOM_FOREST_PARAMS)


def build_adaboost() -> AdaBoostRegressor:
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(**ADABOOST_BASE_ESTIMATOR_PARAMS),
        **ADABOOST_PARAMS,
    )


def build_xgboost() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**XGBOOST_PARAMS)


MODEL_SPECS = [
    ModelSpec("random_forest", build_random_forest, RANDOM_FOREST_PARAMS),
    ModelSpec(
        "adaboost",
        build_adaboost,
        {**{f"base_estimator_{k}": v for k, v in ADABOOST_BASE_ESTIMATOR_PARAMS.items()}, **ADABOOST_PARAMS},
    ),
    ModelSpec("xgboost", build_xgboost, XGBOOST_PARAMS),
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_one_model(df: pd.DataFrame, folds: list, spec: ModelSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    fold_metric_rows = []

    for fold in folds:
        X_train = df.loc[fold.train_idx, FEATURE_COLUMNS]
        y_train = df.loc[fold.train_idx, TARGET_COL].to_numpy()
        X_test = df.loc[fold.test_idx, FEATURE_COLUMNS]
        y_test = df.loc[fold.test_idx, TARGET_COL].to_numpy()

        model = spec.build_fn()
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        prediction_rows.append(pd.DataFrame({
            "fold": fold.fold_id,
            "Date": df.loc[fold.test_idx, "Date"].to_numpy(),
            "regime": df.loc[fold.test_idx, "regime"].to_numpy(),
            "y_true": y_test,
            "y_pred": test_pred,
        }))

        fold_metric_rows.append({
            "fold": fold.fold_id,
            "n_train": len(fold.train_idx),
            "n_test": len(fold.test_idx),
            "test_start_date": fold.test_start_date.strftime("%Y-%m-%d"),
            "test_end_date": fold.test_end_date.strftime("%Y-%m-%d"),
            "train_mae_diagnostic_only": mae(y_train, train_pred),
            "mae": mae(y_test, test_pred),
            "prediction_correlation": prediction_correlation(y_test, test_pred),
            "directional_hit_rate": directional_hit_rate(y_test, test_pred),
            "predictions_nearly_constant": predictions_are_nearly_constant(test_pred),
        })

    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    return predictions, fold_metrics


def run() -> None:
    if not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{BASELINE_SUMMARY_PATH} not found -- run E2-S1's run_baseline.py first"
        )
    baseline_summary = json.loads(BASELINE_SUMMARY_PATH.read_text())

    df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("canonical dataset must be sorted by Date")
    validate_feature_columns(df)
    validate_no_nan_inf(df, FEATURE_COLUMNS + [TARGET_COL], "canonical dataset")

    canonical_sha256 = sha256_of(CANONICAL_PATH)
    if baseline_summary["canonical_dataset_sha256"] != canonical_sha256:
        raise ValueError(
            "baseline_zero_summary.json was generated from a different canonical dataset "
            f"(sha256 {baseline_summary['canonical_dataset_sha256']}) than the one just loaded "
            f"(sha256 {canonical_sha256}) -- rerun E2-S1's run_baseline.py against the current dataset"
        )

    folds = purged_walk_forward_splits(df["Date"], n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE, horizon=HORIZON_TRADING_DAYS)

    for spec in MODEL_SPECS:
        predictions, fold_metrics = train_one_model(df, folds, spec)

        overall_y_true = predictions["y_true"].to_numpy()
        overall_y_pred = predictions["y_pred"].to_numpy()
        overall = {
            "n_oos_rows": len(predictions),
            "n_folds": len(folds),
            "mae": mae(overall_y_true, overall_y_pred),
            "prediction_correlation": prediction_correlation(overall_y_true, overall_y_pred),
            "directional_hit_rate": directional_hit_rate(overall_y_true, overall_y_pred),
            "predictions_nearly_constant": predictions_are_nearly_constant(overall_y_pred),
        }

        model_output_dir = OUTPUT_DIR / spec.name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(model_output_dir / f"{spec.name}_oos_predictions.csv", index=False)
        fold_metrics.to_csv(model_output_dir / f"{spec.name}_fold_metrics.csv", index=False)

        summary = {
            "card": "E2-S6 [Model] Train Random Forest / AdaBoost / XGBoost (follow-up, not a board card)",
            "model": spec.name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "hyperparameters": spec.hyperparameters,
            "target_column": TARGET_COL,
            "feature_columns": FEATURE_COLUMNS,
            "canonical_dataset_path": str(CANONICAL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "canonical_dataset_sha256": canonical_sha256,
            "split_params": {
                "n_folds": N_FOLDS,
                "min_train_size": MIN_TRAIN_SIZE,
                "horizon_trading_days": HORIZON_TRADING_DAYS,
            },
            "overall_metrics": overall,
            "train_metrics_are_diagnostic_only": (
                "train_mae_diagnostic_only measures in-sample fit and is reported for "
                "overfitting diagnostics ONLY. No claim may cite it; all claims rely on "
                "OOS columns computed on held-out test folds."
            ),
            "no_tuning_declaration": (
                f"This is the only configuration run against this dataset for {spec.name}. "
                "Hyperparameters were fixed before any OOS metric was computed and were not "
                "adjusted afterward -- no model zoo, no search."
            ),
            "package_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "xgboost": xgb.__version__,
            },
            "python": platform.python_version(),
        }
        (model_output_dir / f"{spec.name}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        print(f"[{spec.name}] wrote {len(predictions)} OOS predictions across {len(folds)} folds")
        print(json.dumps(overall, indent=2, default=str))


if __name__ == "__main__":
    run()
