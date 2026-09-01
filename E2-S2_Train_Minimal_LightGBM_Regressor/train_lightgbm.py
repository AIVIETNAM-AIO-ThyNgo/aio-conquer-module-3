"""
E2-S2 [P0][Model] Train Minimal LightGBM Regressor
====================================================

Deliverable: a single, fixed LightGBM regression configuration predicting
`forward_return_5d` from the 11 frozen E1-S4 features, scored on the exact
purged walk-forward OOS folds defined in E2-S1 (`splits.py`), with the exact
same metric code (`metrics.py`) that scored the zero baseline -- so LightGBM
and the baseline are compared on identical ground.

Depends on: E2-S1 (splits.py, metrics.py, output/baseline_zero_fold_metrics.csv).

Acceptance discipline enforced here:
  - ONE hyperparameter configuration, chosen up front and never adjusted
    after looking at OOS numbers. There is no search loop in this file.
  - Seed, package version and every hyperparameter are recorded in the
    output summary.
  - Train-set metrics are computed but written to a column named
    `*_diagnostic_only` and never fed into the OOS comparison -- OOS output
    is the only thing claims may rely on.

Run:
    python train_lightgbm.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_S1_DIR = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"
sys.path.insert(0, str(E2_S1_DIR))

from metrics import (  # noqa: E402
    DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION,
    directional_hit_rate,
    mae,
    prediction_correlation,
)
from splits import HORIZON_TRADING_DAYS, MIN_TRAIN_SIZE, N_FOLDS, purged_walk_forward_splits  # noqa: E402

CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
BASELINE_FOLD_METRICS_PATH = E2_S1_DIR / "output" / "baseline_zero_fold_metrics.csv"
BASELINE_SUMMARY_PATH = E2_S1_DIR / "output" / "baseline_zero_summary.json"
TARGET_COL = "forward_return_5d"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Frozen E1-S4 feature set -- must match data/processed/E1-S6_dataset_manifest.json
# "feature_columns" exactly. Checked at runtime in validate_feature_columns().
FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "trend_10d", "trend_20d", "trend_60d",
    "volume_ratio_20d",
]

SEED = 42

# Single fixed configuration, chosen a priori for a small (11-feature), noisy,
# low-signal financial regression target -- shallow trees and conservative
# regularization to guard against the "deep trees overfit" edge case. Not
# tuned: this exact dict is used for every fold, no search, no variants.
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 200,
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    # `deterministic=True` + `force_col_wise=True` make repeated fits on the
    # *same machine* (same thread count, from n_jobs=-1) produce bit-identical
    # predictions -- verified by test_same_seed_same_fold_produces_identical_predictions.
    # This does not guarantee identical output across machines with different
    # core counts: n_jobs=-1 means thread count (and therefore histogram
    # reduction order) varies by machine, which deterministic=True does not
    # correct for. Cross-machine reproducibility of the exact predictions is
    # not claimed; the fixed hyperparameters and pinned package versions are
    # what's reproducible across machines.
    "deterministic": True,
    "force_col_wise": True,
}

DEGENERATE_PREDICTION_STD_THRESHOLD = 1e-6


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"canonical dataset is missing frozen feature columns: {missing}")


def validate_no_nan_inf(df: pd.DataFrame, columns: list[str], context: str) -> None:
    block = df[columns].to_numpy(dtype=float)
    if not np.isfinite(block).all():
        raise ValueError(f"non-finite (NaN/inf) values found in {context} -- refusing to train/predict")


def predictions_are_nearly_constant(y_pred: np.ndarray) -> bool:
    return bool(np.std(y_pred) < DEGENERATE_PREDICTION_STD_THRESHOLD)


def run() -> None:
    if not BASELINE_FOLD_METRICS_PATH.exists() or not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{BASELINE_FOLD_METRICS_PATH} not found -- run E2-S1's run_baseline.py first, "
            "LightGBM is compared against the baseline it depends on."
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
            f"(sha256 {canonical_sha256}) -- rerun E2-S1's run_baseline.py against the current "
            "dataset before comparing LightGBM to it"
        )

    folds = purged_walk_forward_splits(df["Date"], n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE, horizon=HORIZON_TRADING_DAYS)
    baseline_fold_metrics = pd.read_csv(BASELINE_FOLD_METRICS_PATH).set_index("fold")

    prediction_rows = []
    fold_metric_rows = []

    for fold in folds:
        X_train = df.loc[fold.train_idx, FEATURE_COLUMNS]
        y_train = df.loc[fold.train_idx, TARGET_COL].to_numpy()
        X_test = df.loc[fold.test_idx, FEATURE_COLUMNS]
        y_test = df.loc[fold.test_idx, TARGET_COL].to_numpy()

        model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
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

        baseline_mae = float(baseline_fold_metrics.loc[fold.fold_id, "mae"])
        test_mae = mae(y_test, test_pred)

        fold_metric_rows.append({
            "fold": fold.fold_id,
            "n_train": len(fold.train_idx),
            "n_test": len(fold.test_idx),
            "test_start_date": fold.test_start_date.strftime("%Y-%m-%d"),
            "test_end_date": fold.test_end_date.strftime("%Y-%m-%d"),
            "train_mae_diagnostic_only": mae(y_train, train_pred),
            "mae": test_mae,
            "prediction_correlation": prediction_correlation(y_test, test_pred),
            "directional_hit_rate": directional_hit_rate(y_test, test_pred),
            "predictions_nearly_constant": predictions_are_nearly_constant(test_pred),
            "baseline_zero_mae": baseline_mae,
            "mae_improvement_over_baseline": baseline_mae - test_mae,
        })

    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    overall_y_true = predictions["y_true"].to_numpy()
    overall_y_pred = predictions["y_pred"].to_numpy()
    overall_baseline_mae = baseline_summary["overall_metrics"]["mae"]
    overall_test_mae = mae(overall_y_true, overall_y_pred)

    overall = {
        "n_oos_rows": len(predictions),
        "n_folds": len(folds),
        "mae": overall_test_mae,
        "prediction_correlation": prediction_correlation(overall_y_true, overall_y_pred),
        "directional_hit_rate": directional_hit_rate(overall_y_true, overall_y_pred),
        "predictions_nearly_constant": predictions_are_nearly_constant(overall_y_pred),
        "baseline_zero_mae": overall_baseline_mae,
        "mae_improvement_over_baseline": overall_baseline_mae - overall_test_mae,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "lightgbm_oos_predictions.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "lightgbm_fold_metrics.csv", index=False)

    summary = {
        "card": "E2-S2 [P0][Model] Train Minimal LightGBM Regressor",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "hyperparameters": LIGHTGBM_PARAMS,
        "target_column": TARGET_COL,
        "feature_columns": FEATURE_COLUMNS,
        "canonical_dataset_path": str(CANONICAL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "canonical_dataset_sha256": sha256_of(CANONICAL_PATH),
        "split_params": {
            "n_folds": N_FOLDS,
            "min_train_size": MIN_TRAIN_SIZE,
            "horizon_trading_days": HORIZON_TRADING_DAYS,
        },
        "overall_metrics": overall,
        "directional_hit_rate_zero_prediction_convention": DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION,
        "train_metrics_are_diagnostic_only": (
            "train_mae_diagnostic_only measures in-sample fit and is reported for "
            "overfitting diagnostics ONLY. No claim in this card's review may cite it; "
            "all claims rely on the OOS columns (mae, prediction_correlation, "
            "directional_hit_rate) computed on held-out test folds."
        ),
        "no_tuning_declaration": (
            "This is the only configuration run against this dataset for E2-S2. "
            "Hyperparameters were fixed before any OOS metric was computed and were "
            "not adjusted afterward -- no model zoo, no search."
        ),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "python": platform.python_version(),
    }
    (OUTPUT_DIR / "lightgbm_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"Wrote {len(predictions)} OOS predictions across {len(folds)} folds to {OUTPUT_DIR}")
    print(json.dumps(overall, indent=2, default=str))


if __name__ == "__main__":
    run()
