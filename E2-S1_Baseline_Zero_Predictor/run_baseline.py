"""
E2-S1 [P0][Model] Baseline y_hat=0
===================================

Deliverable: baseline y_hat=0 evaluated on the same OOS dates/folds a later
LightGBM model will use (see splits.py). Same target column, same split
code, same metric code -- no special-case advantage or disadvantage for
either model (E2-S1 acceptance).

Depends on: E1-S6 (canonical modeling dataset), E4-S1 (leakage/QA gate).

Run:
    python run_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import (
    DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION,
    directional_hit_rate,
    mae,
    prediction_correlation,
)
from splits import HORIZON_TRADING_DAYS, MIN_TRAIN_SIZE, N_FOLDS, purged_walk_forward_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
TARGET_COL = "forward_return_5d"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("canonical dataset must be sorted by Date")
    if not np.isfinite(df[TARGET_COL].to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite (NaN/inf) values found in {TARGET_COL} -- refusing to score")

    folds = purged_walk_forward_splits(df["Date"])

    prediction_rows = []
    fold_metric_rows = []

    for fold in folds:
        y_true = df.loc[fold.test_idx, TARGET_COL].to_numpy()
        y_pred = np.zeros_like(y_true)

        prediction_rows.append(pd.DataFrame({
            "fold": fold.fold_id,
            "Date": df.loc[fold.test_idx, "Date"].to_numpy(),
            "regime": df.loc[fold.test_idx, "regime"].to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
        }))

        fold_metric_rows.append({
            "fold": fold.fold_id,
            "n_train": len(fold.train_idx),
            "n_test": len(fold.test_idx),
            "test_start_date": fold.test_start_date.strftime("%Y-%m-%d"),
            "test_end_date": fold.test_end_date.strftime("%Y-%m-%d"),
            "mae": mae(y_true, y_pred),
            "prediction_correlation": prediction_correlation(y_true, y_pred),
            "directional_hit_rate": directional_hit_rate(y_true, y_pred),
        })

    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    overall_y_true = predictions["y_true"].to_numpy()
    overall_y_pred = predictions["y_pred"].to_numpy()
    overall = {
        "n_oos_rows": len(predictions),
        "n_folds": len(folds),
        "mae": mae(overall_y_true, overall_y_pred),
        "prediction_correlation": prediction_correlation(overall_y_true, overall_y_pred),
        "directional_hit_rate": directional_hit_rate(overall_y_true, overall_y_pred),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "baseline_zero_oos_predictions.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "baseline_zero_fold_metrics.csv", index=False)

    summary = {
        "card": "E2-S1 [P0][Model] Baseline y_hat=0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COL,
        "canonical_dataset_path": str(CANONICAL_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "canonical_dataset_sha256": sha256_of(CANONICAL_PATH),
        "split_params": {
            "n_folds": N_FOLDS,
            "min_train_size": MIN_TRAIN_SIZE,
            "horizon_trading_days": HORIZON_TRADING_DAYS,
        },
        "overall_metrics": overall,
        "directional_hit_rate_zero_prediction_convention": DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "python": platform.python_version(),
    }
    (OUTPUT_DIR / "baseline_zero_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"Wrote {len(predictions)} OOS predictions across {len(folds)} folds to {OUTPUT_DIR}")
    print(json.dumps(overall, indent=2, default=str))


if __name__ == "__main__":
    run()
