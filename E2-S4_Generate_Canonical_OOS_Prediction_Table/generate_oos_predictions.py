"""
E2-S4 [P0][Model] Generate Canonical OOS Prediction Table
=============================================================

Deliverable: `results/oos_predictions.csv` -- one row per genuine
out-of-sample prediction, with exactly the columns `Date`, `prediction`,
`actual_return_5d`, `regime`, `fold_id`. This is the single canonical
prediction table later E2/E3 stories (regime-conditioned evaluation,
reporting, backtesting) read from, so it must carry only OOS rows and only
these five approved columns -- no train-fold predictions, no diagnostic
columns, no duplicate or out-of-order dates.

Source: E2-S2's `lightgbm_oos_predictions.csv` -- the only model in this
repo producing genuine (non-baseline) predictions, itself produced against
the exact purged walk-forward folds E2-S1's `splits.py` defines and E2-S3
already verified end-to-end.

This script does not trust that source file blindly: every row is
re-checked against the fold each source row claims to belong to (the same
`assert_predictions_trace_to_fold` reasoning E2-S3 uses), so a fold-mixup or
an accidentally-appended diagnostic row is caught here rather than silently
published.

Depends on: E2-S3 (Leakage-Safe Walk-Forward Validation).

Run:
    python generate_oos_predictions.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
E2_S1_DIR = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"
E2_S2_DIR = REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor"
E2_S3_DIR = REPO_ROOT / "E2-S3_Leakage_Safe_Walk_Forward_Validation"
sys.path.insert(0, str(E2_S1_DIR))
sys.path.insert(0, str(E2_S3_DIR))

from splits import HORIZON_TRADING_DAYS, MIN_TRAIN_SIZE, N_FOLDS, purged_walk_forward_splits  # noqa: E402
from validate_walk_forward import assert_predictions_trace_to_fold  # noqa: E402

CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
SOURCE_PREDICTIONS_PATH = E2_S2_DIR / "output" / "lightgbm_oos_predictions.csv"
SOURCE_SUMMARY_PATH = E2_S2_DIR / "output" / "lightgbm_summary.json"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "oos_predictions.csv"

# The only columns the source file may carry -- if E2-S2 ever grows a
# diagnostic column (e.g. train_mae_diagnostic_only) and someone points this
# script at it by mistake, this catches it before it reaches the canonical
# table rather than silently dropping or leaking it through.
SOURCE_COLUMNS = ["fold", "Date", "regime", "y_true", "y_pred"]

# Canonical output schema, in this exact order.
OUTPUT_COLUMNS = ["Date", "prediction", "actual_return_5d", "regime", "fold_id"]

RENAME_MAP = {
    "fold": "fold_id",
    "y_pred": "prediction",
    "y_true": "actual_return_5d",
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_columns(df: pd.DataFrame) -> None:
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"source predictions file has columns {list(df.columns)}, expected exactly "
            f"{SOURCE_COLUMNS} -- refusing to publish, an unexpected (e.g. diagnostic) "
            "column may have been appended"
        )


def validate_every_row_is_genuine_oos(df: pd.DataFrame, dates: pd.Series) -> None:
    """Re-derives the authoritative folds from the canonical dataset and checks
    every source row's Date against the fold it claims (`fold`) -- a row that
    belongs to a training window, a purge gap, or the wrong fold is caught
    here rather than published as if it were OOS."""
    folds = purged_walk_forward_splits(dates, n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE, horizon=HORIZON_TRADING_DAYS)
    for fold in folds:
        assert_predictions_trace_to_fold(df[["fold", "Date"]], fold, dates, "canonical_oos_table_source")


def build_canonical_table(source_df: pd.DataFrame) -> pd.DataFrame:
    table = source_df.rename(columns=RENAME_MAP)[OUTPUT_COLUMNS].copy()
    table["Date"] = pd.to_datetime(table["Date"])

    # Edge case: shuffled row order -- always re-sort by Date rather than
    # trusting the source file's row order.
    table = table.sort_values("Date", kind="stable").reset_index(drop=True)

    # Edge case: duplicate dates from overlapping test folds -- structurally
    # prevented by non-overlapping test blocks in splits.py, but checked here
    # too since this file is the last line of defense before publication.
    duplicate_dates = table.loc[table["Date"].duplicated(keep=False), "Date"]
    if not duplicate_dates.empty:
        raise ValueError(f"canonical OOS table has duplicate dates: {sorted(duplicate_dates.unique())}")

    if not table["Date"].is_monotonic_increasing:
        raise ValueError("canonical OOS table is not in ascending date order after sorting")

    required_no_nan = ["prediction", "actual_return_5d", "regime"]
    nan_counts = table[required_no_nan].isna().sum()
    if nan_counts.any():
        raise ValueError(f"canonical OOS table has NaN values: {nan_counts[nan_counts > 0].to_dict()}")

    if list(table.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"canonical OOS table has columns {list(table.columns)}, expected exactly {OUTPUT_COLUMNS}")

    return table


def run() -> None:
    if not SOURCE_PREDICTIONS_PATH.exists() or not SOURCE_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SOURCE_PREDICTIONS_PATH} not found -- run E2-S2's train_lightgbm.py first"
        )
    source_summary = json.loads(SOURCE_SUMMARY_PATH.read_text())

    canonical_df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    dates = canonical_df["Date"]

    canonical_sha256 = sha256_of(CANONICAL_PATH)
    if source_summary["canonical_dataset_sha256"] != canonical_sha256:
        raise ValueError(
            "lightgbm_summary.json was generated from a different canonical dataset "
            f"(sha256 {source_summary['canonical_dataset_sha256']}) than the one just loaded "
            f"(sha256 {canonical_sha256}) -- rerun E2-S2's train_lightgbm.py against the current "
            "dataset before publishing the canonical OOS table"
        )

    source_df = pd.read_csv(SOURCE_PREDICTIONS_PATH, parse_dates=["Date"])
    validate_source_columns(source_df)
    validate_every_row_is_genuine_oos(source_df, dates)

    table = build_canonical_table(source_df)

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    manifest = {
        "card": "E2-S4 [P0][Model] Generate Canonical OOS Prediction Table",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_predictions_path": str(SOURCE_PREDICTIONS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_predictions_sha256": sha256_of(SOURCE_PREDICTIONS_PATH),
        "output_path": str(OUTPUT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "output_columns": OUTPUT_COLUMNS,
        "n_rows": len(table),
        "n_unique_dates": int(table["Date"].nunique()),
        "n_folds": int(table["fold_id"].nunique()),
        "date_range": [table["Date"].min().strftime("%Y-%m-%d"), table["Date"].max().strftime("%Y-%m-%d")],
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    (RESULTS_DIR / "oos_predictions_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    print(f"Wrote {len(table)} canonical OOS rows to {OUTPUT_PATH}")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    run()
