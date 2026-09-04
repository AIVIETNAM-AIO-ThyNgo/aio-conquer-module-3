"""model.py -- walk-forward training and OOS prediction stages.

This module implements the modeling half of the pipeline:

  Stage A: Baseline y_hat=0        (E2-S1)
  Stage B: LightGBM training       (E2-S2)
  Stage C: Walk-forward validation (E2-S3)
  Stage D: Canonical OOS table     (E2-S4)

Each stage reuses the shared `splits.py` and `metrics.py` from E2-S1
(imported unchanged -- the E2-S1/E2-S2/E2-S3 cards require that every E2
model share one split definition and one metric implementation rather than
reimplementing them).

Contract chaining:
  - Every stage records the canonical dataset hash it consumed and the
    config hash it ran under.
  - A stage refuses to run if its inputs are stale (recorded output hash
    mismatches current input hash) -- catching partial reruns that would
    mix old LightGBM predictions with a freshly regenerated canonical
    dataset, and vice versa.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config
from .contract import (
    StageContract,
    StaleOutputError,
    check_inputs_unchanged,
    check_outputs_current,
    file_hash,
    now_utc_iso,
)

# Reuse the shared E2-S1 splits/metrics modules (must stay unchanged so the
# baseline/LightGBM/validation comparison holds).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_E2_S1_DIR = _REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"
_E2_S2_DIR = _REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor"
_E2_S3_DIR = _REPO_ROOT / "E2-S3_Leakage_Safe_Walk_Forward_Validation"
_E2_S4_DIR = _REPO_ROOT / "E2-S4_Generate_Canonical_OOS_Prediction_Table"

for _p in (_E2_S1_DIR, _E2_S2_DIR, _E2_S3_DIR, _E2_S4_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from splits import (  # noqa: E402
    HORIZON_TRADING_DAYS,
    MIN_TRAIN_SIZE,
    N_FOLDS,
    Fold,
    purged_walk_forward_splits,
)
from metrics import (  # noqa: E402
    directional_hit_rate,
    mae,
    prediction_correlation,
)


class NonFinitePredictionsError(Exception):
    """Raised when a model produces NaN/inf predictions."""


class EmptyFoldError(Exception):
    """Raised when a fold has zero training or test rows."""


def _load_stage_contract(manifest_path: Path, stage_name: str) -> StageContract | None:
    """Load a single stage's contract from the master manifest."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = data.get("stages", {})
    if stage_name in stages:
        return StageContract.from_dict(stages[stage_name])
    if data.get("stage_name") == stage_name:
        return StageContract.from_dict(data)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_canonical(cfg: Config) -> pd.DataFrame:
    path = cfg.resolve(cfg.paths.canonical_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {path} -- run the data_foundation stage first"
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    if df.empty:
        raise EmptyDataError(f"Canonical dataset is empty: {path}")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Canonical dataset must be sorted by Date")
    return df


def _validate_canonical(df: pd.DataFrame, cfg: Config) -> None:
    """Fail loudly on the canonical dataset's invariants."""
    expected_cols = {"Date"} | set(cfg.feature_columns) | {cfg.target.column, cfg.regime.column}
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise MissingColumnError(f"Canonical dataset missing columns: {missing}")

    # No NaN in any column.
    if df.isna().any().any():
        raise DataValidationError("Canonical dataset contains NaN values")

    # No non-finite target or feature values (an inf in a feature column would
    # otherwise reach LGBMRegressor.fit uncaught -- see
    # docs/E2-S2_LightGBM_single_config_audit_report.md Sec. 6).
    numeric_cols = list(cfg.feature_columns) + [cfg.target.column]
    numeric_vals = df[numeric_cols].to_numpy(dtype=float)
    if not np.isfinite(numeric_vals).all():
        raise DataValidationError(f"Non-finite values in one of: {numeric_cols}")

    # Regime labels valid.
    regime_vals = set(df[cfg.regime.column].dropna().unique())
    if not regime_vals.issubset({"LowVol", "HighVol"}):
        raise InvalidRegimeError(f"Invalid regime values: {regime_vals}")


class MissingColumnError(Exception):
    pass


class DataValidationError(Exception):
    pass


class InvalidRegimeError(Exception):
    pass


class EmptyDataError(Exception):
    pass


def _validate_predictions(y_pred: np.ndarray, fold_id: int) -> None:
    if not np.isfinite(y_pred).all():
        raise NonFinitePredictionsError(
            f"fold {fold_id}: model produced non-finite (NaN/inf) predictions"
        )


# ---------------------------------------------------------------------------
# Stage A: Baseline y_hat=0 (E2-S1)
# ---------------------------------------------------------------------------

def run_baseline(cfg: Config, force: bool = False) -> StageContract:
    """Run the zero-baseline stage."""
    df = _load_canonical(cfg)
    _validate_canonical(df, cfg)

    canonical_path = cfg.resolve(cfg.paths.canonical_csv)
    output_dir = cfg.resolve(cfg.paths.baseline_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "baseline_zero_oos_predictions.csv"
    fold_path = output_dir / "baseline_zero_fold_metrics.csv"
    summary_path = output_dir / "baseline_zero_summary.json"

    # Proposed contract.
    input_hashes = {str(canonical_path): file_hash(canonical_path)}
    proposed = StageContract(
        stage_name="baseline",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        params={
            "n_folds": cfg.model.n_folds,
            "min_train_size": cfg.model.min_train_size,
            "horizon_trading_days": cfg.model.horizon_trading_days,
            "target_column": cfg.target.column,
        },
    )

    # Staleness check.
    outputs_exist = pred_path.exists() and fold_path.exists() and summary_path.exists()
    if outputs_exist and not force:
        contract_path = cfg.resolve("pipeline_manifest.json")
        if contract_path.exists():
            previous = _load_stage_contract(contract_path, "baseline")
            if previous is not None:
                check_inputs_unchanged(previous, input_hashes, cfg.config_hash)
                check_outputs_current(previous)
                return previous

    folds = purged_walk_forward_splits(
        df["Date"],
        n_folds=cfg.model.n_folds,
        min_train_size=cfg.model.min_train_size,
        horizon=cfg.model.horizon_trading_days,
    )

    prediction_rows = []
    fold_metric_rows = []
    for fold in folds:
        y_true = df.loc[fold.test_idx, cfg.target.column].to_numpy()
        y_pred = np.zeros_like(y_true)
        _validate_predictions(y_pred, fold.fold_id)

        prediction_rows.append(pd.DataFrame({
            "fold": fold.fold_id,
            "Date": df.loc[fold.test_idx, "Date"].to_numpy(),
            "regime": df.loc[fold.test_idx, cfg.regime.column].to_numpy(),
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

    predictions.to_csv(pred_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)

    summary = {
        "card": "E2-S1 [P0][Model] Baseline y_hat=0",
        "generated_at_utc": now_utc_iso(),
        "target_column": cfg.target.column,
        "canonical_dataset_path": str(canonical_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "canonical_dataset_sha256": file_hash(canonical_path),
        "split_params": {
            "n_folds": cfg.model.n_folds,
            "min_train_size": cfg.model.min_train_size,
            "horizon_trading_days": cfg.model.horizon_trading_days,
        },
        "overall_metrics": overall,
        "config_hash": cfg.config_hash,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "python": platform.python_version(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return StageContract(
        stage_name="baseline",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        output_hashes={
            str(pred_path): file_hash(pred_path),
            str(fold_path): file_hash(fold_path),
        },
        output_records=[str(summary_path)],
        params=proposed.params,
        generated_at_utc=now_utc_iso(),
    )


# ---------------------------------------------------------------------------
# Stage B: LightGBM training (E2-S2)
# ---------------------------------------------------------------------------

def lightgbm_model_params(cfg: Config) -> dict[str, Any]:
    """Effective LGBMRegressor kwargs for this pipeline's LightGBM stage.

    random_state tracks cfg.model.seed (the pipeline's single source of truth
    for the seed) rather than living as a second, independently maintained
    value in pipeline_config.yaml's lightgbm_params -- that duplication is
    exactly what let this path silently diverge from train_lightgbm.py's
    reviewed configuration (missing random_state/n_jobs) and produce a
    different, though individually deterministic, fit. See
    docs/E2-S2_LightGBM_single_config_audit_report.md Sec. 1.
    """
    return {**cfg.model.lightgbm_params, "random_state": cfg.model.seed, "n_jobs": -1}


def run_lightgbm(cfg: Config, force: bool = False) -> StageContract:
    """Run the LightGBM training stage."""
    df = _load_canonical(cfg)
    _validate_canonical(df, cfg)

    canonical_path = cfg.resolve(cfg.paths.canonical_csv)
    baseline_summary_path = cfg.resolve(cfg.paths.baseline_output_dir) / "baseline_zero_summary.json"

    output_dir = cfg.resolve(cfg.paths.lightgbm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "lightgbm_oos_predictions.csv"
    fold_path = output_dir / "lightgbm_fold_metrics.csv"
    summary_path = output_dir / "lightgbm_summary.json"

    # The LightGBM stage depends on the baseline summary existing (the E2-S2
    # card requires LightGBM to compare against the baseline).
    if not baseline_summary_path.exists():
        raise FileNotFoundError(
            f"{baseline_summary_path} not found -- run the baseline stage first"
        )
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))

    # Verify the baseline was scored against the SAME canonical dataset.
    if baseline_summary.get("canonical_dataset_sha256") != file_hash(canonical_path):
        raise StaleOutputError(
            "baseline_zero_summary.json was generated from a different canonical dataset "
            f"(recorded={baseline_summary.get('canonical_dataset_sha256')}) than the one "
            f"just loaded ({file_hash(canonical_path)})"
        )

    model_params = lightgbm_model_params(cfg)

    input_hashes = {
        str(canonical_path): file_hash(canonical_path),
        str(baseline_summary_path): file_hash(baseline_summary_path),
    }
    proposed = StageContract(
        stage_name="lightgbm",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        params={
            "n_folds": cfg.model.n_folds,
            "min_train_size": cfg.model.min_train_size,
            "horizon_trading_days": cfg.model.horizon_trading_days,
            "seed": cfg.model.seed,
            "hyperparameters": dict(model_params),
            "target_column": cfg.target.column,
            "feature_columns": list(cfg.feature_columns),
        },
    )

    outputs_exist = pred_path.exists() and fold_path.exists() and summary_path.exists()
    if outputs_exist and not force:
        contract_path = cfg.resolve("pipeline_manifest.json")
        if contract_path.exists():
            previous = _load_stage_contract(contract_path, "lightgbm")
            if previous is not None:
                check_inputs_unchanged(previous, input_hashes, cfg.config_hash)
                check_outputs_current(previous)
                return previous

    folds = purged_walk_forward_splits(
        df["Date"],
        n_folds=cfg.model.n_folds,
        min_train_size=cfg.model.min_train_size,
        horizon=cfg.model.horizon_trading_days,
    )
    baseline_fold_metrics = pd.read_csv(
        cfg.resolve(cfg.paths.baseline_output_dir) / "baseline_zero_fold_metrics.csv"
    ).set_index("fold")

    prediction_rows = []
    fold_metric_rows = []
    for fold in folds:
        X_train = df.loc[fold.train_idx, cfg.feature_columns]
        y_train = df.loc[fold.train_idx, cfg.target.column].to_numpy()
        X_test = df.loc[fold.test_idx, cfg.feature_columns]
        y_test = df.loc[fold.test_idx, cfg.target.column].to_numpy()

        if len(fold.train_idx) == 0 or len(fold.test_idx) == 0:
            raise EmptyFoldError(f"fold {fold.fold_id}: empty train or test block")

        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train, y_train)

        test_pred = model.predict(X_test)
        _validate_predictions(test_pred, fold.fold_id)

        prediction_rows.append(pd.DataFrame({
            "fold": fold.fold_id,
            "Date": df.loc[fold.test_idx, "Date"].to_numpy(),
            "regime": df.loc[fold.test_idx, cfg.regime.column].to_numpy(),
            "y_true": y_test,
            "y_pred": test_pred,
        }))

        baseline_mae = float(baseline_fold_metrics.loc[fold.fold_id, "mae"])
        test_mae_val = mae(y_test, test_pred)
        fold_metric_rows.append({
            "fold": fold.fold_id,
            "n_train": len(fold.train_idx),
            "n_test": len(fold.test_idx),
            "test_start_date": fold.test_start_date.strftime("%Y-%m-%d"),
            "test_end_date": fold.test_end_date.strftime("%Y-%m-%d"),
            "train_mae_diagnostic_only": mae(y_train, model.predict(X_train)),
            "mae": test_mae_val,
            "prediction_correlation": prediction_correlation(y_test, test_pred),
            "directional_hit_rate": directional_hit_rate(y_test, test_pred),
            "predictions_nearly_constant": bool(np.std(test_pred) < cfg.model.nearly_constant_std_threshold),
            "baseline_zero_mae": baseline_mae,
            "mae_improvement_over_baseline": baseline_mae - test_mae_val,
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
        "predictions_nearly_constant": bool(np.std(overall_y_pred) < cfg.model.nearly_constant_std_threshold),
        "baseline_zero_mae": overall_baseline_mae,
        "mae_improvement_over_baseline": overall_baseline_mae - overall_test_mae,
    }

    predictions.to_csv(pred_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)

    summary = {
        "card": "E2-S2 [P0][Model] Train Minimal LightGBM Regressor",
        "generated_at_utc": now_utc_iso(),
        "seed": cfg.model.seed,
        "hyperparameters": model_params,
        "target_column": cfg.target.column,
        "feature_columns": cfg.feature_columns,
        "canonical_dataset_path": str(canonical_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "canonical_dataset_sha256": file_hash(canonical_path),
        "split_params": {
            "n_folds": cfg.model.n_folds,
            "min_train_size": cfg.model.min_train_size,
            "horizon_trading_days": cfg.model.horizon_trading_days,
        },
        "overall_metrics": overall,
        "train_metrics_are_diagnostic_only": (
            "train_mae_diagnostic_only measures in-sample fit and is reported for "
            "overfitting diagnostics ONLY. No claim may cite it; all claims rely on OOS columns."
        ),
        "no_tuning_declaration": (
            "This is the only configuration run against this dataset for E2-S2. "
            "Hyperparameters were fixed before any OOS metric was computed and were "
            "not adjusted afterward -- no model zoo, no search."
        ),
        "config_hash": cfg.config_hash,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "python": platform.python_version(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return StageContract(
        stage_name="lightgbm",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        output_hashes={
            str(pred_path): file_hash(pred_path),
            str(fold_path): file_hash(fold_path),
        },
        output_records=[str(summary_path)],
        params=proposed.params,
        generated_at_utc=now_utc_iso(),
    )


# ---------------------------------------------------------------------------
# Stage C: Walk-forward validation (E2-S3)
# ---------------------------------------------------------------------------

def run_validation(cfg: Config, force: bool = False) -> StageContract:
    """Run the walk-forward validation audit stage."""
    df = _load_canonical(cfg)
    _validate_canonical(df, cfg)

    canonical_path = cfg.resolve(cfg.paths.canonical_csv)
    baseline_pred_path = cfg.resolve(cfg.paths.baseline_output_dir) / "baseline_zero_oos_predictions.csv"
    lightgbm_pred_path = cfg.resolve(cfg.paths.lightgbm_output_dir) / "lightgbm_oos_predictions.csv"

    output_dir = cfg.resolve(cfg.paths.validation_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = output_dir / "fold_boundary_audit.csv"
    summary_path = output_dir / "walk_forward_validation_summary.json"

    input_hashes = {str(canonical_path): file_hash(canonical_path)}
    if baseline_pred_path.exists():
        input_hashes[str(baseline_pred_path)] = file_hash(baseline_pred_path)
    if lightgbm_pred_path.exists():
        input_hashes[str(lightgbm_pred_path)] = file_hash(lightgbm_pred_path)

    proposed = StageContract(
        stage_name="validation",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        params={
            "n_folds": cfg.model.n_folds,
            "min_train_size": cfg.model.min_train_size,
            "horizon_trading_days": cfg.model.horizon_trading_days,
        },
    )

    outputs_exist = boundary_path.exists() and summary_path.exists()
    if outputs_exist and not force:
        contract_path = cfg.resolve("pipeline_manifest.json")
        if contract_path.exists():
            previous = _load_stage_contract(contract_path, "validation")
            if previous is not None:
                check_inputs_unchanged(previous, input_hashes, cfg.config_hash)
                check_outputs_current(previous)
                return previous

    dates = df["Date"]
    assert_dates_are_contiguous_trading_days(dates)
    folds = purged_walk_forward_splits(
        dates,
        n_folds=cfg.model.n_folds,
        min_train_size=cfg.model.min_train_size,
        horizon=cfg.model.horizon_trading_days,
    )

    boundary_rows = []
    for fold in folds:
        assert_chronological_order(fold, dates)
        assert_purge_removes_label_overlap(fold, cfg.model.horizon_trading_days)
        boundary_rows.append(fold_boundary_row(fold, dates, cfg.model.horizon_trading_days))

    boundary_df = pd.DataFrame(boundary_rows)
    boundary_df.to_csv(boundary_path, index=False)

    # Traceability checks.
    traceability: dict[str, str] = {}
    if baseline_pred_path.exists():
        baseline_predictions = pd.read_csv(baseline_pred_path)
        for fold in folds:
            assert_predictions_trace_to_fold(baseline_predictions, fold, dates, "baseline_zero")
        traceability["baseline_zero"] = "PASS"
    else:
        traceability["baseline_zero"] = "SKIPPED -- run baseline stage first"

    if lightgbm_pred_path.exists():
        lightgbm_predictions = pd.read_csv(lightgbm_pred_path)
        for fold in folds:
            assert_predictions_trace_to_fold(lightgbm_predictions, fold, dates, "lightgbm")
        traceability["lightgbm"] = "PASS"
    else:
        traceability["lightgbm"] = "SKIPPED -- run lightgbm stage first"

    fold_params_frozen = check_fold_params_unchanged_across_outputs()
    first_fold_train_size = len(folds[0].train_idx)

    summary = {
        "card": "E2-S3 [P0][Model] Implement Leakage-Safe Walk-Forward Validation",
        "generated_at_utc": now_utc_iso(),
        "verdict": "PASS",
        "canonical_dataset_path": str(canonical_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "canonical_dataset_sha256": file_hash(canonical_path),
        "split_params": {
            "n_folds": cfg.model.n_folds,
            "min_train_size": cfg.model.min_train_size,
            "horizon_trading_days": cfg.model.horizon_trading_days,
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
        "config_hash": cfg.config_hash,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "python": platform.python_version(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return StageContract(
        stage_name="validation",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        output_hashes={
            str(boundary_path): file_hash(boundary_path),
        },
        output_records=[str(summary_path)],
        params=proposed.params,
        generated_at_utc=now_utc_iso(),
    )


# ---------------------------------------------------------------------------
# Stage D: Canonical OOS table (E2-S4)
# ---------------------------------------------------------------------------

def run_canonical_oos(cfg: Config, force: bool = False) -> StageContract:
    """Build the canonical OOS prediction table."""
    df = _load_canonical(cfg)
    _validate_canonical(df, cfg)

    canonical_path = cfg.resolve(cfg.paths.canonical_csv)
    source_pred_path = cfg.resolve(cfg.paths.lightgbm_output_dir) / "lightgbm_oos_predictions.csv"

    if not source_pred_path.exists():
        raise FileNotFoundError(
            f"{source_pred_path} not found -- run the lightgbm stage first"
        )

    output_path = cfg.resolve(cfg.paths.canonical_oos_table)
    manifest_path = cfg.resolve(cfg.paths.canonical_oos_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        str(canonical_path): file_hash(canonical_path),
        str(source_pred_path): file_hash(source_pred_path),
    }
    proposed = StageContract(
        stage_name="canonical_oos",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        params={"output_columns": ["Date", "prediction", "actual_return_5d", "regime", "fold_id"]},
    )

    outputs_exist = output_path.exists() and manifest_path.exists()
    if outputs_exist and not force:
        contract_path = cfg.resolve("pipeline_manifest.json")
        if contract_path.exists():
            previous = _load_stage_contract(contract_path, "canonical_oos")
            if previous is not None:
                check_inputs_unchanged(previous, input_hashes, cfg.config_hash)
                check_outputs_current(previous)
                return previous

    source_df = pd.read_csv(source_pred_path, parse_dates=["Date"])
    validate_source_columns(source_df)
    validate_every_row_is_genuine_oos(source_df, df["Date"])
    table = build_canonical_table(source_df)

    table.to_csv(output_path, index=False)

    manifest = {
        "card": "E2-S4 [P0][Model] Generate Canonical OOS Prediction Table",
        "generated_at_utc": now_utc_iso(),
        "source_predictions_path": str(source_pred_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "source_predictions_sha256": file_hash(source_pred_path),
        "output_path": str(output_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "output_columns": ["Date", "prediction", "actual_return_5d", "regime", "fold_id"],
        "n_rows": len(table),
        "n_unique_dates": int(table["Date"].nunique()),
        "n_folds": int(table["fold_id"].nunique()),
        "date_range": [table["Date"].min().strftime("%Y-%m-%d"), table["Date"].max().strftime("%Y-%m-%d")],
        "config_hash": cfg.config_hash,
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return StageContract(
        stage_name="canonical_oos",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        output_hashes={
            str(output_path): file_hash(output_path),
        },
        output_records=[str(manifest_path)],
        params=proposed.params,
        generated_at_utc=now_utc_iso(),
    )


# ---------------------------------------------------------------------------
# Re-exports from the shared E2-S3 module (so this module's tests can call
# them without importing the E2-S3 stage file directly).
# ---------------------------------------------------------------------------

from validate_walk_forward import (  # noqa: E402
    assert_chronological_order,
    assert_dates_are_contiguous_trading_days,
    assert_predictions_trace_to_fold,
    assert_purge_removes_label_overlap,
    check_fold_params_unchanged_across_outputs,
    fit_scaler_on_train_fold_only,
    fold_boundary_row,
)

from generate_oos_predictions import (  # noqa: E402
    build_canonical_table,
    validate_every_row_is_genuine_oos,
    validate_source_columns,
)

MAX_TRADING_GAP_DAYS = 7
MIN_ADEQUATE_FIRST_FOLD_TRAIN_SIZE = 1000


# ---------------------------------------------------------------------------
# Stage E: Regime-conditioned evaluation (E2-S5)
# ---------------------------------------------------------------------------

# These mirror the original E2-S5 card's constants but are driven from
# config paths instead of hardcoded values.
REGIMES = ["LowVol", "HighVol"]
SCOPES = ["Overall"] + REGIMES
MODELS = ["baseline_zero", "lightgbm"]

# Columns the canonical OOS table must carry for regime evaluation.
REQUIRED_OOS_COLUMNS = ["Date", "prediction", "actual_return_5d", "regime", "fold_id"]
REQUIRED_BASELINE_COLUMNS = ["Date", "regime", "y_true", "y_pred"]


class RegimeEvaluationError(Exception):
    """Raised when regime evaluation input validation fails."""


def run_regime_evaluation(cfg: Config, force: bool = False) -> StageContract:
    """Stage E: evaluate baseline and LightGBM across Overall/LowVol/HighVol.

    Wraps the E2-S5 logic inside the unified contract-chaining framework so
    the regime evaluation has the same staleness detection, loud failures,
    and single-config sourcing as every other stage.
    """
    canonical_oos_path = cfg.resolve(cfg.paths.canonical_oos_table)
    baseline_pred_path = (
        cfg.resolve(cfg.paths.baseline_output_dir) / "baseline_zero_oos_predictions.csv"
    )
    output_dir = cfg.resolve(cfg.paths.regime_eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perf_path = output_dir / "regime_performance.csv"
    comparison_path = output_dir / "regime_comparison.csv"
    summary_path = output_dir / "regime_performance_summary.json"

    # --- input validation: schema, emptiness, regime labels, finiteness ---
    _require_file(canonical_oos_path, "canonical OOS table")
    _require_file(baseline_pred_path, "baseline OOS predictions")

    canonical_df = pd.read_csv(canonical_oos_path, parse_dates=["Date"])
    baseline_df = pd.read_csv(baseline_pred_path, parse_dates=["Date"])

    _require_columns(canonical_df, REQUIRED_OOS_COLUMNS, "canonical OOS table")
    _require_columns(baseline_df, REQUIRED_BASELINE_COLUMNS, "baseline OOS predictions")
    _require_non_empty(canonical_df, "canonical OOS table")
    _require_non_empty(baseline_df, "baseline OOS predictions")

    # Regime labels must be exactly {LowVol, HighVol} in both frames.
    _validate_regime_labels(canonical_df, cfg.regime.column)
    _validate_regime_labels(baseline_df, cfg.regime.column)

    # No NaN in the columns we score on.
    _require_finite(canonical_df[["prediction", "actual_return_5d"]].to_numpy(),
                    "canonical OOS prediction/actual")
    _require_finite(baseline_df[["y_true", "y_pred"]].to_numpy(),
                    "baseline y_true/y_pred")

    # --- contract + staleness check ---
    input_hashes = {
        str(canonical_oos_path): file_hash(canonical_oos_path),
        str(baseline_pred_path): file_hash(baseline_pred_path),
    }
    proposed = StageContract(
        stage_name="regime_evaluation",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        params={"scopes": SCOPES, "models": MODELS, "regimes": REGIMES},
    )

    outputs_exist = perf_path.exists() and comparison_path.exists() and summary_path.exists()
    if outputs_exist and not force:
        contract_path = cfg.resolve(cfg.paths.pipeline_manifest)
        if contract_path.exists():
            previous = _load_stage_contract(contract_path, "regime_evaluation")
            if previous is not None:
                check_inputs_unchanged(previous, input_hashes, cfg.config_hash)
                check_outputs_current(previous)
                return previous

    # --- load & reshape to a common (Date, regime, y_true, y_pred) frame ---
    lightgbm_df = pd.DataFrame({
        "Date": canonical_df["Date"],
        "regime": canonical_df[cfg.regime.column],
        "y_true": canonical_df["actual_return_5d"],
        "y_pred": canonical_df["prediction"],
    })
    baseline_ready = baseline_df[["Date", cfg.regime.column, "y_true", "y_pred"]].copy()
    baseline_ready = baseline_ready.rename(columns={cfg.regime.column: "regime"})

    # Fairness check: both models scored on the exact same OOS rows.
    _assert_same_oos_rows(baseline_ready, lightgbm_df)

    # --- score ---
    all_rows = []
    for model_name, frame in [("baseline_zero", baseline_ready), ("lightgbm", lightgbm_df)]:
        for scope in SCOPES:
            mask = scope_mask(frame, scope)
            subset = frame[mask]
            y_true = subset["y_true"].to_numpy(dtype=float)
            y_pred = subset["y_pred"].to_numpy(dtype=float)
            n = len(subset)
            all_rows.append({
                "model": model_name,
                "scope": scope,
                "n": n,
                "mae": mae(y_true, y_pred) if n else float("nan"),
                "prediction_correlation": prediction_correlation(y_true, y_pred) if n else float("nan"),
                "directional_hit_rate": directional_hit_rate(y_true, y_pred) if n else float("nan"),
                "predictions_nearly_constant": (bool(np.std(y_pred) < cfg.model.nearly_constant_std_threshold) if n else None),
                "date_start": subset["Date"].min().strftime("%Y-%m-%d") if n else None,
                "date_end": subset["Date"].max().strftime("%Y-%m-%d") if n else None,
            })

    results = pd.DataFrame(all_rows)
    comparison = _build_comparison_table(results)

    results.to_csv(perf_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "card": "E2-S5 [P0][Model] Evaluate Overall, Low-Vol & High-Vol Performance",
        "generated_at_utc": now_utc_iso(),
        "canonical_oos_predictions_path": str(canonical_oos_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "canonical_oos_predictions_sha256": file_hash(canonical_oos_path),
        "baseline_oos_predictions_path": str(baseline_pred_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "baseline_oos_predictions_sha256": file_hash(baseline_pred_path),
        "scopes": SCOPES,
        "models": MODELS,
        "no_cherry_picking_declaration": (
            "This table reports N, MAE, prediction_correlation and directional_hit_rate for "
            "every (model, scope) pair unconditionally -- baseline_zero and lightgbm, each "
            "across Overall/LowVol/HighVol -- regardless of which numbers look favorable. No "
            "metric or scope is omitted based on its value."
        ),
        "config_hash": cfg.config_hash,
        "results": all_rows,
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "python": platform.python_version(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return StageContract(
        stage_name="regime_evaluation",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        output_hashes={
            str(perf_path): file_hash(perf_path),
            str(comparison_path): file_hash(comparison_path),
        },
        output_records=[str(summary_path)],
        params=proposed.params,
        generated_at_utc=now_utc_iso(),
    )


# ---------------------------------------------------------------------------
# Helpers for regime evaluation
# ---------------------------------------------------------------------------

def _require_non_empty(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        raise EmptyDataError(f"{context} is empty")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path} -- run the prerequisite stage first")


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise MissingColumnError(f"{context}: missing required columns {missing}")


def _validate_regime_labels(df: pd.DataFrame, regime_col: str) -> None:
    labels = set(df[regime_col].dropna().unique())
    if not labels.issubset({"LowVol", "HighVol"}):
        raise InvalidRegimeError(
            f"regime column contains invalid values: {labels - {'LowVol', 'HighVol'}}"
        )


def _require_finite(block: np.ndarray, context: str) -> None:
    if not np.isfinite(block).all():
        raise NonFinitePredictionsError(f"non-finite values found in {context}")


def _assert_same_oos_rows(baseline_df: pd.DataFrame, lightgbm_df: pd.DataFrame) -> None:
    """The baseline/LightGBM comparison is only fair if both are scored on
    exactly the same OOS rows -- same dates, same regime labels, same
    target values. Re-verified here rather than assumed."""
    if not baseline_df["Date"].is_unique or not lightgbm_df["Date"].is_unique:
        raise RegimeEvaluationError("a source predictions file has duplicate dates -- cannot compare fairly")
    baseline_dates = np.sort(baseline_df["Date"].to_numpy())
    lightgbm_dates = np.sort(lightgbm_df["Date"].to_numpy())
    if not np.array_equal(baseline_dates, lightgbm_dates):
        raise RegimeEvaluationError("baseline and LightGBM are not scored on the same OOS dates")
    merged = baseline_df.merge(lightgbm_df, on="Date", suffixes=("_baseline", "_lightgbm"))
    if not (merged["regime_baseline"] == merged["regime_lightgbm"]).all():
        raise RegimeEvaluationError("baseline and LightGBM disagree on the regime label for at least one shared date")
    if not np.allclose(merged["y_true_baseline"].to_numpy(dtype=float),
                       merged["y_true_lightgbm"].to_numpy(dtype=float)):
        raise RegimeEvaluationError("baseline and LightGBM disagree on actual_return_5d for at least one shared date")


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "Overall":
        return pd.Series(True, index=df.index)
    if scope not in REGIMES:
        raise ValueError(f"unknown scope {scope!r} -- expected one of {SCOPES}")
    return df["regime"] == scope


def _build_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["model"] == "baseline_zero"].set_index("scope")
    lightgbm = results[results["model"] == "lightgbm"].set_index("scope")
    rows = []
    for scope in SCOPES:
        rows.append({
            "scope": scope,
            "n": int(baseline.loc[scope, "n"]),
            "baseline_zero_mae": baseline.loc[scope, "mae"],
            "lightgbm_mae": lightgbm.loc[scope, "mae"],
            "mae_improvement_over_baseline": baseline.loc[scope, "mae"] - lightgbm.loc[scope, "mae"],
            "baseline_zero_prediction_correlation": baseline.loc[scope, "prediction_correlation"],
            "lightgbm_prediction_correlation": lightgbm.loc[scope, "prediction_correlation"],
            "baseline_zero_directional_hit_rate": baseline.loc[scope, "directional_hit_rate"],
            "lightgbm_directional_hit_rate": lightgbm.loc[scope, "directional_hit_rate"],
        })
    return pd.DataFrame(rows)
