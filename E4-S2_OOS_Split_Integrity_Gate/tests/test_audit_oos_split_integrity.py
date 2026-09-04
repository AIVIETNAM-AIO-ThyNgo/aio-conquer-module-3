"""E4-S2 OOS Split Integrity Gate -- verification suite.

Run:
    python -m pytest E4-S2_OOS_Split_Integrity_Gate/tests/test_audit_oos_split_integrity.py -v

Owner: QA/QC.  Depends on: E1-S6, E2-S1, E2-S2, E2-S3, E2-S4.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Load the audit module (folder has hyphens, so use importlib)
_audit_module_path = REPO_ROOT / "E4-S2_OOS_Split_Integrity_Gate" / "audit_oos_split_integrity.py"
_spec = importlib.util.spec_from_file_location(
    "audit_oos_split_integrity",
    str(_audit_module_path),
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load audit module from {_audit_module_path}")
_audit_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit_module)

IntegrityGate = _audit_module.IntegrityGate
IntegrityViolation = _audit_module.IntegrityViolation
run_audit = _audit_module.run_audit

CANONICAL_PATH = REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
OOS_TABLE_PATH = REPO_ROOT / "results" / "oos_predictions.csv"
BASELINE_PRED_PATH = REPO_ROOT / "E2-S1_Baseline_Zero_Predictor" / "output" / "baseline_zero_oos_predictions.csv"
LIGHTGBM_PRED_PATH = REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor" / "output" / "lightgbm_oos_predictions.csv"


@pytest.fixture(scope="module")
def gate() -> IntegrityGate:
    return IntegrityGate(
        canonical_path=CANONICAL_PATH,
        oos_table_path=OOS_TABLE_PATH,
        baseline_pred_path=BASELINE_PRED_PATH,
        lightgbm_pred_path=LIGHTGBM_PRED_PATH,
    )


@pytest.fixture(scope="module")
def report(gate: IntegrityGate) -> dict:
    return gate.audit()


# ---------------------------------------------------------------------------
# Check 1: Fold chronology
# ---------------------------------------------------------------------------

def test_fold_chronology_passes(report: dict) -> None:
    """max(train_date) < min(test_date) for every fold."""
    assert report["details"]["fold_chronology"]["passed"], (
        f"Fold chronology check failed: {report['violations']}"
    )


def test_all_folds_have_valid_temporal_order(gate: IntegrityGate) -> None:
    """Each fold's train dates must all be before its test dates."""
    canonical_df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    folds = gate._build_folds(canonical_df)
    dates = canonical_df["Date"]

    for fold in folds:
        train_max = dates.iloc[fold.train_idx].max()
        test_min = dates.iloc[fold.test_idx].min()
        assert train_max < test_min, (
            f"Fold {fold.fold_id}: train_max={train_max} not < test_min={test_min}"
        )


def test_folds_cover_entire_oos_region(gate: IntegrityGate) -> None:
    """Test blocks should cover the entire post-warm-up region with no gaps."""
    canonical_df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    folds = gate._build_folds(canonical_df)

    all_test_idx = np.concatenate([f.test_idx for f in folds])
    # No overlaps
    assert len(all_test_idx) == len(set(all_test_idx.tolist()))
    # Contiguous
    assert np.array_equal(all_test_idx, np.arange(all_test_idx[0], all_test_idx[-1] + 1))


# ---------------------------------------------------------------------------
# Check 2: Purge gap
# ---------------------------------------------------------------------------

def test_purge_gap_is_effective(report: dict) -> None:
    """5D boundary purge/gap must be effective."""
    assert report["details"]["purge_gap"]["passed"], (
        f"Purge gap check failed: {report['violations']}"
    )


def test_label_never_reads_test_price(gate: IntegrityGate) -> None:
    """The label for the last training row must never read a price inside the test window."""
    canonical_df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    folds = gate._build_folds(canonical_df)

    for fold in folds:
        if len(fold.train_idx) == 0:
            continue
        last_train_idx = int(fold.train_idx.max())
        first_test_idx = int(fold.test_idx.min())
        label_price_idx = last_train_idx + 5
        assert label_price_idx < first_test_idx, (
            f"Fold {fold.fold_id}: label reads index {label_price_idx} inside test window starting at {first_test_idx}"
        )


def test_purge_gap_is_at_least_horizon(gate: IntegrityGate) -> None:
    """The gap between train and test should be at least HORIZON_TRADING_DAYS."""
    from splits import HORIZON_TRADING_DAYS
    canonical_df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    folds = gate._build_folds(canonical_df)

    for fold in folds:
        if len(fold.train_idx) == 0:
            continue
        gap = int(fold.test_idx.min()) - int(fold.train_idx.max()) - 1
        assert gap >= HORIZON_TRADING_DAYS, (
            f"Fold {fold.fold_id}: gap {gap} < horizon {HORIZON_TRADING_DAYS}"
        )


# ---------------------------------------------------------------------------
# Check 3: No test data in features
# ---------------------------------------------------------------------------

def test_no_test_data_in_features(report: dict) -> None:
    """Features must be computed using only past data (right-aligned windows)."""
    assert report["details"]["no_test_data_in_features"]["passed"], (
        f"Feature leakage check failed: {report['violations']}"
    )


def test_features_are_point_in_time(gate: IntegrityGate) -> None:
    """Feature values at test dates must match values from truncated history."""
    canonical_df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    folds = gate._build_folds(canonical_df)
    feature_cols = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "trend_10d", "trend_20d", "trend_60d",
        "volume_ratio_20d",
    ]

    for fold in folds:
        test_dates = canonical_df["Date"].iloc[fold.test_idx]
        for idx in [0, len(test_dates) // 2, len(test_dates) - 1]:
            test_date = test_dates.iloc[idx]
            date_idx = canonical_df[canonical_df["Date"] == test_date].index[0]
            truncated = canonical_df[canonical_df["Date"] <= test_date]

            for col in feature_cols:
                full_value = canonical_df[col].iloc[date_idx]
                truncated_value = truncated[col].iloc[-1]
                assert np.isclose(full_value, truncated_value, rtol=1e-10, atol=1e-12), (
                    f"Fold {fold.fold_id}, {col} at {test_date}: full={full_value} != truncated={truncated_value}"
                )


# ---------------------------------------------------------------------------
# Check 4: OOS table integrity
# ---------------------------------------------------------------------------

def test_oos_table_integrity(report: dict) -> None:
    """OOS table must have no duplicate dates or in-sample predictions."""
    assert report["details"]["oos_table_integrity"]["passed"], (
        f"OOS table integrity check failed: {report['violations']}"
    )


def test_oos_table_has_no_duplicate_dates() -> None:
    """OOS table must have unique dates."""
    oos_df = pd.read_csv(OOS_TABLE_PATH, parse_dates=["Date"])
    assert oos_df["Date"].is_unique, "OOS table contains duplicate dates"


def test_oos_table_dates_trace_to_folds(gate: IntegrityGate) -> None:
    """Every OOS date must fall in its claimed fold's test window."""
    canonical_df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    oos_df = pd.read_csv(OOS_TABLE_PATH, parse_dates=["Date"])
    folds = gate._build_folds(canonical_df)
    dates = canonical_df["Date"]

    for fold in folds:
        fold_dates = oos_df[oos_df["fold_id"] == fold.fold_id]["Date"].sort_values()
        expected_dates = dates.iloc[fold.test_idx].sort_values()
        assert np.array_equal(fold_dates.values, expected_dates.values), (
            f"Fold {fold.fold_id}: OOS dates don't match test window"
        )


def test_oos_table_has_no_in_sample_predictions(gate: IntegrityGate) -> None:
    """No OOS date should fall in its own fold's training window or purge gap."""
    canonical_df = pd.read_csv(CANONICAL_PATH, parse_dates=["Date"])
    oos_df = pd.read_csv(OOS_TABLE_PATH, parse_dates=["Date"])
    folds = gate._build_folds(canonical_df)
    dates = canonical_df["Date"]

    for _, row in oos_df.iterrows():
        oos_date = row["Date"]
        fold_id = row.get("fold_id")
        if fold_id is None:
            continue
        fold = folds[fold_id]
        # The OOS date must be in this fold's test window
        test_dates = set(dates.iloc[fold.test_idx].tolist())
        assert oos_date in test_dates, (
            f"OOS date {oos_date} is not in fold {fold_id}'s test window"
        )
        # The OOS date must NOT be in this fold's training window
        train_dates = set(dates.iloc[fold.train_idx].tolist())
        assert oos_date not in train_dates, (
            f"OOS date {oos_date} is in fold {fold_id}'s training window"
        )


# ---------------------------------------------------------------------------
# Check 5: Model comparison fairness
# ---------------------------------------------------------------------------

def test_model_comparison_fairness(report: dict) -> None:
    """Baseline and LightGBM must be compared on identical OOS rows."""
    assert report["details"]["model_comparison_fairness"]["passed"], (
        f"Model comparison fairness check failed: {report['violations']}"
    )


def test_baseline_and_lightgbm_same_dates() -> None:
    """Both models must be evaluated on the same dates."""
    baseline_df = pd.read_csv(BASELINE_PRED_PATH, parse_dates=["Date"])
    lightgbm_df = pd.read_csv(LIGHTGBM_PRED_PATH, parse_dates=["Date"])

    baseline_dates = np.sort(baseline_df["Date"].values)
    lightgbm_dates = np.sort(lightgbm_df["Date"].values)
    assert np.array_equal(baseline_dates, lightgbm_dates), "Models have different OOS dates"


def test_baseline_and_lightgbm_same_regime_labels() -> None:
    """Both models must agree on regime labels for each date."""
    baseline_df = pd.read_csv(BASELINE_PRED_PATH, parse_dates=["Date"])
    lightgbm_df = pd.read_csv(LIGHTGBM_PRED_PATH, parse_dates=["Date"])

    merged = baseline_df.merge(lightgbm_df, on="Date", suffixes=("_baseline", "_lightgbm"))
    assert (merged["regime_baseline"] == merged["regime_lightgbm"]).all(), (
        "Models disagree on regime labels"
    )


def test_baseline_and_lightgbm_same_targets() -> None:
    """Both models must have the same actual_return_5d for each date."""
    baseline_df = pd.read_csv(BASELINE_PRED_PATH, parse_dates=["Date"])
    lightgbm_df = pd.read_csv(LIGHTGBM_PRED_PATH, parse_dates=["Date"])

    merged = baseline_df.merge(lightgbm_df, on="Date", suffixes=("_baseline", "_lightgbm"))
    assert np.allclose(merged["y_true_baseline"].values, merged["y_true_lightgbm"].values), (
        "Models disagree on actual_return_5d"
    )


# ---------------------------------------------------------------------------
# Overall audit
# ---------------------------------------------------------------------------

def test_audit_verdict_is_pass(report: dict) -> None:
    """The overall audit verdict must be PASS."""
    assert report["verdict"] == "PASS", (
        f"Audit verdict is {report['verdict']}: {report['violations']}"
    )


def test_audit_report_written() -> None:
    """Running main() should write the audit report to the output directory."""
    # Run main to generate the report
    exit_code = _audit_module.main()
    output_path = REPO_ROOT / "E4-S2_OOS_Split_Integrity_Gate" / "output" / "integrity_gate_report.json"
    assert output_path.exists(), "Audit report file not found after running main()"
    assert exit_code == 0, f"main() returned non-zero exit code: {exit_code}"


def test_run_audit_returns_valid_report() -> None:
    """run_audit() should return a valid report with required fields."""
    report = run_audit()
    assert "verdict" in report
    assert "violations" in report
    assert "checks_passed" in report
    assert "details" in report
    assert report["verdict"] in ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# Check 6: LightGBM cross-artifact consistency (regression tests for the
# real, live drift found this session: results/oos_predictions.csv and
# E2-S6's model-ranking table silently disagreed on LightGBM's own MAE.
# Isolated with tmp_path/synthetic fixtures so they run without depending on
# the real canonical dataset or a live pipeline run.)
# ---------------------------------------------------------------------------

def _write_oos_table(path: Path, mae_value: float) -> None:
    """A minimal 2-row canonical OOS table with a controllable implied MAE."""
    path.write_text(
        "Date,prediction,actual_return_5d,regime,fold_id\n"
        f"2020-01-01,0.0,{mae_value},LowVol,0\n"
        f"2020-01-02,0.0,{-mae_value},LowVol,0\n"
    )


def _write_manifest(path: Path, source_path: Path) -> None:
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path.write_text(json.dumps({
        "source_predictions_path": str(source_path),
        "source_predictions_sha256": source_hash,
    }))


def _write_ranking_table(path: Path, lightgbm_mae: float) -> None:
    path.write_text(
        "rank_by_mae,model,n,mae\n"
        f"1,lightgbm,2,{lightgbm_mae}\n"
    )


def test_cross_artifact_check_passes_when_source_hash_and_mae_agree(tmp_path: Path) -> None:
    source_path = tmp_path / "lightgbm_oos_predictions.csv"
    source_path.write_text("fold,Date,regime,y_true,y_pred\n0,2020-01-01,LowVol,0.02,0.0\n")

    oos_table_path = tmp_path / "oos_predictions.csv"
    _write_oos_table(oos_table_path, mae_value=0.02)

    manifest_path = tmp_path / "oos_predictions_manifest.json"
    _write_manifest(manifest_path, source_path)

    ranking_path = tmp_path / "all_models_overall_ranking.csv"
    _write_ranking_table(ranking_path, lightgbm_mae=0.02)

    gate = IntegrityGate(
        oos_table_path=oos_table_path,
        oos_table_manifest_path=manifest_path,
        model_ranking_path=ranking_path,
    )
    result = gate._check_lightgbm_cross_artifact_consistency()
    assert result["passed"] is True
    assert result["source_hash_current"] is True
    assert result["cross_table_mae_match"] is True
    assert gate.violations == []


def test_cross_artifact_check_fails_on_stale_source_hash(tmp_path: Path) -> None:
    source_path = tmp_path / "lightgbm_oos_predictions.csv"
    source_path.write_text("fold,Date,regime,y_true,y_pred\n0,2020-01-01,LowVol,0.02,0.0\n")

    oos_table_path = tmp_path / "oos_predictions.csv"
    _write_oos_table(oos_table_path, mae_value=0.02)

    manifest_path = tmp_path / "oos_predictions_manifest.json"
    _write_manifest(manifest_path, source_path)

    # Source file changes (a re-run of LightGBM) *after* the manifest was
    # written -- exactly what happened in the real repo this session.
    source_path.write_text("fold,Date,regime,y_true,y_pred\n0,2020-01-01,LowVol,0.02,0.05\n")

    ranking_path = tmp_path / "all_models_overall_ranking.csv"
    _write_ranking_table(ranking_path, lightgbm_mae=0.02)

    gate = IntegrityGate(
        oos_table_path=oos_table_path,
        oos_table_manifest_path=manifest_path,
        model_ranking_path=ranking_path,
    )
    result = gate._check_lightgbm_cross_artifact_consistency()
    assert result["passed"] is False
    assert result["source_hash_current"] is False
    assert any("STALE" in v for v in gate.violations)


def test_cross_artifact_check_fails_on_cross_table_mae_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "lightgbm_oos_predictions.csv"
    source_path.write_text("fold,Date,regime,y_true,y_pred\n0,2020-01-01,LowVol,0.02,0.0\n")

    oos_table_path = tmp_path / "oos_predictions.csv"
    _write_oos_table(oos_table_path, mae_value=0.02)

    manifest_path = tmp_path / "oos_predictions_manifest.json"
    _write_manifest(manifest_path, source_path)

    # Ranking table reports a different LightGBM MAE than the current
    # canonical OOS table implies -- the exact symptom found this session.
    ranking_path = tmp_path / "all_models_overall_ranking.csv"
    _write_ranking_table(ranking_path, lightgbm_mae=0.016056)

    gate = IntegrityGate(
        oos_table_path=oos_table_path,
        oos_table_manifest_path=manifest_path,
        model_ranking_path=ranking_path,
    )
    result = gate._check_lightgbm_cross_artifact_consistency()
    assert result["passed"] is False
    assert result["cross_table_mae_match"] is False
    assert any("MAE disagreement" in v for v in gate.violations)


def test_cross_artifact_check_warns_when_ranking_table_absent(tmp_path: Path) -> None:
    source_path = tmp_path / "lightgbm_oos_predictions.csv"
    source_path.write_text("fold,Date,regime,y_true,y_pred\n0,2020-01-01,LowVol,0.02,0.0\n")

    oos_table_path = tmp_path / "oos_predictions.csv"
    _write_oos_table(oos_table_path, mae_value=0.02)

    manifest_path = tmp_path / "oos_predictions_manifest.json"
    _write_manifest(manifest_path, source_path)

    gate = IntegrityGate(
        oos_table_path=oos_table_path,
        oos_table_manifest_path=manifest_path,
        model_ranking_path=tmp_path / "does_not_exist.csv",
    )
    result = gate._check_lightgbm_cross_artifact_consistency()
    assert result["passed"] is True  # nothing to cross-check yet is not a violation
    assert result["cross_table_mae_match"] is None
    assert any("ranking table not found" in w for w in gate.warnings)
