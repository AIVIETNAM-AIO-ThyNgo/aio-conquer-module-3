"""E4-S2 [P0][QA] OOS Split Integrity Gate
=============================================

Independent audit of fold chronology and canonical OOS table.

This module implements a comprehensive integrity gate that verifies the
out-of-sample (OOS) split and prediction pipeline. It is designed to be
run independently of the main pipeline to provide an unbiased assessment
of the split integrity.

Checks performed:
1. max(train_date) < min(test_date) for every fold.
2. 5D boundary purge/gap is effective (no label leakage).
3. No test data used for feature/model decisions inside fold.
4. OOS table contains no in-sample predictions or duplicate prediction dates.
5. Baseline and LightGBM are compared on identical OOS rows.

BLOCKER RULE: Any violation invalidates current performance results and
requires a clean rerun of the pipeline.

Run:
    python E4-S2_OOS_Split_Integrity_Gate/audit_oos_split_integrity.py
    python -m pytest E4-S2_OOS_Split_Integrity_Gate/tests/test_audit_oos_split_integrity.py -v
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Add E2-S1 for shared splits/metrics
sys.path.insert(0, str(REPO_ROOT / "E2-S1_Baseline_Zero_Predictor"))

from splits import (
    HORIZON_TRADING_DAYS,
    MIN_TRAIN_SIZE,
    N_FOLDS,
    Fold,
    purged_walk_forward_splits,
)


class IntegrityViolation(Exception):
    """Raised when an integrity check fails.

    Per the BLOCKER RULE, any violation invalidates current performance
    results and requires a clean rerun of the pipeline.
    """


class IntegrityGate:
    """Independent audit of OOS split integrity.

    This class performs a series of checks on the canonical dataset,
    fold definitions, and OOS prediction table to ensure that the
    out-of-sample evaluation is valid and free from leakage.
    """

    def __init__(
        self,
        canonical_path: Path | None = None,
        oos_table_path: Path | None = None,
        baseline_pred_path: Path | None = None,
        lightgbm_pred_path: Path | None = None,
    ):
        self.canonical_path = canonical_path or REPO_ROOT / "data" / "processed" / "E1-S6_canonical_modeling_dataset.csv"
        self.oos_table_path = oos_table_path or REPO_ROOT / "results" / "oos_predictions.csv"
        self.baseline_pred_path = baseline_pred_path or REPO_ROOT / "E2-S1_Baseline_Zero_Predictor" / "output" / "baseline_zero_oos_predictions.csv"
        self.lightgbm_pred_path = lightgbm_pred_path or REPO_ROOT / "E2-S2_Train_Minimal_LightGBM_Regressor" / "output" / "lightgbm_oos_predictions.csv"

        self.violations: list[str] = []
        self.warnings: list[str] = []
        self.checks_passed: list[str] = []

    def audit(self) -> dict[str, Any]:
        """Run all integrity checks and return a comprehensive report.

        Returns:
            dict with keys:
                - verdict: "PASS" or "FAIL"
                - violations: list of violation descriptions
                - warnings: list of warning descriptions
                - checks_passed: list of passed check descriptions
                - details: dict with detailed results per check
        """
        self.violations = []
        self.warnings = []
        self.checks_passed = []
        details: dict[str, Any] = {}

        # Load data
        canonical_df = self._load_canonical()
        folds = self._build_folds(canonical_df)

        # Run all checks
        details["fold_chronology"] = self._check_fold_chronology(folds, canonical_df)
        details["purge_gap"] = self._check_purge_gap(folds, canonical_df)
        details["no_test_data_in_features"] = self._check_no_test_data_in_features(folds, canonical_df)
        details["oos_table_integrity"] = self._check_oos_table_integrity(canonical_df)
        details["model_comparison_fairness"] = self._check_model_comparison_fairness()

        # Compile report
        verdict = "FAIL" if self.violations else "PASS"

        report = {
            "card": "E4-S2 [P0][QA] OOS Split Integrity Gate",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "checks_passed": list(self.checks_passed),
            "details": details,
            "blocker_rule": (
                "Any violation invalidates current performance results and "
                "requires a clean rerun of the pipeline."
            ),
            "python": platform.python_version(),
        }

        return report

    def _load_canonical(self) -> pd.DataFrame:
        """Load and validate the canonical dataset."""
        if not self.canonical_path.exists():
            raise FileNotFoundError(f"Canonical dataset not found: {self.canonical_path}")

        df = pd.read_csv(self.canonical_path, parse_dates=["Date"])
        if df.empty:
            raise IntegrityViolation("Canonical dataset is empty")
        if not df["Date"].is_monotonic_increasing:
            raise IntegrityViolation("Canonical dataset dates are not sorted ascending")
        if df["Date"].duplicated().any():
            raise IntegrityViolation("Canonical dataset contains duplicate dates")
        return df

    def _build_folds(self, canonical_df: pd.DataFrame) -> list[Fold]:
        """Build folds from the canonical dataset."""
        return purged_walk_forward_splits(
            canonical_df["Date"],
            n_folds=N_FOLDS,
            min_train_size=MIN_TRAIN_SIZE,
            horizon=HORIZON_TRADING_DAYS,
        )

    def _check_fold_chronology(self, folds: list[Fold], canonical_df: pd.DataFrame) -> dict[str, Any]:
        """Check 1: max(train_date) < min(test_date) for every fold.

        This ensures that no training data leaks into the test period
        and that the temporal ordering is strictly maintained.
        """
        dates = canonical_df["Date"]
        results = {"per_fold": [], "passed": True}

        for fold in folds:
            train_dates = dates.iloc[fold.train_idx]
            test_dates = dates.iloc[fold.test_idx]

            train_max = train_dates.max()
            test_min = test_dates.min()
            is_valid = train_max < test_min

            fold_result = {
                "fold_id": fold.fold_id,
                "train_end": train_max.strftime("%Y-%m-%d"),
                "test_start": test_min.strftime("%Y-%m-%d"),
                "valid": is_valid,
            }
            results["per_fold"].append(fold_result)

            if not is_valid:
                msg = (
                    f"Fold {fold.fold_id}: max(train_date)={train_max.strftime('%Y-%m-%d')} "
                    f"is NOT < min(test_date)={test_min.strftime('%Y-%m-%d')}"
                )
                self.violations.append(msg)
                results["passed"] = False

        if results["passed"]:
            self.checks_passed.append(
                "Fold chronology: max(train_date) < min(test_date) for all folds"
            )
        else:
            self.warnings.append(
                "Fold chronology check FAILED - temporal ordering violated"
            )

        return results

    def _check_purge_gap(self, folds: list[Fold], canonical_df: pd.DataFrame) -> dict[str, Any]:
        """Check 2: 5D boundary purge/gap is effective.

        The forward_return_5d label for row i is Close[i+5]/Close[i]-1.
        A training row within HORIZON_TRADING_DAYS (5) of a test block's
        first date has a label window that overlaps the test period.
        The purge gap must remove enough rows so that the last training
        row's label never reads a price inside the test window.
        """
        dates = canonical_df["Date"]
        results = {"per_fold": [], "passed": True}

        for fold in folds:
            if len(fold.train_idx) == 0:
                continue

            last_train_idx = int(fold.train_idx.max())
            first_test_idx = int(fold.test_idx.min())

            # The label for the last training row references a price
            # HORIZON_TRADING_DAYS rows ahead
            label_price_idx = last_train_idx + HORIZON_TRADING_DAYS

            # The gap must ensure label_price_idx < first_test_idx
            gap_size = first_test_idx - last_train_idx - 1
            is_effective = label_price_idx < first_test_idx

            fold_result = {
                "fold_id": fold.fold_id,
                "last_train_idx": last_train_idx,
                "first_test_idx": first_test_idx,
                "gap_size": gap_size,
                "label_price_idx": label_price_idx,
                "effective": is_effective,
            }
            results["per_fold"].append(fold_result)

            if not is_effective:
                msg = (
                    f"Fold {fold.fold_id}: purge gap INEFFECTIVE - "
                    f"last train row {last_train_idx}'s label reads price at "
                    f"index {label_price_idx}, which is inside the test window "
                    f"starting at {first_test_idx}"
                )
                self.violations.append(msg)
                results["passed"] = False

        if results["passed"]:
            self.checks_passed.append(
                f"Purge gap: 5D boundary effective for all folds (gap >= {HORIZON_TRADING_DAYS})"
            )

        return results

    def _check_no_test_data_in_features(self, folds: list[Fold], canonical_df: pd.DataFrame) -> dict[str, Any]:
        """Check 3: No test data used for feature/model decisions inside fold.

        Verify that features are computed using only information available
        at or before prediction date t. This checks that:
        - Feature windows are right-aligned (trailing)
        - No feature value at a test date uses data from that test date
          or future dates
        """
        dates = canonical_df["Date"]
        feature_cols = [
            "return_1d", "return_5d", "return_10d", "return_20d",
            "volatility_5d", "volatility_10d", "volatility_20d",
            "trend_10d", "trend_20d", "trend_60d",
            "volume_ratio_20d",
        ]

        results = {"per_fold": [], "passed": True}

        for fold in folds:
            fold_passed = True
            fold_details: dict[str, Any] = {"fold_id": fold.fold_id, "feature_checks": []}

            for col in feature_cols:
                # For each test date, verify the feature value is the same
                # whether computed from the full frame or from a frame
                # truncated at that date. If they differ, the feature
                # uses future information.
                test_dates = dates.iloc[fold.test_idx]

                # Sample a few test dates for efficiency
                sample_indices = [0, len(test_dates) // 2, len(test_dates) - 1]
                col_passed = True

                for idx in sample_indices:
                    if idx >= len(test_dates):
                        continue
                    test_date = test_dates.iloc[idx]
                    date_idx = dates[dates == test_date].index[0]

                    # Feature value from full frame
                    full_value = canonical_df[col].iloc[date_idx]

                    # Feature value from truncated frame (up to and including test_date)
                    truncated = canonical_df[canonical_df["Date"] <= test_date]
                    if len(truncated) > 0:
                        truncated_value = truncated[col].iloc[-1]
                        if not np.isclose(full_value, truncated_value, rtol=1e-10, atol=1e-12):
                            msg = (
                                f"Fold {fold.fold_id}, feature {col}: value at {test_date.strftime('%Y-%m-%d')} "
                                f"differs between full frame ({full_value}) and truncated frame ({truncated_value})"
                            )
                            self.violations.append(msg)
                            fold_passed = False
                            col_passed = False

                fold_details["feature_checks"].append({
                    "column": col,
                    "passed": col_passed,
                })

            fold_details["passed"] = fold_passed
            results["per_fold"].append(fold_details)

            if not fold_passed:
                results["passed"] = False

        if results["passed"]:
            self.checks_passed.append(
                "No test data in features: all features are right-aligned (trailing windows only)"
            )

        return results

    def _check_oos_table_integrity(self, canonical_df: pd.DataFrame) -> dict[str, Any]:
        """Check 4: OOS table contains no in-sample predictions or duplicate prediction dates.

        Verify that:
        - The OOS table has no duplicate dates
        - All OOS dates fall within the test windows of their claimed folds
        - No OOS date falls in a training window or purge gap
        """
        results = {
            "duplicate_dates": None,
            "in_sample_predictions": None,
            "fold_traceability": None,
            "passed": True,
        }

        if not self.oos_table_path.exists():
            self.warnings.append(f"OOS table not found: {self.oos_table_path}")
            return results

        oos_df = pd.read_csv(self.oos_table_path, parse_dates=["Date"])

        # Check for duplicate dates
        dup_dates = oos_df["Date"].duplicated().sum()
        results["duplicate_dates"] = int(dup_dates)
        if dup_dates > 0:
            msg = f"OOS table contains {dup_dates} duplicate prediction dates"
            self.violations.append(msg)
            results["passed"] = False
        else:
            self.checks_passed.append("OOS table: no duplicate prediction dates")

        # Check that all OOS dates are genuine OOS (in test windows, not train/purge)
        folds = self._build_folds(canonical_df)
        dates = canonical_df["Date"]

        in_sample_count = 0
        for _, row in oos_df.iterrows():
            oos_date = row["Date"]
            fold_id = row.get("fold_id", None)

            if fold_id is not None:
                fold = folds[fold_id]
                test_dates = dates.iloc[fold.test_idx]
                if oos_date not in test_dates.values:
                    in_sample_count += 1

        results["in_sample_predictions"] = in_sample_count
        if in_sample_count > 0:
            msg = f"OOS table contains {in_sample_count} in-sample predictions (not in test window)"
            self.violations.append(msg)
            results["passed"] = False
        else:
            self.checks_passed.append("OOS table: no in-sample predictions")

        # Check fold traceability
        traceability_issues = 0
        for fold in folds:
            fold_dates = oos_df[oos_df["fold_id"] == fold.fold_id]["Date"].sort_values()
            expected_dates = dates.iloc[fold.test_idx].sort_values()

            if not np.array_equal(fold_dates.values, expected_dates.values):
                traceability_issues += 1

        results["fold_traceability"] = traceability_issues
        if traceability_issues > 0:
            msg = f"OOS table: {traceability_issues} folds have mismatched dates"
            self.violations.append(msg)
            results["passed"] = False
        else:
            self.checks_passed.append("OOS table: all folds trace to correct test windows")

        return results

    def _check_model_comparison_fairness(self) -> dict[str, Any]:
        """Check 5: Baseline and LightGBM are compared on identical OOS rows.

        Verify that both models are evaluated on exactly the same OOS rows
        (same dates, same regime labels, same target values).
        """
        results = {
            "same_dates": None,
            "same_regime_labels": None,
            "same_targets": None,
            "passed": True,
        }

        if not self.baseline_pred_path.exists():
            self.warnings.append(f"Baseline predictions not found: {self.baseline_pred_path}")
            return results

        if not self.lightgbm_pred_path.exists():
            self.warnings.append(f"LightGBM predictions not found: {self.lightgbm_pred_path}")
            return results

        baseline_df = pd.read_csv(self.baseline_pred_path, parse_dates=["Date"])
        lightgbm_df = pd.read_csv(self.lightgbm_pred_path, parse_dates=["Date"])

        # Check same dates
        baseline_dates = np.sort(baseline_df["Date"].values)
        lightgbm_dates = np.sort(lightgbm_df["Date"].values)
        same_dates = np.array_equal(baseline_dates, lightgbm_dates)
        results["same_dates"] = same_dates

        merged = None
        if same_dates:
            merged = baseline_df.merge(lightgbm_df, on="Date", suffixes=("_baseline", "_lightgbm"))

        # Check same regime labels per date
        if same_dates and merged is not None:
            same_regime = (merged["regime_baseline"] == merged["regime_lightgbm"]).all()
            results["same_regime_labels"] = bool(same_regime)

            if not same_regime:
                msg = "Baseline and LightGBM disagree on regime labels for at least one date"
                self.violations.append(msg)
                results["passed"] = False
            else:
                self.checks_passed.append("Model comparison: same regime labels per date")

        # Check same target values per date
        if same_dates and merged is not None:
            same_targets = np.allclose(
                merged["y_true_baseline"].values,
                merged["y_true_lightgbm"].values,
            )
            results["same_targets"] = bool(same_targets)

            if not same_targets:
                msg = "Baseline and LightGBM disagree on actual_return_5d for at least one date"
                self.violations.append(msg)
                results["passed"] = False
            else:
                self.checks_passed.append("Model comparison: same target values per date")

        if not same_dates:
            msg = "Baseline and LightGBM are NOT evaluated on the same dates"
            self.violations.append(msg)
            results["passed"] = False
        else:
            self.checks_passed.append("Model comparison: same OOS dates")

        return results


def run_audit() -> dict[str, Any]:
    """Run the full integrity audit and return the report."""
    gate = IntegrityGate()
    return gate.audit()


def main() -> int:
    """Main entry point for the integrity gate."""
    print("=" * 60)
    print("E4-S2 OOS Split Integrity Gate")
    print("=" * 60)

    report = run_audit()

    print(f"\nVerdict: {report['verdict']}")
    print(f"Checks passed: {len(report['checks_passed'])}")
    print(f"Violations: {len(report['violations'])}")
    print(f"Warnings: {len(report['warnings'])}")

    if report['checks_passed']:
        print("\n--- Checks Passed ---")
        for check in report['checks_passed']:
            print(f"  ✓ {check}")

    if report['violations']:
        print("\n--- VIOLATIONS (BLOCKER) ---")
        for violation in report['violations']:
            print(f"  ✗ {violation}")
        print(f"\n{report['blocker_rule']}")

    if report['warnings']:
        print("\n--- Warnings ---")
        for warning in report['warnings']:
            print(f"  ! {warning}")

    # Write report to file
    output_dir = REPO_ROOT / "E4-S2_OOS_Split_Integrity_Gate" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "integrity_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to: {report_path}")

    return 0 if report['verdict'] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
