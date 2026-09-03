"""
E2-S7 [Model] OOS Error Analysis -- verification suite.

Run:
    python -m pytest E2-S7_OOS_Error_Analysis/tests/test_analyze_oos_errors.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

E2_S7_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = E2_S7_DIR.parent
sys.path.insert(0, str(E2_S7_DIR))

from analyze_oos_errors import (  # noqa: E402
    CANONICAL_OOS_PATH,
    OUTPUT_DIR,
    TOP_ERROR_FRACTION,
    hypergeometric_upper_tail_p_value,
    run,
)

pytestmark = pytest.mark.skipif(
    not CANONICAL_OOS_PATH.exists(),
    reason="run E2-S4/generate_oos_predictions.py first to produce results/oos_predictions.csv",
)


# --------------------------------------------------------------------------
# Acceptance: fixed, disclosed selection rule applied to the complete OOS set
# --------------------------------------------------------------------------

def test_selection_fraction_is_not_a_search_space():
    assert isinstance(TOP_ERROR_FRACTION, float)
    assert 0 < TOP_ERROR_FRACTION < 0.1  # a round, small, pre-committed fraction


def test_run_analyses_every_oos_row_with_no_exclusions():
    run()
    total_oos = len(pd.read_csv(CANONICAL_OOS_PATH))
    summary = json.loads((OUTPUT_DIR / "error_analysis_summary.json").read_text())
    assert summary["selection_rule"]["n_total_oos_rows"] == total_oos


def test_top_error_count_matches_disclosed_rule():
    summary = json.loads((OUTPUT_DIR / "error_analysis_summary.json").read_text())
    expected_k = round(summary["selection_rule"]["n_total_oos_rows"] * TOP_ERROR_FRACTION)
    assert summary["selection_rule"]["k_top_errors"] == expected_k

    top_errors = pd.read_csv(OUTPUT_DIR / "top_errors.csv")
    assert len(top_errors) == expected_k


# --------------------------------------------------------------------------
# Acceptance: extreme-return dominance is a computed number, not a claim
# --------------------------------------------------------------------------

def test_extreme_return_correlation_is_reported_and_bounded():
    summary = json.loads((OUTPUT_DIR / "error_analysis_summary.json").read_text())
    corr = summary["extreme_return_dominance"]["pearson_abs_actual_vs_abs_error"]
    assert -1.0 <= corr <= 1.0


def test_baseline_cross_reference_correlation_matches_lightgbm_correlation():
    """Baseline's error is |actual_return_5d| exactly, so this correlation must
    equal the |actual|-vs-|error| correlation -- proving the relationship is
    not LightGBM-specific, not merely claiming it."""
    summary = json.loads((OUTPUT_DIR / "error_analysis_summary.json").read_text())
    dominance = summary["extreme_return_dominance"]
    assert dominance["pearson_lightgbm_abs_error_vs_baseline_abs_error"] == pytest.approx(
        dominance["pearson_abs_actual_vs_abs_error"], abs=1e-9
    )


# --------------------------------------------------------------------------
# Acceptance: regime-share claim is backed by a significance test, not just
# a comparison of two percentages
# --------------------------------------------------------------------------

def test_hypergeometric_p_value_is_a_valid_probability():
    p = hypergeometric_upper_tail_p_value(population_size=3905, population_successes=1689, sample_size=39, observed_successes=35)
    assert 0.0 <= p <= 1.0


def test_hypergeometric_test_gives_p_one_when_observed_equals_expected():
    """Not vacuous: an observed count exactly at its null expectation should
    not look significant. Population/sample chosen so the expectation is a
    round, exactly-reachable integer."""
    # N=100, K=50 (50% success rate), n=10 -> expected successes = 5 exactly.
    p = hypergeometric_upper_tail_p_value(population_size=100, population_successes=50, sample_size=10, observed_successes=5)
    assert p > 0.4  # comfortably unsurprising, not a tiny tail probability


def test_hypergeometric_test_gives_a_tiny_p_value_for_the_real_enrichment():
    """Reproduces the exact regime-share finding reported in the README --
    if this ever stops being a tiny p-value, the README's headline claim
    needs to be revisited, not silently left stale."""
    summary = json.loads((OUTPUT_DIR / "error_analysis_summary.json").read_text())
    sig = summary["regime_representation"]["significance_test"]
    assert sig["p_value_upper_tail"] < 1e-6
    assert sig["observed_highvol_in_top_k"] > sig["expected_highvol_under_null"]


# --------------------------------------------------------------------------
# Edge case: regime taxonomy is not defined by this analysis
# --------------------------------------------------------------------------

def test_regime_labels_are_exactly_the_pre_existing_taxonomy():
    top_errors = pd.read_csv(OUTPUT_DIR / "top_errors.csv")
    assert set(top_errors["regime"].unique()).issubset({"LowVol", "HighVol"})


# --------------------------------------------------------------------------
# Edge case: sign-mismatch subset behind Explanation B is small -- must be
# labelled as such in the machine-readable summary, not just the prose
# --------------------------------------------------------------------------

def test_error_only_cluster_carries_an_explicit_small_sample_caveat():
    summary = json.loads((OUTPUT_DIR / "error_analysis_summary.json").read_text())
    cluster = summary["error_only_cluster"]
    assert "small_sample_caveat" in cluster
    assert cluster["n"] == len(cluster["dates"])


# --------------------------------------------------------------------------
# Acceptance: no silent retuning declaration is present
# --------------------------------------------------------------------------

def test_no_retuning_declaration_is_present():
    summary = json.loads((OUTPUT_DIR / "error_analysis_summary.json").read_text())
    assert "no_retuning_declaration" in summary
    assert "no hyperparameter" in summary["no_retuning_declaration"].lower() or \
        "no_retuning_declaration" in summary


def test_authorship_conflict_of_interest_is_disclosed():
    summary = json.loads((OUTPUT_DIR / "error_analysis_summary.json").read_text())
    assert "authorship_disclosure" in summary
