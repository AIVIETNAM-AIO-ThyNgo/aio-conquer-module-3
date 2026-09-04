"""
metrics.paired_fold_significance -- verification suite.

Run:
    python -m pytest E2-S1_Baseline_Zero_Predictor/tests/test_metrics.py -v

Owner: Model. Review: QA/QC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import paired_fold_significance  # noqa: E402


def test_consistent_positive_sign_gives_high_t_stat_and_signal_verdict():
    # Random Forest's real per-fold improvement over baseline (all positive).
    improvements = [0.000275, 0.000328, 0.000494, 0.000241, 0.000147, 0.000562]
    result = paired_fold_significance(improvements)
    assert result["sign_consistent_across_all_folds"] is True
    assert result["naive_t_stat"] > 2.0
    assert "distinguishable from zero" in result["verdict"]


def test_mixed_sign_gives_low_t_stat_and_noise_verdict():
    # LightGBM's real per-fold improvement over baseline (mixed sign).
    improvements = [-0.0006812167642641746, -1.280303954008348e-05,
                     0.0002995090601584094, -6.710597194994894e-06,
                     -0.0003804154589398985, 8.055772758539317e-05]
    result = paired_fold_significance(improvements)
    assert result["sign_consistent_across_all_folds"] is False
    assert abs(result["naive_t_stat"]) < 2.0
    assert result["verdict"].startswith("likely noise")


def test_returns_disclosed_caveats():
    result = paired_fold_significance([0.001, 0.002, -0.001, 0.0005, 0.0015])
    assert "degrees of freedom = 4" in result["caveats"]
    assert result["n_folds"] == 5


def test_zero_variance_positive_mean_is_infinite_t():
    result = paired_fold_significance([0.001, 0.001, 0.001])
    assert result["naive_t_stat"] == float("inf")
    assert result["sign_consistent_across_all_folds"] is True


def test_all_zero_improvements_is_zero_t_and_noise():
    result = paired_fold_significance([0.0, 0.0, 0.0])
    assert result["naive_t_stat"] == 0.0
    assert result["verdict"].startswith("likely noise")


def test_raises_on_fewer_than_two_folds():
    with pytest.raises(ValueError, match="at least 2 folds"):
        paired_fold_significance([0.001])


def test_accepts_numpy_array_input():
    result = paired_fold_significance(np.array([0.001, 0.002, 0.0015]))
    assert result["n_folds"] == 3
