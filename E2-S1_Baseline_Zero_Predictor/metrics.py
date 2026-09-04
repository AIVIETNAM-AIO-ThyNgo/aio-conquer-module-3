"""
E2-S1 -- Metric code shared by the baseline and every later E2 model.

Scoring the zero baseline and LightGBM with the same functions is what makes
"no special-case advantage/disadvantage" (E2-S1 acceptance) hold structurally
instead of by promise.
"""
from __future__ import annotations

import numpy as np

DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION = (
    "A prediction of exactly 0 carries no sign, so it cannot be scored as a "
    "correct or incorrect directional call. Rows with y_pred == 0 are excluded "
    "from the directional hit rate's numerator and denominator. If every "
    "prediction in a fold is exactly 0, the hit rate is undefined and reported "
    "as NaN -- never silently as 0.0 or 0.5. The same reasoning makes "
    "prediction correlation undefined (zero variance) for an all-zero "
    "predictor; it is reported as NaN for the same reason."
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def prediction_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.asarray(y_pred)
    if np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(np.asarray(y_true), y_pred)[0, 1])


def directional_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scoreable = y_pred != 0
    if not scoreable.any():
        return float("nan")
    hits = np.sign(y_true[scoreable]) == np.sign(y_pred[scoreable])
    return float(hits.mean())


def paired_fold_significance(fold_improvements: np.ndarray) -> dict:
    """Naive across-fold significance check for a model's per-fold MAE
    improvement over baseline (one value per fold: baseline_mae - model_mae).

    This exists because no model-ranking table in this project ever checked
    whether a reported "overall" improvement (typically 0.0001-0.0006) was
    distinguishable from the fold-to-fold noise already visible in the
    project's own per-fold MAE CSVs, which swing by 3-7x that amount between
    folds. It is deliberately the simplest possible test -- a one-sample
    t-test treating each fold's MAE as one independent observation -- not a
    substitute for a properly-specified test, and callers should treat it as
    a screening heuristic, not a certified significance result.

    Known limitations, disclosed here rather than left implicit:
      - n_folds is small in this project (6 -> 5 degrees of freedom).
      - Folds are not necessarily independent draws: adjacent folds cover
        adjacent, possibly serially-correlated calendar periods (e.g. a
        regime or macro cycle spanning a fold boundary). This test does not
        correct for that.
      - Using per-fold MAE as the unit of replication sidesteps the
        within-fold label-overlap problem (5-day overlapping forward
        returns) but is not a substitute for a block-bootstrap or
        Newey-West-style correction.

    Returns a dict with the mean/sd/naive-SE/naive-t of the per-fold
    improvements, whether the sign is consistent across every fold, and a
    plain-language verdict plus the caveats above (so a caller that just
    dumps this dict into a report doesn't lose the disclosure).
    """
    values = np.asarray(fold_improvements, dtype=float)
    n = values.size
    if n < 2:
        raise ValueError("paired_fold_significance needs at least 2 folds")

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        t_stat = float("inf") if mean != 0 else 0.0
    else:
        t_stat = mean / se
    sign_consistent = bool(np.all(values > 0) or np.all(values < 0))

    verdict = (
        "likely noise: mean improvement is within 2 naive standard errors of zero"
        if abs(t_stat) < 2.0
        else "distinguishable from zero at a naive |t| >= 2 screening threshold"
    )

    return {
        "n_folds": n,
        "mean_improvement": mean,
        "fold_to_fold_sd": sd,
        "naive_se": se,
        "naive_t_stat": t_stat,
        "sign_consistent_across_all_folds": sign_consistent,
        "verdict": verdict,
        "caveats": (
            f"Naive across-fold t-test only (degrees of freedom = {n - 1}); "
            "does not correct for possible serial correlation between "
            "adjacent folds; a screening heuristic, not a certified "
            "significance test. See paired_fold_significance docstring."
        ),
    }
