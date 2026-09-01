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
