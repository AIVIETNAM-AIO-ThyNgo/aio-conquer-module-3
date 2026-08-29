"""
E2-S1 -- Purged walk-forward OOS splits, shared by every E2 model.

`forward_return_5d` is a 5-trading-day-ahead label, so a train row within
HORIZON_TRADING_DAYS before a test block's first date has a label window
that overlaps the test period -- training on it would leak the test outcome.
Those rows are purged from train. See docs/E4-S1_leakage_audit_record.md,
Scope boundary, for the audit finding this implements.

LightGBM (or any later E2 model) must call `purged_walk_forward_splits` with
these same defaults so that "same OOS dates/folds as LightGBM" holds by
construction rather than by convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON_TRADING_DAYS = 5
MIN_TRAIN_SIZE = 1260
N_FOLDS = 6


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def purged_walk_forward_splits(
    dates: pd.Series,
    n_folds: int = N_FOLDS,
    min_train_size: int = MIN_TRAIN_SIZE,
    horizon: int = HORIZON_TRADING_DAYS,
) -> list[Fold]:
    dates = pd.Series(dates).reset_index(drop=True)
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending before splitting")

    n = len(dates)
    remaining = n - min_train_size
    if remaining < n_folds:
        raise ValueError("not enough rows after min_train_size for n_folds test blocks")
    test_size = remaining // n_folds

    folds = []
    for k in range(n_folds):
        test_start_idx = min_train_size + k * test_size
        test_end_idx = n if k == n_folds - 1 else test_start_idx + test_size
        purge_start_idx = test_start_idx - horizon

        folds.append(Fold(
            fold_id=k,
            train_idx=np.arange(0, purge_start_idx),
            test_idx=np.arange(test_start_idx, test_end_idx),
            test_start_date=dates.iloc[test_start_idx],
            test_end_date=dates.iloc[test_end_idx - 1],
        ))
    return folds
