# E2-S1 [P0][Model] Baseline y_hat=0

| Field | Value |
|---|---|
| Epic | E2 |
| Owner | Model |
| Review | Tech Leader |
| Depends on | E1-S6 (Publish Canonical Modeling Dataset & Data Dictionary), E4-S1 (Pre-Model Leakage & Data Quality Gate) |

## Deliverable

Baseline `y_hat = 0` evaluated on the same OOS dates/folds a later LightGBM
model will use. The fold logic lives in [`splits.py`](splits.py) precisely so
that "same OOS dates/folds" holds by construction: any future model calls
`purged_walk_forward_splits` with the same defaults and gets the identical
train/test row indices produced here.

## Acceptance

- **Same target, splits and metric code.** `TARGET_COL = "forward_return_5d"`
  (the E1-S6 canonical target, unchanged). Splits come from
  [`splits.py`](splits.py); metrics come from [`metrics.py`](metrics.py).
  Both are plain importable modules, not baseline-only inline code, so a
  LightGBM story reuses them rather than reimplementing them.
- **Results logged reproducibly.** Running [`run_baseline.py`](run_baseline.py)
  writes, under `output/`:
  - `baseline_zero_oos_predictions.csv` — every OOS row (`fold`, `Date`,
    `regime`, `y_true`, `y_pred`).
  - `baseline_zero_fold_metrics.csv` — MAE / prediction correlation /
    directional hit rate per fold.
  - `baseline_zero_summary.json` — overall metrics, split parameters, the
    canonical dataset's SHA-256 (so results can be tied back to the exact
    E1-S6 artifact that produced them), package versions and a UTC
    timestamp — the same reproducibility pattern as
    `data/processed/E1-S6_dataset_manifest.json`.
  Splits are deterministic (`test_splits_are_deterministic_across_calls`),
  and predictions are always exactly `0.0` (`np.zeros_like`), so reruns
  reproduce identical output byte-for-byte apart from the timestamp.
- **No special-case advantage/disadvantage.** The baseline gets no
  preferential fold boundaries, no metric relaxation and no purge exemption:
  it runs through the exact same `purged_walk_forward_splits` /
  `mae` / `prediction_correlation` / `directional_hit_rate` functions that
  will score LightGBM.

## Splits: purged walk-forward

`forward_return_5d` is a 5-trading-day-ahead label, so a train row within
`HORIZON_TRADING_DAYS` (5) of a test block's first date has a label window
that overlaps the test period — training on it leaks the test outcome. This
was flagged explicitly as an E2 follow-up in the E4-S1 audit
(`docs/E4-S1_leakage_audit_record.md`, Scope boundary). `splits.py` purges
those rows from train for every fold:

- `MIN_TRAIN_SIZE = 1260` (~5 trading years) before the first test block.
- `N_FOLDS = 6`, expanding train window, contiguous non-overlapping test
  blocks covering the entire post-warm-up history (2011-02-02 → 2026-08-13).
- Train for fold *k* = all rows strictly more than 5 trading days before that
  fold's test start; nothing later is ever used to train it.

Verified by `test_purge_gap_is_at_least_horizon_trading_days`,
`test_train_never_contains_a_row_from_its_own_test_block` and
`test_folds_cover_the_oos_region_with_no_gap_or_overlap`.

## Edge cases

**Zero can be a strong MAE baseline for noisy short-horizon returns.**
Observed here: fold MAE ranges 0.0106–0.0201 (overall 0.0160), i.e. a
constant zero forecast is already off by roughly 1–2 percentage points of
5-day forward return per row. LightGBM must clear this bar, not just beat a
naive positive number.

**Hit-rate for exact zero predictions is ambiguous — convention.** A
prediction of exactly `0` has no sign, so it cannot be scored as a correct or
incorrect directional call. `directional_hit_rate` (`metrics.py`) excludes
rows with `y_pred == 0` from both the numerator and denominator; since every
prediction here is `0`, the hit rate is undefined for every fold and
reported as `NaN` — never silently as `0.0` or `0.5`. The same reasoning
makes prediction correlation undefined (zero-variance predictor); it is also
reported as `NaN`. Confirmed empirically: `output/baseline_zero_fold_metrics.csv`
shows `NaN` in both columns for all 6 folds, and
`test_zero_baseline_reports_nan_not_zero_or_half_for_every_fold` asserts it.

**Non-finite target values.** `run_baseline.py` raises before scoring if
`forward_return_5d` contains any NaN/inf — a defect in a dataset
regeneration would otherwise silently produce a NaN baseline MAE that could
be misread as the documented zero-prediction convention above rather than a
real data problem. Verified by `test_run_raises_on_non_finite_target_values`.

## Run

```bash
python E2-S1_Baseline_Zero_Predictor/run_baseline.py
python -m pytest E2-S1_Baseline_Zero_Predictor/tests/test_baseline_zero.py -v
```

## Results (this run)

| Fold | Test window | n_test | MAE | Correlation | Hit rate |
|---|---|---|---|---|---|
| 0 | 2011-02-02 → 2013-09-03 | 650 | 0.01701 | N/A | N/A |
| 1 | 2013-09-04 → 2016-04-04 | 650 | 0.01392 | N/A | N/A |
| 2 | 2016-04-05 → 2018-10-30 | 650 | 0.01057 | N/A | N/A |
| 3 | 2018-10-31 → 2021-06-02 | 650 | 0.02008 | N/A | N/A |
| 4 | 2021-06-03 → 2024-01-02 | 650 | 0.01879 | N/A | N/A |
| 5 | 2024-01-03 → 2026-08-13 | 655 | 0.01538 | N/A | N/A |
| **Overall** | | **3905** | **0.01596** | **N/A** | **N/A** |

## Scope boundary

This card establishes the baseline and the shared splits/metric code only.
LightGBM training and regime-conditioned evaluation belong to later E2
stories, which must import `splits.py` and `metrics.py` unchanged to keep
this comparison valid.
