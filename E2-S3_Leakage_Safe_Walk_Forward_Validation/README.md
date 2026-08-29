# E2-S3 [P0][Model] Implement Leakage-Safe Walk-Forward Validation

| Field | Value |
|---|---|
| Epic | E2 |
| Owner | Model |
| Review | Pipeline + QA/QC |
| Depends on | E2-S2 (Train Minimal LightGBM Regressor -- existing OOS output this card audits) |

## Deliverable

A chronological expanding/walk-forward splitter with an explicit purge/gap
for the 5-trading-day target horizon. The splitter itself,
`purged_walk_forward_splits`, already lives in
[`E2-S1/splits.py`](../E2-S1_Baseline_Zero_Predictor/splits.py) -- it was
built there specifically so E2-S1's baseline, E2-S2's LightGBM model, and
every later E2 model share one fold definition instead of each
reimplementing it (see `splits.py`'s module docstring and the E4-S1 audit's
Scope boundary, which named this as an explicit E2 follow-up).

E2-S3 does not fork or rewrite that splitter. It turns every acceptance item
on the card into an executable, independently-reasoned check
([`validate_walk_forward.py`](validate_walk_forward.py)), runs those checks
against the real canonical dataset and against the OOS prediction files
E2-S1 and E2-S2 already produced, and persists a fold-boundary audit trail
(`output/fold_boundary_audit.csv`) so any OOS prediction row can be traced
back to the exact training window that produced it.

## Acceptance

- **`max(train_date) < min(test_date)` for every fold.**
  `assert_chronological_order` checks this directly against real dates (not
  row indices) for all 6 folds.
  Verified by `test_chronological_order_holds_for_every_real_fold`;
  `test_chronological_order_catches_an_inverted_fold` proves the check is
  not vacuous by feeding it a synthetic inverted fold and requiring it to
  raise.

- **Purge/gap so a training label never reads a test-window price.**
  `assert_purge_removes_label_overlap` reasons from the label's own price
  reference -- `forward_return_5d` for row *i* is
  `Close[i+horizon]/Close[i]-1`, so row *i* leaks into the test window
  whenever `i+horizon` falls inside it -- independent of the
  `purge_start_idx` arithmetic inside `splits.py` itself, so it would still
  catch a purge bug even if that arithmetic were wrong.
  `assert_dates_are_contiguous_trading_days` verifies the assumption
  index-based purging relies on: consecutive rows are consecutive trading
  days with no missing dates, so purging 5 *rows* really does purge 5
  *trading days*.
  Verified by `test_purge_removes_label_overlap_for_every_real_fold`,
  `test_purge_check_catches_an_under_purged_boundary` (adversarial fold with
  only a 3-row gap, must raise),
  `test_purge_check_accepts_an_exactly_sufficient_boundary` (exactly a
  5-row gap, must not raise), `test_dates_are_contiguous_trading_days`, and
  `test_contiguous_check_catches_a_silent_gap`.

- **Preprocessing/model fit occurs only on each training fold.**
  LightGBM (E2-S2) needs no preprocessing, so nothing in this repo currently
  fits a scaler or imputer -- but the next E2 model that does needs a
  fold-safe way to do it. `fit_scaler_on_train_fold_only` is that entry
  point: it fits strictly on `fold.train_idx`, never on the full frame.
  `test_fold_scoped_scaler_is_fit_only_on_train_rows` confirms the fitted
  mean equals the manual mean of the training rows alone;
  `test_fold_scoped_scaler_differs_from_a_full_sample_scaler` demonstrates
  *why* this matters -- a scaler fit on the whole canonical frame sees rows
  the fold-scoped scaler never does, so their statistics measurably diverge.

- **Fold boundaries and `fold_id` saved so every OOS prediction is
  traceable.** `run()` writes `output/fold_boundary_audit.csv` with one row
  per fold: `fold_id`, train/purge/test date ranges, and row counts.
  `assert_predictions_trace_to_fold` then cross-checks E2-S1's
  `baseline_zero_oos_predictions.csv` and E2-S2's
  `lightgbm_oos_predictions.csv` against those boundaries: every prediction
  row tagged with a given `fold` must carry exactly the dates that fold's
  `test_idx` names, no more, no fewer.
  Verified by `test_baseline_oos_predictions_trace_to_documented_fold_boundaries`,
  `test_lightgbm_oos_predictions_trace_to_documented_fold_boundaries`, and
  `test_traceability_check_catches_a_mislabeled_prediction_row` (proves the
  check isn't vacuous).

## Fold design (documented)

Expanding train window, 6 contiguous non-overlapping test blocks covering
the entire post-warm-up history, purge gap of exactly `HORIZON_TRADING_DAYS`
(5) rows before every test block. No shuffling anywhere -- `splits.py`
raises if the input dates aren't sorted ascending, and `run()` raises again
before splitting.

| Fold | n_train | Train start → end | Purge gap | Test start → end | n_test |
|---|---|---|---|---|---|
| 0 | 1255 | 2006-02-01 → 2011-01-25 | 2011-01-26 → 2011-02-01 | 2011-02-02 → 2013-09-03 | 650 |
| 1 | 1905 | 2006-02-01 → 2013-08-26 | 2013-08-27 → 2013-09-03 | 2013-09-04 → 2016-04-04 | 650 |
| 2 | 2555 | 2006-02-01 → 2016-03-28 | 2016-03-29 → 2016-04-04 | 2016-04-05 → 2018-10-30 | 650 |
| 3 | 3205 | 2006-02-01 → 2018-10-23 | 2018-10-24 → 2018-10-30 | 2018-10-31 → 2021-06-02 | 650 |
| 4 | 3855 | 2006-02-01 → 2021-05-25 | 2021-05-26 → 2021-06-02 | 2021-06-03 → 2024-01-02 | 650 |
| 5 | 4505 | 2006-02-01 → 2023-12-22 | 2023-12-26 → 2024-01-02 | 2024-01-03 → 2026-08-13 | 655 |

## Edge cases

- **Overlapping 5D labels at the boundary.** The label for the last
  retained training row references a price `HORIZON_TRADING_DAYS` rows
  ahead of it; the purge gap removes exactly enough rows so that reference
  never crosses into the test block (verified boundary-tight: the last
  train row's label reads the row immediately *before* `test_start_idx`,
  never inside it). See `test_purge_check_accepts_an_exactly_sufficient_boundary`.

- **Too-small early train window.** The purge shrinks fold 0's usable train
  window below the nominal `MIN_TRAIN_SIZE` (1260): it trains on
  `MIN_TRAIN_SIZE - HORIZON_TRADING_DAYS = 1255` rows, confirmed by
  `test_first_fold_train_window_shrinks_by_the_purge_but_stays_adequate`,
  which also checks that shrunk size still clears an adequacy floor
  (`MIN_ADEQUATE_FIRST_FOLD_TRAIN_SIZE = 1000`, ~4 trading years). Separately,
  `purged_walk_forward_splits` raises `ValueError` outright if `min_train_size`
  leaves no room for even one test block (`test_splits_raise_when_min_train_size_leaves_no_room_for_any_test_block`).

- **Final partial fold.** `remaining // n_folds` does not evenly divide the
  post-warm-up history, so the last fold absorbs the remainder instead of
  dropping it: folds 0-4 are 650 rows each, fold 5 is 655.
  `test_final_fold_absorbs_the_remainder_without_dropping_rows` checks this
  and that no test row is ever scored by more than one fold.

- **Full-sample preprocessing.** Guarded structurally by
  `fit_scaler_on_train_fold_only` for future use, with a test that shows a
  full-sample fit and a fold-scoped fit produce genuinely different
  statistics -- see Acceptance above.

- **Fold logic changed after seeing results.** `check_fold_params_unchanged_across_outputs`
  compares `splits.py`'s current `N_FOLDS` / `MIN_TRAIN_SIZE` /
  `HORIZON_TRADING_DAYS` against the `split_params` E2-S1 and E2-S2 already
  recorded in their summary JSONs. If the fold definition had been tweaked
  after either of those runs, this comparison would show a `MISMATCH`
  instead of `MATCH`. Verified by
  `test_fold_params_are_frozen_across_E2_S1_and_E2_S2_recorded_outputs`,
  and reported in `output/walk_forward_validation_summary.json`.

## Run

```bash
python E2-S1_Baseline_Zero_Predictor/run_baseline.py          # if not already run
python E2-S2_Train_Minimal_LightGBM_Regressor/train_lightgbm.py  # if not already run
python E2-S3_Leakage_Safe_Walk_Forward_Validation/validate_walk_forward.py
python -m pytest E2-S3_Leakage_Safe_Walk_Forward_Validation/tests/test_walk_forward_validation.py -v
```

The traceability checks skip (rather than fail) if the E2-S1/E2-S2 output
files don't exist yet; run those scripts first to exercise them for real.

## Results (this run)

`output/walk_forward_validation_summary.json`: `"verdict": "PASS"`, all 4
structural checks true, both traceability checks `PASS` against the
existing `baseline_zero_oos_predictions.csv` (3905 rows) and
`lightgbm_oos_predictions.csv` (3905 rows), and both `fold_params_frozen`
comparisons `MATCH`.

## Scope boundary

This card validates and documents the fold mechanics already implemented in
`E2-S1/splits.py`; it does not change split parameters or add a new
splitter. It adds a fold-scoped preprocessing guard
(`fit_scaler_on_train_fold_only`) for future use but does not itself add
preprocessing to any model -- LightGBM (E2-S2) is tree-based and needs none.
Regime-conditioned evaluation and any model requiring scaling/imputation
belong to later E2 stories, which should call `fit_scaler_on_train_fold_only`
rather than fitting a preprocessor on the full frame.
