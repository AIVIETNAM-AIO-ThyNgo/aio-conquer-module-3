# E2-S4 [P0][Model] Generate Canonical OOS Prediction Table

| Field | Value |
|---|---|
| Epic | E2 |
| Owner | Model |
| Review | Pipeline |
| Depends on | E2-S3 (Leakage-Safe Walk-Forward Validation) |

## Deliverable

`results/oos_predictions.csv` -- one row per genuine out-of-sample
prediction, with exactly the columns `Date`, `prediction`,
`actual_return_5d`, `regime`, `fold_id`. This is the single canonical
prediction table later E2/E3 stories (regime-conditioned evaluation,
reporting, backtesting) read from.

[`generate_oos_predictions.py`](generate_oos_predictions.py) builds it from
E2-S2's `lightgbm_oos_predictions.csv` -- the only model in this repo
producing genuine (non-baseline) predictions -- but does not trust that
source file blindly. It re-derives the authoritative folds from
`E2-S1/splits.py` and re-checks every source row against the fold it claims
to belong to, using the same `assert_predictions_trace_to_fold` reasoning
[E2-S3](../E2-S3_Leakage_Safe_Walk_Forward_Validation) already verified end
to end. A fold mixup or an accidentally-appended diagnostic row is caught
here, before publication, rather than propagated into `results/`.

## Acceptance

- **Every row is genuine OOS.** `validate_every_row_is_genuine_oos`
  rebuilds the folds from the canonical dataset and asserts every source
  row's `Date` falls inside the test window of the fold it's labeled with
  (`fold`) -- reusing E2-S3's traceability check rather than a new one, so
  "genuine OOS" means the same thing across both cards.
  Verified by `test_every_source_row_traces_to_its_claimed_fold`;
  `test_genuine_oos_check_catches_a_row_relabeled_to_the_wrong_fold` proves
  it's not vacuous.

- **One prediction per date; no train predictions.** The source file
  already contains only test-fold rows; `test_row_count_matches_sum_of_fold_test_sizes`
  confirms the published row count equals the sum of every fold's
  `test_idx` size exactly (3905 rows), and `test_one_prediction_per_date`
  confirms `Date` is unique in the published table.

- **No NaN prediction/actual/regime.** `build_canonical_table` raises if
  any of the three required columns contains a NaN.
  `test_no_nan_in_required_columns` confirms none exist in the real run.

- **Date order preserved.** The table is always re-sorted by `Date`
  ascending before writing -- never trusting the source file's row
  order. `test_date_order_is_ascending` and
  `test_shuffled_source_rows_are_still_published_in_date_order` (feeds in a
  randomly shuffled source frame and checks the output is still ordered).

## Edge cases

- **Duplicate dates from overlapping test folds.** Structurally prevented
  by `splits.py`'s non-overlapping test blocks, but checked again here as
  the last line of defense: `build_canonical_table` raises `ValueError` if
  any `Date` appears more than once. `test_duplicate_dates_are_rejected`
  feeds in a source file with one row duplicated and requires a raise.

- **Concatenation/index mismatch.** If a future change to E2-S2 ever
  concatenated fold-wise frames incorrectly (e.g. a `Date` ending up under
  the wrong `fold` label), `validate_every_row_is_genuine_oos` catches it:
  a row's date no longer falling inside its claimed fold's test window
  fails the trace-to-fold check. `test_concatenation_mismatch_is_rejected`
  swaps two rows' dates across fold boundaries and requires a raise.

- **Shuffled row order.** `build_canonical_table` always sorts by `Date`
  before any downstream check runs -- see acceptance above.

- **Accidental in-sample diagnostics appended to canonical file.**
  `validate_source_columns` pins the source file to exactly
  `fold, Date, regime, y_true, y_pred` and raises if any extra column is
  present (e.g. `train_mae_diagnostic_only`), and `build_canonical_table`
  separately asserts the *output* has exactly the five approved columns in
  the approved order. `test_unexpected_extra_column_is_rejected` proves the
  source-side guard; `test_output_columns_are_exactly_the_canonical_schema`
  proves the output-side guard.

## Run

```bash
python E2-S2_Train_Minimal_LightGBM_Regressor/train_lightgbm.py   # if not already run
python E2-S4_Generate_Canonical_OOS_Prediction_Table/generate_oos_predictions.py
python -m pytest E2-S4_Generate_Canonical_OOS_Prediction_Table/tests/test_generate_oos_predictions.py -v
```

## Results (this run)

`results/oos_predictions.csv`: 3905 rows, 3905 unique dates,
2011-02-02 -> 2026-08-13, 6 folds. `results/oos_predictions_manifest.json`
records the source file's SHA-256 alongside the row/date/fold counts so the
table can be tied back to the exact E2-S2 run that produced it.

## Scope boundary

This card publishes the LightGBM model's OOS predictions in the canonical
schema; it does not add a new model, does not change fold logic (owned by
E2-S1/verified by E2-S3), and does not compute any evaluation metric beyond
what's already in `metrics.py` -- regime-conditioned evaluation and
reporting belong to later stories, which should read `results/oos_predictions.csv`
rather than either model's raw `output/*_oos_predictions.csv`.
