# E2-S5 [P0][Model] Evaluate Overall, Low-Vol & High-Vol Performance

| Field | Value |
|---|---|
| Epic | E2 |
| Owner | Model |
| Review | QA/QC |
| Depends on (card) | E4-S2 |
| Depends on (functional) | E2-S4 (Generate Canonical OOS Prediction Table) |

## Deliverable

A canonical results table -- `N`, `MAE`, prediction correlation and
directional hit rate -- for both the zero baseline (E2-S1) and LightGBM
(E2-S2), broken out by **Overall**, **LowVol** and **HighVol**.

[`evaluate_regime_performance.py`](evaluate_regime_performance.py) scores
every (model, scope) combination with the exact same `metrics.py` functions
(`mae`, `prediction_correlation`, `directional_hit_rate`) used everywhere
else in E2, filtering the same OOS row set by `regime` rather than
re-deriving anything per regime.

## Acceptance

- **Same OOS table and metric code across regimes.** `scope_mask` filters
  one already-built prediction frame (LightGBM's from
  `results/oos_predictions.csv`, baseline's from
  `baseline_zero_oos_predictions.csv`); `evaluate_model` calls the same
  three `metrics.py` functions for every scope. `test_scope_mask_partitions_rows_with_no_overlap_or_gap`
  confirms LowVol + HighVol exactly partition every row (no double-count,
  no dropped row); `test_overall_count_equals_sum_of_regime_counts`
  confirms Overall's N equals LowVol's N + HighVol's N.

- **Sample counts always reported.** Every row of `output/regime_performance.csv`
  carries `n`, unconditionally -- there's no path that omits it, even for
  the small HighVol scope. Verified by `test_every_row_reports_n`.

- **Undefined correlation/zero-sign conventions handled explicitly.** The
  zero baseline's `prediction_correlation` and `directional_hit_rate` are
  `NaN` in every scope (constant predictor, no sign) -- the same convention
  `metrics.py` documents and E2-S1 already established, now re-verified to
  hold under regime filtering too, not just on the full OOS set. Verified
  by `test_baseline_correlation_and_hit_rate_are_nan_in_every_scope`
  (also checks it's never silently `0.0` or `0.5`).

- **Baseline and LightGBM comparison is fair.** `assert_same_oos_rows`
  checks, before scoring anything, that both models were evaluated on
  exactly the same dates, the same regime label per date, and the same
  `actual_return_5d` per date -- i.e. genuinely the same OOS rows, not two
  similarly-sized but different samples. Verified by
  `test_baseline_and_lightgbm_share_the_same_oos_rows`, and by three
  adversarial tests that corrupt dates / targets / regime labels and
  require the check to raise.

## Results (this run)

`output/regime_comparison.csv`:

| Scope | N | Baseline MAE | LightGBM MAE | Improvement | Baseline Corr | LightGBM Corr | Baseline Hit Rate | LightGBM Hit Rate |
|---|---|---|---|---|---|---|---|---|
| Overall | 3905 | 0.01596 | 0.01606 | -0.00010 | N/A | 0.079 | N/A | 0.549 |
| LowVol | 2216 | 0.01178 | 0.01171 | +0.00007 | N/A | 0.019 | N/A | 0.550 |
| HighVol | 1689 | 0.02143 | 0.02175 | -0.00032 | N/A | 0.092 | N/A | 0.548 |

## Edge cases

- **HighVol sample much smaller.** 1689 rows vs. 2216 for LowVol (roughly
  43% / 57% of the OOS set) -- reported plainly in `n`, not hidden.
  `test_highvol_sample_is_smaller_but_still_reported` checks the smaller
  count is still present and non-zero.

- **Crisis period dominates metric.** `date_start`/`date_end` are recorded
  per scope so a reader can see *when* each regime's rows fall (HighVol:
  2011-03-10 -> 2026-08-13; LowVol: 2011-02-02 -> 2026-07-31) rather than
  only the aggregate number -- a metric dominated by one clustered episode
  (e.g. a volatility spike) is visible as a date-range fact, not hidden
  inside a single MAE. Verified by `test_every_scope_reports_a_date_range`.

- **MAE / correlation / hit rate disagree.** They do, here: HighVol has the
  *worst* MAE (0.0217 vs. 0.0117 for LowVol -- returns are simply larger in
  magnitude in high-volatility periods) but the *best* prediction
  correlation (0.092 vs. 0.019), while directional hit rate is close to
  50% and roughly flat across all three scopes (0.548-0.550). No single
  metric is treated as authoritative; all three are reported side by side
  precisely so this kind of disagreement stays visible.

- **Near-zero or constant predictions.** `predictions_are_nearly_constant`
  (std < `1e-6`) is computed per scope for both models: `True` for
  `baseline_zero` in all three scopes (it is exactly constant by
  definition) and `False` for `lightgbm` in all three (real prediction
  variance, not a collapsed model). Verified by
  `test_baseline_predictions_are_flagged_nearly_constant_in_every_scope`
  and `test_lightgbm_predictions_are_not_flagged_nearly_constant`.

- **Report all metrics -- no cherry-picking.** `output/regime_performance.csv`
  always has exactly `len(MODELS) * len(SCOPES)` = 6 rows (both models,
  all three scopes), and every row carries `mae` (never NaN, since N > 0 in
  every scope here). `regime_performance_summary.json` carries an explicit
  `no_cherry_picking_declaration`. Verified by
  `test_output_has_every_model_scope_combination` and
  `test_output_carries_every_required_metric_column`.

## Run

```bash
python E2-S1_Baseline_Zero_Predictor/run_baseline.py                       # if not already run
python E2-S2_Train_Minimal_LightGBM_Regressor/train_lightgbm.py            # if not already run
python E2-S4_Generate_Canonical_OOS_Prediction_Table/generate_oos_predictions.py  # if not already run
python E2-S5_Evaluate_Overall_LowVol_HighVol_Performance/evaluate_regime_performance.py
python -m pytest E2-S5_Evaluate_Overall_LowVol_HighVol_Performance/tests/test_evaluate_regime_performance.py -v
```

## Scope boundary

The card names **E4-S2** as its dependency; this repository has not yet
produced an E4-S2 artifact or gate. The functional dependency this script
actually has is E2-S4's canonical OOS table
(`results/oos_predictions.csv`) plus E2-S1's baseline predictions, both of
which already exist. If E4-S2 is added later as a QA gate over E2's
outputs, this script does not bypass it -- it should be re-run (or its
result re-reviewed) after that gate exists, the same way E2 as a whole was
gated by `docs/E4-S1_leakage_audit_record.md` before any E2 model was
trained.

This card evaluates two models (baseline, LightGBM) across three scopes; it
does not add a new model, does not change regime construction (owned by
E1), and does not decide whether LightGBM "passes" -- reading the table
honestly (see Results above: LightGBM does not clearly beat the baseline on
MAE in any scope) is left to the reviewer, not asserted here.
