# Canonical Results Table by Volatility Regime (E2-S5) — Independent Audit

| Field | Value |
|---|---|
| Object under audit | `E2-S5_Evaluate_Overall_LowVol_HighVol_Performance/output/regime_performance.csv` + `regime_comparison.csv` — N, MAE, prediction correlation, hit rate for Overall/LowVol/HighVol × {baseline_zero, lightgbm} |
| Repository | `aio-conquer-module-3` @ `main` (`a76ecb0`) |
| Auditor | Independent audit, principal-data-scientist persona (Claude Code session) |
| Date | 2026-09-03 |
| Evidence | Direct source read of the regime-construction notebook cell, `evaluate_regime_performance.py`, `metrics.py`, `E3-S2/pipeline/model.py::run_regime_evaluation`; live computation against `results/oos_predictions.csv` (stdlib `csv`/`float`, no pandas available); independent replication of a crisis-block sensitivity cut and per-regime variance ratios not present in the deliverable itself |
| Result | **Conditional pass** — but on different, more specific terms than the three pre-set boxes; see verdict |

## Verdict up front

The regime *definition* — the highest-risk area named in this audit — is clean and unusually well-proven: the threshold is a strictly past-only expanding-median (`.expanding(min_periods=252).median().shift(1)`), and the notebook itself contains a self-contained counterfactual proof at three sampled positions (first eligible, middle, last) confirming the stored threshold matches a manual past-only recomputation to `1e-12`, distinct from the earlier `E4-S1` audit's finding that a full-sample version would disagree on 260 rows. Row-count reconciliation is exact, both overall and per regime, verified against the live file rather than the summary's self-report. Where this table falls short of a clean Pass is narrower and more concrete than the two edge-case sections the rubric anticipated (§6/§7): the actual-return-side zero-sign convention named explicitly as new for this deliverable is not implemented in code at all, and it has already fired once, silently, on the live 3,905-row file — a small-magnitude but real, demonstrated gap, not a hypothetical one. Separately, §6 (small-N/clustering treatment) and §7 (crisis-block sensitivity cut) are genuinely absent from the deliverable, exactly as the rubric's conditional-pass carve-out anticipates — I ran the missing §7 check myself as the auditor and it does not materially change the headline HighVol result, which is reported below rather than left as an open question.

---

## 1. Regime definition integrity — **Pass**

**The exact rule, as code** (`E1-S3_to_E1-S6...ipynb`, cell 16): volatility measure = `volatility_20d`; threshold = `df["volatility_20d"].expanding(min_periods=252).median().shift(1)`; equality rule = `volatility_20d >= threshold → HighVol`, else `LowVol` (matches `pipeline_config.yaml`'s `equality_rule: HighVol`); a row is only eligible for a regime label once 252 prior observations exist.

**Lookahead check — verified in code, not documentation, twice over.** `.expanding()` uses only rows up to and including the current one, and `.shift(1)` then pushes that value one row forward so day *t*'s threshold is computed from data strictly before *t* — construction (b) from the standard, not a full-sample quantile. This is not merely asserted: cell 17 of the same notebook independently recomputes the threshold at three sampled positions (first eligible row, middle, last) using `df[REGIME_VOLATILITY_COL].iloc[:position]` (a hard positional truncation, not a call into the expanding-window code) and confirms `np.isclose(stored_threshold, manual_threshold, atol=1e-12, rtol=1e-12)`. This corroborates and extends the previously-audited `E4-S1` finding that a full-sample-median version of the same threshold disagrees with the published labels on 260 rows — i.e., the check can and does distinguish a leaky construction from this one.

**Single canonical object, shared identically.** The `regime` column is computed once inside the `E1-S6` canonical-dataset build and carried through unchanged into both `E2-S1`'s baseline predictions and `E2-S2`'s LightGBM predictions (both scripts copy `df.loc[fold.test_idx, "regime"]` directly from the same canonical file — no second, independently-computed regime classification exists anywhere). `evaluate_regime_performance.py::assert_same_oos_rows` explicitly checks the two models' regime labels agree for every shared date and is not vacuous — `test_fairness_check_catches_mismatched_regime_labels` proves it raises on a synthetic mismatch — and it passed without raising on the real run (confirmed: `regime_performance_summary.json` exists with no error, and the live per-regime N values agree exactly between models, see §2/§3).

**Stability under feature-availability filtering.** The canonical dataset's single `dropna()` (E1-S6 build, cell 20) drops rows failing on *any* of features/target/regime together, in one pass — a row cannot end up with a feature but a missing regime, or vice versa, by construction, not by a downstream reconciliation step.

## 2. Split/metric code parity across regimes — **Pass, with one structural risk disclosed**

**Row-count parity within each regime, not just aggregate**: verified live, directly against `results/oos_predictions.csv`'s own `regime` column (recomputed independently for the E2-S4 audit and re-used here): LowVol N = 2,216, HighVol N = 1,689 — and the summary confirms both `baseline_zero` and `lightgbm` report identically 2,216 / 1,689, per regime, not just matching in aggregate. **Overall = LowVol + HighVol exactly**: 2,216 + 1,689 = 3,905 = Overall N, confirmed both from my own live recomputation and from `test_overall_count_equals_sum_of_regime_counts` / `test_scope_mask_partitions_rows_with_no_overlap_or_gap` (the latter proves the partition has no overlap *and* no gap, not just that the totals happen to sum). The same three `metrics.py` functions (`mae`, `prediction_correlation`, `directional_hit_rate`) are imported once and called identically for every (model, scope) pair inside `evaluate_model` — no per-regime reimplementation exists in `evaluate_regime_performance.py`.

**Structural risk, not a currently-active bug**: `E3-S2/pipeline/model.py::run_regime_evaluation` does not import `scope_mask`, `assert_same_oos_rows`, or `build_comparison_table` from `evaluate_regime_performance.py` — it redefines its own `scope_mask`, `_assert_same_oos_rows`, `_build_comparison_table` (confirmed by direct source read; the redefinitions are textually identical to the originals today). This is the same duplication pattern the prior LightGBM-stage audit found had already diverged with a live bug (a dropped seed) in that case. Here, both copies are currently identical and both call the same shared `metrics.py` functions for the actual scoring, so there is no demonstrated numeric divergence today — but it is a latent single-source-of-truth risk: a future edit to one copy's `scope_mask`/fairness-check logic without the other would silently reintroduce exactly this failure mode, and nothing in the repository would catch it.

## 3. Sample count (N) discipline — **Pass**

`n` is reported unconditionally for every row of `regime_performance.csv` — verified live (all 6 rows carry a positive `n`) and by `test_every_row_reports_n`. Correlation and hit rate share the same denominator as `n` here (no separate exclusion-adjusted count is needed for this file, since the zero-prediction exclusion in `directional_hit_rate` only removes rows from *baseline* rows, which are reported as `NaN` overall rather than a partial hit rate over a smaller denominator — see §5). HighVol's reported N (1,689) was cross-checked independently against the raw `results/oos_predictions.csv` file's `regime` column directly (not the summary's self-report) and matches exactly.

## 4. Correlation validity for degenerate predictions — **Pass**

`metrics.py::prediction_correlation` computes Pearson correlation via `np.corrcoef`, explicitly, everywhere — one function, imported once, used for every (model, scope) cell; no Spearman or alternate correlation type appears anywhere in this codebase. Degenerate case handled correctly and verified live: all three baseline rows report `NaN` (std of an all-zero predictor is exactly 0, triggering the function's explicit `if np.std(y_pred) == 0: return nan` branch) — confirmed in the live `regime_performance_summary.json` and by `test_baseline_correlation_and_hit_rate_are_nan_in_every_scope`, which explicitly asserts the value is never silently `0.0` (a real check, not just "not None"). LightGBM shows genuine non-NaN correlation in all three scopes (0.025–0.092), consistent with `predictions_nearly_constant: false` everywhere — no regime currently triggers LightGBM's degenerate-correlation path, so that branch is correctly implemented but not exercised by today's data (a distinct, disclosed observation from "handled" — see §9).

## 5. Hit-rate zero-sign convention, extended to both sides — **Fail**

Prediction-side convention (`y_pred == 0` excluded from numerator/denominator) is implemented and correctly verified across every regime for the baseline (§4). **The actual-side convention explicitly named as new for this deliverable does not exist in code.** `metrics.py::directional_hit_rate` masks only on `y_pred != 0`; it contains no check for `y_true == 0`. When `y_true == 0` and `y_pred != 0` (in-mask), `np.sign(0) == 0` is compared against `np.sign(y_pred) ∈ {-1, +1}` and is always `False` — i.e. such a row is silently scored as a directional **miss**, not excluded from the denominator, and this convention (or the absence of one) is stated nowhere in `metrics.py`'s own `DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION` docstring. **This is not hypothetical**: I checked the live 3,905-row file directly and found **exactly one row** with `actual_return_5d == 0.0` (float-exact). Its effect on the headline numbers is negligible (1 row out of 3,905, ~0.03%), but the standard's own bar here is the presence of a stated, implemented, and tested convention for this side — which does not exist — not the magnitude of today's impact.

## 6. Small-N and clustering risk in HighVol — **Gap (not implemented)**

No confidence interval, standard error, effective-sample-size adjustment, or block-bootstrap treatment exists anywhere in `E2-S5`'s code, tests, or README. The only mitigation present is disclosure of `n` itself (1,689 vs 2,216) and a statement that it is "smaller... not hidden" — real and useful, but not what §6 asks for. Volatility regimes are demonstrably autocorrelated in this data (see §7's block analysis: 61 distinct contiguous HighVol episodes rather than 1,689 independent draws), so treating HighVol's nominal N as the effective sample size would overstate precision; nothing in the deliverable corrects for this.

## 7. Crisis-period dominance and robustness — **Gap in the deliverable; independently checked by this audit**

The deliverable reports `date_start`/`date_end` per scope (a real, useful transparency step) but never computes the sensitivity cut the standard asks for. I ran it myself, directly against the live file: HighVol is **not** dominated by a single crisis episode — there are 61 distinct contiguous HighVol blocks, and the largest (2021-11-30 → 2023-04-14, 345 rows) accounts for only ~20.4% of HighVol's 1,689 rows. Recomputing LightGBM's HighVol metrics with that single largest block excluded: MAE 0.02176 → 0.02108 (≈3% relative decrease), prediction correlation 0.079 → 0.096 (a mild increase), hit rate 0.554 → 0.558 (essentially flat). Reported here regardless of direction, per the standard's own instruction: the headline HighVol row is not fragile to its single largest contiguous episode, but this finding exists only because I computed it as auditor — it is absent from the deliverable itself, which is the actual finding for this section.

## 8. Metric disagreement reconciliation — **Pass**

The README states the disagreement plainly and by name: HighVol has the *worst* MAE (0.0217 vs 0.0117 for LowVol) but the *best* prediction correlation (0.092 vs 0.019), with directional hit rate roughly flat across all three scopes (0.548–0.554) — explicitly attributed to larger-magnitude returns in high-volatility periods rather than left as an unexplained tension, and presented as "no single metric is treated as authoritative," not led with whichever number favours LightGBM.

## 9. Near-zero or constant predictions, per regime — **Pass on the stated check; a real reporting gap disclosed**

The boolean `predictions_nearly_constant` (std < 1e-6) is computed per (model, scope) and verified live: `True` for baseline in all three scopes (exactly constant by construction), `False` for LightGBM in all three scopes — correctly cross-referenced with §4's NaN/non-NaN correlation split (baseline NaN everywhere, LightGBM non-NaN everywhere), consistent as the standard requires. **Gap**: the deliverable reports only this boolean, not a graded variance measure. I computed the prediction-variance-to-actual-variance ratio per regime directly from the live file: Overall 0.098, **LowVol 0.029**, HighVol 0.125. LowVol's ratio is markedly lower than the other two — LightGBM's predictions there capture proportionally far less of the (smaller) actual variance in LowVol than elsewhere, a real, non-trivial difference in signal strength across regimes that the current boolean-only reporting does not surface, even though it correctly does not cross the "nearly constant" threshold.

## 10. No cherry-picking — completeness of what's reported — **Pass**

Live output has exactly `len(MODELS) × len(SCOPES) = 6` rows (2 models × 3 scopes), matching the schema exactly — no additional regime cut (MedVol, tercile, sub-period) appears anywhere. Checked the regime-construction notebook and repository history directly for evidence of an alternate definition explored and dropped: the notebook has exactly one commit in its entire history (`31a58bb`), `MIN_HISTORICAL_VOL_OBSERVATIONS` and `equality_rule` have never changed, and a repo-wide grep for `MedVol`/`tercile`/`quartile`/`bull_market`/`bear_market`/sub-period terms returns only one hit — a test (`test_scope_mask_rejects_unknown_scope`) that deliberately probes `"MediumVol"` to prove the function *rejects* an unrecognised scope, not evidence one was ever computed.

---

## 11. Sign-off checklist

- [x] Regime classification rule stated as code: `volatility_20d`, 20-day window, `.expanding(min_periods=252).median().shift(1)` threshold, `>=` → HighVol (§1)
- [x] Lookahead check performed on the volatility threshold — past-only expanding median, independently re-derived at 3 sampled positions in the notebook itself (§1)
- [x] Single canonical regime-label object, shared by baseline and LightGBM evaluations, cross-checked live rather than assumed (§1)
- [x] Row-count parity between baseline and LightGBM confirmed within each regime (2,216/2,216 LowVol, 1,689/1,689 HighVol), not just overall (§2)
- [x] Overall N (3,905) reconciles exactly to LowVol N (2,216) + HighVol N (1,689) (§2/§3)
- [x] Correlation type (Pearson) stated and identical across regimes and models (§4)
- [x] N/A convention for degenerate correlation stated and applied consistently; never fabricated as 0 (§4)
- [ ] Hit-rate zero-sign convention covers the actual-side zero case — **missing in code, and already triggered once on the live file** (§5)
- [ ] HighVol row carries an explicit small-N/clustering caveat or interval — **absent** (§6)
- [x] Crisis-block sensitivity cut — absent from the deliverable, but performed by this audit and reported regardless of direction (§7)
- [x] Metric disagreement across MAE/correlation/hit rate explicitly reconciled in the write-up (§8)
- [ ] Per-regime prediction variance reported, not just the pooled/boolean flag — **boolean only; graded ratio computed by this audit, not the deliverable** (§9)
- [x] No additional regime cuts explored and silently dropped from the final table (§10)

## Overall verdict: **Conditional pass, on narrower grounds than the pre-set boxes**

The rubric's "Conditional pass" box names §6/§7 as the expected gap, and both are indeed gaps here — but §5 is also a genuine failure, not an edge-case omission: the actual-side zero-sign convention is unimplemented and has already fired once (silently) on the live 3,905-row file. It does not meet the rubric's explicit hard-Fail bar (no lookahead threshold, no row-count mismatch, no degenerate correlation misreported as 0, no undisclosed cherry-picked cut), so a full Fail is not warranted, and every other section — including the two the audit was told to scrutinise hardest, §1 and §2's row-count-within-regime parity — passed on direct, independently-computed evidence rather than the deliverable's self-report. This is reported as Conditional pass with a corrected, more specific list of what closes it out.

### What resolves this to a full Pass

1. Extend `metrics.py::directional_hit_rate` (and its documented convention string) to also exclude rows where `y_true == 0` from the numerator and denominator, matching the existing `y_pred == 0` treatment, and report the excluded count explicitly — not just the prediction-side case.
2. Add a small-N treatment for HighVol: at minimum a standard error or bootstrap confidence interval on MAE/correlation/hit rate, or an explicit effective-sample-size statement given the 61-episode clustering structure found in §7.
3. Add the crisis-block sensitivity cut (§7) as a standing, reported row in the deliverable itself — this audit's replication (HighVol MAE 0.0218 → 0.0211 excluding the largest block) should not have to be produced by an external auditor to exist at all.
4. Report per-regime prediction-variance ratio (or at minimum a non-boolean prediction std) alongside the existing `predictions_nearly_constant` flag, so LowVol's markedly weaker relative signal (variance ratio 0.029 vs 0.098 Overall) is visible in the table itself.
5. Either import `E2-S5`'s `scope_mask`/`assert_same_oos_rows`/`build_comparison_table` into `E3-S2/pipeline/model.py::run_regime_evaluation` rather than maintaining a second copy, or add a test that fails if the two copies ever diverge — closing the same class of risk the LightGBM-stage audit found had already turned into a live bug elsewhere in this pipeline.
