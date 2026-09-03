# E2-S7 [Model] OOS Error Analysis — Largest Errors, Weak Periods, Regime-Specific Failure Patterns

| Field | Value |
|---|---|
| Epic | E2 |
| Owner | Model |
| Depends on | E2-S4 (canonical `results/oos_predictions.csv`), E2-S1 (baseline predictions), E1-S4/E1-S6 (regime taxonomy) |
| Model under analysis | LightGBM, E2-S2's single frozen configuration |

## Disclosure before anything else

**This document and the independent audit that will follow it were produced within the same review session.** That is a conflict of interest for the audit, not something the audit resolves — it is stated here so a reader of the audit doesn't mistake self-review for independent review. Every number below is computed by [`analyze_oos_errors.py`](analyze_oos_errors.py) and written to [`output/error_analysis_summary.json`](output/error_analysis_summary.json) / [`output/top_errors.csv`](output/top_errors.csv) — nothing in this narrative is asserted without a traceable field in that output.

**No model, feature, or hyperparameter change has resulted from this analysis, and none is proposed to happen silently.** Where a follow-up experiment is suggested below, it is named explicitly as a new, separately-logged experiment against the standard already established for E2-S2 — not an implied adjustment to "the" current model.

## Method (fixed before inspection)

- **Selection rule**: top 1% of OOS rows by `|actual_return_5d − prediction|`, applied to the complete, unfiltered `results/oos_predictions.csv` (3,905 rows). 1% is a round number chosen for its own sake, not tuned after a first look — it yields **N = 39**.
- **No exclusions.** All 3,905 rows were ranked; none were removed as outliers, for readability, or for any other reason. 3,905 analysed = 3,905 evaluated in `E2-S4`'s canonical table — reconciled exactly, not approximately.
- **Regime labels are not defined here.** `LowVol`/`HighVol` are read unchanged from `results/oos_predictions.csv`, sourced from the `E1-S4`/`E1-S6` volatility-regime construction — audited independently, before this document existed, in `docs/E4-S1_leakage_audit_record.md` and `docs/E2-S5_Regime_Performance_Table_audit_report.md`. No regime boundary was created or adjusted for this analysis.

---

## Observations (numbers only — no causal claims in this section)

1. **HighVol is overrepresented in the top-39 errors, by a wide and quantified margin — tested, not eyeballed.** HighVol is 43.2% of all 3,905 OOS rows but 89.7% (35/39) of the top-39 errors (`output/error_analysis_summary.json:regime_representation`). LowVol is 56.8% of rows but only 10.3% (4/39) of top errors. **Significance test**: under the null that top-error membership is independent of regime (i.e. a random 39-row draw from the 3,905-row population), the expected HighVol count is 16.9 (sd 3.08); the observed count of 35 is a z ≈ 5.9 departure, and the exact one-sided hypergeometric tail probability is **p = 1.43×10⁻⁹** (`regime_representation.significance_test` in the summary JSON). This is a descriptive-statistics significance test against a regime-blind-sampling null, not a claim about *why* the enrichment exists — that remains the subject of the Speculative Explanations section below.

2. **The `|actual| ↔ |error|` relationship is strong and quantified, not asserted.** Pearson correlation between `|actual_return_5d|` and `|error|` across the full 3,905-row OOS set is **0.9225**; the rank (Spearman) correlation is **0.9026**.

3. **This same correlation is identical when computed against the ŷ=0 baseline's error instead of LightGBM's.** Because the baseline always predicts 0, its error is `|actual_return_5d|` exactly. Pearson(`|LightGBM error|`, `|baseline error|`) = **0.9225** — the same value as observation 2, to floating-point precision (`pearson_lightgbm_abs_error_vs_baseline_abs_error` in the summary JSON).

4. **69.2% (27/39) of top-error dates are also top-1%-by-`|actual|` dates.** The remaining **12/39 (30.8%)** are top-errors without being top-magnitude days. On **all 12 of these 12 rows**, `sign(actual_return_5d) ≠ sign(prediction)` — a 100% sign-mismatch rate in this specific 12-row subset (`error_only_cluster` in the summary JSON). n=12 — see small-sample caveat below.

5. **The single largest calendar-contiguous cluster inside the top-39 (dates within 10 days of a neighbour) is 2020-02-19 → 2020-03-23, 13 dates — 33.3% of the top-39, not a majority.** The next-largest clusters are 2022-06-06→06-10 (5 dates) and 2011-08-01→08-10 plus 2025-03-28→04-10 (3–4 dates each); the remainder are isolated single- or double-date events spread across 2011, 2015, 2018, 2022, and 2025.

6. **Fold 3 (test window 2018-10-31 → 2021-06-02) contributes 19/39 (48.7%) of top errors**, against a 16.6% share of all OOS rows by fold. This is the only fold whose test window contains both the December 2018 selloff and the February–April 2020 COVID crash.

7. **Per-regime MAE on the full OOS set** (already established in `docs/E2-S5_Regime_Performance_Table_audit_report.md`, restated here for reference): LowVol n=2,216, MAE=0.01174; HighVol n=1,689, MAE=0.02176.

---

## Speculative explanations (clearly separated from Observations above — none of this is asserted as established fact)

### Explanation A — the "boring"/null explanation: this is arithmetic, not a LightGBM-specific weakness

Observation 3 is a direct test of this. If the `|actual| ↔ |error|` relationship reflected something specific to LightGBM's modelling choices, the correlation should differ depending on whose error is being measured. It does not — the correlation against the literal ŷ=0 baseline's error is identical to the correlation against LightGBM's error. A previously-audited fact about this model (`docs/E2-S4_Canonical_OOS_Table_audit_report.md`, `docs/E2-S5_Regime_Performance_Table_audit_report.md`) is that its predictions are heavily shrunk toward zero (pooled prediction-variance-to-actual-variance ratio ≈ 0.098) — any model whose outputs stay small in magnitude will, mechanically, show `error ≈ |actual|` on days when `|actual|` is large. **Consistent with, not refuted by, the data above**: this explanation predicts exactly what observations 2 and 3 show, and predicts it for *any* near-zero-output model, not a LightGBM-specific failure.

*Distinguishing test proposed, not run*: if a genuinely non-degenerate model (larger prediction variance, e.g. Random Forest from `E2-S6`) were substituted, Explanation A predicts its `|error|` vs `|actual|` correlation would be measurably *lower* than 0.92, since larger-magnitude predictions would absorb more of the large moves rather than being swamped by them. This is a falsifiable follow-up, not run here.

### Explanation B — sign-mismatch on trend reversals (a genuinely distinct, more specific mechanism)

Observation 4 isolates a different, smaller pattern than Explanation A: 12 rows are top-errors *without* being top-magnitude days, and every one of them has a wrong-sign prediction. Looking at those 12 dates directly (`output/error_analysis_summary.json:error_only_cluster`), several sit right at a sharp local reversal inside a larger episode already identified in the observations — e.g. 2020-04-06/07/09 are the first days of the sharp COVID *rebound*, immediately following the crash captured in the largest cluster (observation 5); 2018-12-13/14 sit inside the December 2018 selloff's own reversal days. A plausible, distinct mechanism: the model's feature set is dominated by trailing return/trend windows (`return_1d…20d`, `trend_10d/20d/60d`), which by construction still encode the *prior* direction immediately after a sharp reversal — the model may be extrapolating a trend that has just broken, rather than failing simply because the move was large. **This is a hypothesis, not a finding** — it is a different, testable mechanism from Explanation A, not the same explanation in different words.

*Distinguishing test proposed, not run*: compute the sign-mismatch rate specifically in the 1–3 trading days immediately following a local trend reversal (e.g. a sign change in `return_10d`) versus mid-trend continuation days, across the full OOS set, not just the 12-row illustrative subset. If elevated at reversal points specifically, that supports Explanation B over a generic noise account; if flat, it does not.

**Small-sample caveat, attached here rather than only at the top**: the 12-row subset behind Explanation B is below the ~dozen-point threshold for treating a pattern as more than illustrative. A 100% sign-mismatch rate on 12 rows is a clean-looking number that could plausibly arise by chance in noisy financial data at this sample size — it motivates the distinguishing test above, it does not substitute for it.

---

## What this analysis does not show

- It does not show LightGBM is uniquely bad in HighVol relative to a reasonable alternative — the baseline cross-reference (observation 3) suggests the *same* large-error pattern would appear for any near-zero-output model on these dates, and `E2-S6`'s multi-model comparison already found Random Forest, not LightGBM, is the best-performing model on this data.
- It does not establish that the model "fails during crises" as a general property — the largest single episode (COVID, observation 5) accounts for a third, not a majority, of the top-error list; five other distinct macro episodes across 14 years (2011 debt-ceiling stress, 2015 China deval, Feb 2018 vol spike, Dec 2018 selloff, 2022 rate-hike bear market, 2025 volatility) each contribute independently.
- It does not test Explanations A or B against each other statistically — both are reported as candidate, falsifiable mechanisms with a proposed distinguishing test, per the discipline this deliverable is required to hold itself to, not as competing conclusions to be weighed by prose alone.

## Next steps (proposed, not enacted)

If Explanation B is pursued, it should be scoped as a **new, separately-logged experiment** — its own hypothesis, its own test computed against the full 3,905-row OOS set (not the 12-row illustrative subset), and, if it motivates a feature or hyperparameter change, a new frozen configuration with its own seed and independent OOS evaluation, exactly as `E2-S2`'s single-configuration standard requires. Nothing in this document changes the current model.
