# OOS Error Analysis — Independent Audit (Final)

| Field | Value |
|---|---|
| Object under audit | `E2-S7_OOS_Error_Analysis/README.md`, `analyze_oos_errors.py`, `output/error_analysis_summary.json`, `output/top_errors.csv` — a written analysis of the largest OOS errors, weak periods, and regime-specific failure patterns for the LightGBM 5D-return model |
| Repository | `aio-conquer-module-3` @ `main` |
| Auditor | Independent-audit persona, Claude Code session |
| Date | 2026-09-03 |
| Evidence | Direct source read of `analyze_oos_errors.py` and `README.md`; every quoted statistic independently recomputed from `results/oos_predictions.csv` and `E2-S1_Baseline_Zero_Predictor/output/baseline_zero_oos_predictions.csv` using stdlib `csv`/`math`/`fractions` (no pandas in this shell); two robustness checks run beyond what the document itself contains (N-sensitivity sweep, and a fold-3 significance cross-check) |
| Result | **Conditional pass** |

## Disclosure, stated up front because it governs how to read every verdict below

**This is a self-audit.** The document under audit was authored in this same session, by the same model. This is disclosed in the document's own header, not something discovered by this audit. Because the biggest risk in a self-audit is unconsciously grading one's own narrative choices favourably, this audit's method was: recompute every number from the raw files rather than trust the summary JSON, and add checks the document does not perform itself (an N-sensitivity sweep on the top-error cutoff, and an independent significance test for the fold-3 concentration claim) rather than merely re-reading what's already there.

---

## 1. Systematic, non-cherry-picked inspection — **Pass**

The rule (top 1% by absolute error, N=39) is a module-level constant in `analyze_oos_errors.py`, set once, applied to the complete unfiltered 3,905-row file with no exclusions — independently reconfirmed by recomputing the same ranking from raw data and matching all 39 dates in `output/top_errors.csv` exactly. Row-count reconciliation is exact (3,905 analysed = 3,905 in `E2-S4`'s canonical table). Every top-error row carries a `Date`. **One limitation, honestly scoped**: "the rule was fixed before inspection" is the author's own account and cannot be verified from a source independent of the author in a self-audit — this audit does not assume it is true; it instead tested whether the headline finding is robust to the specific N chosen (see "Extra scrutiny" below), which is the closest available substitute for a timestamped log.

## 2. Quantify whether extreme-return periods dominate — **Pass**

Reported and independently reproduced: Pearson r = 0.9225, Spearman ρ = 0.9026 between `|actual_return_5d|` and `|error|` over the full 3,905-row set. The document does the one thing the standard specifically warns is easy to skip: it does not present this correlation as a LightGBM-specific discovery. It computes a second number — the same correlation against the ŷ=0 baseline's error (which is `|actual|` exactly) — and finds it identical (0.9225 to floating-point precision), then explicitly frames the first correlation as "arithmetic... not a LightGBM-specific failure" (Explanation A). This is the correct handling of the exact framing risk named in the standard's own note.

## 3. Regime definitions must pre-exist the failure analysis — **Pass**

`analyze_oos_errors.py` contains no threshold, quantile, or labelling logic of any kind — `regime` is read unchanged from `results/oos_predictions.csv`, confirmed by direct source read. Its provenance is cited to `docs/E4-S1_leakage_audit_record.md` (2026-08-24) and `docs/E2-S5_Regime_Performance_Table_audit_report.md`, both of which predate this document and were produced by independently auditing the volatility-regime construction in `E1-S4`/`E1-S6`. No boundary could have been shaped to fit the error cluster, because no boundary is computed here.

## 4. Error-distribution comparison must show counts, not just shape — **Conditional pass**

Every comparison carries an `n`: regime shares (HighVol 35/39 vs. 43.2% baseline, both counts shown), fold shares (fold 3: 19/39 vs. 16.6% baseline), per-regime MAE (LowVol n=2,216; HighVol n=1,689). One consistent statistic (MAE, or a plain share) is used throughout — no switching between mean/median or MAE/RMSE across buckets.

**The regime-share claim now has a real significance test, not just two percentages.** `analyze_oos_errors.py::hypergeometric_upper_tail_p_value` computes the exact one-sided hypergeometric tail: under the null that top-error membership is independent of regime, expected HighVol count is 16.9 (sd 3.08), observed is 35, **p = 1.43×10⁻⁹**. Independently re-derived in this audit via the exact combinatorial computation and confirmed with a 200,000-trial Monte Carlo simulation (0/200,000 draws reached 35) — not vacuous, either: a dedicated test (`test_hypergeometric_test_gives_p_one_when_observed_equals_expected`) confirms the function returns an unsurprising p-value when the observed count sits at its null expectation.

**Remaining gap**: the fold-share claim (19/39 fold 3) and the largest-calendar-cluster claim (13/39, 2020-02-19→2020-03-23) still have no dispersion or significance treatment in the deliverable itself. As additional scrutiny in this audit (not part of the document), I computed the same style of test for fold-3: under a regime-blind/fold-blind null, expected count 6.5 (sd 2.31), observed 19, **p = 3.13×10⁻⁶** — also highly significant, suggesting the gap is a rigor/documentation omission in the artefact rather than evidence the claim is spurious. The cluster-size claim is a harder, different kind of test (a scan-statistic problem over an adaptively-chosen window, not a fixed-population draw) and this audit does not compute a number for it — marked **unverified**, not assumed passing, rather than reported with a misleadingly simple p-value.

## 5. Alternative explanations must be genuine, distinct, and falsifiable — **Pass**

Two explanations, two different mechanisms: Explanation A (arithmetic — near-zero predictions swamped by large moves, true for any similarly-shrunk model) is the required "boring" null; Explanation B (sign-mismatch specifically on trend reversals, scoped to the 12-row subset A does not explain) is a distinct, narrower claim about feature lag, not a restatement of A in different words. Each carries a stated, falsifiable distinguishing test that is explicitly marked **proposed, not run** — for A, substituting a higher-variance model and checking whether its `|error|`/`|actual|` correlation drops; for B, checking whether sign-mismatch rate is elevated specifically near detected trend reversals versus mid-trend days across the full OOS set. Neither test is presented as already-supporting evidence.

## 6. Observation and causal explanation must be structurally separated — **Pass**

Two headed sections exist: "Observations (numbers only — no causal claims in this section)" and "Speculative explanations (clearly separated from Observations above...)." Scanned for fact-stated-as-mechanism phrasing (`"...because..."`, `"failed due to..."`) — none found; all causal language is hedged ("a plausible, distinct mechanism," "the model *may* be extrapolating," "this is a hypothesis, not a finding").

## 7. Hard cases must not be removed for looking bad — **Pass**

No filtering, winsorising, or smoothing step exists anywhere in `analyze_oos_errors.py` — the only transformation applied is computing `error`/`abs_error`/`abs_actual` from the unmodified source columns. The 3,905-row reconciliation (§1) independently confirms no row was silently dropped before ranking.

## 8. No silent re-tuning triggered by this analysis — **Pass**

Both the README and the JSON (`no_retuning_declaration`) state no model/feature/hyperparameter change has been made. The one proposed follow-up (testing Explanation B) is explicitly scoped as "a new, separately-logged experiment... with its own seed and independent OOS evaluation" — consistent with the single-configuration standard already established for `E2-S2`, not an implied silent adjustment.

## 9. Small-sample discipline, applied per claim — **Conditional pass**

The thinnest claim in the document (the 12-row, 100%-sign-mismatched Explanation-B subset) carries an explicit, adjacent caveat in both prose and the machine-readable summary (`error_only_cluster.small_sample_caveat`) — correctly done, and now backed by the regime-share significance test rather than a bare count. **Gap, unchanged from the prior review**: the fold-3 claim (n=19 within the top-39) and the secondary-cluster mentions (2011: 3 dates; 2015: 2 dates) are shown as counts but not individually flagged as illustrative — a reader could read "five other distinct macro episodes... each contribute independently" as a settled multi-episode pattern rather than several 1–5-date anecdotes. Showing the count is necessary but the standard asks for more than the number alone at each claim.

## 10. Cross-reference against the baseline — **Pass**

This is the document's strongest section and does real work rather than being a checkbox: `analyze_oos_errors.py` explicitly computes the baseline's `|error|` (`= |actual_return_5d|`, since baseline predicts 0 always) and correlates it against LightGBM's `|error|`, finding the identical 0.9225 value. Used correctly — as evidence *against* a model-specific-weakness narrative (Explanation A), not selectively invoked only where convenient.

---

## Extra scrutiny performed in this audit, beyond re-reading the document

1. **N-sensitivity sweep** (not in the document): reran the top-error ranking at 0.5% (k=20), 2% (k=78), and 5% (k=195). HighVol share of the top-error list: 85.0%, 89.7% (document's own N), 91.0%, 88.7% — stable across a 10× range in N, consistent with (not proof of) the claim that N=1% was not tuned to manufacture the finding.
2. **Fold-3 significance cross-check** (not in the document): p = 3.13×10⁻⁶ under the same hypergeometric-null style of test used for the regime-share claim — see §4.

---

## 11. Sign-off checklist

- [x] Top-error selection rule (1%, N=39) fixed and disclosed in code before the narrative; robustness to N confirmed by this audit
- [x] Total analysed-row count (3,905) matches total OOS-evaluated-row count exactly
- [x] Correlation for extreme-return dominance reported as a number (Pearson 0.9225, Spearman 0.9026), correctly framed as arithmetic, not a model-specific discovery
- [x] Regime taxonomy sourced from a pre-existing, independently-audited definition; no boundary computed in this analysis
- [x] Per-bucket sample sizes reported alongside every distributional comparison
- [x] Regime-share claim backed by an exact significance test (p = 1.43×10⁻⁹), independently re-derived in this audit
- [ ] Fold-share and largest-cluster claims still lack a significance/dispersion treatment or an explicit "descriptive only" qualifier
- [x] ≥2 genuinely distinct, falsifiable alternative explanations, including one null/boring explanation
- [x] Observation and causal explanation visibly separated in document structure; no causal-as-fact language found
- [x] No exclusions of hard/outlier cases; declared and independently verified
- [x] No model/feature change made; any follow-up explicitly scoped as a new, separately-logged experiment
- [ ] Small-sample caveats attached at the claim level for every thin-sample claim, not only the ones that already have it (12-row subset, now also regime-share)

## Overall verdict: **Conditional pass**

Per the standard's own rubric: narrative and structure are sound — §5 (distinct, falsifiable explanations with a null included), §6 (structural observation/explanation separation), and the disciplined half of §9 all pass cleanly, and every quantified check the standard treats as load-bearing is present as an actual number, not a claim: §2's extreme-return correlation, §3's pre-existing regime provenance, §10's baseline cross-reference, and now §4's regime-share significance test (p = 1.43×10⁻⁹), all independently reproduced in this audit rather than taken on the document's word. This does not clear a full Pass because §4 and §9 are only partially closed: the fold-share (19/39, fold 3) and largest-cluster (13/39) claims remain unquantified, shown only as counts. It does not fall to Fail — no regime definition is post-hoc or circular, no hard case was removed, no model change occurred without being logged as a new experiment, and no causal claim is stated as settled fact without a hedge and a proposed distinguishing test.

### What closes this out to a full Pass

1. Add a significance test (the fold-3 hypergeometric check computed in this audit, p = 3.13×10⁻⁶, or an equivalent) to the deliverable itself for the fold-share claim, matching the treatment now given to the regime-share claim.
2. Either compute an appropriate scan-statistic test for the largest-cluster claim, or explicitly label it "descriptive only, not tested for significance" at the point of the claim.
3. Attach an adjacent illustrative-only qualifier to the secondary single/double-date episode mentions (2011, 2015, 2018 vol spike), not only to the 12-row Explanation-B subset.
4. If Explanation B's reversal hypothesis is pursued, log it as a new, separately-numbered card with its own hypothesis statement and pre-registered test — not folded into a future revision of this document.
