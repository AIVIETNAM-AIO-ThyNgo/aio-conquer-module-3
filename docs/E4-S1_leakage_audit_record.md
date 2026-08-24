# E4-S1 [P0][QA] Pre-Model Leakage & Data Quality Gate — Audit Record

| Field | Value |
|---|---|
| Card | E4-S1 Pre-Model Leakage & Data Quality Gate |
| Depends on | E1-S6 (Publish Canonical Modeling Dataset & Data Dictionary) |
| Repository | `aio-conquer-module-3` @ `feature/e1-s3-to-e1-s6-data-foundation` |
| Auditor | Huy Pham |
| Date | 2026-08-24 |
| Evidence | `python -m pytest tests/test_E1_S6_canonical_dataset.py -v` → **18 passed** |
| Result | **PASS** |

## Verdict

**No leakage found.** All six leakage checks pass against an independent rebuild
of the pipeline from the immutable E1-S1 raw artifact. The suite does not import
the notebook: it re-derives every feature, the target and the regime label from
raw OHLCV and requires the published dataset to agree.

One defect was found and fixed during the audit (§ D1). E2 is unblocked.

## Artifacts under audit

| Artifact | SHA-256 (first 16) | Verified against |
|---|---|---|
| `data/raw/E1-S1_SPY_OHLCV_auto_adjusted.csv` | `1007a7421e108dd8` | provenance + manifest |
| `data/processed/E1-S6_canonical_modeling_dataset.csv` | `96f79c15ff277bf0` | manifest |
| `docs/E1-S6_data_dictionary.csv` | `3e478195e58d1a4c` | manifest |

Raw: 5442 rows, 2005-01-03 → 2026-08-20.
Canonical: 5165 rows, 2006-02-01 → 2026-08-13.

## Results at a glance

| # | Checklist item | Result |
|---|---|---|
| 1 | Feature timestamps documented and verified ≤ t | PASS |
| 2 | No centered rolling, backfill, future imputation or full-sample statistics | PASS |
| 3 | Target / feature / regime indices aligned after NaN removal | PASS |
| 4 | Audit result recorded | PASS |

---

## 1. For every feature, document the latest timestamp of information used and verify it is ≤ prediction time t

**PASS.**

**Documentation** — `test_data_dictionary_documents_every_canonical_column`: all
14 canonical columns carry a non-empty `formula`, `window`,
`timestamp_semantics` and `role`; the dictionary schema matches the manifest
exactly.

**Verification** — `test_features_are_point_in_time`, the decisive test. For 14
sampled prediction dates (first and last admitted date always included), all 11
features are rebuilt from a raw frame **physically truncated at t** and compared
to the published values at rtol 1e-9. All 154 comparisons match. A feature that
reads the future cannot survive having the future deleted.

**Sanity** — `test_no_feature_correlates_implausibly_with_target`: the strongest
feature/target correlation is `return_5d` at **−0.0814**; all 11 sit well under
the 0.20 alarm threshold. Consistent with weak genuine signal, inconsistent with
leakage.

**Target integrity** — `test_target_matches_manual_forward_return` confirms
`y[t] = Close[t+5]/Close[t] − 1` by hand on 4 dates.
`test_target_is_not_a_feature` confirms the label and the regime column are
absent from `feature_columns` (11 unique).
`test_target_horizon_is_not_silently_shifted` confirms exactly 5 unlabelled
trailing rows — an off-by-one horizon would surface here.

## 2. Verify no centered rolling, backfill, future-derived imputation or full-sample statistics are used in features/regime construction

**PASS.**

`test_pipeline_source_has_no_forward_looking_constructs` scans every code cell of
**both** notebooks in the repository
(`E1-S3_to_E1-S6_Data_Foundation_and_Regime_Construction.ipynb`, 14 cells;
`project.ipynb`, 10 cells) for `center=True`, `.bfill(`, `method='backfill'`,
`.interpolate(`, `.fit_transform(`, and any negative `.shift()` other than the
sanctioned `shift(-5)` target horizon. **No findings.** Every rolling window is
declared `center=False, min_periods=window`; `pct_change(fill_method=None)`
performs no filling.

`test_regime_threshold_uses_only_prior_observations` rebuilds 12 sampled regime
labels from truncated history — threshold = median of `volatility_20d` values
**strictly before t**, minimum 252 prior observations — and all 12 match. The
guard is the `.shift(1)` applied after `.expanding().median()`.

The same test carries a **non-vacuity check**: a full-sample median threshold
disagrees with the published labels on **260 rows**. The test can therefore
distinguish a leaky threshold from a safe one; it is not passing trivially.

`test_regime_labels_are_clean` — labels are exactly `{LowVol, HighVol}`, split
2590 / 2575. Both regimes are thick enough for regime-conditioned evaluation.

## 3. Verify target/feature/regime indices remain aligned after rolling-window NaNs and target shift are removed

**PASS.**

`test_canonical_row_set_matches_independent_rebuild` — the published file is
compared column-by-column against a rebuild that never imports the notebook.
Dates match exactly; all 12 numeric columns match at rtol 1e-9; all 5165 regime
labels match.

Structurally, alignment is safe by construction: `dropna()` runs once on a single
frame holding X, y and regime together, so a row can never be dropped from one
and kept in another.

`test_no_missing_values_and_no_duplicate_dates` — zero NaNs; dates unique and
monotonically increasing.

`test_row_count_accounting_is_explainable` — the row loss is fully explained:

```
5442  raw
− 272  warm-up (volatility_20d valid from index 20, + 252 prior observations
                for the regime threshold, + 1 for shift(1))
−   5  unlabelled tail (t+5 target unavailable)
= 5165  canonical   ← matches the file and the manifest
```

A silent misalignment would break this identity.

## 4. Record audit result as PASS/BLOCKED

**PASS.** Recorded in this document.

Supporting integrity checks: `test_raw_artifact_matches_its_provenance_hash`,
`test_published_artifacts_match_manifest_hashes`,
`test_manifest_points_at_files_that_exist`, `test_pipeline_is_deterministic`
(two runs, identical frame hash), `test_reproduce_instructions_are_actionable`
(every file the README names exists), and
`test_notebook_was_executed_top_to_bottom` — execution counts 1→14 and 1→10, no
gaps, no duplicates, no unexecuted cells. The "manual notebook edits" edge case
is clear.

---

## Defects

### D1 — Working tree diverged from the hash-pinned artifacts — **FIXED**

**Severity:** P1 (integrity of the audit trail; not a leakage finding, so the
BLOCKER RULE was not triggered).

**Found.** All three hashed artifacts failed their manifest/provenance hash in
the working copy:

| Artifact | On disk | Manifest |
|---|---|---|
| raw SPY | `e3c6b0e5…` | `1007a7421e108dd8` |
| canonical dataset | `654b42ce…` | `96f79c15ff277bf0` |
| data dictionary | `8a93020f…` | `3e478195e58d1a4c` |

**Cause.** Line-ending conversion, not data corruption. Every tracked file in the
working tree had been rewritten CRLF while the committed blobs remained LF.
`git diff --ignore-cr-at-eol` was **empty**, and `git diff --stat` showed a
symmetric `14430 insertions(+), 14430 deletions(−)` across 11 files — the
signature of pure line-ending churn with zero content change. Stripping CR from
the on-disk files reproduced the manifest hashes exactly.

`core.autocrlf` was unset both locally and globally, so git did not perform the
conversion; an editor or sync tool rewrote the files outside git.

**Impact.** A SHA-256 is computed over bytes. Once every artifact fails its hash
for a benign reason, a genuinely modified dataset becomes indistinguishable from
line-ending noise, and the tamper-evidence the manifest exists to provide is
silently lost.

**Fix applied.**

1. All 11 tracked files in the working tree were rewritten byte-for-byte from
   their committed blobs. `git diff --stat` is now empty and the three hashes
   match the manifest.
2. Added `.gitattributes` pinning `*.csv`, `*.json` and `*.ipynb` to `-text`
   (never converted), with `* text=auto eol=lf` as the repository default, so
   the hashed artifacts stay byte-identical on any platform.

**Re-run after fix:** 18 passed.

---

## Scope boundary

This gate certifies the **E1 dataset only**. It does not certify E2.

No scaler, imputer or centering statistic exists anywhere in E1, so the
"preprocessing-before-split" and "full-sample medians/scalers" edge cases named
on the card have no surface here — they move to E2. When LightGBM training and
purged walk-forward splits land, re-audit for:

- preprocessing fitted before the split rather than inside each fold;
- purge/embargo around the **5-day overlapping target** — consecutive rows share
  forward windows, so adjacent train/test rows leak into each other unless a gap
  of at least 5 trading days separates them;
- regime labels or regime-conditioned statistics computed across fold
  boundaries.

## Sign-off

| Role | Name | Date |
|---|---|---|
| QA/QC (auditor) | Huy Pham | 2026-08-24 |
| Data (review) | | |
| Model (review) | | |
