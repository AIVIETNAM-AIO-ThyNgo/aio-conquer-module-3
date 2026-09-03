# y-hat = 0 Baseline vs LightGBM — Independent Audit Report

| Field | Value |
|---|---|
| Object under audit | E2-S1 zero baseline vs E2-S2 LightGBM regressor, `forward_return_5d` OOS comparison |
| Depends on | E1-S6 (canonical modeling dataset), E2-S1 `splits.py`/`metrics.py` |
| Repository | `aio-conquer-module-3` @ `main` (`a76ecb0`) — audit began on `feature/e4-s2-implementation` (`aae9758`); branch moved mid-audit, see §0 |
| Auditor | Independent audit, principal-data-scientist persona (Claude Code session) |
| Date | 2026-09-02 |
| Evidence | Direct source read of `run_baseline.py`, `train_lightgbm.py`, `metrics.py`, `splits.py`; independent recomputation from raw OOS prediction CSVs; `git grep --all` provenance checks |
| Result | **Conditional pass** |

## Verdict

Target parity, split/fold/row-count parity, and metric-code parity are all **independently verified**, not assumed — recomputed by hand from the raw prediction CSVs, bypassing the pipeline's own summary claims. That part of the engineering is genuinely sound.

It does not clear a full **Pass**. Three things block it: a reproducibility-logging gap (no git commit hash, and one artifact field — `config_hash` — that no code in this repository produces), a statistical-significance step that does not exist anywhere in the pipeline itself, and — separate from process, and the most important finding in this report — **the underlying result does not support "LightGBM beats the baseline."** LightGBM's pooled OOS MAE is nominally *worse* than the zero baseline's, it wins in only 2 of 6 folds, and the delta is statistically indistinguishable from zero in either direction.

None of this rises to the rubric's own **Fail** bar (no parity check came back non-identical, no post-hoc filtering or metric-shopping was found), so it is reported as Conditional pass with an explicit, non-negotiable remediation list — not softened, and not rounded up.

---

## §0. Environment state anomaly (discovered mid-audit, not part of the 8-point checklist)

During this audit, `HEAD` moved from `feature/e4-s2-implementation` (`aae9758`) to `main` (`a76ecb0`) outside of any command issued in this audit session — evidenced by `git reflog` showing `checkout: moving from feature/e4-s2-implementation to main` followed by `pull --tags origin main: Fast-forward`, neither of which this session invoked. Two consequences:

1. `E4-S2_OOS_Split_Integrity_Gate/` (the independent split-integrity gate script) is no longer present in the checked-out tree — it exists only on the feature branch, not on `main`.
2. `data/processed/E1-S6_canonical_modeling_dataset.csv` — the canonical input every number in this report traces back to — is **missing from disk**. It is gitignored (`data/processed/*.csv`), so a plain branch switch should not have removed it; something else did.

Consequence for this audit: a live, from-scratch re-run of `run_baseline.py` was attempted and failed with `FileNotFoundError` on the canonical dataset. This is an **environment gap, not a pipeline defect** — the script correctly refused to fabricate a result from a missing input. All findings below that require "independent reproduction" were verified by recomputing directly from the existing OOS prediction CSVs (hashed and backed up before this was discovered), not by a fresh end-to-end execution. That distinction is preserved throughout — see §1 and §8.

---

## Artifacts under audit

| Artifact | SHA-256 (first 16) | Rows | Notes |
|---|---|---|---|
| `E2-S1_Baseline_Zero_Predictor/output/baseline_zero_oos_predictions.csv` | `00d32fed21970f57` | 3,905 | ŷ=0 for every row, verified |
| `E2-S1_Baseline_Zero_Predictor/output/baseline_zero_fold_metrics.csv` | `dcad39affd35928` | 6 | per-fold MAE |
| `E2-S1_Baseline_Zero_Predictor/output/baseline_zero_summary.json` | `2c36d36c1555 9a7e` | — | contains unexplained `config_hash` — see §1 |
| `E2-S2_Train_Minimal_LightGBM_Regressor/output/lightgbm_oos_predictions.csv` | `37c82fe60868fd5e` | 3,905 | 0/3,905 exact-zero predictions |
| `E2-S2_Train_Minimal_LightGBM_Regressor/output/lightgbm_fold_metrics.csv` | `60ce352340b16e94` | 6 | per-fold MAE + baseline delta |
| `E2-S2_Train_Minimal_LightGBM_Regressor/output/lightgbm_summary.json` | `93fa6cfad9343a5c` | — | same unexplained `config_hash` value as baseline's |
| `data/processed/E1-S6_canonical_modeling_dataset.csv` | `96f79c15ff277bf0` (as recorded in both summaries) | — | **file itself currently absent from disk — see §0** |

Both summary JSONs record `canonical_dataset_sha256 = 96f79c15ff277bf0a8f8107a6313ac86e85888d4dbdbe7f7bd4ddc0c297b4da6` — identical between the two runs, which is what E2-S2's hard-coded equality gate (raises `ValueError` on mismatch) is designed to guarantee.

## Results at a glance

| § | Checklist item | Verdict |
|---|---|---|
| 1 | Reproducibility | **FAIL** |
| 2 | Target parity | PASS |
| 3 | Split/fold parity | PASS (one architectural caveat) |
| 4 | Metric code parity | PASS |
| 5 | No special-case advantage | PASS |
| 6 | Edge cases (zero-prediction convention) | PASS |
| 7 | Statistical validity | **FAIL** |
| 8 | Logging/artefact completeness | Partial — see checklist |
| — | **Overall** | **Conditional pass** |

---

## 1. Reproducibility — FAIL

**What was checked.** Commit/version pinning, fixed seed, logged config, and whether the reported numbers can be independently reproduced from logged inputs.

**Pass evidence.** `SEED = 42` is fixed and logged in `lightgbm_summary.json`. `package_versions` in both summaries (`numpy 2.5.2`, `pandas 3.0.5`, `lightgbm 4.6.0`) match `requirements.txt` exactly. I independently recomputed both MAE values from the raw `y_true`/`y_pred` columns using `np.mean(np.abs(...))` — bypassing `metrics.py` entirely — and matched the logged figures to 14+ significant digits (baseline: `0.015956810096261882` both ways; LightGBM: `0.016073403849985016` logged vs `0.016073403849985013` recomputed, a float-precision artifact, not a discrepancy).

**Fail evidence.**
- **No git commit hash is logged anywhere.** I grepped both summary JSONs' key sets directly (`sorted(json.load(...).keys())`) — no `git_commit`, `commit_hash`, or equivalent field exists. Per the audit standard: a result without a commit hash is not a logged result, regardless of how much else is logged alongside it.
- **`config_hash` is present but unexplained.** Both `baseline_zero_summary.json` and `lightgbm_summary.json` carry a `config_hash` field with the **identical** 64-character value (`7d4b36aa712c99f54340809fabf166517d23b9bc53a18779a185960f28e51887`) despite the baseline having zero hyperparameters and LightGBM having twelve. I ran `git grep -n "config_hash"` across every commit on every branch in this repository (`git rev-list --all`); the only hits are in an unrelated module (`E3-S2_Data_Model_Integration_Flow/pipeline/config.py`), never in `run_baseline.py` or `train_lightgbm.py`, on any commit. **No code in this repository, at any point in its history, produces this field.** Per audit standard, an unverifiable claim is marked unverified, not assumed benign — this field cannot be trusted as evidence of anything, and its identical value across two different configs is itself inconsistent with it meaning what its name implies.
- **Live re-run is currently blocked**, per §0 — `data/processed/E1-S6_canonical_modeling_dataset.csv` is absent from disk. Reproduction in this audit was therefore CSV-level (recomputing from already-materialized predictions), not full-pipeline (raw data → trained model → predictions). These are not the same bar, and I am not representing the weaker one as the stronger one.

## 2. Target parity — PASS

`TARGET_COL = "forward_return_5d"` is the identical hardcoded string constant in both `run_baseline.py:36` and `train_lightgbm.py:53`. Both read from the same `CANONICAL_PATH`. `train_lightgbm.py:138-144` hard-gates on `canonical_dataset_sha256` matching the value recorded in the baseline's summary and raises `ValueError` on mismatch — this is an enforced precondition, not a documentation promise. I independently merged the two OOS prediction CSVs on `Date` and confirmed `y_true` is bit-identical (`np.allclose`) on all 3,905 shared dates, with zero mismatched rows.

## 3. Split/fold parity — PASS, with one architectural caveat

**Row counts** (independently recomputed from the OOS CSVs via `groupby('fold').size()`, not trusted from either summary): identical in every one of 6 folds — 650, 650, 650, 650, 650, 655 on both sides.

**Dates:** `np.array_equal` on the sorted date arrays from both CSVs — identical, 3,905/3,905.

**Caveat:** the split is not generated once and passed by reference to both evaluations — `purged_walk_forward_splits()` is called separately, once per script. Empirically this produces identical folds here because (a) it is a pure deterministic function of `(dates, n_folds, min_train_size, horizon)` with no other inputs, (b) the sha256 dataset-identity gate in §2 blocks a silent dataset drift between the two calls, and (c) a separate downstream script, `E2-S3/validate_walk_forward.py`'s `check_fold_params_unchanged_across_outputs()`, diffs the *recorded* `split_params` between the baseline and LightGBM summaries — **but that script is currently not present on `main`** (see §0) and is opt-in even when present, not enforced inline by either training script. Three separate, correct mechanisms compensating for one architectural gap is a defensible design, not "the same object passed to both evaluations" as the strict standard asks for.

## 4. Metric code parity — PASS

Confirmed by reading the import statements directly, not by trusting a comment: both `run_baseline.py:26-31` and `train_lightgbm.py:42-47` have the identical `from metrics import (directional_hit_rate, mae, prediction_correlation)`, importing from the same file, with no baseline-specific wrapper, try/except, or re-implementation anywhere in either script. Combined with the independent hand-recomputation of MAE in §1, this is verified twice over, not assumed once.

## 5. No special-case advantage — PASS

No `StandardScaler`, no inverse-transform, no unit conversion exists in either script — `forward_return_5d` is read and predicted/compared in the same raw-return space for both models, so ŷ=0 genuinely means "0 raw return," not "0 in some transformed space that happens to differ from the metric's space." No evidence of post-hoc date or metric filtering was found; both scripts' header comments explicitly declare a single fixed configuration run before any OOS number was seen, and no artifact contradicts that declaration.

## 6. Edge cases (zero-prediction convention) — PASS

The convention is documented once, as a shared string constant (`DIRECTIONAL_HIT_RATE_ZERO_PREDICTION_CONVENTION` in `metrics.py`), imported and surfaced verbatim in both summary JSONs rather than restated in prose per-script. It is implemented generically — `directional_hit_rate()` and `prediction_correlation()` mask on `y_pred != 0` for *whichever* array is passed in, so the same code path would apply to LightGBM if it ever predicted exactly zero, not a baseline-only special case. Verified: baseline is 3,905/3,905 exact zero (hit-rate and correlation correctly `NaN`, not silently `0.0` or `0.5`); LightGBM is 0/3,905 exact zero. The symmetric branch is written correctly but was **never exercised** by this particular run — noted as untested-in-practice, not unhandled.

## 7. Statistical validity — FAIL

No Diebold-Mariano test, block/paired bootstrap, or per-fold variance disclosure exists anywhere in `E2-S1`, `E2-S2`, or `E2-S5`'s own code for the baseline-vs-LightGBM pair specifically. One was computed independently in this audit (paired bootstrap over the 6 shared folds, 10,000 resamples): mean ΔMAE (baseline − LightGBM) = **−0.000117**, 95% CI **[−0.000392, 0.000122]** (spans zero), Wilcoxon signed-rank p = 0.56 raw. Not significant in either direction.

**This is the headline finding of the audit.** LightGBM's pooled OOS MAE (**0.016073**) is not merely "not significantly better" than the baseline's (**0.015957**) — it is nominally *worse*. Per-fold breakdown (`mae_improvement_over_baseline` column, recomputed and cross-checked):

| Fold | Test window | LightGBM vs baseline MAE |
|---|---|---|
| 0 | 2011-02-02 → 2013-09-03 | **worse** by 0.000681 |
| 1 | 2013-09-04 → 2016-04-04 | worse by 0.000013 (negligible) |
| 2 | 2016-04-05 → 2018-10-30 | better by 0.000300 |
| 3 | 2018-10-31 → 2021-06-02 | worse by 0.000007 (negligible) |
| 4 | 2021-06-03 → 2024-01-02 | **worse** by 0.000380 |
| 5 | 2024-01-03 → 2026-08-13 | better by 0.000081 |

LightGBM wins on 2 of 6 folds. This is not "a statistically indistinguishable win" — it is a nominal loss on the primary metric that also fails to clear significance in either direction. Report it as such; do not report it as "LightGBM roughly matches the baseline," which implies a tie this data doesn't support either.

## 8. Logging/artefact completeness

| Item | Status |
|---|---|
| Target definition + transformation, version-controlled | ✅ |
| Fold/date list, identical object reused for both models, diff-proof | ⚠️ shared function (diff-verified empirically identical), not a shared object — see §3 |
| Per-fold row counts for both models, matching | ✅ independently verified |
| Metric function source/import, shared between both models | ✅ verified by import inspection |
| Explicit hit-rate-at-zero convention, documented and applied symmetrically | ✅ written symmetrically; LightGBM branch untested in this run |
| Significance test or variance disclosure alongside point estimates | ❌ absent from the repository; supplied externally in this audit |
| Environment/version/config/data-snapshot metadata attached | ⚠️ present and pinned, but no git commit hash and one unexplained field (`config_hash`) |
| Re-run performed independently by the auditor | ⚠️ CSV-level recomputation done and matched; live fresh execution currently blocked (§0) |

---

## Remediation checklist (required to reach Pass)

1. Log the git commit hash (`git rev-parse HEAD`) into every summary JSON produced by `run_baseline.py` and `train_lightgbm.py`.
2. Explain or remove the `config_hash` field — as it stands it is an artifact of unknown provenance and cannot be relied on for anything.
3. Add a Diebold-Mariano or paired-bootstrap significance test to the E2 pipeline itself (E2-S5 or a new story), rather than leaving it to an ad-hoc external audit.
4. Restore `data/processed/E1-S6_canonical_modeling_dataset.csv` and re-establish a consistent checked-out branch before relying on any of these numbers for a decision.
5. Merge `E4-S2_OOS_Split_Integrity_Gate` and `E2-S3`'s `check_fold_params_unchanged_across_outputs()` into `main` and wire them into the same run that produces the comparison, so the split-parity compensating controls in §3 are enforced inline rather than opt-in.
6. Correct any downstream reporting that currently states or implies "LightGBM beats the zero baseline" — on this data, with this configuration, it does not.
