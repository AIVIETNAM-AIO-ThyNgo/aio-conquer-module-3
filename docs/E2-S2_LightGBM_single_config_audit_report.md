# Single LightGBM Configuration for 5D Forward Return — Independent Audit

| Field | Value |
|---|---|
| Object under audit | E2-S2's "one fixed LightGBM configuration" predicting `forward_return_5d` from the 11 frozen E1-S4 features |
| Repository | `aio-conquer-module-3` @ `main` (`a76ecb0`) |
| Auditor | Independent audit, principal-data-scientist persona (Claude Code session) |
| Date | 2026-09-02 |
| Evidence | Direct source read of `train_lightgbm.py`, `E3-S2/pipeline/{model,config}.py`, `pipeline_config.yaml`, `pipeline_manifest.json`; `git log`/`git diff` across all local branches; independent recomputation from raw OOS prediction CSVs (stdlib `csv`, no pandas available in this shell) |
| Result | **Fail** |

## Verdict up front

There are, right now, in this repository's own history, **two different LightGBM implementations that both claim to be "the E2-S2 configuration"** and they produce **two different OOS numbers** from nominally the same hyperparameters. The second, currently-on-disk implementation silently drops the recorded seed. That is a positively demonstrated reproducibility failure, not a suspicion — and it independently corroborates this session's own working hypothesis about where this kind of bug hides: in a duplicate code path, not the one anyone re-reads. Separately, this project ran a disclosed but real four-algorithm-family comparison (`E2-S6`) after the single-LightGBM deliverable was frozen, and Random Forest beat LightGBM on every metric, in every regime. Neither finding was fabricated for this report — both are named, dated, and traceable to specific lines and commits below.

---

## Artifacts under audit

| Artifact | Status |
|---|---|
| `E2-S2_Train_Minimal_LightGBM_Regressor/train_lightgbm.py` | Standalone script, single commit `b8eec2f`, never modified since |
| `E3-S2_Data_Model_Integration_Flow/pipeline/model.py::run_lightgbm()` | Second, independent implementation of the *same* stage, added in `936e81b` |
| `E3-S2_Data_Model_Integration_Flow/pipeline_config.yaml` | YAML source of hyperparameters for the pipeline path |
| `E2-S2_Train_Minimal_LightGBM_Regressor/output/lightgbm_summary.json` (on disk) | Gitignored since `936e81b` (`**/output/*.json`) — currently produced by the pipeline path, not the standalone script |
| `data/processed/E1-S6_canonical_modeling_dataset.csv` | **Absent from disk** — gitignored intermediate; full end-to-end re-run is currently blocked |
| `E4-S2_OOS_Split_Integrity_Gate/output/integrity_gate_report.json` | PASS, all 9 checks, `generated_at_utc` 2026-08-30T07:12:04 |
| `E2-S6_Multi_Model_Comparison/output/*` | RandomForest/AdaBoost/XGBoost trained and ranked against LightGBM, `generated_at_utc` 2026-09-01T05:10:23 |

---

## 1. Reproducibility & environment pinning — **FAIL**

**Seed recorded, but not actually used by the code path that generated the current artifacts.** `train_lightgbm.py:65,84` fixes `SEED = 42` and embeds it in `LIGHTGBM_PARAMS["random_state"]` — that script, run alone, is genuinely deterministic (its own test, `test_same_seed_same_fold_produces_identical_predictions`, confirms `np.array_equal` across two fits). But the checked-in `lightgbm_summary.json`'s `hyperparameters` block (verified by direct `json.load` in this session) contains **no `random_state` key at all**, and matches `pipeline_config.yaml`'s `model.lightgbm_params` (lines 50–65) key-for-key, which also has no `random_state`/`n_jobs`. Tracing the actual call: `E3-S2/pipeline/model.py:364` does `model = lgb.LGBMRegressor(**cfg.model.lightgbm_params)` — `cfg.model.seed` (=42) is read at `config.py:171` into `ModelConfig.seed`, but it is **never merged into `lightgbm_params`** before that constructor call; it is used only for logging (`model.py:326,419`, `"seed": cfg.model.seed`). Grepped both files for every occurrence of `random_state` and `lightgbm_params` to confirm no other merge point exists — there is none.

**This is not hypothetical — it already produced two different numbers.** `git diff b8eec2f 936e81b -- E2-S2.../output/lightgbm_summary.json` shows the version generated on 2026-08-29T11:17:23 (by the standalone script, `random_state: 42` present, MAE **0.0160559**, correlation **0.07885**, hit rate **0.5493**) was **deleted from git** when the unified pipeline landed, and the version now on disk (2026-08-30T07:11:57, no `random_state`, MAE **0.0160734**, correlation **0.07060**, hit rate **0.5536**) is a different run of nominally the same configuration. Correlation moved ~10% relative on identical data, identical folds, identical nominal hyperparameters — consistent with an unseeded fit, not floating-point noise. Every downstream number in `E2-S4`, `E2-S5`, `E2-S6`, and the prior audit report's headline finding trace back to whichever of these two the pipeline last happened to produce.

**Package versions**: recorded and pinned. `numpy==2.5.2`, `pandas==3.0.5`, `lightgbm==4.6.0` in the summary match `requirements.txt` exactly (`==` pins, no lockfile/hash-pinning or container digest — a lesser but real gap for §10).

**Independent re-run**: attempted and blocked — `data/processed/E1-S6_canonical_modeling_dataset.csv` is absent from disk (gitignored). Given the seed-drop defect above, a fresh re-run of the pipeline path would not be expected to reproduce the current numbers even if the dataset were present.

## 2. Single-configuration proof ("no model zoo") — **FAIL (disclosed, but real)**

Within `E2-S2` itself: clean. `git log --all -- train_lightgbm.py` returns exactly one commit (`b8eec2f`); the 12-key `LIGHTGBM_PARAMS` dict has never changed. No `Optuna`/`Hyperopt`/`GridSearchCV`/`RandomizedSearchCV`/`BayesSearchCV`/`.study(` import or artifact exists anywhere in the repository (`grep -rniE` across every `.py`/`.ipynb`, zero hits) or in either notebook.

Outside `E2-S2`: not clean. `E2-S6_Multi_Model_Comparison` (commit `b0cbde8`, added *after* `E2-S2`'s LightGBM config was already frozen and OOS-scored) trains **Random Forest, AdaBoost, and XGBoost** — three additional algorithm families — on the identical purged folds/features/target, and explicitly ranks all five models. This is transparently disclosed (README: *"Is LightGBM the best model? No."*) and explicitly scoped as non-canonical (*"This does not replace LightGBM as 'the' E2 model... E2-S4's canonical `results/oos_predictions.csv` still points at LightGBM's output, unchanged"*), and each additional model's hyperparameters were fixed a priori per its own README table, not searched. Per the audit standard this is still a finding, not a clean pass: the deliverable was scoped as "single LightGBM configuration, not a model-selection exercise," and a four-family comparison against the same OOS-scoring machinery is a model-selection exercise, regardless of how honestly its result (Random Forest wins on every metric, in both regimes; LightGBM ranks 4th of 5, worse than the zero baseline) was reported and *not* acted on.

## 3. No broad hyperparameter search — **Pass**

`max_depth=4, num_leaves=15, learning_rate=0.05, min_child_samples=30, subsample/colsample_bytree=0.8` are round, domain-plausible values with a stated rationale in-code (*"small (11-feature), noisy, low-signal financial regression target ... shallow trees and conservative regularization"*) — not suspiciously over-precise (no `num_leaves=47`-style artifacts). Combined with §2's negative search-tool grep, there is no positive evidence of an automated or hidden sweep specifically over LightGBM's own hyperparameters. The caveat from §2 (a disclosed cross-family comparison exists) does not itself imply LightGBM's 12 values were searched — they were not, in any commit.

## 4. Train vs OOS metric discipline — **Pass**

Every claims-bearing number (`mae`, `prediction_correlation`, `directional_hit_rate`, `mae_improvement_over_baseline`) is computed only on `fold.test_idx`, in both code paths. `train_mae_diagnostic_only` is explicitly labelled and both summaries carry a `train_metrics_are_diagnostic_only` field disclaiming any claim may cite it. Computed independently from the checked-in `lightgbm_fold_metrics.csv`: mean train MAE **0.015599**, mean OOS MAE **0.016074**, gap **+0.000475** (OOS worse by ~3% relative) — small, and in 2 of 6 folds OOS MAE is actually *lower* than train MAE, consistent with a genuinely shallow, non-overfit model rather than a train/OOS gap being quietly omitted.

## 5. Frozen-feature integrity — **Pass**

`FEATURE_COLUMNS` in `train_lightgbm.py` is checked at runtime (`validate_feature_columns`) against `data/processed/E1-S6_dataset_manifest.json["feature_columns"]`, and `E2-S2/tests/test_train_lightgbm.py::test_feature_columns_match_frozen_manifest` enforces this in CI. `E4-S1_leakage_audit_record.md` (2026-08-24, pre-dates E2-S2's first commit of 2026-08-29/30 by several days) independently rebuilt every feature point-in-time from raw OHLCV and found zero mismatches — the freeze predates training. Both the standalone script and the pipeline path load features from the same `CANONICAL_PATH`/`canonical_csv`, gated by an explicit sha256 equality check against the value recorded when the baseline was scored (`train_lightgbm.py:138-144`; `model.py:307-312`) — same file used for train and OOS by construction, not by convention.

## 6. Input robustness: NaN / inf — **Pass, with one gap**

`train_lightgbm.py::validate_no_nan_inf` checks **both NaN and inf** on `FEATURE_COLUMNS + [TARGET_COL]` and raises before training (tested explicitly: `test_validate_no_nan_inf_raises_on_nan/_inf`). The pipeline path's equivalent, `model.py::_validate_canonical`, checks NaN across the whole frame and checks `np.isfinite` **only on the target column**, not on the feature columns — an inf in a feature column would not be caught by this second implementation before fitting. In practice this is moot today: independently recomputed from the checked-in `lightgbm_oos_predictions.csv`, NaN count = 0 and inf count = 0 in both `y_true` and `y_pred`, and `E4-S1` established the canonical dataset itself is NaN/inf-free at the source. But it is a real latent gap in the second implementation, consistent with §1/§2's broader theme that the duplicate pipeline code path is less rigorous than the script it re-implements.

## 7. Degenerate / near-constant prediction check — **Pass**

Recomputed directly from `lightgbm_oos_predictions.csv` (n=3,905): predictions min **-0.0982**, max **0.0620**, mean **0.00184**, std **0.00695**; actual returns mean **0.00287**, std **0.02223**. `predictions_nearly_constant` (threshold std < 1e-6) is correctly `False` — not a collapsed predictor. Variance ratio (pred var / actual var) = **0.098**: the model explains a real but small fraction of target variance and is meaningfully shrunk toward the mean (expected for shallow trees + 0.05 learning rate on a weak-signal target), not literally constant, and its mean is close to the target's mean rather than a leaked or arbitrary constant.

## 8. Overfitting controls (deep trees) — **Pass**

Regularizers are present and reasoned about (`max_depth=4`, `num_leaves=15` well under 2^4, `min_child_samples=30`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`), and `E2-S2/tests` pins bounds on them (`max_depth<=6`, `num_leaves<=31`, `min_child_samples>=20`). **No early stopping is used anywhere** in either code path (`grep` for `early_stopping`/`eval_set`/`callbacks` across both `E2-S2` and `E3-S2/pipeline`: zero hits) — `n_estimators=200` is a fixed, non-adaptive boosting-round count, so the early-stopping-leakage failure mode named in the standard does not apply here; there is no validation set to accidentally conflate with the OOS set. The small, consistent train/OOS gap from §4 (+0.000475) is fully consistent with the shallow-tree, high-regularization configuration actually used.

## 9. No repeated tuning against future/OOS results (the peeking problem) — **FAIL**

No experiment tracker (no MLflow/W&B directory, no `.db` study file) exists to give a full timeline, so this can't be answered from logs. But the OOS set was demonstrably **scored at least twice** by two different implementations on two different dates with two different results — see §1. That is exactly the failure mode this section exists to catch, just arrived at via an engineering defect (silently dropped seed across a code duplication) rather than manual hyperparameter iteration. The hyperparameter *values* were never changed in response to either scoring event (both commits show identical `objective/metric/n_estimators/max_depth/num_leaves/learning_rate/min_child_samples/subsample/subsample_freq/colsample_bytree/reg_alpha/reg_lambda`), so this is not "peeking-driven tuning" in the classic sense — but it does mean the number currently reported as "the" OOS MAE is one of at least two non-identical candidates produced from an unseeded process, and a third run would plausibly produce a third number. That is disqualifying under the standard's own framing regardless of intent.

## 10. Package-version behaviour risk — **Unverified (partial pass)**

`lightgbm==4.6.0` param names used (`n_estimators`, `max_depth`, `num_leaves`, `learning_rate`, `min_child_samples`, `subsample`, `subsample_freq`, `colsample_bytree`, `reg_alpha`, `reg_lambda`) are the current sklearn-wrapper native names for this version, not deprecated aliases — no silent-fallback risk identified there. `requirements.txt` pins exact versions (`==`) for numpy/pandas/lightgbm, which is adequate for detecting drift on `pip install`, but there is no lockfile (`uv.lock`/`poetry.lock`/hash-pinned `requirements.txt`) or container digest in this repository (the `uv.lock` that does exist is inside `rcs/`, an unrelated vendored skills repo, not this project's environment) — marked unverified rather than pass because a `pip install -r requirements.txt` today would resolve to the pinned versions but would not catch a compromised/altered package release at the same version number.

---

## 11. Sign-off checklist

- [x] Seed, LightGBM version, and full hyperparameter dict recorded — **but the seed recorded in the artifact currently on disk was not the seed actually used to fit the model** (§1)
- [ ] Independent re-run reproduces the OOS metric — **blocked** (canonical dataset absent) **and would not be expected to succeed** given the seed defect
- [x] Experiment/run history reviewed for hidden alternative configs or algorithms — reviewed; found a **disclosed** four-algorithm-family comparison outside `E2-S2` (§2)
- [x] Explicit statement of how the config was chosen — present, consistent with `train_lightgbm.py`'s single commit
- [x] Feature-set hash/timestamp predates tuning; same file used for train and OOS (§5)
- [x] Train metrics labelled diagnostic-only; all claims traced to OOS output (§4)
- [x] NaN/inf data-quality summary present — for the standalone script; **partial** for the pipeline path (no inf check on features) (§6)
- [x] OOS prediction distribution reported here (min/max/std/variance ratio), not left for the reader to infer (§7)
- [x] Regularisation hyperparameters disclosed and reasoned about; no early stopping used, so no early-stopping/OOS-set conflation is possible (§8)
- [ ] Timeline of OOS scoring events — **not logged by the project**; reconstructed here from git history, showing at least 2 distinct scoring events with diverging results (§1, §9)
- [ ] Environment pinned via lockfile/container — exact-version `requirements.txt` only, no hash lock or container digest (§10)

## Overall verdict: **Fail**

Per the standard's own bar: *"Fail — reproduction fails, an undisclosed alternative model/search is found, early stopping used the final OOS set, or the OOS set was demonstrably scored more than once during tuning."* Two of these four conditions are met directly: **reproduction cannot be shown to succeed and has concrete evidence of having already failed once** (§1), and **the OOS set was demonstrably scored more than once**, with different results, from what was represented as one fixed configuration (§1, §9). The model-zoo finding (§2) is disclosed rather than hidden, which is materially better than a concealed alternative, but does not on its own change the overall verdict given the two Fail items above.

### Required before this can be re-audited as Pass or Conditional pass

1. Fix `E3-S2/pipeline/model.py::run_lightgbm` (and `config.py`) to actually pass `cfg.model.seed` into the `LGBMRegressor` constructor (either add `random_state`/`n_jobs` to `pipeline_config.yaml`'s `lightgbm_params` block, or merge `cfg.model.seed` in before the `**cfg.model.lightgbm_params` call) — then add a determinism test for `run_lightgbm` equivalent to `test_same_seed_same_fold_produces_identical_predictions`, which currently only covers the standalone script's hardcoded params, not the YAML-driven path that actually produced the artifacts under audit.
2. Decide, once, which of `train_lightgbm.py` or `pipeline/model.py::run_lightgbm` is the canonical generator of `E2-S2`'s outputs, and delete or clearly deprecate the other — two live implementations of "the one frozen configuration" is itself the mechanism that produced this audit's headline finding.
3. Restore `data/processed/E1-S6_canonical_modeling_dataset.csv` (or otherwise make the pipeline runnable end-to-end) so a from-scratch reproduction can actually be attempted, not just reasoned about from committed artifacts.
4. Either fold `E2-S6`'s finding into the record as a scope amendment (the deliverable becomes "single LightGBM configuration, known not to be the best of the models informally compared") or remove the comparison from the repository if the "single-model deliverable" framing is meant to hold strictly — as it stands, the repository simultaneously asserts "single LightGBM configuration" and contains a dated, disclosed exercise proving a different model wins on every metric.
5. Add an inf check on feature columns to `model.py::_validate_canonical` to match `train_lightgbm.py::validate_no_nan_inf`'s coverage.
6. Pin the environment with a lockfile or container digest, not exact-version `requirements.txt` alone.
