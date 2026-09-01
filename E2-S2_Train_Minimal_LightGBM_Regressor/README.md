# E2-S2 [P0][Model] Train Minimal LightGBM Regressor

| Field | Value |
|---|---|
| Epic | E2 |
| Owner | Model |
| Review | QA/QC |
| Depends on | E2-S1 (Baseline y_hat=0 -- `splits.py`, `metrics.py`, `output/baseline_zero_fold_metrics.csv`) |

## Deliverable

A single LightGBM regression configuration predicting `forward_return_5d`
from the 11 frozen E1-S4 features, trained and scored on the exact purged
walk-forward OOS folds defined in E2-S1's [`splits.py`](../E2-S1_Baseline_Zero_Predictor/splits.py)
(imported unchanged, not reimplemented) and scored with E2-S1's
[`metrics.py`](../E2-S1_Baseline_Zero_Predictor/metrics.py) (also imported
unchanged), so LightGBM and the zero baseline are compared on identical
ground.

## Acceptance

- **Seed + package version + hyperparameters recorded.** `output/lightgbm_summary.json`
  records `seed`, the full `hyperparameters` dict, `package_versions`
  (`lightgbm`, `numpy`, `pandas`), and the interpreter's Python version.
- **No model zoo; no broad hyperparameter search.** [`train_lightgbm.py`](train_lightgbm.py)
  defines exactly one `LIGHTGBM_PARAMS` dict, used for every fold. There is
  no search loop, no list of candidate configs, and no code path that could
  produce more than one trained configuration per run.
  `test_single_fixed_configuration_not_a_search_space` asserts no
  hyperparameter value is a list/tuple/set.
- **Train metrics used only diagnostically.** Per-fold train MAE is written
  to a column literally named `train_mae_diagnostic_only`, and
  `lightgbm_summary.json` carries an explicit
  `train_metrics_are_diagnostic_only` statement. Nothing downstream reads
  this column to make a performance claim.
- **Claims rely on OOS output.** The only metrics fed into the
  baseline-vs-LightGBM comparison (`mae_improvement_over_baseline`) are
  computed on each fold's held-out test block, via the same `mae` /
  `prediction_correlation` / `directional_hit_rate` functions E2-S1 uses.

## Hyperparameters (fixed, chosen before any OOS metric was seen)

```python
{
    "objective": "regression", "metric": "mae",
    "n_estimators": 200, "max_depth": 4, "num_leaves": 15,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_alpha": 0.0, "reg_lambda": 1.0,
    "random_state": 42, "deterministic": True, "force_col_wise": True,
}
```

Shallow depth (`max_depth=4`, `num_leaves=15`) and a raised
`min_child_samples` (30) were chosen a priori because the target is a noisy,
low-signal financial return over only 11 features (the E4-S1 audit found the
strongest feature/target correlation is -0.08) -- deep trees have nothing
genuine to fit and would overfit noise. `subsample` / `colsample_bytree`
add further regularization. This exact configuration ran once against the
OOS folds and was not adjusted afterward.

## Edge cases

- **NaN/inf input.** `validate_no_nan_inf` checks every feature and the
  target column before any fold trains; it raises rather than silently
  passing corrupted rows to LightGBM. Verified by
  `test_validate_no_nan_inf_raises_on_nan` / `_on_inf`.
- **Stale baseline comparison.** `run()` checks that
  `baseline_zero_summary.json`'s recorded `canonical_dataset_sha256` matches
  the dataset just loaded, and raises before training if they differ --
  otherwise a canonical-dataset regeneration where only `train_lightgbm.py`
  gets rerun would silently compare fresh LightGBM OOS error against a
  baseline scored on different rows. Verified by
  `test_run_raises_if_baseline_summary_was_generated_from_a_different_dataset`.
- **Predictions nearly constant.** `predictions_are_nearly_constant` flags
  any fold whose OOS prediction std falls under `1e-6`, logged per fold in
  `lightgbm_fold_metrics.csv`. Observed: `False` for all 6 folds this run --
  the model is not collapsing to a constant output.
- **Deep trees overfit.** Guarded structurally by the fixed shallow
  hyperparameters above, not by early-stopping or a validation-set search
  (which would reintroduce tuning). `train_mae_diagnostic_only` vs `mae`
  per fold gives a visible overfit gap for QA/QC to inspect (e.g. fold 0:
  train 0.0164 vs OOS 0.0177 -- close, not blown out).
- **Repeated tuning against future/OOS results.** Procedurally avoided:
  this file contains one hyperparameter dict and one training pass per
  fold; `lightgbm_summary.json` carries a `no_tuning_declaration` stating
  the configuration was fixed before any OOS metric existed.
- **Package-version behavior differences.** `lightgbm==4.6.0`,
  `scikit-learn==1.9.0` (required by `lightgbm.sklearn`) are pinned in
  `requirements.txt` and the installed version is recorded in
  `lightgbm_summary.json` on every run. `deterministic=True` and
  `force_col_wise=True` make repeated fits *on the same machine* (same
  thread count, from `n_jobs=-1`) bit-identical -- verified by
  `test_same_seed_same_fold_produces_identical_predictions`. This does not
  guarantee identical predictions across machines with different core
  counts (`n_jobs=-1` picks up each machine's own core count, which
  `deterministic=True` does not correct for); what's reproducible across
  machines is the fixed hyperparameters and pinned package versions, not
  the exact floating-point predictions.

## Run

```bash
python E2-S1_Baseline_Zero_Predictor/run_baseline.py   # produces the baseline this script compares against
python E2-S2_Train_Minimal_LightGBM_Regressor/train_lightgbm.py
python -m pytest E2-S2_Train_Minimal_LightGBM_Regressor/tests/test_train_lightgbm.py -v
```

## Results (this run) -- OOS only

| Fold | Test window | n_test | train MAE (diagnostic) | OOS MAE | Baseline MAE | Improvement | Correlation | Hit rate |
|---|---|---|---|---|---|---|---|---|
| 0 | 2011-02-02 → 2013-09-03 | 650 | 0.01640 | 0.01770 | 0.01701 | -0.00069 | 0.040 | 0.475 |
| 1 | 2013-09-04 → 2016-04-04 | 650 | 0.01631 | 0.01398 | 0.01392 | -0.00005 | 0.071 | 0.505 |
| 2 | 2016-04-05 → 2018-10-30 | 650 | 0.01573 | 0.01024 | 0.01057 | +0.00033 | 0.098 | 0.557 |
| 3 | 2018-10-31 → 2021-06-02 | 650 | 0.01461 | 0.01999 | 0.02008 | +0.00009 | 0.127 | 0.592 |
| 4 | 2021-06-03 → 2024-01-02 | 650 | 0.01514 | 0.01914 | 0.01879 | -0.00035 | 0.050 | 0.555 |
| 5 | 2024-01-03 → 2026-08-13 | 655 | 0.01561 | 0.01530 | 0.01538 | +0.00007 | 0.105 | 0.611 |
| **Overall** | | **3905** | -- | **0.01606** | **0.01596** | **-0.0001** | **0.079** | **0.549** |

**Reading this honestly:** on pooled MAE, this LightGBM configuration is
essentially tied with (marginally worse than) the zero baseline -- an MAE
win is not established. Prediction correlation is positive but weak
(0.04-0.13 per fold) and directional hit rate sits mostly just above 50%
(above chance in 5 of 6 folds). This is a legitimate, expected outcome for a
noisy 5-day-forward equity return signal and is exactly why E2-S1 exists: it
keeps this model honest against a trivial competitor instead of reporting
OOS numbers in isolation.

## Scope boundary

This card trains one configuration and reports OOS numbers. It does not
claim the model beats the baseline, does not search hyperparameters, and
does not perform regime-conditioned evaluation -- that belongs to a later
E2 story, which should reuse `splits.py`/`metrics.py` and the `regime`
column already carried through `lightgbm_oos_predictions.csv`.
