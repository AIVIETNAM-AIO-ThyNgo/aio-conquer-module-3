# E2-S6 [Model] Compare Random Forest, AdaBoost, XGBoost & LightGBM

| Field | Value |
|---|---|
| Epic | E2 (follow-up -- **not a board card**, requested directly) |
| Owner | Model |
| Depends on | E2-S1 (splits/metrics), E2-S2 (feature set + validation helpers), E2-S4 (canonical LightGBM OOS table), E2-S5 (regime-evaluation code) |

## Purpose

E2-S2 trained exactly one model (LightGBM) and never asked whether it was
the *best available* model, only whether it beat the trivial zero baseline.
This story answers that follow-up question directly: train three more
regressors -- **Random Forest, AdaBoost, XGBoost** -- on the identical
purged walk-forward folds, features and target, and rank all five models
(baseline + 4) together.

## What this is not

Not a Trello card. There is no `DELIVERABLE`/`ACCEPTANCE`/`EDGE CASES` text
to satisfy here -- the discipline applied is the one already established by
E2-S2 (one fixed config per model, no tuning after seeing OOS numbers,
train metrics diagnostic-only, fairness re-verified against the baseline),
applied to three more models instead of introducing a new standard.

## Files

- [`train_additional_models.py`](train_additional_models.py) -- trains
  Random Forest, AdaBoost and XGBoost, one per fold, writing per-model
  `output/<model>/` (OOS predictions, fold metrics, summary) in the same
  schema E2-S1/E2-S2 already use.
- [`compare_all_models.py`](compare_all_models.py) -- aggregates all five
  models (baseline_zero, lightgbm, random_forest, adaboost, xgboost) using
  E2-S5's `evaluate_model`/`assert_same_oos_rows` **imported, not
  reimplemented** -- writes a ranked Overall table and a full
  Overall/LowVol/HighVol breakdown.

Nothing here forks `splits.py`, `metrics.py`, `FEATURE_COLUMNS`,
`validate_no_nan_inf` or the regime-evaluation logic -- all imported from
E2-S1/E2-S2/E2-S5 so the fifth model is scored exactly like the first four.

## Hyperparameters (fixed per model, chosen before any OOS metric was seen)

Same rationale as E2-S2: 11-feature, noisy, low-signal target (E4-S1 audit:
strongest feature/target correlation is -0.08) -- every model here is
deliberately shallow/regularized rather than tuned for capacity.

```python
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200, "max_depth": 4, "min_samples_leaf": 30,
    "max_features": 0.8, "random_state": 42, "n_jobs": -1,
}
ADABOOST_BASE_ESTIMATOR_PARAMS = {"max_depth": 3, "min_samples_leaf": 30, "random_state": 42}
ADABOOST_PARAMS = {"n_estimators": 100, "learning_rate": 0.05, "loss": "linear", "random_state": 42}
XGBOOST_PARAMS = {
    "objective": "reg:squarederror", "n_estimators": 200, "max_depth": 4,
    "learning_rate": 0.05, "min_child_weight": 30, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0,
    "random_state": 42, "n_jobs": -1, "tree_method": "hist", "verbosity": 0,
}
```

## New dependency

`xgboost==3.4.1` was not previously in `requirements.txt` -- added it.
Install with `pip install -r requirements.txt` (or `pip install xgboost`
directly) before running this folder's scripts.

## Results (this run) -- Overall, ranked by OOS MAE

| Rank | Model | N | MAE | Improvement vs. baseline | Correlation | Hit rate | Nearly constant |
|---|---|---|---|---|---|---|---|
| 1 | **random_forest** | 3905 | **0.015615** | **+0.000341** | **0.103** | **0.595** | False |
| 2 | baseline_zero | 3905 | 0.015957 | +0.000000 | N/A | N/A | True |
| 3 | xgboost | 3905 | 0.016006 | -0.000049 | 0.061 | 0.556 | False |
| 4 | lightgbm | 3905 | 0.016056 | -0.000099 | 0.079 | 0.549 | False |
| 5 | adaboost | 3905 | 0.016229 | -0.000272 | 0.090 | 0.391 | False |

**Random Forest is the best model here, on all three metrics at once** (lowest
MAE, highest correlation, highest hit rate) -- and it is the *only* model of
the four that beats the zero baseline on MAE. LightGBM, the only model E2-S2
tried, ranks **4th of 5** -- worse than doing nothing.

## Results by regime (Overall / LowVol / HighVol)

| Model | Scope | N | MAE | Correlation | Hit rate |
|---|---|---|---|---|---|
| random_forest | LowVol | 2216 | 0.01156 | 0.040 | 0.588 |
| random_forest | HighVol | 1689 | 0.02094 | 0.114 | 0.605 |
| lightgbm | LowVol | 2216 | 0.01172 | 0.019 | 0.550 |
| lightgbm | HighVol | 1689 | 0.02175 | 0.092 | 0.548 |
| xgboost | LowVol | 2216 | 0.01170 | 0.016 | 0.556 |
| xgboost | HighVol | 1689 | 0.02166 | 0.070 | 0.555 |
| adaboost | LowVol | 2216 | 0.01200 | **-0.052** | 0.407 |
| adaboost | HighVol | 1689 | 0.02177 | 0.112 | 0.369 |

Random Forest beats every other model in **both** regimes, not just
Overall -- it is not winning by averaging a good LowVol result against a
bad HighVol one; it is ahead in each separately (full table:
[`output/all_models_regime_performance.csv`](output/all_models_regime_performance.csv)).

## An honest anomaly worth reporting, not hiding: AdaBoost

AdaBoost's Overall `directional_hit_rate` is **0.391** -- *below* chance
(0.50) -- despite a **positive** `prediction_correlation` (0.090), and its
LowVol correlation is outright **negative** (-0.052) while HighVol
correlation is positive (0.112). This is a genuine metric disagreement
(the exact edge case E2-S5 named: "MAE/correlation/hit rate disagree"), not
a bug: `AdaBoostRegressor` predicts via a **weighted median** of its weak
learners, not a weighted mean, so a few large, correctly-signed errors can
produce positive linear correlation while the *typical* prediction still
lands on the wrong side of zero more often than not. No hyperparameter was
adjusted after seeing this -- it is reported as observed.

## Is LightGBM the best model?

**No.** Random Forest has lower OOS MAE, higher prediction correlation and
higher directional hit rate, Overall and in both regimes. LightGBM is
2nd-to-last of five, narrowly ahead only of AdaBoost. This does not mean
LightGBM was mis-implemented -- E2-S2's hyperparameters were fixed a priori
and never tuned, same as every model here -- it means that on *this*
dataset, with *this* feature set, at *these* fixed shallow-model
hyperparameters, Random Forest generalizes better.

## Run

```bash
pip install -r requirements.txt   # picks up xgboost==3.4.1 if not already installed
python E2-S1_Baseline_Zero_Predictor/run_baseline.py                             # if not already run
python E2-S6_Multi_Model_Comparison/train_additional_models.py
python E2-S6_Multi_Model_Comparison/compare_all_models.py
python -m pytest E2-S6_Multi_Model_Comparison/tests -v
```

## Scope boundary

This does not replace LightGBM as "the" E2 model in any downstream card --
E2-S4's canonical `results/oos_predictions.csv` still points at LightGBM's
output, unchanged. This folder is a standalone comparison; promoting
Random Forest to the canonical model (if desired) would be a separate,
deliberate decision, not an automatic consequence of this table.
