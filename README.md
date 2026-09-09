# Effect of Data Drift on Gradient Boosting Models

A leakage-safe machine-learning study of whether signals learned from historical **SPY** data remain predictive when market volatility regimes change.

## Overview

This project predicts the **5-trading-day forward return of SPY** from historical price and volume features, then evaluates whether model performance changes between **LowVol** and **HighVol** market regimes.

The repository implements the full research workflow:

1. Acquire and validate auto-adjusted daily SPY OHLCV data.
2. Build a canonical modeling dataset with historical-only features.
3. Construct volatility regimes without using future information.
4. Train baseline and tree-based regression models.
5. Generate genuine out-of-sample predictions with purged walk-forward validation.
6. Compare overall and regime-conditioned performance.
7. Audit data integrity, split boundaries, and out-of-sample errors.

> This is a predictive research project, not an executable trading strategy. It does not model transaction costs, turnover, portfolio sizing, slippage, or causal effects.

## Research Question

**Do gradient-boosting signals trained on historical SPY data retain predictive value when the market moves between low- and high-volatility regimes?**

## Key Findings

Results reported in the accompanying technical report show that:

- The zero-return baseline achieved an overall MAE of approximately **0.01596**.
- LightGBM achieved an overall MAE of approximately **0.01606**.
- LightGBM slightly improved on the baseline in **LowVol** periods but underperformed it in **HighVol** periods.
- In the fixed multi-model comparison, **Random Forest** produced the strongest overall result with MAE **0.015615**, correlation **0.103**, and directional hit rate **0.595**.
- Predictive performance is therefore regime-dependent, and the primary LightGBM signal does not remain uniformly robust when volatility rises.

These values describe the frozen experiment reported by the project. Re-running with live market data may produce different row counts and metrics.

## Methodology

### Data

- **Instrument:** SPDR S&P 500 ETF Trust (SPY)
- **Frequency:** Daily
- **Source:** Yahoo Finance through `yfinance`
- **Adjustment:** Auto-adjusted OHLCV
- **Configured start date:** `2005-01-01`
- **Configured end date:** `null` — download through the most recent trading day

### Target

```text
forward_return_5d = Close(t+5) / Close(t) - 1
```

The target is used only as `y`; it is never included in the feature set.

### Features

The canonical dataset contains 11 historical features:

| Group | Features |
|---|---|
| Returns | `return_1d`, `return_5d`, `return_10d`, `return_20d` |
| Realized volatility | `volatility_5d`, `volatility_10d`, `volatility_20d` |
| Trend | `trend_10d`, `trend_20d`, `trend_60d` |
| Volume | `volume_ratio_20d` |

Every feature uses information available at or before prediction date `t`.

### Market Regimes

The regime label compares `volatility_20d` with an expanding median calculated from volatility observations strictly before `t`.

- At least **252 prior observations** are required.
- Values below the historical threshold are labeled `LowVol`.
- Equality and values above the threshold are labeled `HighVol`.
- No full-sample threshold or future information is used.

### Validation

The experiment uses expanding walk-forward validation with:

- **6 folds**
- **1,260 observations** minimum initial training window
- **5-trading-day purge** matching the prediction horizon
- Chronological train/test separation
- One prediction per genuine out-of-sample row

## Models

| Model | Role |
|---|---|
| Zero-return predictor | Naive benchmark |
| LightGBM regressor | Primary gradient-boosting model |
| Random Forest | Bagging-based comparison |
| XGBoost | Alternative boosting comparison |
| AdaBoost | Alternative boosting comparison |

Model parameters and pipeline paths are defined in `E3-S2_Data_Model_Integration_Flow/pipeline_config.yaml`.

## Repository Structure

```text
.
├── data/                                      # Raw and processed datasets
├── docs/                                      # Data dictionary and audit records
├── E2-S1_Baseline_Zero_Predictor/             # Zero-return benchmark
├── E2-S2_Train_Minimal_LightGBM_Regressor/    # Primary LightGBM model
├── E2-S3_Leakage_Safe_Walk_Forward_Validation/# Split and leakage audit
├── E2-S4_Generate_Canonical_OOS_Prediction_Table/
├── E2-S5_Evaluate_Overall_LowVol_HighVol_Performance/
├── E2-S6_Multi_Model_Comparison/              # RF, XGBoost, AdaBoost comparison
├── E2-S7_OOS_Error_Analysis/                  # OOS diagnostic analysis
├── E3-S2_Data_Model_Integration_Flow/         # Unified reproducible pipeline
├── E4-S2_OOS_Split_Integrity_Gate/            # OOS integrity checks
├── results/                                   # Canonical OOS outputs
├── tests/                                     # Canonical dataset tests
├── pipeline_manifest.json                     # Stage-level contract manifest
└── requirements.txt
```

Each experiment folder contains its own README, scripts, tests, and generated outputs.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/AIVIETNAM-AIO-ThyNgo/aio-conquer-module-3.git
cd aio-conquer-module-3

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies include NumPy, pandas, scikit-learn, LightGBM, XGBoost, yfinance, matplotlib, and pytest.

## Reproduce the Pipeline

Run the unified data and primary-model pipeline:

```bash
cd E3-S2_Data_Model_Integration_Flow
python -m pipeline.run_pipeline --force
cd ..
```

Run the additional model comparison:

```bash
python E2-S6_Multi_Model_Comparison/train_additional_models.py
python E2-S6_Multi_Model_Comparison/compare_all_models.py
```

Run out-of-sample error analysis and the split-integrity audit:

```bash
python E2-S7_OOS_Error_Analysis/analyze_oos_errors.py
python E4-S2_OOS_Split_Integrity_Gate/audit_oos_split_integrity.py
```

## Tests

Run the project test suites from the repository root:

```bash
python -m pytest \
  tests/ \
  E2-S1_Baseline_Zero_Predictor/tests/ \
  E2-S2_Train_Minimal_LightGBM_Regressor/tests/ \
  E2-S3_Leakage_Safe_Walk_Forward_Validation/tests/ \
  E2-S4_Generate_Canonical_OOS_Prediction_Table/tests/ \
  E3-S2_Data_Model_Integration_Flow/pipeline/tests/ \
  E4-S2_OOS_Split_Integrity_Gate/tests/ \
  E2-S7_OOS_Error_Analysis/tests/ -q
```

## Main Outputs

| Artifact | Description |
|---|---|
| `data/processed/E1-S6_canonical_modeling_dataset.csv` | Canonical modeling dataset |
| `data/processed/E1-S6_dataset_manifest.json` | Dataset schema, hashes, and provenance |
| `docs/E1-S6_data_dictionary.csv` | Feature and target definitions |
| `results/oos_predictions.csv` | Canonical out-of-sample prediction table |
| `results/oos_predictions_manifest.json` | OOS table manifest |
| `pipeline_manifest.json` | Aggregated stage contracts and hash chain |

## Reproducibility Note

The default configuration uses `end_date: null`, so a fresh run downloads the latest available SPY observations. Yahoo Finance may also revise historical data. As a result, live runs can differ from the committed artifacts and the metrics reported above.

For a strictly frozen replication, set a fixed end date in `E3-S2_Data_Model_Integration_Flow/pipeline_config.yaml` and use the committed raw dataset and manifests.

## Limitations

- The study evaluates statistical prediction, not net trading profitability.
- No transaction costs, slippage, turnover, or portfolio constraints are modeled.
- Results are specific to SPY daily data and the selected feature set.
- Regime definitions depend on realized volatility and may not capture every form of market drift.
- Reported model differences are modest and should not be interpreted as causal evidence.

## Contributors

- [Vincent Dao](https://github.com/AIVIETNAM-AIO-VincentDao25)
- [Phạm Quang Huy](https://github.com/AIVIETNAM-AIO-Huycomputervision)
- [Jasie Ngo](https://github.com/AIVIETNAM-AIO-ThyNgo)

## License

This project is released under the [MIT License](LICENSE).
