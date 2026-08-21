# M03 — Gradient Boosting Signals Across Market Regimes

Research question: does a LightGBM signal for SPY five-trading-day forward returns retain different genuine out-of-sample predictive performance in Low-Vol versus High-Vol regimes?

This branch implements the Trello Data Foundation cards through E1-S6:

- [E1-S1 — Acquire & Version Raw SPY OHLCV](https://trello.com/c/XZnNG075)
- [E1-S2 — Validate Raw Market Data](https://trello.com/c/ylsf1Ag0)
- [E1-S3 — Build Returns & 5-Trading-Day Forward Target](https://trello.com/c/dRR58WDg)
- [E1-S4 — Build Historical Feature Set](https://trello.com/c/1nBr3JrS)
- [E1-S5 — Construct Leakage-Safe Low/High Volatility Regime](https://trello.com/c/G8hsrnJR)
- [E1-S6 — Publish Canonical Modeling Dataset & Data Dictionary](https://trello.com/c/4wwdJQnv)

## Primary artifacts

- `E1-S3_to_E1-S6_Data_Foundation_and_Regime_Construction.ipynb`: executable implementation, manual spot checks, charts, and quality gates.
- `data/raw/E1-S1_SPY_OHLCV_auto_adjusted.csv`: immutable auto-adjusted SPY OHLCV input.
- `data/raw/E1-S1_SPY_OHLCV_auto_adjusted.provenance.json`: source, date range, acquisition convention, and raw SHA-256.
- `data/processed/E1-S6_canonical_modeling_dataset.csv`: the only admitted modeling row set.
- `data/processed/E1-S6_dataset_manifest.json`: schema, hashes, package versions, and regime definition.
- `docs/E1-S6_data_dictionary.csv`: formula, window, units, role, timestamp semantics, and missing-value policy for every canonical column.

## Frozen E1-S4 features

The canonical dataset contains 11 features:

- Returns: `return_1d`, `return_5d`, `return_10d`, `return_20d`.
- Annualized realized volatility: `volatility_5d`, `volatility_10d`, `volatility_20d` (`ddof=1`).
- Trend versus right-aligned moving average: `trend_10d`, `trend_20d`, `trend_60d`.
- Volume: `volume_ratio_20d`.

Every feature uses information available at or before prediction date `t`. The target `forward_return_5d = Close_(t+5) / Close_t - 1` is used only as `y`.

## Leakage-safe E1-S5 regime

`volatility_20d` is compared with the expanding median of 20-day volatility values strictly before `t`. The threshold requires at least 252 prior volatility observations. Equality is classified as `HighVol`; no smoothing or full-sample threshold is used.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter nbconvert --execute --to notebook --inplace \
  E1-S3_to_E1-S6_Data_Foundation_and_Regime_Construction.ipynb
python tests/test_E1_S6_canonical_dataset.py
```

The committed raw artifact is reused on rerun, so Yahoo history cannot be silently revised. Delete neither the raw CSV nor its provenance file during a normal reproduction run.

## Scope boundary

This work establishes the E1 modeling dataset only. LightGBM training, purged walk-forward splits, canonical OOS predictions, and regime-conditioned evaluation belong to E2.
