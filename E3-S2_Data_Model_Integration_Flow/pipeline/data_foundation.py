"""data_foundation.py -- port of E1-S3→E1-S6 notebook logic into deterministic Python.

This module reproduces the entire data-foundation notebook as a single
deterministic stage with an explicit input/output contract:

  Inputs (consumed):
    - pipeline_config.yaml (config hash recorded)
    - data/raw/E1-S1_SPY_OHLCV_auto_adjusted.csv  [if present, reused]
    - data/raw/E1-S1_SPY_OHLCV_auto_adjusted.provenance.json  [if present]

  Outputs (produced):
    - data/raw/E1-S1_SPY_OHLCV_auto_adjusted.csv  [frozen raw artifact]
    - data/raw/E1-S1_SPY_OHLCV_auto_adjusted.provenance.json
    - data/processed/E1-S6_canonical_modeling_dataset.csv
    - data/processed/E1-S6_dataset_manifest.json
    - docs/E1-S6_data_dictionary.csv

  Guarantees:
    - Fails loudly on missing OHLCV columns, empty data, duplicate dates,
      non-positive prices, invalid OHLC relationships, negative volume.
    - Fails loudly on invalid regime labels (anything outside {LowVol, HighVol}).
    - Fails loudly on non-finite target values in canonical rows.
    - Deterministic: same raw artifact + same config → byte-identical outputs.
    - Raw artifact immutability: if a committed raw CSV exists, its hash is
      verified against the recorded contract before reuse; a provider-revised
      file is detected, not silently adopted.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .config import Config
from .contract import (
    StageContract,
    StaleOutputError,
    check_inputs_unchanged,
    check_outputs_current,
    file_hash,
    file_hash_or_none,
    now_utc_iso,
)


# Tolerance for floating-point OHLC consistency checks.
_ATOL = 1e-10
_RTOL = 1e-12


class DataValidationError(Exception):
    """Raised when raw market data fails a domain validation check."""


class EmptyDataError(Exception):
    """Raised when an input DataFrame is empty."""


class MissingColumnError(Exception):
    """Raised when a required column is missing."""


class InvalidRegimeError(Exception):
    """Raised when the regime column contains values outside {LowVol, HighVol}."""


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise MissingColumnError(f"{context}: missing required columns {missing}")


def _require_non_empty(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        raise EmptyDataError(f"{context}: DataFrame is empty")


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# E1-S1: Acquire & version raw SPY OHLCV
# ---------------------------------------------------------------------------

def _acquire_raw(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Download raw OHLCV or reuse the committed immutable artifact.

    Returns (raw_df, provenance_dict, source_mode).
    """
    raw_path = cfg.resolve(cfg.paths.raw_csv)
    provenance_path = cfg.resolve(cfg.paths.raw_provenance)

    if raw_path.exists():
        # Committed raw artifact exists -- reuse it, but verify its hash
        # matches the previous contract so a silently-revised file is caught.
        if not provenance_path.exists():
            raise FileNotFoundError(
                f"Raw data exists without provenance: {provenance_path}"
            )
        raw = pd.read_csv(raw_path, parse_dates=["Date"]).set_index("Date")
        _require_columns(raw, cfg.data.ohlcv_columns, "raw OHLCV")
        raw = raw[cfg.data.ohlcv_columns].copy()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        return raw, provenance, "immutable committed raw artifact"

    # No committed artifact -- download from Yahoo Finance.
    downloaded = yf.download(
        cfg.data.ticker,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        auto_adjust=cfg.data.auto_adjust,
        progress=False,
    )
    if downloaded.empty:
        raise EmptyDataError("yfinance returned an empty SPY dataset")
    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)
    _require_columns(downloaded, cfg.data.ohlcv_columns, "downloaded OHLCV")
    raw = downloaded[cfg.data.ohlcv_columns].copy()
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "Date"

    # Freeze the artifact so later reruns are bitwise deterministic.
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, date_format="%Y-%m-%d")

    provenance = {
        "card": "E1-S1 [P0][Data] Acquire & Version Raw SPY OHLCV",
        "ticker": cfg.data.ticker,
        "provider": "Yahoo Finance via yfinance",
        "downloaded_at_utc": now_utc_iso(),
        "requested_start_date": cfg.data.start_date,
        "requested_end_date": cfg.data.end_date,
        "auto_adjust": cfg.data.auto_adjust,
        "price_convention": "auto-adjusted OHLC; volume from yfinance",
        "rows": int(len(raw)),
        "first_trading_date": raw.index.min().date().isoformat(),
        "last_trading_date": raw.index.max().date().isoformat(),
        "columns": list(cfg.data.ohlcv_columns),
    }
    provenance["sha256"] = _sha256_of(raw_path)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Re-read from disk so the in-memory frame matches the frozen bytes.
    raw = pd.read_csv(raw_path, parse_dates=["Date"]).set_index("Date")
    raw = raw[cfg.data.ohlcv_columns].copy()
    return raw, provenance, "fresh yfinance download frozen as immutable raw artifact"


# ---------------------------------------------------------------------------
# E1-S2: Validate raw market data
# ---------------------------------------------------------------------------

def _validate_raw(raw: pd.DataFrame, cfg: Config) -> None:
    """Domain validation on raw OHLCV. Raises DataValidationError on any
    violation. Calendar gaps are NOT flagged -- weekends/holidays are valid."""
    _require_non_empty(raw, "raw OHLCV")
    _require_columns(raw, cfg.data.ohlcv_columns, "raw OHLCV")

    price_columns = ["Open", "High", "Low", "Close"]

    if not isinstance(raw.index, pd.DatetimeIndex):
        raise DataValidationError("Date index must be DatetimeIndex")
    if not raw.index.is_monotonic_increasing:
        raise DataValidationError("Dates are not ascending")
    if raw.index.duplicated().any():
        raise DataValidationError(
            f"Found {int(raw.index.duplicated().sum())} duplicate dates"
        )
    if raw[cfg.data.ohlcv_columns].isna().any().any():
        raise DataValidationError("Missing OHLCV values")
    for col in cfg.data.ohlcv_columns:
        if not pd.api.types.is_numeric_dtype(raw[col]):
            raise DataValidationError(f"Non-numeric OHLCV column: {col}")
    if (raw[price_columns] <= 0).any().any():
        raise DataValidationError("Found non-positive adjusted prices")
    if (raw["Volume"] < 0).any():
        raise DataValidationError("Found negative volume")

    max_oc = raw[["Open", "Close"]].max(axis=1)
    min_oc = raw[["Open", "Close"]].min(axis=1)
    # Allow tiny tolerance for auto-adjusted OHLC inconsistencies.
    if ((raw["High"] < raw["Low"]) & ~np.isclose(raw["High"], raw["Low"], atol=_ATOL, rtol=_RTOL)).any():
        raise DataValidationError("Found High < Low")
    if ((raw["High"] < max_oc) & ~np.isclose(raw["High"], max_oc, atol=_ATOL, rtol=_RTOL)).any():
        raise DataValidationError("Found High < Open/Close")
    if ((raw["Low"] > min_oc) & ~np.isclose(raw["Low"], min_oc, atol=_ATOL, rtol=_RTOL)).any():
        raise DataValidationError("Found Low > Open/Close")


# ---------------------------------------------------------------------------
# E1-S3: Build returns & 5-trading-day forward target
# ---------------------------------------------------------------------------

def _build_target(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = raw.copy()
    df[cfg.daily_return_column] = df["Close"].pct_change(fill_method=None)
    df[cfg.target.column] = (
        df["Close"].shift(-cfg.target.horizon_trading_days) / df["Close"] - 1
    )
    return df


# ---------------------------------------------------------------------------
# E1-S4: Build historical feature set
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    for window in cfg.features.return_windows:
        df[f"return_{window}d"] = df["Close"] / df["Close"].shift(window) - 1

    for window in cfg.features.volatility_windows:
        df[f"volatility_{window}d"] = (
            df[cfg.daily_return_column]
            .rolling(window=window, min_periods=window, center=False)
            .std(ddof=1)
            * np.sqrt(cfg.data.trading_days_per_year)
        )

    for window in cfg.features.trend_windows:
        ma = df["Close"].rolling(
            window=window, min_periods=window, center=False
        ).mean()
        df[f"trend_{window}d"] = df["Close"] / ma - 1

    vol_ma = df["Volume"].rolling(
        window=cfg.features.volume_ma_window,
        min_periods=cfg.features.volume_ma_window,
        center=False,
    ).mean()
    df[f"volume_ratio_{cfg.features.volume_ma_window}d"] = df["Volume"] / vol_ma

    return df


# ---------------------------------------------------------------------------
# E1-S5: Construct leakage-safe Low/High volatility regime
# ---------------------------------------------------------------------------

def _build_regime(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rc = cfg.regime
    df[rc.threshold_column] = (
        df[rc.volatility_column]
        .expanding(min_periods=rc.min_historical_observations)
        .median()
        .shift(1)
    )
    df[rc.column] = pd.Series(pd.NA, index=df.index, dtype="string")
    eligible = df[[rc.volatility_column, rc.threshold_column]].notna().all(axis=1)

    if rc.equality_rule == "HighVol":
        high_mask = eligible & (df[rc.volatility_column] >= df[rc.threshold_column])
        low_mask = eligible & (df[rc.volatility_column] < df[rc.threshold_column])
    else:
        high_mask = eligible & (df[rc.volatility_column] > df[rc.threshold_column])
        low_mask = eligible & (df[rc.volatility_column] <= df[rc.threshold_column])

    df.loc[high_mask, rc.column] = "HighVol"
    df.loc[low_mask, rc.column] = "LowVol"

    # Validate: every eligible row must have a valid regime label.
    valid_labels = {"LowVol", "HighVol"}
    actual = set(df[rc.column].dropna().unique())
    if not actual.issubset(valid_labels):
        raise InvalidRegimeError(
            f"Regime column contains invalid values: {actual - valid_labels}"
        )
    if df.loc[eligible, rc.column].isna().any():
        raise InvalidRegimeError("Eligible rows with missing regime labels")
    if df.loc[~eligible, rc.column].notna().any():
        raise InvalidRegimeError("Ineligible rows with regime labels")

    return df


# ---------------------------------------------------------------------------
# E1-S6: Publish canonical modeling dataset & data dictionary
# ---------------------------------------------------------------------------

def _build_data_dictionary(cfg: Config) -> pd.DataFrame:
    """Build the E1-S6 data dictionary from config-derived metadata."""
    records: list[dict[str, Any]] = [
        {
            "column": "Date",
            "role": "identifier",
            "dtype": "datetime64[ns]",
            "formula": "SPY trading date",
            "window": "none",
            "timestamp_semantics": "prediction date t",
            "unit": "date",
            "missing_value_policy": "not allowed in canonical rows",
            "source": "E1-S1 immutable SPY raw artifact",
        },
        {
            "column": cfg.daily_return_column,
            "role": "feature",
            "dtype": "float64",
            "formula": "Close_t / Close_(t-1) - 1",
            "window": "1 trading observation",
            "timestamp_semantics": "uses prices through t only",
            "unit": "decimal return",
            "missing_value_policy": "warm-up row excluded at E1-S6",
            "source": "auto-adjusted Close",
        },
    ]
    for window in cfg.features.return_windows:
        records.append({
            "column": f"return_{window}d",
            "role": "feature",
            "dtype": "float64",
            "formula": f"Close_t / Close_(t-{window}) - 1",
            "window": f"{window} trading observations",
            "timestamp_semantics": "uses prices through t only",
            "unit": "decimal return",
            "missing_value_policy": "warm-up rows excluded at E1-S6",
            "source": "auto-adjusted Close",
        })
    for window in cfg.features.volatility_windows:
        records.append({
            "column": f"volatility_{window}d",
            "role": "feature",
            "dtype": "float64",
            "formula": f"std(return_1d[t-{window-1}:t], ddof=1) * sqrt(252)",
            "window": f"{window} trading observations",
            "timestamp_semantics": "right-aligned; uses returns through t only",
            "unit": "annualized decimal volatility",
            "missing_value_policy": "warm-up rows excluded at E1-S6",
            "source": "derived from return_1d",
        })
    for window in cfg.features.trend_windows:
        records.append({
            "column": f"trend_{window}d",
            "role": "feature",
            "dtype": "float64",
            "formula": f"Close_t / mean(Close[t-{window-1}:t]) - 1",
            "window": f"{window} trading observations",
            "timestamp_semantics": "right-aligned; uses prices through t only",
            "unit": "decimal ratio minus one",
            "missing_value_policy": "warm-up rows excluded at E1-S6",
            "source": "auto-adjusted Close",
        })
    records.append({
        "column": f"volume_ratio_{cfg.features.volume_ma_window}d",
        "role": "feature",
        "dtype": "float64",
        "formula": f"Volume_t / mean(Volume[t-{cfg.features.volume_ma_window-1}:t])",
        "window": f"{cfg.features.volume_ma_window} trading observations",
        "timestamp_semantics": "right-aligned; uses volume through t only",
        "unit": "ratio",
        "missing_value_policy": "warm-up rows excluded at E1-S6",
        "source": "SPY Volume",
    })
    records.append({
        "column": cfg.target.column,
        "role": "target",
        "dtype": "float64",
        "formula": f"Close_(t+{cfg.target.horizon_trading_days}) / Close_t - 1",
        "window": f"{cfg.target.horizon_trading_days} subsequent trading observations",
        "timestamp_semantics": "label attached to prediction date t; never enters X",
        "unit": "decimal forward return",
        "missing_value_policy": "final five unavailable labels excluded at E1-S6",
        "source": "auto-adjusted Close",
    })
    records.append({
        "column": cfg.regime.column,
        "role": "segment label",
        "dtype": "string",
        "formula": f"HighVol if {cfg.regime.volatility_column} >= median(prior {cfg.regime.volatility_column}); else LowVol",
        "window": f"20-day volatility + expanding past-only threshold (min {cfg.regime.min_historical_observations} observations)",
        "timestamp_semantics": "threshold excludes current and all future rows",
        "unit": "{LowVol, HighVol}",
        "missing_value_policy": "insufficient-history rows excluded at E1-S6",
        "source": f"derived from {cfg.regime.volatility_column}",
    })
    return pd.DataFrame(records)


def _publish_canonical(
    df: pd.DataFrame,
    cfg: Config,
    provenance: dict[str, Any],
    source_mode: str,
) -> pd.DataFrame:
    """Drop warm-up / unavailable-target rows, write canonical CSV + manifest + data dictionary."""
    feature_cols = cfg.feature_columns
    target_col = cfg.target.column
    regime_col = cfg.regime.column
    numeric_cols = feature_cols + [target_col]

    canonical_cols = ["Date"] + feature_cols + [target_col, regime_col]

    # Drop rows with any NaN in feature/target/regime (alignment by construction).
    block = df[feature_cols + [target_col, regime_col]]
    _require_non_empty(block, "feature/target/regime block before dropna")
    canonical_indexed = block.dropna().copy()
    _require_non_empty(canonical_indexed, "canonical rows after dropna")

    # Validate no non-finite values in numeric columns.
    numeric_vals = canonical_indexed[numeric_cols].to_numpy(dtype=float)
    if not np.isfinite(numeric_vals).all():
        raise DataValidationError("Non-finite values in canonical numeric columns")

    # Validate regime labels one more time.
    regime_vals = set(canonical_indexed[regime_col].unique())
    if not regime_vals.issubset({"LowVol", "HighVol"}):
        raise InvalidRegimeError(f"Invalid regime values in canonical data: {regime_vals}")

    canonical_dataset = canonical_indexed.reset_index()
    canonical_dataset = canonical_dataset[canonical_cols].copy()
    canonical_dataset[numeric_cols] = canonical_dataset[numeric_cols].astype("float64")
    canonical_dataset[regime_col] = canonical_dataset[regime_col].astype("string")
    canonical_dataset["Date"] = pd.to_datetime(canonical_dataset["Date"])

    # Write canonical CSV.
    canonical_path = cfg.resolve(cfg.paths.canonical_csv)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_dataset.to_csv(canonical_path, index=False, date_format="%Y-%m-%d")

    # Write data dictionary.
    data_dictionary = _build_data_dictionary(cfg)
    dd_path = cfg.resolve(cfg.paths.data_dictionary)
    dd_path.parent.mkdir(parents=True, exist_ok=True)
    data_dictionary.to_csv(dd_path, index=False)

    # Write manifest.
    raw_path = cfg.resolve(cfg.paths.raw_csv)
    manifest = {
        "card": "E1-S6 [P0][Data] Publish Canonical Modeling Dataset & Data Dictionary",
        "generated_at_utc": now_utc_iso(),
        "source_mode": source_mode,
        "source_raw_path": str(raw_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "source_raw_sha256": file_hash(raw_path),
        "canonical_path": str(canonical_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "canonical_sha256": file_hash(canonical_path),
        "data_dictionary_path": str(dd_path.relative_to(cfg._repo_root)).replace("\\", "/"),
        "data_dictionary_sha256": file_hash(dd_path),
        "rows": int(len(canonical_dataset)),
        "columns": canonical_cols,
        "first_date": canonical_dataset["Date"].min().date().isoformat(),
        "last_date": canonical_dataset["Date"].max().date().isoformat(),
        "feature_columns": feature_cols,
        "target_column": target_col,
        "regime_column": regime_col,
        "regime_threshold": {
            "volatility_column": cfg.regime.volatility_column,
            "method": "expanding median of values strictly before t",
            "minimum_prior_observations": cfg.regime.min_historical_observations,
            "equality_rule": cfg.regime.equality_rule,
        },
        "config_hash": cfg.config_hash,
        "python": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "yfinance": yf.__version__,
        },
    }
    manifest_path = cfg.resolve(cfg.paths.manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return canonical_dataset


def _load_stage_contract(manifest_path: Path, stage_name: str) -> StageContract | None:
    """Load a single stage's contract from the master manifest.

    The master manifest has a different structure than a single-stage contract
    (it wraps all stages under a `stages` key). This helper extracts one
    stage's contract from it.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = data.get("stages", {})
    if stage_name in stages:
        return StageContract.from_dict(stages[stage_name])
    # Also accept a flat single-stage contract.
    if data.get("stage_name") == stage_name:
        return StageContract.from_dict(data)
    return None


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_data_foundation(cfg: Config, force: bool = False) -> StageContract:
    """Run the full data-foundation stage.

    Args:
        cfg: loaded pipeline config.
        force: if True, regenerate outputs even if inputs are unchanged.
            If False (default), raises StaleOutputError when outputs exist
            but were produced from different inputs than the current run
            would use.

    Returns:
        StageContract recording inputs, outputs, and params.
    """
    raw_path = cfg.resolve(cfg.paths.raw_csv)
    provenance_path = cfg.resolve(cfg.paths.raw_provenance)
    canonical_path = cfg.resolve(cfg.paths.canonical_csv)
    manifest_path = cfg.resolve(cfg.paths.manifest)
    dd_path = cfg.resolve(cfg.paths.data_dictionary)

    # Build the contract we *would* produce.
    input_hashes: dict[str, str] = {}
    if raw_path.exists():
        input_hashes[str(raw_path)] = file_hash(raw_path)
    if provenance_path.exists():
        input_hashes[str(provenance_path)] = file_hash(provenance_path)

    proposed = StageContract(
        stage_name="data_foundation",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        params={
            "ticker": cfg.data.ticker,
            "start_date": cfg.data.start_date,
            "horizon_trading_days": cfg.target.horizon_trading_days,
            "feature_columns": cfg.feature_columns,
            "regime_equality_rule": cfg.regime.equality_rule,
        },
    )

    # Check staleness: if outputs exist and inputs are unchanged, we can skip.
    # If inputs changed but outputs exist, they are stale -- raise.
    output_paths = [raw_path, provenance_path, canonical_path, manifest_path, dd_path]
    outputs_exist = all(p.exists() for p in output_paths)

    if outputs_exist and not force:
        # Load previous contract and compare.
        contract_path = cfg.resolve("pipeline_manifest.json")
        if contract_path.exists():
            previous = _load_stage_contract(contract_path, "data_foundation")
            if previous is not None:
                check_inputs_unchanged(previous, input_hashes, cfg.config_hash)
                check_outputs_current(previous)
                # Inputs unchanged and outputs current -- nothing to do.
                return previous

    # Run the stage.
    raw, provenance, source_mode = _acquire_raw(cfg)
    _validate_raw(raw, cfg)
    df = _build_target(raw, cfg)
    df = _build_features(df, cfg)
    df = _build_regime(df, cfg)
    _publish_canonical(df, cfg, provenance, source_mode)

    # Build the actual contract with output hashes.
    # Deterministic artifacts (CSVs) get hash-checked; the manifest JSON
    # contains a timestamp so it is only existence-checked.
    output_hashes = {
        str(raw_path): file_hash(raw_path),
        str(provenance_path): file_hash(provenance_path),
        str(canonical_path): file_hash(canonical_path),
        str(dd_path): file_hash(dd_path),
    }
    output_records = [str(manifest_path)]

    contract = StageContract(
        stage_name="data_foundation",
        config_hash=cfg.config_hash,
        input_hashes=input_hashes,
        output_hashes=output_hashes,
        output_records=output_records,
        params=proposed.params,
        generated_at_utc=now_utc_iso(),
    )
    return contract
