"""Config loader -- reads pipeline_config.yaml into frozen dataclasses.

There is exactly one way to get configuration into a stage: load this file.
Every path declared relative to the repository root is resolved to an absolute
path via Config.resolve() before any stage sees it, so no stage ever contains
a hardcoded absolute path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The config file lives in the E3-S2_Data_Model_Integration_Flow/ folder.
# The repo root (where data/, E2-S1_*, etc. live) is one level up.
CONFIG_PATH = Path(__file__).resolve().parent.parent / "pipeline_config.yaml"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DataConfig:
    ticker: str
    start_date: str
    end_date: str | None
    auto_adjust: bool
    trading_days_per_year: int
    ohlcv_columns: list[str]


@dataclass(frozen=True)
class FeaturesConfig:
    return_windows: list[int]
    volatility_windows: list[int]
    trend_windows: list[int]
    volume_ma_window: int


@dataclass(frozen=True)
class TargetConfig:
    column: str
    horizon_trading_days: int


@dataclass(frozen=True)
class RegimeConfig:
    column: str
    volatility_column: str
    threshold_column: str
    min_historical_observations: int
    equality_rule: str


@dataclass(frozen=True)
class ModelConfig:
    seed: int
    horizon_trading_days: int
    min_train_size: int
    n_folds: int
    nearly_constant_std_threshold: float
    lightgbm_params: dict[str, Any]


@dataclass(frozen=True)
class PathsConfig:
    raw_csv: str
    raw_provenance: str
    canonical_csv: str
    manifest: str
    data_dictionary: str
    baseline_output_dir: str
    lightgbm_output_dir: str
    validation_output_dir: str
    canonical_oos_table: str
    canonical_oos_manifest: str
    pipeline_manifest: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    features: FeaturesConfig
    target: TargetConfig
    regime: RegimeConfig
    model: ModelConfig
    paths: PathsConfig
    _repo_root: Path = field(default=CONFIG_PATH.parent, repr=False)
    _config_hash: str = field(default="", repr=False)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a repo-root-relative path string to an absolute Path."""
        return (self._repo_root / relative_path).resolve()

    @property
    def config_hash(self) -> str:
        return self._config_hash

    # -- derived feature/target column names (must match E1-S6 manifest) --
    @property
    def feature_columns(self) -> list[str]:
        cols = ["return_1d"]
        cols += [f"return_{w}d" for w in self.features.return_windows]
        cols += [f"volatility_{w}d" for w in self.features.volatility_windows]
        cols += [f"trend_{w}d" for w in self.features.trend_windows]
        cols += [f"volume_ratio_{self.features.volume_ma_window}d"]
        return cols

    @property
    def daily_return_column(self) -> str:
        return "return_1d"


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load and validate the pipeline config from YAML.

    The raw YAML text is hashed and stored on the returned Config so every
    downstream manifest can record which config produced it.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    config_hash = _sha256_str(raw_text)
    cfg = yaml.safe_load(raw_text)

    # Validate top-level sections exist -- fail loudly on a malformed config.
    for section in ("data", "features", "target", "regime", "model", "paths"):
        if section not in cfg:
            raise ValueError(f"pipeline_config.yaml missing required section: {section}")

    data = DataConfig(
        ticker=str(cfg["data"]["ticker"]),
        start_date=str(cfg["data"]["start_date"]),
        end_date=cfg["data"]["end_date"] if cfg["data"]["end_date"] is not None else None,
        auto_adjust=bool(cfg["data"]["auto_adjust"]),
        trading_days_per_year=int(cfg["data"]["trading_days_per_year"]),
        ohlcv_columns=list(cfg["data"]["ohlcv_columns"]),
    )
    features = FeaturesConfig(
        return_windows=[int(w) for w in cfg["features"]["return_windows"]],
        volatility_windows=[int(w) for w in cfg["features"]["volatility_windows"]],
        trend_windows=[int(w) for w in cfg["features"]["trend_windows"]],
        volume_ma_window=int(cfg["features"]["volume_ma_window"]),
    )
    target = TargetConfig(
        column=str(cfg["target"]["column"]),
        horizon_trading_days=int(cfg["target"]["horizon_trading_days"]),
    )
    regime = RegimeConfig(
        column=str(cfg["regime"]["column"]),
        volatility_column=str(cfg["regime"]["volatility_column"]),
        threshold_column=str(cfg["regime"]["threshold_column"]),
        min_historical_observations=int(cfg["regime"]["min_historical_observations"]),
        equality_rule=str(cfg["regime"]["equality_rule"]),
    )

    # model section has a nested dict; pass it through.
    model = ModelConfig(
        seed=int(cfg["model"]["seed"]),
        horizon_trading_days=int(cfg["model"]["horizon_trading_days"]),
        min_train_size=int(cfg["model"]["min_train_size"]),
        n_folds=int(cfg["model"]["n_folds"]),
        nearly_constant_std_threshold=float(cfg["model"]["nearly_constant_std_threshold"]),
        lightgbm_params=dict(cfg["model"]["lightgbm_params"]),
    )

    paths = PathsConfig(**cfg["paths"])

    # Validate regime equality_rule.
    if regime.equality_rule not in ("LowVol", "HighVol"):
        raise ValueError(
            f"regime.equality_rule must be 'LowVol' or 'HighVol', got {regime.equality_rule!r}"
        )

    # Validate target horizon matches model horizon (they are the same concept
    # and must be identical for the purge gap to be correct).
    if target.horizon_trading_days != cfg["model"].get("horizon_trading_days", target.horizon_trading_days):
        raise ValueError(
            f"target.horizon_trading_days ({target.horizon_trading_days}) must match "
            "model.horizon_trading_days in pipeline_config.yaml"
        )

    return Config(
        data=data,
        features=features,
        target=target,
        regime=regime,
        model=model,
        paths=paths,
        _repo_root=REPO_ROOT,
        _config_hash=config_hash,
    )


def feature_columns_from_config(cfg: Config) -> list[str]:
    """Return the canonical 11-feature column list derived from config."""
    return cfg.feature_columns
