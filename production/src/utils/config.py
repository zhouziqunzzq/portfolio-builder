from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class StrategyConfig:
    name: str
    rebalance: str
    rebalance_day: str
    warmup_days: int


@dataclass
class UniverseConfig:
    name: str
    membership_csv: str
    filter_unknown_sectors: bool = True


@dataclass
class MarketDataConfig:
    root: str
    source: str = "yfinance"
    session: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalsConfig:
    momentum_windows: List[int]
    momentum_weights: List[float]
    vol_window: int
    vol_penalty: float


@dataclass
class SectorConfig:
    weights: Dict[str, float]
    smoothing_alpha: float
    smoothing_beta: float = 0.3
    risk_on_equity_frac: float = 1.0
    risk_off_equity_frac: float = 0.7
    trend_filter: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StocksConfig:
    top_k_per_sector: int
    weighting: str


@dataclass
class CostsConfig:
    turnover_bps_per_100pct: float


@dataclass
class RuntimeConfig:
    output_root: str
    save: Dict[str, bool]
    plotting: bool
    log_level: str = "INFO"


@dataclass
class AppConfig:
    strategy: StrategyConfig
    universe: UniverseConfig
    market_data: MarketDataConfig
    signals: SignalsConfig
    sectors: SectorConfig
    stocks: StocksConfig
    costs: CostsConfig
    runtime: RuntimeConfig

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def output_root_path(self) -> Path:
        return (self.base_dir / self.runtime.output_root).resolve()

    @property
    def membership_csv_path(self) -> Path:
        return (self.base_dir / self.universe.membership_csv).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_app_config(app_config_yaml: Path) -> AppConfig:
    raw = load_yaml(app_config_yaml)

    cfg = AppConfig(
        strategy=StrategyConfig(**raw["strategy"]),
        universe=UniverseConfig(**raw["universe"]),
        market_data=MarketDataConfig(**raw["market_data"]),
        signals=SignalsConfig(**raw["signals"]),
        sectors=SectorConfig(**raw["sectors"]),
        stocks=StocksConfig(**raw["stocks"]),
        costs=CostsConfig(**raw["costs"]),
        runtime=RuntimeConfig(**raw["runtime"]),
    )

    # Validate some basics
    assert cfg.strategy.rebalance in {"monthly"}
    assert cfg.stocks.weighting in {"equal-weight", "inverse-vol"}

    return cfg
