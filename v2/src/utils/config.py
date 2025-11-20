from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
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
    current_constituents_filename: str = "current_constituents.csv"
    membership_raw_filename: str | None = None
    membership_enriched_filename: str | None = None


@dataclass
class MarketDataConfig:
    root: str
    source: str = "yfinance"
    session: Dict[str, Any] = field(default_factory=dict)


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
        strategy=StrategyConfig(**raw.get("strategy", {})),
        universe=UniverseConfig(**raw.get("universe", {})),
        market_data=MarketDataConfig(**raw.get("market_data", {})),
        costs=CostsConfig(**raw.get("costs", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )

    # Validate some basics
    assert cfg.strategy.rebalance in {"monthly"}

    return cfg
