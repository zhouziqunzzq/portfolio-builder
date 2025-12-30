from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from regime_engine import RegimeConfig
from sleeves.trend.trend_config import TrendConfig
from sleeves.defensive.defensive_config import DefensiveConfig
from sleeves.sideways_base.sideways_base_config import SidewaysBaseConfig
from allocator.multi_sleeve_config import MultiSleeveConfig
from friction_control.friction_control_config import FrictionControlConfig
from iml.config import IMLConfig
from at.config import ATConfig


@dataclass
class UniverseManagerConfig:
    membership_csv: str
    sectors_yaml: str


@dataclass
class MarketDataStoreConfig:
    data_root: str
    source: str = "yfinance"


@dataclass
class RuntimeConfig:
    log_level: str = "INFO"
    log_to_file: bool = True
    log_root: Optional[str] = None
    # Optional path to a JSON file used for runtime state persistence.
    # FileStateManager will require this to be set (non-None).
    state_file: Optional[str] = None
    state_persist_interval_secs: float = 30.0

    # Event loops config
    graceful_shutdown_timeout_secs: float = 5.0


@dataclass
class AppConfig:
    # Runtime
    runtime: RuntimeConfig

    # Infrastructures
    universe_manager: UniverseManagerConfig
    market_data_store: MarketDataStoreConfig
    regime_engine: RegimeConfig

    # Sleeves
    trend_sleeve: TrendConfig
    defensive_sleeve: DefensiveConfig
    sideways_base_sleeve: SidewaysBaseConfig

    # Multi-sleeve allocator
    multi_sleeve_allocator: MultiSleeveConfig

    # IML
    iml: IMLConfig

    # AT
    at: ATConfig

    @staticmethod
    def load_from_yaml(path: Path) -> AppConfig:
        raw = _load_yaml(path)

        multi_raw = dict(raw.get("multi_sleeve_allocator", {}) or {})
        friction_raw = multi_raw.get("friction_control_config")
        if isinstance(friction_raw, dict):
            multi_raw["friction_control_config"] = FrictionControlConfig(**friction_raw)
        elif friction_raw is None or isinstance(friction_raw, FrictionControlConfig):
            pass
        else:
            raise TypeError(
                "multi_sleeve_allocator.friction_control_config must be a mapping (dict)"
            )

        cfg = AppConfig(
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            universe_manager=UniverseManagerConfig(**raw.get("universe_manager", {})),
            market_data_store=MarketDataStoreConfig(**raw.get("market_data_store", {})),
            regime_engine=RegimeConfig(**raw.get("regime_engine", {})),
            trend_sleeve=TrendConfig(**raw.get("trend_sleeve", {})),
            defensive_sleeve=DefensiveConfig(**raw.get("defensive_sleeve", {})),
            sideways_base_sleeve=SidewaysBaseConfig(
                **raw.get("sideways_base_sleeve", {})
            ),
            multi_sleeve_allocator=MultiSleeveConfig(**multi_raw),
            iml=IMLConfig(**raw.get("iml", {})),
            at=ATConfig(**raw.get("at", {})),
        )

        return cfg


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
