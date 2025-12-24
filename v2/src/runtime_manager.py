from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import sys

# Make v2/src importable by adding it to sys.path. This mirrors the pattern
# used throughout v2 runners and allows direct imports.
_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from configs import AppConfig
from universe_manager import UniverseManager
from market_data_store import MarketDataStore
from signal_engine import SignalEngine
from vec_signal_engine import VectorizedSignalEngine
from regime_engine import RegimeEngine
from sleeves.trend.trend_sleeve import TrendSleeve
from sleeves.defensive.defensive_sleeve import DefensiveSleeve
from sleeves.sideways_base.sideways_base_sleeve import SidewaysBaseSleeve
from allocator.multi_sleeve_allocator import MultiSleeveAllocator


@dataclass(frozen=True)
class RuntimeManagerOptions:
    """Construction-time options for RuntimeManager singletons."""

    # For MarketDataStore and UniverseManager
    # Default to local-only mode for performance
    local_only: bool = True
    # For MarketDataStore
    use_memory_cache: bool = True

    # SignalEngine cache behavior knobs (kept here because backtests sometimes
    # disable margin for very-frequent sampling schedules).
    # Disable margin by default to ensure correctness in live
    disable_signal_cache_margin: bool = True
    disable_signal_cache_extension: bool = True


class RuntimeManager:
    """Factory + registry for commonly used V2 runtime singletons.

    This centralizes runtime wiring (UniverseManager, MarketDataStore, SignalEngine,
    sleeves, allocator, etc.) so backtest and live runners can share one consistent
    construction path.

    The construction mirrors `build_runtime()` in `v2/src/backtest_runner.py`.

    Notes
    -----
    - Instances are constructed eagerly in `from_app_config()`.
    - Use `get(name)` or `rm[name]` to retrieve a singleton.
    """

    def __init__(
        self,
        *,
        app_config: AppConfig,
        options: Optional[RuntimeManagerOptions] = None,
        objects: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._app_config = app_config
        self._options = options or RuntimeManagerOptions()
        self._objects: Dict[str, object] = dict(objects or {})

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        app_config_path: Path,
        *,
        options: Optional[RuntimeManagerOptions] = None,
    ) -> "RuntimeManager":
        app_cfg = AppConfig.load_from_yaml(app_config_path)
        return cls.from_app_config(app_cfg, options=options)

    @classmethod
    def from_app_config(
        cls,
        app_cfg: AppConfig,
        *,
        options: Optional[RuntimeManagerOptions] = None,
    ) -> "RuntimeManager":
        opt = options or RuntimeManagerOptions()

        # Universe
        um = UniverseManager(
            membership_csv=Path(app_cfg.universe_manager.membership_csv),
            sectors_yaml=Path(app_cfg.universe_manager.sectors_yaml),
            local_only=bool(opt.local_only),
        )

        # Market data store
        mds = MarketDataStore(
            data_root=str(app_cfg.market_data_store.data_root),
            source=str(app_cfg.market_data_store.source),
            local_only=bool(opt.local_only),
            use_memory_cache=bool(opt.use_memory_cache),
        )

        # Signals
        signals = SignalEngine(
            mds,
            disable_cache_margin=bool(opt.disable_signal_cache_margin),
            disable_cache_extension=bool(opt.disable_signal_cache_extension),
        )
        vec_engine = VectorizedSignalEngine(um, mds)

        # Regime engine
        regime_engine = RegimeEngine(
            signals=signals,
            config=app_cfg.regime_engine,
        )

        # Sleeves
        trend = TrendSleeve(
            universe=um,
            mds=mds,
            signals=signals,
            vec_engine=vec_engine,
            config=app_cfg.trend_sleeve,
        )
        defensive = DefensiveSleeve(
            universe=um,
            mds=mds,
            signals=signals,
            config=app_cfg.defensive_sleeve,
        )
        sideways_base = SidewaysBaseSleeve(
            mds=mds,
            signals=signals,
            config=app_cfg.sideways_base_sleeve,
        )

        allocator = MultiSleeveAllocator(
            regime_engine=regime_engine,
            sleeves={
                "defensive": defensive,
                "trend": trend,
                "sideways_base": sideways_base,
            },
            config=app_cfg.multi_sleeve_allocator,
        )

        objects: Dict[str, object] = {
            "app_config": app_cfg,
            "options": opt,
            "um": um,
            "universe_manager": um,
            "mds": mds,
            "market_data_store": mds,
            "signals": signals,
            "signal_engine": signals,
            "vec_engine": vec_engine,
            "vectorized_signal_engine": vec_engine,
            "regime_engine": regime_engine,
            "trend": trend,
            "trend_sleeve": trend,
            "defensive": defensive,
            "defensive_sleeve": defensive,
            "sideways_base": sideways_base,
            "sideways_base_sleeve": sideways_base,
            "allocator": allocator,
            "multi_sleeve_allocator": allocator,
        }

        return cls(app_config=app_cfg, options=opt, objects=objects)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    @property
    def options(self) -> RuntimeManagerOptions:
        return self._options

    def get(self, name: str) -> object:
        """Return a singleton by name.

        Raises
        ------
        KeyError
                If `name` is not registered.
        """

        key = str(name)
        if key in self._objects:
            return self._objects[key]

        available = ", ".join(sorted(self._objects.keys()))
        raise KeyError(f"Unknown runtime singleton '{key}'. Available: {available}")

    def __getitem__(self, name: str) -> object:
        """Shorthand for `get(name)`."""

        return self.get(name)

    def keys(self) -> set[str]:
        return set(self._objects.keys())
