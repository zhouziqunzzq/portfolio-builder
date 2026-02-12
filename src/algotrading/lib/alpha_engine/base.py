from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Type

from algotrading.lib.alpha.base import BaseAlpha, BaseAlphaConfig, BaseAlphaOutput
from algotrading.lib.alpha.base import MarketDataEvent
from algotrading.lib.types.instruments import InstrumentRef
from algotrading.lib.types.market_data import Timeframe


@dataclass(frozen=True)
class AlphaKey:
    """Stable key for locating a specific alpha instance."""

    ref: InstrumentRef
    tf: Timeframe
    alpha_type: Type[BaseAlpha]


class BaseAlphaEngine(ABC):
    """Interface for event-driven alpha routing and lookup."""

    @abstractmethod
    def subscribe(
        self,
        ref: InstrumentRef,
        tf: Timeframe,
        alpha_type: Type[BaseAlpha],
        config: BaseAlphaConfig,
    ) -> BaseAlpha:
        """Register or return an alpha instance for the given key."""

    @abstractmethod
    def update(self, event: MarketDataEvent) -> Dict[AlphaKey, BaseAlphaOutput]:
        """Route a market data event to matching alphas and return outputs."""

    @abstractmethod
    def ready(self, key: AlphaKey) -> bool:
        """Return whether the specified alpha is ready for consumption."""

    @abstractmethod
    def get(self, key: AlphaKey) -> Optional[BaseAlphaOutput]:
        """Return the most recent output for a specific alpha."""

    @abstractmethod
    def keys(self) -> Iterable[AlphaKey]:
        """Return all registered alpha keys."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all registered alpha instances."""
