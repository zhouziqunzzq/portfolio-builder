from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from algotrading.lib.types.market_data import Timeframe
from algotrading.lib.eventing.md_events import BaseBarUpserted
from algotrading.lib.types.trading import InstrumentRef


@dataclass(frozen=True)
class BaseMarketDataAdapterConfig:
    name: str  # "alpaca"
    base_timeframe: Timeframe  # e.g. 1m, 5s
    # Provider-specific configs, e.g. auth config, environment, etc.,
    # should be added in subclasses.


class BaseMarketDataAdapter(ABC):
    """
    Provider-specific adapter.
    Responsibility: translate provider stream/poll into BarBaseUpserted(base_tf) events.
    """

    def __init__(self, config: BaseMarketDataAdapterConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def base_timeframe(self) -> Timeframe:
        return self.config.base_timeframe

    @abstractmethod
    async def subscribe_bars(self, refs: Iterable[InstrumentRef]) -> None:
        """Start streaming base bars for refs."""

    @abstractmethod
    async def unsubscribe_bars(self, refs: Iterable[InstrumentRef]) -> None:
        """Stop streaming base bars for refs."""

    # TODO: add methods for ticks, quotes, trades as needed

    @abstractmethod
    async def on_bars(self, bars: Any) -> Iterable[BaseBarUpserted]:
        """
        Provider-specific callback function to process incoming bar data.
        Should yield BarBaseUpserted events.
        """
