from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from algotrading.lib.types.market_data import Timeframe
from algotrading.lib.types.trading import InstrumentRef
from algotrading.lib.eventing.md_events import (
    MDBarSubscribeRequest,
    MDBarUnsubscribeRequest,
    MDBarBatchSubscribeRequest,
    MDBarBatchUnsubscribeRequest,
    BaseBarUpserted,
    BarCompleted,
    BarUpdated,
    BarClosed,
    BarBatchCompleted,
)


@dataclass(frozen=True)
class BaseMarketDataAggregatorConfig:
    source_name: str  # "alpaca"
    base_timeframe: Timeframe  # e.g. 1m, 5s
    # Provider-specific configs, e.g. auth config, environment, etc.,
    # should be added in subclasses.


MDBarEvents = BarCompleted | BarUpdated | BarClosed | BarBatchCompleted


class BaseMarketDataAggregator(ABC):
    """
    Consumes base-bar upserts from market data adapters and produces derived-bar upserts.

    Key design contract:
    - Input bars are identified by (InstrumentRef, base_tf, start_ts) and are UPSERTED.
    - Implementations may maintain subscription state to decide which derived TFs to compute.
    - Implementations must be deterministic and idempotent under replay (same sequence of upserts -> same outputs).
    - Derived TFs should be integer multiples of base_tf (including base_tf itself, which simply passes through).
    """

    def __init__(self, config: BaseMarketDataAggregatorConfig) -> None:
        self._config = config

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source name (e.g. 'alpaca', 'polygon')."""

    @property
    @abstractmethod
    def base_timeframe(self) -> Timeframe:
        """The minimal granularity timeframe the adapter supplies to this aggregator (e.g. 1m, 5s)."""

    # ----- subscription management -----

    @abstractmethod
    def on_subscribe(self, msg: MDBarSubscribeRequest) -> None:
        """
        Register downstream interest in bars for a set of instruments/timeframes.

        In case the requested timeframe equals to base_timeframe, implementations should simply pass through the bars.
        """

    @abstractmethod
    def on_subscribe_batch(self, msg: MDBarBatchSubscribeRequest) -> None:
        """
        Register downstream interest in a batch of bars for cross-instrument sync.

        Implementation should call `on_subscribe()` internally for each instrument/timeframe pair in the batch if requested.
        """

    @abstractmethod
    def on_unsubscribe(self, msg: MDBarUnsubscribeRequest) -> None:
        """
        Unregister downstream interest in bars for a set of instruments/timeframes.

        Implementations should be robust to unsubscribing unknown refs/timeframes.
        """

    @abstractmethod
    def on_unsubscribe_batch(self, msg: MDBarBatchUnsubscribeRequest) -> None:
        """
        Unregister downstream interest in a batch of bars for cross-instrument sync.

        Implementation should call `on_unsubscribe()` internally for each instrument/timeframe pair in the batch if requested.
        """

    @abstractmethod
    def subscribed_timeframes(self, ref: InstrumentRef) -> tuple[Timeframe, ...]:
        """Return currently active derived timeframes for the given instrument (including base_timeframe if applicable)."""

    # ----- ingestion + derivation -----

    @abstractmethod
    def on_base_upsert(
        self, ev: BaseBarUpserted, *, now: Optional[datetime] = None
    ) -> Iterable[MDBarEvents]:
        """
        Ingest a base bar UPSERT and return zero or more derived bar closed/updated events.

        Requirements:
        - `ev.key.tf` MUST equal `base_timeframe`; implementations may raise or ignore otherwise.
        - Must treat input as an UPSERT (update-if-exists, insert-if-not).
        - Must propagate corrections deterministically to any affected derived bars.

        Returning an iterable allows callers to publish derived events.
        """
