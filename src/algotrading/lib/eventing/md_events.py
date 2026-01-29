from dataclasses import dataclass, field
from typing import Optional

from .topic import Topic
from .base import BaseEvent

from algotrading.lib.types.market_data import BarKey, BarBatchKey, OHLCVBar, Timeframe
from algotrading.lib.types.instruments import InstrumentRef


@dataclass(frozen=True, kw_only=True)
class BaseBarUpserted(BaseEvent):
    """Event indicating a base (raw) bar data upsert from market data provider."""

    topic: Topic = field(default=Topic.MD_BAR_BASE_UPSERT, init=False)

    key: BarKey
    curr: OHLCVBar
    prev: Optional[OHLCVBar]
    is_correction: bool
    source: str  # e.g. "alpaca", "polygon", etc


@dataclass(frozen=True, kw_only=True)
class BarCompleted(BaseEvent):
    """Event indicating a completed bar from market data aggregator."""

    topic: Topic = field(default=Topic.MD_BAR_COMPLETED, init=False)

    key: BarKey
    bar: OHLCVBar


@dataclass(frozen=True, kw_only=True)
class BarBatchCompleted(BaseEvent):
    """Event indicating a completed batch of bars for cross-instrument sync."""

    topic: Topic = field(default=Topic.MD_BAR_BATCH_COMPLETED, init=False)

    key: BarBatchKey
    # The actual bars are not included in this event because the event is just a sync signal.
    # Consumers are expected to have received the individual BarCompleted events for each instrument in the batch.


@dataclass(frozen=True, kw_only=True)
class BarUpdated(BaseEvent):
    """Event indicating an updated bar (e.g. after late ticks) from market data aggregator."""

    topic: Topic = field(default=Topic.MD_BAR_UPDATED, init=False)

    key: BarKey
    bar: OHLCVBar
    prev: Optional[OHLCVBar] = None


@dataclass(frozen=True, kw_only=True)
class BarClosed(BaseEvent):
    """Event indicating a closed bar from market data aggregator."""

    topic: Topic = field(default=Topic.MD_BAR_CLOSED, init=False)

    key: BarKey
    bar: OHLCVBar


@dataclass(frozen=True, kw_only=True)
class MDBarSubscribeRequest(BaseEvent):
    """Event requesting subscription to bar market data for given instruments and timeframes."""

    topic: Topic = field(default=Topic.MD_BAR_SUBSCRIBE, init=False)

    instrument_refs: tuple[InstrumentRef, ...]
    timeframes: tuple[Timeframe, ...]


@dataclass(frozen=True, kw_only=True)
class MDBarUnsubscribeRequest(BaseEvent):
    """Event requesting unsubscription from bar market data for given instruments and timeframes."""

    topic: Topic = field(default=Topic.MD_BAR_UNSUBSCRIBE, init=False)

    instrument_refs: tuple[InstrumentRef, ...]
    timeframes: tuple[Timeframe, ...]


@dataclass(frozen=True, kw_only=True)
class MDBarBatchSubscribeRequest(BaseEvent):
    """Event requesting subscription to a batch of bar market data (sync signal)."""

    topic: Topic = field(default=Topic.MD_BAR_BATCH_SUBSCRIBE, init=False)

    instrument_refs: tuple[InstrumentRef, ...]  # Batch of instruments to subscribe to
    timeframe: Timeframe  # Timeframe to subscribe to for all instruments in the batch
    auto_subscribe_constituents: bool = (
        True  # Whether to auto-subscribe to constituent instruments of the batch
    )


@dataclass(frozen=True, kw_only=True)
class MDBarBatchUnsubscribeRequest(BaseEvent):
    """Event requesting unsubscription from a batch of bar market data (sync signal)."""

    topic: Topic = field(default=Topic.MD_BAR_BATCH_UNSUBSCRIBE, init=False)

    instrument_refs: tuple[
        InstrumentRef, ...
    ]  # Batch of instruments to unsubscribe from
    timeframe: (
        Timeframe  # Timeframe to unsubscribe from for all instruments in the batch
    )
    auto_unsubscribe_constituents: bool = (
        False  # Whether to auto-unsubscribe from constituent instruments of the batch
    )
