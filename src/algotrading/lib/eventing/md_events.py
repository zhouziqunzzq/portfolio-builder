from dataclasses import dataclass, field
from typing import Optional

from .topic import Topic
from .base import BaseEvent

from algotrading.lib.types.market_data import BarKey, OHLCVBar, Timeframe
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
