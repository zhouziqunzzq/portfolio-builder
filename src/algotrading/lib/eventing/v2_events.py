from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from .topic import Topic
from .base import *


@dataclass(frozen=True)
class V2MarketClockEvent(BaseEvent):
    """V2 Market clock event indicating market open/close status."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.V2_MARKET_CLOCK, init=False)

    now: datetime
    is_market_open: bool
    # If market is NOT open, next_market_open will be set.
    next_market_open: Optional[datetime] = None
    # If market IS open, next_market_close will be set.
    next_market_close: Optional[datetime] = None


@dataclass(frozen=True)
class V2NewBarsEvent(BaseEvent):
    """V2 Event indicating new bar data is available."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.V2_BAR, init=False)

    # TODO: Add more fields as needed, e.g., bar data payload


@dataclass(frozen=True)
class V2BarsCheckedEvent(BaseEvent):
    """V2 Event indicating that new bars have been checked."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.V2_BAR, init=False)


@dataclass(frozen=True)
class V2RebalancePlanRequestEvent(BaseEvent):
    """V2 Event indicating a rebalance plan request."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.V2_REBALANCE_PLAN, init=False)

    rebalance_id: str
    weights: Dict[str, float]  # Mapping of tickers to target weights


@dataclass(frozen=True)
class V2RebalancePlanConfirmationEvent(BaseEvent):
    """V2 Event indicating a rebalance plan has been confirmed."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.V2_REBALANCE_PLAN, init=False)

    rebalance_id: str
    confirmed_ts: float


@dataclass(frozen=True)
class V2PositionCleanupIntent:
    """V2 Intent to clean up a specific position."""

    ticker: str
    reason: str  # e.g., "below_min_size", "delisted", etc.

    # Optional audit fields
    observed_qty: Optional[Decimal] = None
    qty_threshold: Optional[Decimal] = None
    observed_market_value: Optional[Decimal] = None
    market_value_threshold: Optional[Decimal] = None


@dataclass(frozen=True)
class V2PositionCleanupPlanRequestEvent(BaseEvent):
    """V2 Event indicating a position cleanup plan request."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.V2_POSITION_CLEANUP_PLAN, init=False)

    request_id: str
    intents: Dict[str, V2PositionCleanupIntent]  # Mapping of ticker to intent


@dataclass(frozen=True)
class V2PositionCleanupPlanConfirmationEvent(BaseEvent):
    """V2 Event indicating a position cleanup plan has been confirmed."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.V2_POSITION_CLEANUP_PLAN, init=False)

    request_id: str
    confirmed_ts: float
