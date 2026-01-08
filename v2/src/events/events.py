from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
from .topic import Topic

from pathlib import Path
import sys

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from models import BrokerAccount, BrokerPosition


@dataclass(frozen=True)
class BaseEvent:
    """Base class for events on the event bus."""

    topic: Topic
    ts: float
    source: str = field(default="", kw_only=True)
    correlation_id: str = field(default="", kw_only=True)


@dataclass(frozen=True)
class MarketClockEvent(BaseEvent):
    """Market clock event indicating market open/close status."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.MARKET_CLOCK, init=False)

    now: datetime
    is_market_open: bool
    # If market is NOT open, next_market_open will be set.
    next_market_open: Optional[datetime] = None
    # If market IS open, next_market_close will be set.
    next_market_close: Optional[datetime] = None


@dataclass(frozen=True)
class NewBarsEvent(BaseEvent):
    """Event indicating new bar data is available."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.BAR, init=False)

    # TODO: Add more fields as needed, e.g., bar data payload


@dataclass(frozen=True)
class BarsCheckedEvent(BaseEvent):
    """Event indicating that new bars have been checked."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.BAR, init=False)


@dataclass(frozen=True)
class RebalancePlanRequestEvent(BaseEvent):
    """Event indicating a rebalance plan request."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.REBALANCE_PLAN, init=False)

    rebalance_id: str
    weights: Dict[str, float]  # Mapping of tickers to target weights


@dataclass(frozen=True)
class RebalancePlanConfirmationEvent(BaseEvent):
    """Event indicating a rebalance plan has been confirmed."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.REBALANCE_PLAN, init=False)

    rebalance_id: str
    confirmed_ts: float


@dataclass(frozen=True)
class AccountSnapshotEvent(BaseEvent):
    """Event containing broker account + positions snapshot.

    Intended to be published periodically by EML services.
    """

    topic: Topic = field(default=Topic.ACCOUNT, init=False)

    account: BrokerAccount
    positions: List[BrokerPosition] = field(default_factory=list)


@dataclass(frozen=True)
class PositionCleanupIntent:
    """Intent to clean up a specific position."""

    ticker: str
    reason: str  # e.g., "below_min_size", "delisted", etc.

    # Optional audit fields
    observed_qty: Optional[Decimal] = None
    qty_threshold: Optional[Decimal] = None
    observed_market_value: Optional[Decimal] = None
    market_value_threshold: Optional[Decimal] = None


@dataclass(frozen=True)
class PositionCleanupPlanRequestEvent(BaseEvent):
    """Event indicating a position cleanup plan request."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.POSITION_CLEANUP_PLAN, init=False)

    request_id: str
    intents: Dict[str, PositionCleanupIntent]  # Mapping of ticker to intent


@dataclass(frozen=True)
class PositionCleanupPlanConfirmationEvent(BaseEvent):
    """Event indicating a position cleanup plan has been confirmed."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.POSITION_CLEANUP_PLAN, init=False)

    request_id: str
    confirmed_ts: float
