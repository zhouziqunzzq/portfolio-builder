from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from .topic import Topic


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
    topic: Topic = field(default=Topic.NEW_BARS, init=False)

    # TODO: Add more fields as needed, e.g., bar data payload


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
class BrokerAccount:
    """Normalized broker account snapshot.

    This is intentionally a small, stable set of fields needed by the app.
    Execution adapters (EML) should map broker-specific payloads into this model.
    """

    id: Optional[str] = None
    status: Optional[str] = None
    cash: Optional[float] = None
    buying_power: Optional[float] = None
    portfolio_value: Optional[float] = None
    equity: Optional[float] = None
    last_equity: Optional[float] = None
    adj_equity: Optional[float] = None  # Adjusted equity after cash buffers, if any


@dataclass(frozen=True)
class BrokerPosition:
    """Normalized broker position snapshot."""

    symbol: str
    qty: Optional[float] = None
    market_value: Optional[float] = None
    avg_entry_price: Optional[float] = None
    side: Optional[str] = None
    unrealized_pl: Optional[float] = None


@dataclass(frozen=True)
class AccountSnapshotEvent(BaseEvent):
    """Event containing broker account + positions snapshot.

    Intended to be published periodically by EML services.
    """

    topic: Topic = field(default=Topic.ACCOUNT, init=False)

    account: BrokerAccount
    positions: List[BrokerPosition] = field(default_factory=list)
