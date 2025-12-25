from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
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
