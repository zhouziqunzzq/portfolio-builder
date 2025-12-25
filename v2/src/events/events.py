from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from .topic import Topic


@dataclass(frozen=True)
class BaseEvent:
    """Base class for events on the event bus."""

    topic: Topic
    ts: float
    source: str = ""
    correlation_id: str = ""


@dataclass(frozen=True)
class MarketClockEvent(BaseEvent):
    """Market clock event indicating market open/close status."""

    topic: Topic = Topic.MARKET_CLOCK
    now: datetime
    is_market_open: bool
    # If market is NOT open, next_market_open will be set.
    next_market_open: Optional[datetime] = None
    # If market IS open, next_market_close will be set.
    next_market_close: Optional[datetime] = None
