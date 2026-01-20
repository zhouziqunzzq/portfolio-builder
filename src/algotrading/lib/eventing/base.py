from dataclasses import dataclass, field

from .topic import Topic


@dataclass(frozen=True)
class BaseEvent:
    """Base class for events on the event bus."""

    topic: Topic
    ts: float
    source: str = field(default="", kw_only=True)
    correlation_id: str = field(default="", kw_only=True)
