from dataclasses import dataclass, field

from .topic import Topic
from .base import BaseEvent


@dataclass(frozen=True, kw_only=True)
class StopEvent(BaseEvent):
    """Event indicating a system stop signal."""

    # Fixed topic for this event type (not part of __init__).
    topic: Topic = field(default=Topic.SYSTEM_STOP, init=False)
