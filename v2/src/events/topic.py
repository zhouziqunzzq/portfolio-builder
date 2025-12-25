from enum import Enum


class Topic(str, Enum):
    """Event topics for the event bus."""

    MARKET_CLOCK = "market_clock"
    BAR = "bar"
    NEW_BARS = "new_bars"
    REBALANCE_DUE = "rebalance_due"
    PLAN = "plan"
    ORDER = "order"
    FILL = "fill"
    LOG = "log"
    STOP = "stop"
