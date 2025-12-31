from enum import Enum


class Topic(str, Enum):
    """Event topics for the event bus."""

    MARKET_CLOCK = "market_clock"
    BAR = "bar"
    REBALANCE_PLAN = "rebalance_plan"
    ACCOUNT = "account"
    ORDER = "order"
    FILL = "fill"
    LOG = "log"
    STOP = "stop"
