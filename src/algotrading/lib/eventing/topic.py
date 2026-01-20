from enum import Enum


class Topic(str, Enum):
    """Event topics for the event bus."""

    # System-level topics
    SYSTEM_STOP = "system_stop"
    SYSTEM_LOG = "system_log"

    # Market data topics
    # TODO: Add topics

    # Execution-related topics
    EXEC_ACCOUNT_SNAPSHOT = "exec_account_snapshot"
    EXEC_ORDER = "exec_order"
    EXEC_FILL = "exec_fill"

    # V2-specific topics (keep for backwards compatibility)
    V2_MARKET_CLOCK = "v2_market_clock"  # legacy market clock topic
    V2_BAR = "v2_bar"  # legacy bar topic
    V2_REBALANCE_PLAN = "v2_rebalance_plan"
    V2_POSITION_CLEANUP_PLAN = "v2_position_cleanup_plan"

    # V3-specific topics
    # TODO: Add topics
