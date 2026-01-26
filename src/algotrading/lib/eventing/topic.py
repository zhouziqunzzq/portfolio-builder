from enum import Enum


class Topic(str, Enum):
    """Event topics for the event bus."""

    # System-level topics
    SYSTEM_STOP = "system_stop"
    SYSTEM_LOG = "system_log"

    # Market data topics
    # Note: "upsert" simply means "insert or update"
    MD_BAR_BASE_UPSERT = "md_bar_base_upsert" # Base (raw) bar data upsert from md provider
    MD_BAR_COMPLETED = "md_bar_completed"  # Signal that all bars for a given timeframe have been ingested
    MD_BAR_UPDATED = "md_bar_updated"  # Updated bar data (e.g. after late ticks) from md aggregator
    MD_BAR_CLOSED = "md_bar_closed"  # Closed bar data from md aggregator
    MD_BARS_BATCH_CLOSED = "md_bars_batch_closed"  # Cross-instrument sync signal for closed bars batch
    MD_BAR_SUBSCRIBE = "md_bar_subscribe"  # Request to subscribe to bar market data
    MD_BAR_UNSUBSCRIBE = "md_bar_unsubscribe"  # Request to unsubscribe from bar market data
    # TODO: Add tick topics

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
