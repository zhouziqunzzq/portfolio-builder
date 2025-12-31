from dataclasses import dataclass
from typing import Optional


@dataclass
class EMLConfig:
    # Polling interval for EML background loop.
    polling_interval_secs: float = 30.0

    # If True, fetch positions alongside account snapshot.
    include_positions: bool = True

    # Safety: cancel any outstanding/open orders on startup.
    cancel_open_orders_on_startup: bool = True

    # Safety: cancel any outstanding/open orders on shutdown (best-effort).
    # Default False to avoid surprising behavior; enable explicitly in app config.
    cancel_open_orders_on_shutdown: bool = False

    # Timeout for waiting for order fills (in seconds).
    wait_for_order_fill_timeout_secs: float = 30.0

    # Rebalance execution knobs
    min_order_size: float = 1.0  # Minimum order size to place in USD
    cash_buffer_pct: Optional[float] = (
        0.01  # Keep this % of account value in cash; mutually exclusive with cash_buffer_abs
    )
    cash_buffer_abs: Optional[float] = (
        None  # Or keep this absolute amount in cash; mutually exclusive with cash_buffer_pct
    )

    # Execution history retention knobs
    max_execution_history_days: int = 365  # Retain execution history for this many days; set to 0 or less to disable
