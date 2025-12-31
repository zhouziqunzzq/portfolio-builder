from dataclasses import dataclass
from typing import Optional


@dataclass
class EMLConfig:
    # Polling interval for EML background loop.
    polling_interval_secs: float = 30.0

    # If True, fetch positions alongside account snapshot.
    include_positions: bool = True

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
