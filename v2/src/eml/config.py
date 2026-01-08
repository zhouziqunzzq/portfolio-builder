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
    max_execution_history_days: int = (
        365  # Retain execution history for this many days; set to 0 or less to disable
    )

    # Pending rebalance retry cap
    # If a pending rebalance execution attempt fails this many times, it is moved
    # to `failed_rebalance_requests` in EML state for manual intervention.
    max_pending_rebalance_execution_retries: int = 10

    # Pending position cleanup retry cap
    # If a pending position cleanup execution attempt fails this many times, it is moved
    # to `failed_position_cleanup_requests` in EML state for manual intervention.
    max_pending_position_cleanup_execution_retries: int = 5

    # Safety guard for position cleanup execution: refuse to submit cleanup SELLs when
    # the absolute position quantity exceeds this threshold.
    #
    # Rationale: cleanup intents are intended for residual positions. This prevents an
    # upstream bug or stale state from accidentally liquidating a large holding.
    #
    # Set to None to disable this guard.
    position_cleanup_max_abs_qty: Optional[float] = 1.0

    def validate(self) -> None:
        """Validate configuration values."""
        if self.cash_buffer_pct is not None and self.cash_buffer_abs is not None:
            raise ValueError(
                "EMLConfig: cash_buffer_pct and cash_buffer_abs are mutually exclusive; only one may be set."
            )
        if self.cash_buffer_pct is not None:
            if not (0.0 <= self.cash_buffer_pct < 1.0):
                raise ValueError(
                    "EMLConfig: cash_buffer_pct must be in the range [0.0, 1.0)."
                )
        if self.min_order_size < 0.0:
            raise ValueError("EMLConfig: min_order_size must be non-negative.")
        if self.polling_interval_secs <= 0.0:
            raise ValueError("EMLConfig: polling_interval_secs must be positive.")
        if self.wait_for_order_fill_timeout_secs <= 0.0:
            raise ValueError(
                "EMLConfig: wait_for_order_fill_timeout_secs must be positive."
            )
        if self.max_pending_rebalance_execution_retries < 0:
            raise ValueError(
                "EMLConfig: max_pending_rebalance_execution_retries must be non-negative."
            )
        if self.max_pending_position_cleanup_execution_retries < 0:
            raise ValueError(
                "EMLConfig: max_pending_position_cleanup_execution_retries must be non-negative."
            )

        if self.position_cleanup_max_abs_qty is not None:
            try:
                max_qty = float(self.position_cleanup_max_abs_qty)
            except Exception as e:
                raise ValueError(
                    "EMLConfig: position_cleanup_max_abs_qty must be a float or None."
                ) from e
            if max_qty < 0:
                raise ValueError(
                    "EMLConfig: position_cleanup_max_abs_qty must be non-negative when set."
                )
