from dataclasses import dataclass


@dataclass
class ATConfig:
    # Polling interval for event loop
    polling_interval_secs: float = 30.0
    # Lookback weeks for precomputing signals / scores
    precompute_lookback_weeks: int = 52  # 1 year
    # Separate lookback window for bootstrapping sleeve state on first deployment.
    # This is used to forward-simulate historical rebalances to warm sleeve internals
    # (e.g., trend sector-weight smoothing) without emitting any orders.
    bootstrap_lookback_weeks: int = 52  # 1 year

    # Position cleanup settings
    # Enable or disable residual position cleanup
    position_cleanup_enabled: bool = False
    # Interval in days to run position cleanup
    position_cleanup_interval_days: int = 1
    # Threshold for market value below which positions are considered residual
    position_cleanup_market_value_threshold: float = 0.10
    # Threshold for quantity below which positions are considered residual
    position_cleanup_qty_threshold: float = 0.001

    def validate(self) -> None:
        """Validate configuration values."""
        if self.polling_interval_secs <= 0:
            raise ValueError("polling_interval_secs must be positive")

        if self.precompute_lookback_weeks <= 0:
            raise ValueError("precompute_lookback_weeks must be positive")

        if self.bootstrap_lookback_weeks < 0:
            raise ValueError("bootstrap_lookback_weeks cannot be negative")

        if self.position_cleanup_enabled:
            if self.position_cleanup_interval_days <= 0:
                raise ValueError("position_cleanup_interval_days must be positive")

            if self.position_cleanup_market_value_threshold < 0:
                raise ValueError(
                    "position_cleanup_market_value_threshold cannot be negative"
                )

            if self.position_cleanup_qty_threshold < 0:
                raise ValueError("position_cleanup_qty_threshold cannot be negative")
