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
