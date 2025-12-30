from dataclasses import dataclass


@dataclass
class ATConfig:
    # Polling interval for event loop
    polling_interval_secs: float = 30.0
    # Lookback weeks for precomputing signals / scores
    precompute_lookback_weeks: int = 52  # 1 year
