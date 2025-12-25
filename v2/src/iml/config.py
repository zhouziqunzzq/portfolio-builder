from dataclasses import dataclass


@dataclass
class IMLConfig:
    # Polling interval for event loop
    polling_interval_secs: float = 30.0
    # Bar interval
    bar_interval: str = "1d"
    # Interval to check for new bars
    bar_polling_interval_secs: float = 60 * 60 * 2.0  # Every 2 hours
    # Lookback weeks for fetching historical bars
    bar_fetch_lookback_weeks: int = 52 * 5  # 5 years
