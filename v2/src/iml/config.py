from dataclasses import dataclass


@dataclass
class IMLConfig:
    # Polling interval for event loop
    polling_interval_secs: float = 30.0

    # Historical data (bars) fetch configuration
    # Whether bar polling is enabled
    bar_polling_enabled: bool = True
    # Bar interval
    bar_interval: str = "1d"
    # Interval to check for new bars
    bar_polling_interval_secs: float = 60 * 60 * 2.0  # Every 2 hours
    # Lookback weeks for fetching historical bars
    bar_fetch_lookback_weeks: int = 52 * 5  # 5 years

    # Universe refresh configuration
    # Whether universe polling is enabled
    universe_polling_enabled: bool = True
    # Interval to check for universe updates
    universe_polling_interval_secs: float = 60 * 60 * 24.0  # Every 24 hours

    def validate(self) -> None:
        if self.polling_interval_secs <= 0:
            raise ValueError("polling_interval_secs must be positive")
        if self.bar_polling_enabled:
            if self.bar_polling_interval_secs <= 0:
                raise ValueError("bar_polling_interval_secs must be positive")
            if self.bar_fetch_lookback_weeks < 0:
                raise ValueError("bar_fetch_lookback_weeks must be non-negative")
        if self.universe_polling_enabled:
            if self.universe_polling_interval_secs <= 0:
                raise ValueError("universe_polling_interval_secs must be positive")
