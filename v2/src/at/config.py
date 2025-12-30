from dataclasses import dataclass


@dataclass
class ATConfig:
    # Polling interval for event loop
    polling_interval_secs: float = 30.0
