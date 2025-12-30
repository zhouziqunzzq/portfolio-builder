from dataclasses import dataclass


@dataclass
class EMLConfig:
    # Polling interval for EML background loop.
    polling_interval_secs: float = 30.0

    # If True, fetch positions alongside account snapshot.
    include_positions: bool = True
