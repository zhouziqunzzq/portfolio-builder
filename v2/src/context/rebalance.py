from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RebalanceContext:
    """
    Context information for a rebalance operation.
    """
    # rebalance timestamp for which this context applies (e.g. for which a sleeve generates weights)
    rebalance_ts: datetime | str

    # Regime context
    primary_regime: str = ""  # e.g. "bull", "bear", etc.
    # Regime scores at the time of rebalance
    regime_scores: dict[str, float] = field(default_factory=dict)
