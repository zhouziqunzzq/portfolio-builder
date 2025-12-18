from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RebalanceContext:
    """
    Context information for a rebalance operation.
    """
    # rebalance timestamp for which this context applies (e.g. for which a sleeve generates weights)
    rebalance_ts: datetime | str
