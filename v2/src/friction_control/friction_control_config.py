from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FrictionControlConfig:
    """
    Config for the Friction Controls.
    """

    # ------------------------------------------------------------------
    # Hysteresis parameters
    # ------------------------------------------------------------------
    dw_min: float = (0.5 / 100)  # Minimum change in weight to trigger rebalancing
