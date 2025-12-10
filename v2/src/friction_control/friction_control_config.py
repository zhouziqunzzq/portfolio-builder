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
    hysteresis_dw_min: float = 0.5 / 100  # Minimum change in weight to trigger rebalancing

    # ------------------------------------------------------------------
    # Minimum Trade Notional parameters
    # ------------------------------------------------------------------
    min_trade_notional_abs: float = 25  # Minimum trade size in dollars
    min_trade_pct_of_aum: float = 0.3 / 100  # Minimum trade size as a fraction of AUM

    # ------------------------------------------------------------------
    # Minimum holding period parameters
    # Note: This is in number of rebalance steps, not days; Has to be
    #   greater than 1 to have any effect.
    # ------------------------------------------------------------------
    min_holding_rebalances: int = 0


    # hysteresis_dw_min: float = 0.0  # Minimum change in weight to trigger rebalancing
    # min_trade_notional_abs: float = 0.0  # Minimum trade size in dollars
    # min_trade_pct_of_aum: float = 0.0  # Minimum trade size as a fraction of AUM
    # min_holding_rebalances: int = 0
