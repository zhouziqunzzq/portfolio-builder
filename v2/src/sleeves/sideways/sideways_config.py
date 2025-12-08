from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SidewaysConfig:
    """
    Config for the stock-based Sideways Sleeve.

    Adds:
      - Liquidity filters (ADV, median volume, price)
      - Z-score winsorization
    """

    # ------------------------------------------------------------------
    # Signals / stock scoring
    # ------------------------------------------------------------------
    mom_windows: List[int] = field(default_factory=lambda: [63, 126, 252])
    mom_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    vol_window: int = 20
    vol_penalty: float = 0.5

    # ------------------------------------------------------------------
    # Liquidity filters
    # ------------------------------------------------------------------
    use_liquidity_filters: bool = False

    # Windows for liquidity statistics (ADV, median volume)
    liquidity_window: int = 20  # convenience primary window
    adv_window: int = 20  # used by ADV signal
    median_volume_window: int = 20  # used by median volume signal

    # Hard minimums required for eligibility
    min_adv20: float = 10_000_000.0  # dollars
    min_median_volume20: int = 200_000  # shares
    min_price: float = 5.0  # dollars

    # Z-score winsorization / clipping
    use_zscore_winsorization: bool = False
    zscore_clip: float = 3.0  # clip all z-scores to [-3, +3]

    # ------------------------------------------------------------------
    # Sector weighting
    # ------------------------------------------------------------------
    sector_softmax_alpha: float = 1.0
    sector_w_min: float = 0.0
    sector_w_max: float = 0.30
    sector_smoothing_beta: float = 0.3
    sector_top_k: int = 5

    # ------------------------------------------------------------------
    # Stock selection within each sector
    # ------------------------------------------------------------------
    top_k_per_sector: int = 2
    weighting_mode: str = "equal"  # "equal" | "inverse-vol"

    # ------------------------------------------------------------------
    # Regime-based gating for the Trend sleeve
    # ------------------------------------------------------------------
    use_regime_gating: bool = False
    # By default, turn the sleeve OFF in strongly trending/crisis regimes.
    gated_off_regimes: List[str] = field(
        default_factory=lambda: ["bull", "bear", "crisis"]
    )
