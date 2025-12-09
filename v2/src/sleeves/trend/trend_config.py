from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrendConfig:
    """
    Config for the stock-based Trend Sleeve (Momentum V2-style).

    Adds:
      - Liquidity filters (ADV, median volume, price)
      - Z-score winsorization
    """

    # ------------------------------------------------------------------
    # Signals / stock scoring
    # ------------------------------------------------------------------
    # CS-momentum (Cross-Sectional momentum)
    mom_windows: List[int] = field(default_factory=lambda: [63, 126, 252])
    mom_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    vol_window: int = 20
    vol_penalty: float = 0.5  # negative contribution from volatility
    cs_weight: float = 1.0  # overall scale for CS-mom contribution

    # NEW: TS-momentum (Time-Series momentum)
    use_ts_mom: bool = False
    ts_mom_windows: List[int] = field(default_factory=lambda: [63, 126, 252])
    ts_mom_weights: List[float] = field(default_factory=lambda: [0.3, 1.0, 0.7])
    use_ts_gate: bool = True
    ts_gate_threshold: float = -0.3  # min TS-mom to be "long"
    ts_weight: float = 0.7  # overall scale for TS-mom contribution

    # Z-score winsorization / clipping
    use_zscore_winsorization: bool = False
    zscore_clip: float = 3.0  # clip all z-scores to [-3, +3]

    # ------------------------------------------------------------------
    # Liquidity filters (new)
    # ------------------------------------------------------------------
    use_liquidity_filters: bool = False

    # Windows for liquidity statistics (ADV, median volume)
    liquidity_window: int = 20  # convenience primary window
    adv_window: int = 20  # used by ADV signal
    median_volume_window: int = 20  # used by median volume signal

    # Hard minimums required for eligibility
    min_adv20: float = 5_000_000.0  # USD
    min_median_volume20: int = 50_000  # shares
    min_price: float = 0.5  # dollars

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
    # Rebalancing (may be overridden by global scheduler)
    # ------------------------------------------------------------------
    rebalance_freq: str = "M"

    # ------------------------------------------------------------------
    # Regime-based gating for the Trend sleeve
    # ------------------------------------------------------------------
    use_regime_gating: bool = True
    gated_off_regimes: Tuple[str, ...] = ("crisis",)
    # gated_off_regimes: Tuple[str, ...] = ("crisis", "bear")
