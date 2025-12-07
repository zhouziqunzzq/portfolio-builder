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
    mom_windows: List[int] = field(default_factory=lambda: [63, 126, 252])
    mom_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    vol_window: int = 20
    vol_penalty: float = 0.5  # negative contribution from volatility

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

    # Trend filter on benchmark for risk-on/risk-off scaling
    # Note: this is currently NOT used; Imported from V1.5 for completeness.
    # trend_filter_enabled: bool = False
    # trend_benchmark: str = "SPY"
    # trend_window: int = 200
    # risk_on_equity_frac: float = 1.0
    # risk_off_equity_frac: float = 0.7

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
