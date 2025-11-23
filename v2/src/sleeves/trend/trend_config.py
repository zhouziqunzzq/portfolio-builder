from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrendConfig:
    """
    Config for the stock-based Trend Sleeve (Momentum V2-style).

    This is intentionally very close to the Momentum V1.5 production defaults:

    - Universe: S&P 500 (handled by UniverseManager)
    - Signals:
        * multi-horizon momentum (63, 126, 252)
        * realized volatility (20d)
        * stock_score = momentum_score - vol_penalty * vol_score
    - Sectors:
        * sector scores = mean(stock_score) per sector
        * sector weights via softmax + caps/floors + smoothing + top-k
        * trend filter on SPY (200d SMA) for risk-on / risk-off equity fraction
    - Stocks:
        * top_k_per_sector = 2
        * equal-weight or inverse-vol within each sector
    """

    # ------------------------------------------------------------------
    # Signals / stock scoring (from V1.5 "signals" section)
    # ------------------------------------------------------------------
    mom_windows: List[int] = field(default_factory=lambda: [63, 126, 252])
    mom_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    vol_window: int = 20

    # In V1.5: stock_score â momentum_score - vol_penalty * vol
    # Weâll implement this as:
    #   stock_score = momentum_score + (-vol_penalty) * vol_zscore
    vol_penalty: float = 0.5

    # ------------------------------------------------------------------
    # Sector weighting (from V1.5 "sectors" section / SectorWeightEngine)
    # ------------------------------------------------------------------
    # Softmax sharpness for sector scores
    sector_softmax_alpha: float = 1.0  # sectors.smoothing_alpha

    # Per-sector min/max weights BEFORE top-k truncation
    sector_w_min: float = 0.0  # sectors.weights.w_min
    sector_w_max: float = 0.30  # sectors.weights.w_max

    # Smoothing factor for sector weights (0 = stick, 1 = jump)
    sector_smoothing_beta: float = 0.3  # sectors.smoothing_beta

    # Keep only top-K sectors at each rebalance (others zeroed, then renormalized)
    sector_top_k: int = 5  # sectors.top_k_sectors

    # Trend filter on benchmark (risk-on / risk-off scaling)
    trend_filter_enabled: bool = True
    trend_benchmark: str = "SPY"  # sectors.trend_filter.benchmark
    trend_window: int = 200  # sectors.trend_filter.window

    # Equity fraction in risk-on vs risk-off (remaining becomes implicit cash)
    risk_on_equity_frac: float = 1.0  # sectors.risk_on_equity_frac
    risk_off_equity_frac: float = 0.7  # sectors.risk_off_equity_frac

    # ------------------------------------------------------------------
    # Stock selection within each sector (from V1.5 "stocks" section)
    # ------------------------------------------------------------------
    # Number of stocks to pick per sector at each rebalance
    top_k_per_sector: int = 2  # stocks.top_k_per_sector

    # How to allocate weight within each sector (equal-weight or inverse-vol)
    weighting_mode: str = "equal"  # "equal" | "inverse-vol"

    # ------------------------------------------------------------------
    # Rebalancing (sleeve-level; global scheduler may override)
    # ------------------------------------------------------------------
    # pandas offset alias ("M" ~ month-end); V1.5 used monthly rebalancing.
    rebalance_freq: str = "M"

    # Regime-based gating for the Trend sleeve
    # If True, the sleeve can be turned completely OFF under certain regimes
    use_regime_gating: bool = True

    # Regimes (by name) under which this sleeve should be fully disabled.
    # Comparison is case-insensitive; MultiSleeveAllocator passes the primary
    # regime as a lowercase string ("bull", "bear", "crisis", etc.)
    gated_off_regimes: Tuple[str, ...] = ("crisis", "bear")
