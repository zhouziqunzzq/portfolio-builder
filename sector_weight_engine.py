from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SectorWeightEngine:
    """
    Convert sector scores into sector weights over time, with:
    - softmax weighting
    - cap/floor constraints
    - smoothing/hysteresis
    - trend-based risk-on/risk-off scaling
    """
    sector_scores: pd.DataFrame           # index: Date, columns: sectors
    benchmark_prices: pd.Series          # index: Date, benchmark (SPY / ^GSPC) close prices

    alpha: float = 1.0                   # softmax sharpness
    w_min: float = 0.03                  # min sector weight
    w_max: float = 0.30                  # max sector weight
    beta: float = 0.3                    # smoothing factor (0=stick, 1=jump)
    trend_window: int = 200              # SMA window for trend filter
    risk_on_equity_frac: float = 1.0     # equity fraction in risk-on
    risk_off_equity_frac: float = 0.7    # equity fraction in risk-off
    top_k_sectors: Optional[int] = None  # keep only top-k sectors each rebalance (others zeroed, then renormalize)

    def __post_init__(self):
        # Align benchmark index to sector_scores index
        self.sector_scores = self.sector_scores.sort_index()
        self.benchmark_prices = self.benchmark_prices.sort_index()

        # Reindex benchmark to the sector_scores index (forward/backfill if needed)
        self.benchmark_prices = (
            self.benchmark_prices.reindex(self.sector_scores.index)
                                 .ffill()
                                 .bfill()
        )

        # Ensure no negative or weird alpha values
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if not (0 <= self.w_min <= self.w_max <= 1.0):
            raise ValueError("Invalid w_min/w_max")
        if not (0 <= self.beta <= 1.0):
            raise ValueError("beta should be in [0, 1]")
        if not (0 <= self.risk_off_equity_frac <= self.risk_on_equity_frac <= 1.0):
            raise ValueError("Equity fractions must be between 0 and 1")

        # Resolve default for top_k_sectors to number of sectors
        n_sectors = len(self.sector_scores.columns)
        if self.top_k_sectors is None or int(self.top_k_sectors) <= 0 or int(self.top_k_sectors) > n_sectors:
            self.top_k_sectors = n_sectors

    # ------------------------------------------------------------
    # Trend filter
    # ------------------------------------------------------------
    def compute_trend_state(self) -> pd.Series:
        """
        Simple long-term trend filter on the benchmark:
        risk_on = price > SMA(trend_window)
        Returns Series[Date] ∈ {0, 1}.
        """
        sma = self.benchmark_prices.rolling(self.trend_window).mean()
        trend_state = (self.benchmark_prices > sma).astype(int)
        trend_state.name = "trend_state"
        return trend_state

    # ------------------------------------------------------------
    # Core weighting logic for a single cross-section
    # ------------------------------------------------------------
    def _softmax_weights(self, scores: pd.Series) -> pd.Series:
        # Use raw scores, allow negatives
        x = self.alpha * scores.astype(float)
        if x.isna().all():
            # Fallback: equal-weight
            return pd.Series(1.0 / len(scores), index=scores.index, dtype=float)

        # For numerical stability
        x = x.fillna(0.0)
        x = x - x.max()
        e = np.exp(x)
        w = e / e.sum()
        return w

    def _apply_caps_and_floors(self, w: pd.Series) -> pd.Series:
        """
        Enforce min/max per sector and renormalize.
        """
        w_clipped = w.clip(lower=self.w_min, upper=self.w_max)

        # If everything got clipped to zero (edge case), fall back to equal-weight
        total = w_clipped.sum()
        if total <= 0:
            return pd.Series(1.0 / len(w_clipped), index=w_clipped.index, dtype=float)

        return w_clipped / total

    def _smooth_weights(
        self,
        prev_weights: Optional[pd.Series],
        new_weights: pd.Series,
    ) -> pd.Series:
        """
        Exponential smoothing / hysteresis:
        w_smoothed = (1 - beta) * prev + beta * new

        If no prev_weights, just return new_weights.
        """
        if prev_weights is None:
            return new_weights

        # Align indices
        prev_weights = prev_weights.reindex(new_weights.index).fillna(0.0)

        w_smoothed = (1 - self.beta) * prev_weights + self.beta * new_weights
        # Renormalize to sum to 1
        total = w_smoothed.sum()
        if total <= 0:
            return pd.Series(
                1.0 / len(w_smoothed),
                index=w_smoothed.index,
                dtype=float,
            )
        return w_smoothed / total

    def _apply_top_k(self, w: pd.Series) -> pd.Series:
        """Keep only top-k sectors by weight, zero others, then renormalize to 1.

        If k is invalid or >= number of sectors, return w normalized.
        """
        if w.empty:
            return w
        k = int(self.top_k_sectors or len(w))
        if k >= len(w) or k <= 0:
            total = float(w.sum())
            return (w / total) if total > 0 else pd.Series(1.0 / len(w), index=w.index, dtype=float)

        keep_idx = w.nlargest(k).index
        w2 = w.copy()
        w2.loc[~w2.index.isin(keep_idx)] = 0.0
        total = float(w2.sum())
        if total > 0:
            return w2 / total
        # Fallback: equal-weight among kept indices
        w2.loc[:] = 0.0
        w2.loc[keep_idx] = 1.0 / float(k)
        return w2

    # ------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------
    def compute_weights(self) -> pd.DataFrame:
        """
        Compute sector weights for each date in sector_scores.index.
        Returns DataFrame[Date × Sector] of equity weights
        (you can later multiply by equity_frac per date if you want).
        """
        sectors = list(self.sector_scores.columns)
        index = self.sector_scores.index

        trend_state = self.compute_trend_state()

        weights = pd.DataFrame(index=index, columns=sectors, dtype=float)

        prev_weights: Optional[pd.Series] = None

        for dt in index:
            scores_t = self.sector_scores.loc[dt]

            # 1) softmax -> base weights
            w_raw = self._softmax_weights(scores_t)

            # 2) caps/floors
            w_capped = self._apply_caps_and_floors(w_raw)

            # 3) smoothing vs previous
            w_smoothed = self._smooth_weights(prev_weights, w_capped)

            # 3.5) enforce top-k selection and renormalize among selected sectors
            w_selected = self._apply_top_k(w_smoothed)

            # 4) trend-based equity scaling
            if trend_state.loc[dt] == 1:
                equity_frac = self.risk_on_equity_frac
            else:
                equity_frac = self.risk_off_equity_frac

            w_final = w_selected * equity_frac   # remaining 1 - equity_frac = implicit cash

            weights.loc[dt] = w_final
            prev_weights = w_smoothed

        return weights
