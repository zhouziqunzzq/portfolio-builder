from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StockAllocator:
    """
    Allocate sector weights down to individual stocks by:
    - ranking stocks within each sector by stock_score
    - selecting top-k per sector
    - allocating sector weight across those stocks (equal or inverse-vol)
    """
    sector_weights: pd.DataFrame           # index: Date, columns: sectors
    stock_scores: pd.DataFrame            # index: Date, columns: tickers
    sector_map: Dict[str, str]            # ticker -> sector

    stock_vol: Optional[pd.DataFrame] = None  # index: Date, columns: tickers (for inverse-vol)
    top_k: int = 2
    weighting_mode: str = "equal"         # "equal" or "inverse_vol"
    preserve_cash: bool = True            # if True, don't force weights to sum to 1 (keep cash)

    def __post_init__(self):
        # Normalize ticker cases
        self.stock_scores = self.stock_scores.copy()
        self.stock_scores.columns = [c.upper() for c in self.stock_scores.columns]

        if self.stock_vol is not None:
            self.stock_vol = self.stock_vol.copy()
            self.stock_vol.columns = [c.upper() for c in self.stock_vol.columns]

        self.sector_map = {t.upper(): s for t, s in self.sector_map.items()}

        # Align date index: we will use sector_weights.index as master
        self.stock_scores = (
            self.stock_scores.reindex(self.sector_weights.index)
                            .ffill()
        )
        if self.stock_vol is not None:
            self.stock_vol = (
                self.stock_vol.reindex(self.sector_weights.index)
                              .ffill()
            )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _tickers_in_sector(self, sector: str) -> List[str]:
        return [t for t, s in self.sector_map.items() if s == sector]

    def _select_top_k_for_sector_on_date(
        self,
        dt: pd.Timestamp,
        sector: str,
    ) -> List[str]:
        """
        For a given date & sector, return the list of selected tickers (top_k).
        """
        tickers = self._tickers_in_sector(sector)
        if not tickers:
            return []

        # Filter to tickers that exist in stock_scores
        tickers = [t for t in tickers if t in self.stock_scores.columns]
        if not tickers:
            return []

        scores_t = self.stock_scores.loc[dt, tickers]

        # Drop NaN scores
        scores_t = scores_t.dropna()
        if scores_t.empty:
            return []

        # Rank descending and take top_k
        top = scores_t.sort_values(ascending=False).head(self.top_k)
        return top.index.tolist()

    def _intra_sector_weights(
        self,
        dt: pd.Timestamp,
        tickers: List[str],
    ) -> pd.Series:
        """
        Compute intra-sector weights for a given set of tickers at a date,
        according to weighting_mode.
        """
        if not tickers:
            return pd.Series(dtype=float)

        if self.weighting_mode == "equal" or self.stock_vol is None:
            # equal-weight
            n = len(tickers)
            return pd.Series(1.0 / n, index=tickers, dtype=float)

        # inverse-vol weighting
        vol_t = self.stock_vol.loc[dt, tickers]
        vol_t = vol_t.replace({0.0: np.nan})
        # Drop NaN vol; if all NaN, fallback to equal
        if vol_t.isna().all():
            n = len(tickers)
            return pd.Series(1.0 / n, index=tickers, dtype=float)

        inv_vol = 1.0 / vol_t
        inv_vol = inv_vol.replace({np.inf: np.nan}).dropna()

        # Might lose some tickers if vol was NaN; re-add them equal-weight if needed
        if inv_vol.empty:
            n = len(tickers)
            return pd.Series(1.0 / n, index=tickers, dtype=float)

        w = inv_vol / inv_vol.sum()

        # If we lost some tickers, reintroduce them with zero or small weight; simplest is zero
        w = w.reindex(tickers).fillna(0.0)
        return w

    # ------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------
    def compute_stock_weights(self) -> pd.DataFrame:
        """
        Allocate sector weights to stocks for each date.
        Returns DataFrame[Date Ã Ticker] of portfolio weights.
        """
        dates = self.sector_weights.index
        all_tickers = sorted(set(self.stock_scores.columns))
        stock_weights = pd.DataFrame(index=dates, columns=all_tickers, dtype=float)

        prev_row = None

        for dt in dates:
            sector_row = self.sector_weights.loc[dt]
            w_row = pd.Series(0.0, index=all_tickers, dtype=float)

            for sector, sector_w in sector_row.items():
                if pd.isna(sector_w) or sector_w <= 0:
                    continue

                tickers = self._select_top_k_for_sector_on_date(dt, sector)
                if not tickers:
                    continue

                intra_w = self._intra_sector_weights(dt, tickers)
                # scale by sector weight
                w_row.loc[intra_w.index] += sector_w * intra_w

            if not self.preserve_cash:
                # normalize so that total weight sums to 1 (no cash)
                total = w_row.sum()
                if total > 0:
                    w_row = w_row / total

            stock_weights.loc[dt] = w_row

            prev_row = w_row

        return stock_weights
