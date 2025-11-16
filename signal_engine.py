from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from universe_manager import UniverseManager

# If you want to call from UniverseManager:
# from universe_manager import UniverseManager


@dataclass
class SignalEngine:
    """
    Compute price-based signals (momentum, volatility, composite scores)
    for a universe of stocks, optionally grouped by sector.
    """
    prices: pd.DataFrame                # index: Date, columns: tickers
    sector_map: Optional[Dict[str, str]] = None

    def __post_init__(self):
        # Normalize tickers to uppercase and align sector_map to columns
        self.prices = self.prices.copy()
        self.prices.columns = [c.upper() for c in self.prices.columns]

        if self.sector_map is not None:
            self.sector_map = {k.upper(): v for k, v in self.sector_map.items()}
            # Keep sector_map only for tickers present in prices
            self.sector_map = {
                t: s for t, s in self.sector_map.items()
                if t in self.prices.columns
            }

    # ------------------------------------------------------------------
    # Convenience constructor from UniverseManager (optional)
    # ------------------------------------------------------------------
    @classmethod
    def from_universe(
        cls,
        universe,      # type: UniverseManager
        start,
        end,
        field: str = "Close",
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> "SignalEngine":
        """
        Build SignalEngine directly from a UniverseManager by pulling
        a price matrix and its sector_map.
        """
        prices = universe.get_price_matrix(
            start=start,
            end=end,
            field=field,
            interval=interval,
            auto_adjust=auto_adjust,
        )
        sector_map = universe.sector_map if hasattr(universe, "sector_map") else None
        return cls(prices=prices, sector_map=sector_map)

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _zscore_within_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score each column within its sector, per date.
        If no sector_map is provided, z-score across all tickers.
        """
        df = df.copy()
        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

        if not self.sector_map:
            # Single cross-section (no sectors)
            mean = df.mean(axis=1)
            std = df.std(axis=1).replace(0, np.nan)
            result = df.sub(mean, axis=0).div(std, axis=0)
            return result

        # Group tickers by sector
        sector_to_tickers: Dict[str, list[str]] = {}
        for ticker, sector in self.sector_map.items():
            sector_to_tickers.setdefault(sector, []).append(ticker)

        for sector, tickers in sector_to_tickers.items():
            # Only keep tickers that exist in df
            cols = [t for t in tickers if t in df.columns]
            if not cols:
                continue
            sub = df[cols]
            mean = sub.mean(axis=1)
            std = sub.std(axis=1).replace(0, np.nan)
            z = sub.sub(mean, axis=0).div(std, axis=0)
            result[cols] = z

        return result
    
    def _zscore_across_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score each column per date across ALL tickers (no sector grouping).
        """
        mean = df.mean(axis=1)
        std = df.std(axis=1).replace(0, np.nan)
        return df.sub(mean, axis=0).div(std, axis=0)

    # ------------------------------------------------------------------
    # Returns & momentum
    # ------------------------------------------------------------------
    def compute_simple_returns(self) -> pd.DataFrame:
        """Daily simple returns: r_t = P_t / P_{t-1} - 1."""
        # Explicitly disable implicit forward-filling (FutureWarning removal)
        rets = self.prices.pct_change(fill_method=None)
        return rets

    def compute_momentum_components(
        self,
        windows: Sequence[int] = (63, 126, 252),
    ) -> Dict[int, pd.DataFrame]:
        """
        Compute raw momentum for each lookback window:
        mom_w(t) = P_t / P_{t-w} - 1

        Returns a dict: window -> DataFrame[Date × Ticker]
        """
        mom_dict: Dict[int, pd.DataFrame] = {}
        for w in windows:
            mom = self.prices / self.prices.shift(w) - 1.0
            mom_dict[w] = mom
        return mom_dict

    def compute_momentum_score(
        self,
        windows: Sequence[int] = (63, 126, 252),
        weights: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """
        Compute a composite momentum score by:
        1) computing raw momentum for each window in `windows`
        2) z-scoring each window within sector, per date
        3) combining as weighted sum of z-scores
        """
        if weights is None:
            weights = [1.0] * len(windows)
        if len(weights) != len(windows):
            raise ValueError("weights and windows must have the same length")

        mom_dict = self.compute_momentum_components(windows)
        mom_z_list = []
        for w in windows:
            mom = mom_dict[w]
            mom_z = self._zscore_across_universe(mom)
            mom_z_list.append(mom_z)

        # Weighted sum of z-scores
        momentum_score = pd.DataFrame(
            0.0, index=self.prices.index, columns=self.prices.columns
        )
        for w, z, w_weight in zip(windows, mom_z_list, weights):
            momentum_score = momentum_score.add(z * w_weight)

        return momentum_score

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------
    def compute_volatility(self, window: int = 20) -> pd.DataFrame:
        """
        Realized volatility from daily returns over a rolling window,
        annualized: sigma = std(returns_window) * sqrt(252)
        """
        rets = self.compute_simple_returns()
        vol = rets.rolling(window=window).std() * np.sqrt(252)
        return vol

    def compute_vol_score(self, window: int = 20) -> pd.DataFrame:
        """
        Z-scored volatility within sector, with sign flipped so that
        lower vol => higher score.
        """
        vol = self.compute_volatility(window=window)
        vol_z = self._zscore_across_universe(vol)
        vol_score = -vol_z  # lower vol is better
        return vol_score

    # ------------------------------------------------------------------
    # Composite stock score
    # ------------------------------------------------------------------
    def compute_stock_score(
        self,
        mom_windows: Sequence[int] = (63, 126, 252),
        mom_weights: Optional[Sequence[float]] = None,
        vol_window: int = 20,
        vol_weight: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compute the composite signal for each stock:
        stock_score = momentum_score + vol_weight * vol_score

        - `mom_windows`: lookback windows for momentum horizons
        - `mom_weights`: weights for each horizon (default equal)
        - `vol_window`: window for realized volatility
        - `vol_weight`: weight for volatility score (typical ~1.0–1.2)
        """
        momentum_score = self.compute_momentum_score(
            windows=mom_windows,
            weights=mom_weights,
        )
        vol_score = self.compute_vol_score(window=vol_window)

        stock_score = momentum_score.add(vol_weight * vol_score, fill_value=0.0)

        # DO NOT z-score final layer by sector
        # keep cross-sector differences intact
        # allow some sectors to have higher mean scores

        return stock_score
    
    def compute_sector_scores_from_stock_scores(
        self,
        stock_score: pd.DataFrame,
        sector_map: dict[str, str],
    ) -> pd.DataFrame:
        """
        Average stock scores within each sector, per date.
        Returns DataFrame[Date x Sector].
        """
        # Build a small helper DataFrame: cols=tickers, row: sector label
        sectors = pd.Series(
            {ticker: sector_map.get(ticker.upper(), None) for ticker in stock_score.columns},
            name="sector",
        )

        # Drop tickers without a known sector
        valid_cols = sectors.dropna().index.tolist()
        stock_score = stock_score[valid_cols]
        sectors = sectors[valid_cols]

        # Group columns by sector and take mean per date
        # Trick: stack → groupby → unstack
        stacked = stock_score.copy()
        stacked.columns = pd.MultiIndex.from_arrays([sectors, stacked.columns])
        # level 0: sector, level 1: ticker
        # Avoid deprecated groupby with axis=1 by transposing
        sector_scores = stacked.T.groupby(level=0).mean().T

        return sector_scores

