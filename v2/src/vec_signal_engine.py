from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Dict

from datetime import datetime
import numpy as np
import pandas as pd

# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from universe_manager import UniverseManager
from market_data_store import MarketDataStore


@dataclass
class VectorizedSignalEngine:
    """
    Vectorized signal engine for cross-sectional (many-ticker) work.

    Responsibilities:
      - Build price matrices (Date x Ticker) from MarketDataStore
      - Optionally make those matrices membership-aware using S&P500 membership
        from UniverseManager, to reduce survivorship bias
      - Serve as the basis for vectorized momentum / vol / stock-score, etc.
    """

    universe: UniverseManager
    mds: MarketDataStore

    # --- core matrix builders -------------------------------------------------

    def get_field_matrix(
        self,
        tickers: Optional[Iterable[str]],
        start: datetime | str,
        end: datetime | str,
        field: str = "Close",
        interval: str = "1d",
        local_only: bool = False,
        auto_adjust: bool = True,
        membership_aware: bool = True,
        treat_unknown_as_always_member: bool = True,
    ) -> pd.DataFrame:
        """
        Build a field matrix: Date x Ticker.

        Parameters
        ----------
        tickers:
            Iterable of tickers to include. If None:
              - try universe.tickers
              - else fall back to universe.sector_map keys
        start, end:
            Calendar start/end (will be converted to Timestamp).
        field:
            OHLCV field to use; must be a column in MarketDataStore.get_ohlcv
            result (e.g. "Close", "Adjclose").
        interval:
            Bar interval ("1d", "1wk", ...).
        local_only:
            Passed through to MarketDataStore.get_ohlcv.
        auto_adjust:
            Passed through to MarketDataStore.get_ohlcv.
        membership_aware:
            If True, apply S&P500 membership mask for *known* index tickers.
        treat_unknown_as_always_member:
            Only used when membership_aware=True.
            If True:
              - SP500 names are masked by membership (time-varying).
              - Unknown / non-index names (GLD, BND, etc.) are treated
                as 'always member' (mask = True on all dates).
            If False:
              - Unknown tickers are dropped entirely from the matrix.
        """
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # -----------------------------
        # Resolve tickers
        # -----------------------------
        if tickers is None:
            # Prefer explicit universe.tickers if available
            universe_tickers = getattr(self.universe, "tickers", None)
            if universe_tickers is not None:
                tickers = list(universe_tickers)
            else:
                # Fallback to sector_map keys
                smap = getattr(self.universe, "sector_map", {}) or {}
                tickers = list(smap.keys())

        tickers = [t.upper() for t in tickers]
        if not tickers:
            return pd.DataFrame()

        # -----------------------------
        # Build membership mask (optional)
        # -----------------------------
        mem_mask: Optional[pd.DataFrame] = None
        # known_index_tickers: Sequence[str] = []
        # unknown_tickers: Sequence[str] = []

        if membership_aware:
            try:
                mem_mask = self.universe.membership_mask(
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                )
            except Exception:
                mem_mask = None

            if mem_mask is not None and not mem_mask.empty:
                mem_mask.index = pd.to_datetime(mem_mask.index)
                mem_mask = mem_mask.sort_index()

                # Restrict membership matrix to the requested tickers
                mem_mask = mem_mask.reindex(columns=tickers)

                # known_index_tickers = [t for t in tickers if t in mem_mask.columns]
                # unknown_tickers = [t for t in tickers if t not in mem_mask.columns]
            else:
                # If membership fails, fall back to non-membership-aware behavior
                mem_mask = None
                # known_index_tickers = []
                # unknown_tickers = tickers
        # else:
        #     unknown_tickers = tickers

        # -----------------------------
        # Fetch per-ticker series
        # -----------------------------
        field_dict: Dict[str, pd.Series] = {}

        for t in tickers:
            try:
                df = self.mds.get_ohlcv(
                    ticker=t,
                    start=start_dt,
                    end=end_dt,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    local_only=local_only,
                )
                if df is None or df.empty:
                    print(
                        f"[VectorizedSignalEngine] WARNING: no data for {t}, skipping"
                    )
                    continue

                if field not in df.columns:
                    # Try some reasonable fallbacks
                    if field.lower() == "close" and "Close" in df.columns:
                        col = "Close"
                    elif field.lower() == "adjclose" and "Adjclose" in df.columns:
                        col = "Adjclose"
                    else:
                        # Just skip this ticker if desired field not present
                        print(
                            f"[VectorizedSignalEngine] WARNING: {field} not found for {t}, skipping"
                        )
                        continue
                else:
                    col = field

                s = df[col].astype(float)
                s.name = t
                field_dict[t] = s
                # print(f"[VectorizedSignalEngine] loaded {field} for {t}, s={s}")
            except Exception as e:
                print(
                    f"[VectorizedSignalEngine] WARNING: failed to load {field} for {t}: {e}"
                )
                import traceback

                traceback.print_exc()
                continue

        if not field_dict:
            return pd.DataFrame()

        # Combine into a single matrix, align on union of dates
        field_mat = pd.concat(field_dict.values(), axis=1)
        field_mat.index = pd.to_datetime(field_mat.index)
        field_mat = field_mat.sort_index()
        # Reindex to business days only if interval is daily
        if interval == "1d":
            field_mat = field_mat.reindex(
                index=pd.date_range(
                    start=field_mat.index.min(),
                    end=field_mat.index.max(),
                    freq="B",
                )
            )

        # Keep only our requested tickers (in canonical order)
        field_mat = field_mat.reindex(columns=tickers)

        # -----------------------------
        # Apply membership mask if requested
        # -----------------------------
        if membership_aware:
            # membership_mask: [Date x Ticker] of bools (True = in index)
            mem_mask = self.universe.membership_mask(start=start_dt, end=end_dt)
            # Align dates to price matrix
            mem_mask = mem_mask.reindex(index=field_mat.index)
            # Align columns to price matrix, keep unknown tickers as NaN for now
            mem_mask = mem_mask.reindex(columns=field_mat.columns)

            if treat_unknown_as_always_member:
                # Unknown tickers => all-NaN column, treat as always in
                mem_mask = mem_mask.fillna(True)
            else:
                # Unknown tickers => out of universe
                mem_mask = mem_mask.fillna(False)

            # Apply membership mask: where False -> set price to NaN
            field_mat = field_mat.where(mem_mask)

        return field_mat

    # Simple helper that uses membership-aware SP500-only prices by default
    def get_field_matrix_sp500(
        self,
        start: datetime | str,
        end: datetime | str,
        field: str = "Close",
        interval: str = "1d",
        local_only: bool = False,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Convenience wrapper: use the universe's SP500 membership as-of each date,
        drop non-SP500 tickers, and apply time-varying membership mask.
        """
        return self.get_field_matrix(
            tickers=None,
            start=start,
            end=end,
            field=field,
            interval=interval,
            local_only=local_only,
            auto_adjust=auto_adjust,
            membership_aware=True,
            treat_unknown_as_always_member=False,
        )

    def get_price_matrix(self, *args, **kwargs):
        """
        Convenience wrapper around get_field_matrix with field="Close".
        """
        if "field" not in kwargs:
            kwargs["field"] = "Close"
        return self.get_field_matrix(*args, **kwargs)

    def get_price_matrix_sp500(self, *args, **kwargs):
        """
        Convenience wrapper around get_field_matrix_sp500 with field="Close".
        """
        if "field" not in kwargs:
            kwargs["field"] = "Close"
        return self.get_field_matrix_sp500(*args, **kwargs)

    # ======================================================================
    #  VECTORIZED APIs
    # ======================================================================

    # -----------------------------------------------------------
    # Get return matrix
    # -----------------------------------------------------------
    def get_returns(
        self,
        price_mat: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute return matrix (same shape as price_mat).
        """
        if price_mat.empty:
            return price_mat
        return price_mat.pct_change(fill_method=None).fillna(0.0)

    # -----------------------------------------------------------
    # Multi-horizon vectorized momentum
    # -----------------------------------------------------------
    def get_momentum(
        self,
        price_mat: pd.DataFrame,
        lookbacks: Sequence[int],
    ) -> Dict[int, pd.DataFrame]:
        """
        Vectorized cross-sectional time-series momentum.

        Uses row-based .shift() to count TRADING BARS.
        This matches the non-vectorized SignalEngine behavior.

        IMPORTANT: price_mat must NOT contain NaN values, as .shift() counts
        all rows including NaNs. Call .dropna(how='all', axis=0) or .ffill()
        to clean the data before passing to this method.

        Returns
        -------
        dict: window -> momentum matrix (Date x Ticker)
        """
        mom_dict: Dict[int, pd.DataFrame] = {}

        if price_mat.empty:
            for w in lookbacks:
                mom_dict[w] = pd.DataFrame(index=price_mat.index)
            return mom_dict

        for w in lookbacks:
            # Row-based shift = trading bars (assuming no NaN rows)
            mom = price_mat / price_mat.shift(w) - 1.0
            mom_dict[w] = mom

        return mom_dict

    def get_ts_momentum(
        self,
        price_mat: pd.DataFrame,
        lookbacks: Sequence[int],
    ) -> Dict[int, pd.DataFrame]:
        """
        Time-series momentum: Sharpe-like (mu/sigma) per asset, per window.
        Returns dict[window] -> DataFrame[Date x Ticker]
        """
        ts_dict: Dict[int, pd.DataFrame] = {}
        if price_mat.empty:
            for w in lookbacks:
                ts_dict[w] = pd.DataFrame(index=price_mat.index)
            return ts_dict

        # log returns (optional but clean)
        rets = np.log(price_mat).diff()

        for w in lookbacks:
            mu = rets.rolling(w).mean()
            sigma = rets.rolling(w).std()
            # Make sure to use sqrt(w) to make "Sharpe-like" ts-mom comparable across windows
            ts_raw = (mu.div(sigma) * np.sqrt(w)).replace([np.inf, -np.inf], np.nan)
            ts_dict[w] = ts_raw

        return ts_dict

    def get_spread_momentum(
        self,
        price_mat: pd.DataFrame,
        lookbacks: Sequence[int],
        benchmark: Optional[
            str
        ] = "SPY",  # Use a single benchmark ticker for all for now
        price_col: str = "Close",
        interval: str = "1d",
    ) -> Dict[int, pd.DataFrame]:
        """
        Vectorized spread momentum vs benchmark.

        Returns
        -------
        dict: window -> spread momentum matrix (Date x Ticker)
        """
        spread_mom_dict: Dict[int, pd.DataFrame] = {}

        if price_mat.empty:
            for w in lookbacks:
                spread_mom_dict[w] = pd.DataFrame(index=price_mat.index)
            return spread_mom_dict

        # Fetch benchmark prices from market data store directly
        df_bench = self.mds.get_ohlcv(
            ticker=benchmark,
            start=price_mat.index.min(),
            end=price_mat.index.max(),
            interval=interval,
        )
        if df_bench is None or df_bench.empty:
            for w in lookbacks:
                spread_mom_dict[w] = pd.DataFrame(index=price_mat.index)
            return spread_mom_dict
        bench_price = df_bench[price_col].astype(float)
        bench_price.index = pd.to_datetime(bench_price.index)
        bench_price = bench_price.reindex(index=price_mat.index).ffill()

        # Compute log returns
        # Replace zeros/inf before taking log to avoid warnings
        bench_price_clean = bench_price.replace([0, np.inf, -np.inf], np.nan)
        price_mat_clean = price_mat.replace([0, np.inf, -np.inf], np.nan)
        log_bench = np.log(bench_price_clean)
        log_px = np.log(price_mat_clean)

        for w in lookbacks:
            ret_i = log_px - log_px.shift(w)
            ret_b = log_bench - log_bench.shift(w)
            # broadcast benchmark returns across all tickers
            ret_b_mat = pd.DataFrame(
                np.tile(ret_b.values.reshape(-1, 1), (1, price_mat.shape[1])),
                index=price_mat.index,
                columns=price_mat.columns,
            )
            spread_w = ret_i - ret_b_mat
            spread_mom_dict[w] = spread_w

        return spread_mom_dict

    # -----------------------------------------------------------
    # Vectorized realized volatility
    # -----------------------------------------------------------
    def get_volatility(
        self,
        price_mat: pd.DataFrame,
        window: int = 20,
        annualize: bool = True,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Vectorized rolling realized volatility (Date x Ticker).

        Uses log returns to match non-vectorized SignalEngine behavior.
        """
        if price_mat.empty:
            return price_mat

        # Use log returns to match non-vec SignalEngine
        # Handle zeros and inf/nan from division
        price_ratio = price_mat / price_mat.shift(1)
        # Replace inf/nan/zeros with NaN to avoid log warnings
        price_ratio = price_ratio.replace([0, np.inf, -np.inf], np.nan)
        log_returns = np.log(price_ratio)
        vol = log_returns.rolling(window).std()

        if annualize:
            factor = self._annualize_factor_for_interval(interval)
            vol *= np.sqrt(factor)

        return vol

    def get_ewm_volatility(
        self,
        price_mat: pd.DataFrame,
        halflife: int = 20,
        annualize: bool = True,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Vectorized EWM realized volatility (Date x Ticker).
        """
        if price_mat.empty:
            return price_mat

        returns = price_mat.pct_change(fill_method=None)
        vol = returns.ewm(halflife=halflife).std()

        if annualize:
            factor = self._annualize_factor_for_interval(interval)
            vol *= np.sqrt(factor)

        return vol

    def _annualize_factor_for_interval(self, interval: str) -> int:
        """Return an annualization factor (periods per year) for `interval`.

        Recognizes weekly and monthly-like intervals; defaults to 252 for daily.
        """
        lower = (interval or "1d").lower()
        if "wk" in lower or lower.endswith("w"):
            return 52
        if "mo" in lower or "month" in lower:
            return 12
        return 252

    # -----------------------------------------------------------
    # Convenience for slicing large precomputed matrices
    # -----------------------------------------------------------
    @staticmethod
    def slice_by_date(
        mat: pd.DataFrame,
        as_of: datetime | str,
        history: int | None = None,
    ) -> pd.DataFrame:
        """
        Slice matrix up to as_of.
        Optionally keep only the last `history` days.
        """
        as_of = pd.to_datetime(as_of)
        out = mat.loc[mat.index <= as_of]
        if history is not None and history > 0:
            out = out.tail(history)
        return out
