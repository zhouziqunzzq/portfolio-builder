from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional, Tuple

from datetime import datetime
import pandas as pd
import numpy as np

# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from market_data_store import MarketDataStore


# ---------- Signal cache key & params normalization ----------


def _normalize_params(params: Dict[str, Any]) -> Tuple[Tuple[str, Hashable], ...]:
    """
    Turn a params dict into a sorted, hashable tuple.

    Example:
      {"window": 63, "method": "simple"}
    -> (("method", "simple"), ("window", 63))
    """
    if not params:
        return tuple()
    # Ensure values are hashable â convert lists to tuples etc.
    normalized_items = []
    for k, v in sorted(params.items()):
        if isinstance(v, dict):
            # Nested dicts: recursively normalize
            v = _normalize_params(v)
        elif isinstance(v, (list, tuple)):
            v = tuple(v)
        normalized_items.append((k, v))
    return tuple(normalized_items)


@dataclass(frozen=True)
class SignalKey:
    ticker: str
    interval: str
    name: str
    params_key: Tuple[Tuple[str, Hashable], ...]


# ---------- SignalStore: in-memory per-run cache ----------


class SignalStore:
    """
    Simple in-memory cache: (asset, interval, signal, params) -> pd.Series
    """

    def __init__(self) -> None:
        self._cache: Dict[SignalKey, pd.Series] = {}

    def get(self, key: SignalKey) -> Optional[pd.Series]:
        return self._cache.get(key)

    def set(self, key: SignalKey, series: pd.Series) -> None:
        # Always store sorted by index
        self._cache[key] = series.sort_index()

    def clear(self) -> None:
        self._cache.clear()


# ---------- SignalEngine ----------


class SignalEngine:
    """
    Central signal computation & caching engine.

    All sleeves + RegimeEngine should request signals via this class.
    """

    def __init__(self, mds: MarketDataStore):
        self.mds = mds
        self.store = SignalStore()

    # ----- public API -----

    def get_series(
        self,
        ticker: str,
        signal: str,
        start: datetime | str,
        end: datetime | str,
        interval: str = "1d",
        **params: Any,
    ) -> pd.Series:
        """
        Get a full time series for a given (ticker, signal, params) over [start, end].

        - Cache key is (ticker, interval, signal, params).
        - If cached series fully covers [start, end] *up to a small calendar margin*
          (to account for weekends/holidays), we slice & return.
        - Otherwise we extend the cached range to the union and update the cache.
        """
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        key = SignalKey(
            ticker=ticker.upper(),
            interval=interval,
            name=signal,
            params_key=_normalize_params(params),
        )

        cached = self.store.get(key)
        if cached is not None and not cached.empty:
            cached_start = cached.index.min()
            cached_end = cached.index.max()

            # For daily/weekly/monthly data, give ourselves a calendar margin
            if interval in ("1d", "1wk", "1mo"):
                margin = pd.Timedelta(days=7)
            else:
                margin = pd.Timedelta(0)

            effective_start = cached_start - margin
            effective_end = cached_end + margin

            # Do we already effectively cover the requested window?
            if start_dt >= effective_start and end_dt <= effective_end:
                # Full coverage: fast path
                # print(f"[SignalEngine] CACHE HIT: {key}")
                return cached.loc[(cached.index >= start_dt) & (cached.index <= end_dt)]

            # Partial coverage: extend range (recompute over union)
            new_start = min(start_dt, cached_start)
            new_end = max(end_dt, cached_end)

            # print(f"[SignalEngine] CACHE HIT (EXTEND RANGE): {key}")
            series = self._compute_signal_full(
                ticker=ticker,
                signal=signal,
                start=new_start,
                end=new_end,
                interval=interval,
                **params,
            )
            self.store.set(key, series)
            return series.loc[(series.index >= start_dt) & (series.index <= end_dt)]

        # Cache miss: compute the full series for requested range
        # print(f"[SignalEngine] CACHE MISS: {key}")
        series = self._compute_signal_full(
            ticker=ticker,
            signal=signal,
            start=start_dt,
            end=end_dt,
            interval=interval,
            **params,
        )

        self.store.set(key, series)
        return series.loc[(series.index >= start_dt) & (series.index <= end_dt)]

    # ----- internal dispatcher -----

    def _compute_signal_full(
        self,
        ticker: str,
        signal: str,
        start: datetime,
        end: datetime,
        interval: str,
        **params: Any,
    ) -> pd.Series:
        """
        Compute the *full* signal series required, given the requested date range.

        Important: for indicators with lookback (e.g. window=63),
        we need to fetch extra price history before `start` so the first value
        at `start` is valid.
        """
        signal = signal.lower()

        if signal == "ts_mom":
            return self._compute_ts_mom_full(ticker, start, end, interval, **params)
        elif signal == "vol":
            return self._compute_vol_full(ticker, start, end, interval, **params)
        elif signal == "trend_score":
            return self._compute_trend_score_full(
                ticker, start, end, interval, **params
            )
        elif signal == "beta":
            return self._compute_beta_full(ticker, start, end, interval, **params)
        elif signal == "adv":
            return self._compute_adv_full(ticker, start, end, interval, **params)
        elif signal == "median_volume":
            return self._compute_median_volume_full(
                ticker, start, end, interval, **params
            )
        elif signal == "last_price":
            return self._compute_last_price_full(ticker, start, end, interval, **params)
        else:
            raise ValueError(f"Unknown signal: {signal}")

    # ----- concrete signal implementations -----

    def _compute_ts_mom_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 252,
        price_col: str = "Close",
    ) -> pd.Series:
        """
        Time-series momentum: P / P.shift(window) - 1

        We fetch (start - extra_lookback, end) so that value at 'start'
        has a valid lookback.
        """
        extra = window + 5  # add a bit of cushion
        start_for_data = start - pd.tseries.offsets.BDay(extra)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        mom = price / price.shift(window) - 1.0
        mom.name = f"ts_mom_{window}"

        return mom

    def _compute_vol_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        price_col: str = "Close",
        annualize: bool = True,
    ) -> pd.Series:
        """
        Realized volatility over 'window', computed from daily log returns.
        """
        # Need window extra days
        extra = window + 5
        start_for_data = start - pd.tseries.offsets.BDay(extra)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        rets = np.log(price / price.shift(1)).dropna()

        vol = rets.rolling(window=window).std()
        if annualize:
            vol = vol * np.sqrt(252)

        vol.name = f"vol_{window}"
        return vol

    def _compute_trend_score_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        fast_window: int = 50,
        slow_window: int = 200,
        price_col: str = "Close",
    ) -> pd.Series:
        """
        Example 'trend_score' combining fast-vs-slow MA and price vs slow MA.

        Simple heuristic:
          trend_score = 0.5 * sign(MA_fast - MA_slow) + 0.5 * (price - MA_slow) / MA_slow
        (You can refine this later, this is just a placeholder.)
        """
        extra = slow_window + 10
        start_for_data = start - pd.tseries.offsets.BDay(extra)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)

        ma_fast = price.rolling(window=fast_window).mean()
        ma_slow = price.rolling(window=slow_window).mean()

        ma_diff_sign = np.sign(ma_fast - ma_slow)
        rel_to_slow = (price - ma_slow) / ma_slow

        trend_score = 0.5 * ma_diff_sign + 0.5 * rel_to_slow
        trend_score.name = f"trend_score_{fast_window}_{slow_window}"

        return trend_score

    def _compute_beta_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 63,
        benchmark: str = "SPY",
        price_col: str = "Close",
    ) -> pd.Series:
        """
        Rolling beta of `ticker` vs `benchmark` using log returns.

        Definition:
            beta_t = Cov(R_asset, R_benchmark) / Var(R_benchmark)
        where Cov and Var are computed over a rolling window of length `window`.

        Parameters:
            ticker:    asset whose beta you want
            benchmark: reference index/ETF (default: SPY)
            window:    lookback window in bars (default: 63)
        """
        extra = window + 5
        start_for_data = start - pd.tseries.offsets.BDay(extra)

        # Fetch both asset and benchmark prices
        df_asset = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        df_bench = self.mds.get_ohlcv(
            ticker=benchmark,
            start=start_for_data,
            end=end,
            interval=interval,
        )

        if df_asset.empty or df_bench.empty:
            return pd.Series(dtype=float)

        pa = df_asset[price_col].astype(float)
        pm = df_bench[price_col].astype(float)

        # Log returns
        ra = np.log(pa / pa.shift(1))
        rm = np.log(pm / pm.shift(1))

        # Align on common dates
        rets = pd.concat(
            [
                ra.rename("ra"),
                rm.rename("rm"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if rets.empty:
            return pd.Series(dtype=float)

        # Rolling covariance & variance
        cov = rets["ra"].rolling(window=window).cov(rets["rm"])
        var_m = rets["rm"].rolling(window=window).var()

        beta = cov / var_m
        beta.name = f"beta_{benchmark}_{window}"

        return beta

    # ----------------------------------------------------------------------
    # Liquidity signals
    # ----------------------------------------------------------------------

    def _compute_adv_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        price_col: str = "Close",
        volume_col: str = "Volume",
    ) -> pd.Series:
        """
        ADV = rolling mean of (Close * Volume)

        Returns: Series indexed by date, name = "adv_{window}".
        """

        extra = window + 5
        start_for_data = start - pd.tseries.offsets.BDay(extra)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty or price_col not in df or volume_col not in df:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        vol = df[volume_col].astype(float)

        adv = (price * vol).rolling(window=window).mean()
        adv.name = f"adv_{window}"
        return adv

    def _compute_median_volume_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        volume_col: str = "Volume",
    ) -> pd.Series:
        """
        20-day median volume (or configurable window).
        """

        extra = window + 5
        start_for_data = start - pd.tseries.offsets.BDay(extra)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty or volume_col not in df:
            return pd.Series(dtype=float)

        vol = df[volume_col].astype(float)
        mv = vol.rolling(window=window).median()
        mv.name = f"median_vol_{window}"
        return mv

    def _compute_last_price_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        price_col: str = "Close",
    ) -> pd.Series:
        """
        Useful for liquidity filters (avoid penny stocks, etc.)
        Simple pass-through of adjusted close.
        """

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
        )
        if df.empty or price_col not in df:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        price.name = "last_price"
        return price
