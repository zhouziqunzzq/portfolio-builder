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
    # Ensure values are hashable -> convert lists to tuples etc.
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

    def __init__(
        self,
        mds: MarketDataStore,
        disable_cache_margin: bool = False,
        # The current implementation of cache extension drags down performance
        # by too much in some cases; default disable for now.
        disable_cache_extension: bool = True,
    ) -> None:
        self.mds = mds
        self.store = SignalStore()
        self.annualize_factor = {
            "1d": 252,
            "1wk": 52,
            "1mo": 12,
        }

        self.disable_cache_margin = disable_cache_margin
        if self.disable_cache_margin:
            print("[SignalEngine] Cache margin DISABLED")
        self.disable_cache_extension = disable_cache_extension
        if self.disable_cache_extension:
            print("[SignalEngine] Cache extension DISABLED")

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
        rst_series = self._get_series(
            ticker=ticker,
            signal=signal,
            start=start,
            end=end,
            interval=interval,
            **params,
        )
        # Print a warning if:
        # - returned series is empty
        # - returned series end drifts too far from requested end date
        if rst_series.empty:
            # print(
            #     f"[SignalEngine] WARNING: returned empty series for {ticker} {signal} from {pd.to_datetime(start).date()} to {pd.to_datetime(end).date()} (interval={interval}, params={params})"
            # )
            pass
        else:
            requested_end = pd.to_datetime(end)
            actual_end = rst_series.index.max()
            if actual_end < requested_end - pd.Timedelta(days=5):
                print(
                    f"[SignalEngine] WARNING: returned series for {ticker} {signal} ends at {actual_end.date()}, which is more than 5 days before requested end date {requested_end.date()} (interval={interval}, params={params})"
                )
        return rst_series

    # ----- internal dispatcher -----

    def _get_series(
        self,
        ticker: str,
        signal: str,
        start: datetime | str,
        end: datetime | str,
        interval: str = "1d",
        **params: Any,
    ) -> pd.Series:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        # auto_ffill = params.pop("auto_ffill", False)
        # auto_ffill_limit = params.pop("auto_ffill_limit", 5)

        key = SignalKey(
            ticker=ticker.upper(),
            interval=interval,
            name=signal,
            params_key=_normalize_params(params),
        )

        cached = self.store.get(key)
        # Cache hit: check coverage
        if cached is not None and not cached.empty:
            cached_start = cached.index.min()
            cached_end = cached.index.max()

            # For daily/weekly/monthly data, give ourselves a calendar margin
            if self.disable_cache_margin:
                margin = pd.Timedelta(0)
            elif interval == "1d":
                margin = pd.Timedelta(days=3)
            elif interval == "1wk":
                margin = pd.Timedelta(days=10)
            elif interval == "1mo":
                margin = pd.Timedelta(days=40)
            else:
                margin = pd.Timedelta(0)

            effective_start = cached_start - margin
            effective_end = cached_end + margin

            # Do we already effectively cover the requested window?
            if start_dt >= effective_start and end_dt <= effective_end:
                # Full coverage: fast path
                # print(f"[SignalEngine] CACHE HIT: {key}; Requested [{start_dt.date()} to {end_dt.date()}], Cached [{cached_start.date()} to {cached_end.date()}]")
                return cached.loc[(cached.index >= start_dt) & (cached.index <= end_dt)]

            # Partial coverage: extend range (recompute over union)
            if not self.disable_cache_extension:
                new_start = min(start_dt, cached_start)
                new_end = max(end_dt, cached_end)
                # print(f"[SignalEngine] CACHE HIT (EXTEND RANGE): {key}; Requested [{start_dt.date()} to {end_dt.date()}], Cached [{cached_start.date()} to {cached_end.date()}], New range [{new_start.date()} to {new_end.date()}]")
                series = self._compute_signal_full(
                    ticker=ticker,
                    signal=signal,
                    start=new_start,
                    end=new_end,
                    interval=interval,
                    # auto_ffill=auto_ffill,
                    # auto_ffill_limit=auto_ffill_limit,
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
            # auto_ffill=auto_ffill,
            # auto_ffill_limit=auto_ffill_limit,
            **params,
        )

        self.store.set(key, series)
        if series.empty:
            return series
        return series.loc[(series.index >= start_dt) & (series.index <= end_dt)]

    def _compute_signal_full(
        self,
        ticker: str,
        signal: str,
        start: datetime,
        end: datetime,
        interval: str,
        # auto_ffill: bool = False,  # whether to forward-fill missing values by default
        # auto_ffill_limit: Optional[
        #     int
        # ] = 5,  # max number of consecutive missing values to ffill over
        **params: Any,
    ) -> pd.Series:
        """
        Compute the *full* signal series required, given the requested date range.

        Important: for indicators with lookback (e.g. window=63),
        we need to fetch extra price history before `start` so the first value
        at `start` is valid.
        """
        signal = signal.lower()
        possible_signal_variants = [
            signal,
            signal.replace(" ", "_"),
            signal.replace("-", "_"),
        ]

        # Prefer direct naming convention: `_compute_{signal}_full`.
        # This keeps a single place to add new signals and avoids long
        # duplicated if/elif chains.
        for signal in possible_signal_variants:
            method = getattr(self, f"_compute_{signal}_full", None)
            if method is not None:
                break
        # Fallback: try some common aliases
        if method is None:
            # Common alias mappings (extendable)
            aliases = {
                "median-volume": "median_volume",
                "last-price": "last_price",
            }
            normalized = aliases.get(signal, signal)
            method = getattr(self, f"_compute_{normalized}_full", None)

        if method is None:
            raise ValueError(f"Unknown signal: {signal}")

        sig = method(ticker, start, end, interval, **params)

        # if auto_ffill:
        #     # Extend to requested date range
        #     sig = sig.reindex(pd.date_range(start, end, freq="D"))
        #     sig = sig.ffill(
        #         limit=auto_ffill_limit
        #     )  # ffill because the requested end date may not be a trading day
        return sig

    def _calc_start_for_data(
        self, start: datetime, extra_bars: int, interval: str
    ) -> datetime:
        """
        Compute an approximate calendar start date to fetch `extra_bars` of
        history before `start`, depending on `interval`.

        - For daily data (`1d`) use business days (`BDay`).
        - For weekly data (`1wk`) subtract `7 * extra_bars` days.
        - For monthly data (`1mo`) use `DateOffset(months=...)`.
        - For other/undetected intervals fall back to business-day estimate.

        This is a conservative approximation; if `MarketDataStore` later
        supports fetching N-bars before a date, prefer that instead.
        """
        lower = (interval or "").lower()
        # daily
        if lower in ("1d", "d", "day") or lower.endswith("d"):
            return start - pd.tseries.offsets.BDay(extra_bars)

        # weekly-ish
        if "wk" in lower or lower.endswith("w") or "week" in lower:
            return start - pd.Timedelta(days=7 * extra_bars)

        # monthly-ish
        if "mo" in lower or "month" in lower:
            return start - pd.DateOffset(months=extra_bars)

        # fallback: assume daily business-day spacing (conservative)
        return start - pd.tseries.offsets.BDay(max(1, extra_bars))

    # ----- concrete signal implementations -----

    def _compute_ret_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 1,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Simple return over `window`: P / P.shift(window) - 1

        We fetch (start - extra_lookback, end) so that value at 'start'
        has a valid lookback.
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        ret = price / price.shift(window) - 1.0
        ret.name = f"ret_{window}"

        return ret

    def _compute_log_ret_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 1,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Log return over `window`: log(P / P.shift(window))

        We fetch (start - extra_lookback, end) so that value at 'start'
        has a valid lookback.
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        log_ret = np.log(price / price.shift(window))
        log_ret.name = f"log_ret_{window}"

        return log_ret

    def _compute_ts_mom_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 252,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Time-series momentum: P / P.shift(window) - 1

        We fetch (start - extra_lookback, end) so that value at 'start'
        has a valid lookback.
        """
        ret = self.get_series(
            ticker,
            "ret",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )

        mom = ret
        mom.name = f"ts_mom_{window}"

        return mom

    def _compute_ts_mom_sharpe_like_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 252,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        log_ret = self.get_series(
            ticker,
            "log_ret",
            start,
            end,
            interval,
            window=1,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )

        mu = log_ret.rolling(window=window).mean()
        sigma = log_ret.rolling(window=window).std()
        ts_raw = (mu.div(sigma) * np.sqrt(window)).replace([np.inf, -np.inf], np.nan)
        ts_raw.name = f"ts_mom_sharpe_like_{window}"

        return ts_raw

    def _compute_vol_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        price_col: str = "Close",
        annualize: bool = True,
        buffer_bars: int = 5,
    ) -> pd.Series:
        """
        Realized volatility over 'window', computed from daily log returns.
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        log_rets = self.get_series(
            ticker,
            "log_ret",
            start_for_data,
            end,
            interval,
            window=1,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )
        log_rets = log_rets.dropna()

        vol = log_rets.rolling(window=window).std()
        if annualize:
            annualize_factor = self.annualize_factor.get(interval, 252)
            vol = vol * np.sqrt(annualize_factor)

        vol.name = f"vol_{window}"
        return vol

    def _compute_ewm_vol_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        halflife: int = 20,
        price_col: str = "Close",
        annualize: bool = True,
    ) -> pd.Series:
        """
        Exponentially-weighted moving volatility estimator.

        Parameters:
            halflife: halflife in trading days for EWM
        """
        # Need extra bars for EWM to stabilize (approx in calendar time)
        extra = halflife * 10  # EWM needs more data to stabilize
        start_for_data = self._calc_start_for_data(start, extra, interval)

        log_rets = self.get_series(
            ticker,
            "log_ret",
            start_for_data,
            end,
            interval,
            window=1,
            price_col=price_col,
            buffer_bars=extra,
        )
        log_rets = log_rets.dropna()
        if log_rets.empty:
            return pd.Series(dtype=float)

        vol = log_rets.ewm(halflife=halflife, adjust=False).std()
        if annualize:
            annualize_factor = self.annualize_factor.get(interval, 252)
            vol = vol * np.sqrt(annualize_factor)
        vol.name = f"ewm_vol_{halflife}"
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
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Example 'trend_score' combining fast-vs-slow MA and price vs slow MA.

        Simple heuristic:
          trend_score = 0.5 * sign(MA_fast - MA_slow) + 0.5 * (price - MA_slow) / MA_slow
        (You can refine this later, this is just a placeholder.)
        """
        extra = slow_window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

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

    def _compute_sma_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 50,
        price_col: str = "Close",
        buffer_bars: int = 5,
    ) -> pd.Series:
        """
        Simple moving average (SMA) over `window` bars.

        Returns a Series indexed by date, name = "sma_{window}".
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty or price_col not in df:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        sma = price.rolling(window=window).mean()
        sma.name = f"sma_{window}"
        return sma

    def _compute_beta_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 63,
        benchmark: str = "SPY",
        price_col: str = "Close",
        buffer_bars: int = 10,
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
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)
        # Log returns
        ra = self.get_series(
            ticker,
            "log_ret",
            start_for_data,
            end,
            interval,
            window=1,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )
        rm = self.get_series(
            benchmark,
            "log_ret",
            start_for_data,
            end,
            interval,
            window=1,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )

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

        beta = cov / var_m.replace(0, np.nan)
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
        buffer_bars: int = 5,
    ) -> pd.Series:
        """
        ADV = rolling mean of (Close * Volume)

        Returns: Series indexed by date, name = "adv_{window}".
        """

        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

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
        buffer_bars: int = 5,
    ) -> pd.Series:
        """
        20-bar median volume (or configurable window).
        """

        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

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
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

        price = df[price_col].astype(float)
        price.name = "last_price"
        # print(f"[SignalEngine] _compute_last_price_full: Retrieved {len(price)} data points for {ticker} from {start.date()} to {end.date()}")
        return price

    # ----------------------------------------------------------------------
    # Spread signals
    # ----------------------------------------------------------------------

    def _compute_spread_mom_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        price_col: str = "Close",
        benchmark: str = "SPY",
        buffer_bars: int = 5,
    ) -> pd.Series:
        """
        Spread momentum = R_ticker - R_benchmark over `window` days.

        Where R_ticker = log(P_t / P_{t-window}) is the log return over `window`.
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        # Log returns
        ra = self.get_series(
            ticker,
            "log_ret",
            start_for_data,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )
        rm = self.get_series(
            benchmark,
            "log_ret",
            start_for_data,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )

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

        spread_mom = rets["ra"] - rets["rm"]
        spread_mom.name = f"spread_mom_{benchmark}_{window}"

        return spread_mom

    def _compute_log_spread_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        price_col: str = "Close",
        benchmark: str = "SPY",
        buffer_bars: int = 5,
    ) -> pd.Series:
        """
        Log spread = log(P_ticker / P_benchmark)
                   = log(P_ticker) - log(P_benchmark)
        """
        extra = buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df_ticker = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        df_benchmark = self.mds.get_ohlcv(
            ticker=benchmark,
            start=start_for_data,
            end=end,
            interval=interval,
        )

        if df_ticker.empty or df_benchmark.empty:
            return pd.Series(dtype=float)

        price_ticker = df_ticker[price_col].astype(float)
        price_benchmark = df_benchmark[price_col].astype(float)

        # Align on common dates
        prices = pd.concat(
            [
                price_ticker.rename("pt"),
                price_benchmark.rename("pb"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if prices.empty:
            return pd.Series(dtype=float)

        log_spread = np.log(prices["pt"]) - np.log(prices["pb"])
        log_spread.name = f"log_spread_{benchmark}"

        return log_spread

    def _compute_log_spread_beta_hedged_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        price_col: str = "Close",
        benchmark: str = "SPY",
        hedge_window: int = 63,
        buffer_bars: int = 5,
    ) -> pd.Series:
        """
        Log spread beta-hedged = log(P_ticker / P_benchmark^beta)
                   = log(P_ticker) - beta * log(P_benchmark)
        Where beta is the rolling beta of ticker vs benchmark.
        """
        extra = buffer_bars + hedge_window * 2
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df_ticker = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        df_benchmark = self.mds.get_ohlcv(
            ticker=benchmark,
            start=start_for_data,
            end=end,
            interval=interval,
        )

        if df_ticker.empty or df_benchmark.empty:
            return pd.Series(dtype=float)

        price_ticker = df_ticker[price_col].astype(float)
        price_benchmark = df_benchmark[price_col].astype(float)

        # Align on common dates
        prices = pd.concat(
            [
                price_ticker.rename("pt"),
                price_benchmark.rename("pb"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if prices.empty or len(prices) < hedge_window + 2:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

        lp_s = np.log(prices["pt"])
        lp_b = np.log(prices["pb"])

        # use log returns for beta
        r_s = lp_s.diff()
        r_b = lp_b.diff()

        # rolling beta = cov(rs, rb)/var(rb)
        cov = r_s.rolling(hedge_window).cov(r_b)
        var = r_b.rolling(hedge_window).var()
        beta = cov / (var + 1e-12)

        # residualized spread (level)
        spread = lp_s - beta * lp_b

        # optional: remove rolling mean so z-score is purely about deviations
        spread = spread - spread.rolling(hedge_window).mean()

        spread.name = f"log_spread_betahedged_{benchmark}"
        return spread.dropna()

    # ----------------------------------------------------------------------
    # Bollinger signals
    # ----------------------------------------------------------------------

    def _compute_bb_mid_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Bollinger Bands middle line = simple moving average over `window`.
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty or price_col not in df:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        bb_mid = price.rolling(window=window).mean()
        bb_mid.name = f"bb_mid_{window}"
        return bb_mid

    def _compute_bb_std_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        ddof: int = 0,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Bollinger Bands standard deviation over `window`.
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start_for_data,
            end=end,
            interval=interval,
        )
        if df.empty or price_col not in df:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        bb_std = price.rolling(window=window).std(ddof=ddof)
        bb_std.name = f"bb_std_{window}"
        return bb_std

    def _compute_bb_upper_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        k: float = 2.0,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Bollinger Bands upper line = mid + k * std
        """
        mid = self.get_series(
            ticker,
            "bb_mid",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )
        std = self.get_series(
            ticker,
            "bb_std",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )

        upper = mid + float(k) * std
        upper.name = f"bb_upper_{window}_{k:g}"
        return upper

    def _compute_bb_lower_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        k: float = 2.0,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Bollinger Bands lower line = mid - k * std
        """
        mid = self.get_series(
            ticker,
            "bb_mid",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )
        std = self.get_series(
            ticker,
            "bb_std",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )

        lower = mid - float(k) * std
        lower.name = f"bb_lower_{window}_{k:g}"
        return lower

    def _compute_bb_z_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Bollinger Bands Z-score = (price - mid) / std
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

        mid = self.get_series(
            ticker,
            "bb_mid",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )
        std = self.get_series(
            ticker,
            "bb_std",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )

        bb_z = (price - mid) / std.replace(0, np.nan)
        bb_z.name = f"bb_z_{window}"
        return bb_z

    def _compute_bb_bandwidth_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        k: float = 2.0,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Bollinger Bands Bandwidth = (upper - lower) / mid = (2 * k * std) / mid
        """
        mid = self.get_series(
            ticker,
            "bb_mid",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        ).replace(0.0, np.nan)
        std = self.get_series(
            ticker,
            "bb_std",
            start,
            end,
            interval,
            window=window,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )

        # (upper - lower)/mid = (2*k*std)/mid
        bw = (2.0 * float(k) * std) / mid
        bw.name = f"bb_bandwidth_{window}_{k:g}"
        return bw

    def _compute_bb_percent_b_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        k: float = 2.0,
        price_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Bollinger Bands %B = (price - lower) / (upper - lower)
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

        upper = self.get_series(
            ticker,
            "bb_upper",
            start,
            end,
            interval,
            window=window,
            k=k,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )
        lower = self.get_series(
            ticker,
            "bb_lower",
            start,
            end,
            interval,
            window=window,
            k=k,
            price_col=price_col,
            buffer_bars=buffer_bars,
        )
        denom = (upper - lower).replace(0.0, np.nan)

        pb = (price - lower) / denom
        pb.name = f"bb_percent_b_{window}_{k:g}"
        return pb

    def _compute_trend_slope_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 50,
        price_col: str = "Close",
        use_log_price: bool = True,
        buffer_bars: int = 20,
    ) -> pd.Series:
        """
        Trend slope over `window` bars, computed via OLS linear regression.
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df = self.mds.get_ohlcv(
            ticker=ticker, start=start_for_data, end=end, interval=interval
        )
        if df.empty or price_col not in df:
            return pd.Series(dtype=float)

        price = df[price_col].astype(float)
        y = np.log(price) if use_log_price else price

        # Precompute constants for slope with x = 0..window-1
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_demean = x - x_mean
        denom = float(np.sum(x_demean * x_demean))  # > 0

        def _ols_slope(arr: np.ndarray) -> float:
            # arr is length=window, may contain nan
            if np.isnan(arr).any():
                return np.nan
            y_demean = arr - float(np.mean(arr))
            return float(np.sum(x_demean * y_demean) / denom)

        slope = y.rolling(window=window).apply(_ols_slope, raw=True)
        # Units: log-price per bar (or price per bar if use_log_price=False)
        slope.name = f"trend_slope_{window}" + ("_log" if use_log_price else "")
        return slope

    def _compute_donchian_pos_full(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        window: int = 20,
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        buffer_bars: int = 10,
    ) -> pd.Series:
        """
        Donchian Channel position over `window` bars:
            pos = (close - Donchian_Low) / (Donchian_High - Donchian_Low)
        """
        extra = window + buffer_bars
        start_for_data = self._calc_start_for_data(start, extra, interval)

        df = self.mds.get_ohlcv(
            ticker=ticker, start=start_for_data, end=end, interval=interval
        )
        if df.empty or any(c not in df for c in (high_col, low_col, close_col)):
            return pd.Series(dtype=float)

        high = df[high_col].astype(float)
        low = df[low_col].astype(float)
        close = df[close_col].astype(float)

        hi = high.rolling(window=window).max()
        lo = low.rolling(window=window).min()
        rng = (hi - lo).replace(0.0, np.nan)

        pos = (close - lo) / rng
        pos.name = f"donchian_pos_{window}"
        return pos
