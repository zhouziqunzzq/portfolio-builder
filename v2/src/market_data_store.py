from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from pathlib import Path
from datetime import UTC, datetime, timedelta
import json
import logging
import pandas as pd
import yfinance as yf

import sys

_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from utils.tz import to_canonical_eastern_naive


class BaseMarketDataStore(ABC):
    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        start: datetime | str,
        end: datetime | str,
        interval: str = "1d",
        auto_adjust: bool = True,
        local_only: bool = False,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def get_ohlcv_matrix(
        self,
        tickers: List[str],
        start: datetime | str,
        end: datetime | str,
        field: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        local_only: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        raise NotImplementedError()


class MarketDataStore(BaseMarketDataStore):
    def __init__(
        self,
        data_root: str,
        source: str = "yfinance",
        local_only: bool = False,
        use_memory_cache: bool = False,
    ):
        """
        Args:
            data_root: root directory for price cache (parquet files).
            source: data provider (currently only 'yfinance' is supported).
            local_only: if True, never attempt online fetches; only use local cache.
            use_memory_cache: if True, keep per-(ticker, interval) OHLCV DataFrames
                              in memory so repeated calls avoid disk reads.
        """
        super().__init__()

        # Logger for this class
        self.log = logging.getLogger(self.__class__.__name__)

        self.data_root = Path(data_root)
        self.source = source
        self.data_root.mkdir(parents=True, exist_ok=True)
        # If True, overrides online fetching for individual calls.
        self.local_only = local_only

        # In-memory cache of already-loaded OHLCV data.
        # Key: (ticker_upper, interval, 'adj'|'raw') -> pd.DataFrame
        self.use_memory_cache = bool(use_memory_cache)
        self._memory_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    # ---------- public API ----------

    def get_ohlcv(
        self,
        ticker: str,
        start: datetime | str,
        end: datetime | str,
        interval: str = "1d",
        auto_adjust: bool = True,
        local_only: bool = Optional[False],
    ) -> pd.DataFrame:
        """Get OHLCV data for the given ticker and date range.
        Timezone convention
        -------------------
        This class stores and returns OHLCV indices as tz-naive timestamps.
        For daily (and coarser) bars, these timestamps should be interpreted as
        US/Eastern calendar dates.

        - If callers pass tz-aware `start`/`end`, they are converted to US/Eastern
            and then the tzinfo is dropped (tz-naive) before any comparisons.
        - If callers pass tz-naive `start`/`end`, they are assumed to already be
            in the US/Eastern convention.

        Args:
            ticker: Ticker symbol.
            start: Start date (inclusive).
            end: End date (inclusive).
            interval: Data interval (e.g., '1d', '1wk', '1mo').
            auto_adjust: Whether to auto-adjust prices if supported.
            local_only: If set, overrides instance local_only for this call.
        Returns:
            DataFrame with DateTimeIndex and OHLCV columns.
        """
        local_only = local_only if local_only is not None else self.local_only

        start_dt = to_canonical_eastern_naive(start)
        end_dt = to_canonical_eastern_naive(end)

        # For daily (and coarser) data, normalize to midnight so we compare by date
        if interval in ("1d", "1wk", "1mo"):
            start_dt = start_dt.normalize()
            end_dt = end_dt.normalize()

        # If requesting coarser-than-daily intervals, fetch daily bars and
        # aggregate client-side to guarantee reproducible rules. Build a
        # single `df_source` and apply common slicing/cleanup below to avoid
        # duplicated code paths.
        df_source: pd.DataFrame | None = None

        if interval != "1d":
            # Expand start/end to ensure full weeks/months are present
            # before resampling so we don't drop partial periods.
            if interval == "1wk":
                # include the previous 6 days to contain the full week
                fetch_start = start_dt - timedelta(days=6)
                fetch_end = end_dt + timedelta(days=6)
            elif interval == "1mo":
                # include an extra month on each side to ensure month-end coverage
                fetch_start = start_dt - timedelta(days=31)
                fetch_end = end_dt + timedelta(days=31)
            else:
                # Fallback: fetch a bit of padding for unknown coarse intervals
                fetch_start = start_dt - timedelta(days=7)
                fetch_end = end_dt + timedelta(days=7)

            df_daily = self._ensure_coverage(
                ticker=ticker,
                start=fetch_start,
                end=fetch_end,
                interval="1d",
                auto_adjust=auto_adjust,
                local_only=local_only,
            )

            if df_daily is None or df_daily.empty:
                return pd.DataFrame()

            # Slice to requested date range and aggregate to requested interval
            df_daily_sliced = df_daily.loc[
                (df_daily.index >= start_dt) & (df_daily.index <= end_dt)
            ]
            df_source = self._aggregate_daily_to_interval(
                df_daily_sliced, interval, ticker=ticker
            )
        else:
            # interval == '1d'
            df_full = self._ensure_coverage(
                ticker=ticker,
                start=start_dt,
                end=end_dt,
                interval=interval,
                auto_adjust=auto_adjust,
                local_only=local_only,
            )

            if df_full is None or df_full.empty:
                return pd.DataFrame()

            df_source = df_full

        # Common post-processing: slice to requested date range and clean duplicates
        if df_source is None or df_source.empty:
            return pd.DataFrame()
        df = df_source.loc[(df_source.index >= start_dt) & (df_source.index <= end_dt)]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def get_ohlcv_matrix(
        self,
        tickers: List[str],
        start: datetime | str,
        end: datetime | str,
        field: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        local_only: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        """Assemble a ohlcv matrix [date x ticker] for the given tickers and date range.

        Args:
            tickers: List of ticker symbols to fetch.
            start: Start date (inclusive).
            end: End date (inclusive).
            field: OHLCV field to extract (e.g., 'Adjclose', 'Close'); if None, auto-detect.
            interval: Data interval (e.g., '1d').
            auto_adjust: Whether to auto-adjust prices if supported.
            local_only: If set, overrides instance local_only for this call.
        """

        def _fetch_one(sym: str) -> Optional[pd.Series]:
            try:
                ohlcv = self.get_ohlcv(
                    sym,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    local_only=(
                        local_only if local_only is not None else self.local_only
                    ),
                )
                if ohlcv is None or len(ohlcv) == 0:
                    return None

                # Determine field
                if field is not None and field in ohlcv.columns:
                    col = field
                else:
                    if "Adjclose" in ohlcv.columns:
                        col = "Adjclose"
                    elif "Close" in ohlcv.columns:
                        col = "Close"
                    else:
                        self.log.debug("No Adjclose/Close for %s", sym)
                        return None

                s = ohlcv[col].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                s.name = sym
                return s
            except Exception as e:
                self.log.exception("Failed loading ohlcv for %s", sym)
                return None

        out = pd.DataFrame()
        for t in tickers:
            s = _fetch_one(t)
            if s is None:
                continue
            if out.empty:
                out = s.to_frame()
            else:
                out = out.join(s, how="outer")

        if out.empty:
            return None
        # Restrict to requested window and sort
        start_dt = to_canonical_eastern_naive(start).normalize()
        end_dt = to_canonical_eastern_naive(end).normalize()
        out = out.loc[start_dt:end_dt].sort_index()

        return out

    def has_cached(
        self, ticker: str, interval: str = "1d", auto_adjust: bool = True
    ) -> bool:
        """
        Return True if we have any cached data for (ticker, interval),
        either in memory (if enabled) or on disk.
        """
        key = self._cache_key(ticker, interval, auto_adjust)
        if self.use_memory_cache and key in self._memory_cache:
            return True
        return self._data_path(ticker, interval, auto_adjust).exists()

    def get_cached_coverage(
        self, ticker: str, interval: str = "1d", auto_adjust: bool = True
    ):
        df = self._load_cached_df(ticker, interval, auto_adjust)
        if df is None or df.empty:
            return None
        return df.index.min(), df.index.max()

    def reset_memory_cache(self) -> None:
        """
        Clear the in-memory OHLCV cache.
        """
        self._memory_cache.clear()

    # ---------- internal helpers ----------

    def _cache_key(
        self, ticker: str, interval: str, auto_adjust: bool = True
    ) -> tuple[str, str, str]:
        return (ticker.upper(), interval, "adj" if auto_adjust else "raw")

    def _ticker_dir(self, ticker: str, interval: str, auto_adjust: bool = True) -> Path:
        group = "adj" if auto_adjust else "raw"
        return self.data_root / "ohlcv" / group / interval / ticker.upper()

    def _data_path(self, ticker: str, interval: str, auto_adjust: bool = True) -> Path:
        return self._ticker_dir(ticker, interval, auto_adjust) / "data.parquet"

    def _meta_path(self, ticker: str, interval: str, auto_adjust: bool = True) -> Path:
        return self._ticker_dir(ticker, interval, auto_adjust) / "meta.json"

    def _load_cached_df(
        self, ticker: str, interval: str, auto_adjust: bool = True
    ) -> pd.DataFrame | None:
        """
        Load OHLCV for (ticker, interval) from in-memory cache (if enabled) or disk.

        NOTE: The cache now stores separate adjusted/unadjusted variants
        under distinct paths so `auto_adjust` is respected when loading.
        """
        key = self._cache_key(ticker, interval, auto_adjust)

        # First, try in-memory cache
        if self.use_memory_cache and key in self._memory_cache:
            df = self._memory_cache[key]
            # We assume df is already sorted & DateTimeIndex
            return df

        # Fallback to disk
        path = self._data_path(ticker, interval, auto_adjust)
        if not path.exists():
            return None

        df = pd.read_parquet(path)
        idx = pd.to_datetime(df.index)
        # Be defensive: parquet round-trips can preserve tz-awareness.
        # Canonicalize any tz-aware index to US/Eastern, then drop tzinfo.
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("US/Eastern").tz_localize(None)
        df.index = idx
        df = df.sort_index()

        # Populate memory cache for future calls
        if self.use_memory_cache:
            self._memory_cache[key] = df

        return df

    def _save_cached_df(
        self, ticker: str, interval: str, df: pd.DataFrame, auto_adjust: bool = True
    ) -> None:
        """
        Save OHLCV to disk and, if enabled, update in-memory cache.
        """
        if interval != "1d":
            raise ValueError("Only 1d supported in _save_cached_df")

        tdir = self._ticker_dir(ticker, interval, auto_adjust)
        tdir.mkdir(parents=True, exist_ok=True)

        df_sorted = df.sort_index()
        df_sorted.to_parquet(self._data_path(ticker, interval, auto_adjust))

        # Update memory cache
        if self.use_memory_cache:
            key = self._cache_key(ticker, interval, auto_adjust)
            self._memory_cache[key] = df_sorted

        meta = {
            "ticker": ticker.upper(),
            "interval": interval,
            "adjusted": bool(auto_adjust),
            "start": df_sorted.index.min().strftime("%Y-%m-%d"),
            "end": df_sorted.index.max().strftime("%Y-%m-%d"),
            "last_updated": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "source": self.source,
        }
        with open(self._meta_path(ticker, interval, auto_adjust), "w") as f:
            json.dump(meta, f, indent=2)

    def _fetch_online(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for (ticker, interval) from the online source.
        Only supports 'yfinance' for now.
        - start: inclusive
        - end: inclusive for '1d' intervals, exclusive for coarser intervals
        """
        if self.source != "yfinance":
            raise NotImplementedError("Only yfinance source currently supported")

        self.log.info(
            "Fetching online: %s %s -> %s %s",
            ticker,
            start.date(),
            end.date(),
            interval,
        )

        df = yf.download(
            tickers=ticker,
            start=start,
            end=(
                end + timedelta(days=1) if interval == "1d" else end
            ),  # <--- IMPORTANT: yfinance is exclusive of end date
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",  # <--- IMPORTANT: avoid MultiIndex (Price, Ticker)
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Handle possible MultiIndex columns just in case
        if isinstance(df.columns, pd.MultiIndex):
            # Common layout: level 0 = Price, level 1 = Ticker
            # For single-ticker download we can just take the first level
            # or cross-section the specific ticker.
            try:
                # If indexed as (Price, Ticker) with ticker level:
                df = df.xs(ticker, axis=1, level=-1)
            except Exception:
                # Fallback: keep only the first level
                df.columns = df.columns.get_level_values(0)

        # Normalize column names (remove spaces, title-case)
        df = df.rename(columns=lambda c: c.replace(" ", "").title())

        # Ensure DatetimeIndex and force tz-naive
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()

        return df

    def _ensure_coverage(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        auto_adjust: bool,
        local_only: bool,
    ) -> pd.DataFrame:
        """
        Ensure that cached data covers [start, end] for (ticker, interval).
        May fetch missing segments online (unless local_only).
        """
        if interval != "1d":
            raise ValueError("Only 1d supported in _ensure_coverage")

        # Canonicalize for safe comparisons against cached index values.
        start = to_canonical_eastern_naive(start).to_pydatetime()
        end = to_canonical_eastern_naive(end).to_pydatetime()

        df_cached = self._load_cached_df(ticker, interval, auto_adjust)

        # If running in local-only mode, never attempt to fetch online.
        if local_only:
            if df_cached is None:
                return pd.DataFrame()
            return df_cached

        if df_cached is None or df_cached.empty:
            df_new = self._fetch_online(ticker, start, end, interval, auto_adjust)
            df_combined = df_new
        else:
            cached_start = df_cached.index.min()
            cached_end = df_cached.index.max()

            dfs = [df_cached]

            # left gap
            if start < cached_start:
                missing_left_start = start
                missing_left_end = cached_start
                df_left = self._fetch_online(
                    ticker, missing_left_start, missing_left_end, interval, auto_adjust
                )
                if not df_left.empty:
                    dfs.append(df_left)

            # right gap (note +1 day to avoid overlapping last cached bar)
            if end > cached_end:
                missing_right_start = (
                    cached_end + timedelta(days=1) if interval == "1d" else cached_end
                )
                if missing_right_start < end:
                    df_right = self._fetch_online(
                        ticker, missing_right_start, end, interval, auto_adjust
                    )
                    if not df_right.empty:
                        dfs.append(df_right)

            df_combined = (
                pd.concat(dfs)
                .sort_index()
                .loc[lambda x: ~x.index.duplicated(keep="last")]
            )

        if df_combined is None or df_combined.empty:
            return pd.DataFrame()

        self._save_cached_df(ticker, interval, df_combined, auto_adjust)
        return df_combined

    def _aggregate_daily_to_interval(
        self,
        df_daily: pd.DataFrame,
        interval: str,
        ticker: str = "",  # optional, for logging only
    ) -> pd.DataFrame:
        """
        Aggregate daily `df_daily` into a coarser `interval`.

        Supported intervals: '1wk', '1mo'. For '1wk' we anchor weeks to
        Fridays and use OHLCV aggregation. For '1mo' we aggregate month-end
        using the same OHLCV rules.
        """
        if df_daily is None or df_daily.empty:
            return pd.DataFrame()

        # Ensure we have the common columns; if not present, attempt to select what exists
        cols = [
            c
            for c in ["Open", "High", "Low", "Close", "Volume"]
            if c in df_daily.columns
        ]
        if not cols:
            return pd.DataFrame()

        df_src = df_daily[cols]

        if interval == "1wk":
            df_agg = (
                df_src.resample("W-FRI")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )
            self.log.debug(
                "Aggregated weekly data for %s %s -> %s",
                ticker,
                df_src.index.min().date(),
                df_src.index.max().date(),
            )
            return df_agg

        if interval == "1mo":
            df_agg = (
                df_src.resample("ME")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )
            self.log.debug(
                "Aggregated monthly data for %s %s -> %s",
                ticker,
                df_src.index.min().date(),
                df_src.index.max().date(),
            )
            return df_agg

        # Unknown coarse interval: return the daily frame (fallback)
        return df_src
