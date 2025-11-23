from __future__ import annotations

"""
Inlined production copy of the project's MarketDataStore.

This file used to be a thin wrapper that modified sys.path and re-exported
the project-level implementation. For production packaging we inline the
implementation to avoid runtime path hacks.
"""

from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import yfinance as yf


class MarketDataStore:
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
        self.data_root = Path(data_root)
        self.source = source
        self.data_root.mkdir(parents=True, exist_ok=True)
        # If True, overrides online fetching for individual calls.
        self.local_only = local_only

        # In-memory cache of already-loaded OHLCV data.
        # Key: (ticker_upper, interval) -> pd.DataFrame
        self.use_memory_cache = bool(use_memory_cache)
        self._memory_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ---------- public API ----------

    def get_ohlcv(
        self,
        ticker: str,
        start: datetime | str,
        end: datetime | str,
        interval: str = "1d",
        auto_adjust: bool = True,
        local_only: bool = False,
    ) -> pd.DataFrame:
        local_only = self.local_only or local_only
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # For daily (and coarser) data, normalize to midnight so we compare by date
        if interval in ("1d", "1wk", "1mo"):
            start_dt = start_dt.normalize()
            end_dt = end_dt.normalize()

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

        return df_full.loc[(df_full.index >= start_dt) & (df_full.index <= end_dt)]

    def has_cached(self, ticker: str, interval: str = "1d") -> bool:
        """
        Return True if we have any cached data for (ticker, interval),
        either in memory (if enabled) or on disk.
        """
        key = self._cache_key(ticker, interval)
        if self.use_memory_cache and key in self._memory_cache:
            return True
        return self._data_path(ticker, interval).exists()

    def get_cached_coverage(self, ticker: str, interval: str = "1d"):
        df = self._load_cached_df(ticker, interval)
        if df is None or df.empty:
            return None
        return df.index.min(), df.index.max()

    # ---------- internal helpers ----------

    def _cache_key(self, ticker: str, interval: str) -> tuple[str, str]:
        return (ticker.upper(), interval)

    def _ticker_dir(self, ticker: str, interval: str) -> Path:
        return self.data_root / "ohlcv" / interval / ticker.upper()

    def _data_path(self, ticker: str, interval: str) -> Path:
        return self._ticker_dir(ticker, interval) / "data.parquet"

    def _meta_path(self, ticker: str, interval: str) -> Path:
        return self._ticker_dir(ticker, interval) / "meta.json"

    def _load_cached_df(self, ticker: str, interval: str) -> pd.DataFrame | None:
        """
        Load OHLCV for (ticker, interval) from in-memory cache (if enabled) or disk.

        NOTE: The cache does not distinguish auto_adjust variants; behavior is
        consistent with the existing implementation which stores a single
        adjusted/unadjusted variant in the parquet.
        """
        key = self._cache_key(ticker, interval)

        # First, try in-memory cache
        if self.use_memory_cache and key in self._memory_cache:
            df = self._memory_cache[key]
            # We assume df is already sorted & DateTimeIndex
            # Print a lighter log than disk loads to avoid noise if desired
            # print(f"[MarketDataStore] Memory cache hit: {ticker} {interval}")
            return df

        # Fallback to disk
        path = self._data_path(ticker, interval)
        if not path.exists():
            return None

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        # print(f"[MarketDataStore] Loaded cached data: {ticker} {interval}")

        # Populate memory cache for future calls
        if self.use_memory_cache:
            self._memory_cache[key] = df

        return df

    def _save_cached_df(self, ticker: str, interval: str, df: pd.DataFrame) -> None:
        """
        Save OHLCV to disk and, if enabled, update in-memory cache.
        """
        tdir = self._ticker_dir(ticker, interval)
        tdir.mkdir(parents=True, exist_ok=True)

        df_sorted = df.sort_index()
        df_sorted.to_parquet(self._data_path(ticker, interval))

        # Update memory cache
        if self.use_memory_cache:
            key = self._cache_key(ticker, interval)
            self._memory_cache[key] = df_sorted

        meta = {
            "ticker": ticker.upper(),
            "interval": interval,
            "start": df_sorted.index.min().strftime("%Y-%m-%d"),
            "end": df_sorted.index.max().strftime("%Y-%m-%d"),
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "source": self.source,
        }
        with open(self._meta_path(ticker, interval), "w") as f:
            json.dump(meta, f, indent=2)

    def _fetch_online(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        if self.source != "yfinance":
            raise NotImplementedError("Only yfinance source currently supported")

        print(
            f"[MarketDataStore] Fetching online: {ticker} {start.date()} â {end.date()} {interval}"
        )

        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
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

        # Ensure DatetimeIndex
        df.index = pd.to_datetime(df.index)
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
        df_cached = self._load_cached_df(ticker, interval)

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
                missing_right_start = cached_end + timedelta(days=1)
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

        self._save_cached_df(ticker, interval, df_combined)
        return df_combined
