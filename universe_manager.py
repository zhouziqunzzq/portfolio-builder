from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

from market_data_store import MarketDataStore


@dataclass
class UniverseManager:
    data_store: MarketDataStore
    _tickers: List[str]
    sector_map: Optional[Dict[str, str]] = None
    name: Optional[str] = None

    def __post_init__(self):
        # Normalize tickers to uppercase and de-duplicate
        self._tickers = sorted({t.upper() for t in self._tickers})
        if self.sector_map is not None:
            # Normalize keys to uppercase
            self.sector_map = {k.upper(): v for k, v in self.sector_map.items()}

    # -------- alternative constructor from CSV --------

    @classmethod
    def from_csv(
        cls,
        data_store: MarketDataStore,
        csv_path: str | Path,
        ticker_col: str = "Symbol",
        sector_col: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "UniverseManager":
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        if ticker_col not in df.columns:
            raise ValueError(f"Ticker column '{ticker_col}' not found in {csv_path}")

        tickers = df[ticker_col].astype(str).str.upper().tolist()

        sector_map = None
        if sector_col is not None and sector_col in df.columns:
            sector_map = {
                str(t).upper(): str(s)
                for t, s in zip(df[ticker_col], df[sector_col])
            }

        return cls(
            data_store=data_store,
            _tickers=tickers,
            sector_map=sector_map,
            name=name or csv_path.stem,
        )

    # -------- basic info --------

    def tickers(self) -> List[str]:
        """Return list of tickers in the universe."""
        return list(self._tickers)

    def sectors(self) -> set[str]:
        """Return set of sectors present in the universe (if sector_map is available)."""
        if not self.sector_map:
            return set()
        return set(self.sector_map.values())

    def get_sector(self, ticker: str) -> Optional[str]:
        """Return sector for a given ticker, if known."""
        if not self.sector_map:
            return None
        return self.sector_map.get(ticker.upper())

    # -------- data management --------

    def ensure_ohlcv(
        self,
        start,
        end,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> None:
        """
        Ensure OHLCV data for all tickers in [start, end].
        This will populate the local cache via MarketDataStore.
        """
        print(
            f"[UniverseManager] Ensuring OHLCV for {len(self._tickers)} tickers "
            f"({self.name or 'universe'}) {start} → {end} ({interval})"
        )
        for i, ticker in enumerate(self._tickers, start=1):
            print(f"[UniverseManager] [{i}/{len(self._tickers)}] {ticker}")
            _ = self.data_store.get_ohlcv(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
            )

    def get_ohlcv_dict(
        self,
        start,
        end,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return a dict mapping ticker -> OHLCV DataFrame for [start, end].
        Uses the underlying cache (and will fetch any missing data).
        """
        result: Dict[str, pd.DataFrame] = {}
        for ticker in self._tickers:
            df = self.data_store.get_ohlcv(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
            )
            if not df.empty:
                result[ticker] = df
        return result

    def get_price_matrix(
        self,
        start,
        end,
        field: str = "Close",
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Build a wide price matrix: index = Date, columns = tickers, values = [field].
        Missing tickers (no data) are silently skipped.
        """
        series_list = []
        for ticker in self._tickers:
            df = self.data_store.get_ohlcv(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
            )
            if df.empty:
                continue

            if field not in df.columns:
                # You could raise here if you want stricter behavior.
                print(
                    f"[UniverseManager] WARNING: field '{field}' not found for {ticker}, skipping."
                )
                continue

            s = df[field].rename(ticker.upper())
            series_list.append(s)

        if not series_list:
            return pd.DataFrame()

        price_matrix = pd.concat(series_list, axis=1).sort_index()
        return price_matrix
