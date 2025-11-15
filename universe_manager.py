from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

from market_data_store import MarketDataStore


@dataclass
class UniverseManager:
    data_store: MarketDataStore
    membership: pd.DataFrame   # full membership, including Unknown
    name: str = "SP500_hist"

    def __post_init__(self):
        df = self.membership.copy()
        # Normalize sector column name
        if "GICS Sector" in df.columns and "Sector" not in df.columns:
            df = df.rename(columns={"GICS Sector": "Sector"})

        # Keep full membership
        self.membership = df

        # Tradable subset = known sector only
        self.membership_tradable = df[df["Sector"] != "Unknown"].copy()

        # Optional: small diagnostic
        n_total = df["Symbol"].nunique()
        n_tradable = self.membership_tradable["Symbol"].nunique()
        print(f"[UniverseManager] Symbols total: {n_total}, tradable: {n_tradable}")

    # -------- alternative constructor from CSV --------

    @classmethod
    def from_membership_csv(
        cls,
        data_store: MarketDataStore,
        csv_path: str,
        ticker_col: str = "Symbol",
        sector_col: str = "GICS Sector",
        date_added_col: str = "DateAdded",
        date_removed_col: str = "DateRemoved",
        name: str = "SP500_hist",
    ) -> "UniverseManager":
        df = pd.read_csv(csv_path)
        df = df.rename(
            columns={
                ticker_col: "Symbol",
                sector_col: "Sector",
                date_added_col: "DateAdded",
                date_removed_col: "DateRemoved",
            }
        )
        tickers = df["Symbol"].astype(str).str.upper().tolist()
        df["DateAdded"] = pd.to_datetime(df["DateAdded"]).dt.normalize()
        df["DateRemoved"] = pd.to_datetime(df["DateRemoved"]).dt.normalize()
        return cls(
            data_store=data_store,
            membership=df,
            name=name,
        )

    # -------- basic info --------
    @property
    def all_tickers(self, include_unknown: bool = False) -> List[str]:
        if include_unknown:
            return sorted(self.membership["Symbol"].unique())
        # Only tickers with known sector
        return sorted(self.membership_tradable["Symbol"].unique())

    @property
    def sector_map(self) -> dict[str, str]:
        df = self.membership_tradable.sort_values("DateAdded")
        sectors = {}
        for sym, g in df.groupby("Symbol"):
            # last known sector is fine
            sectors[sym] = g.iloc[-1]["Sector"]
        return sectors

    def tickers(self) -> List[str]:
        """Return list of tickers in the universe."""
        return self.all_tickers

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
            f"[UniverseManager] Ensuring OHLCV for {len(self.all_tickers)} tickers "
            f"({self.name or 'universe'}) {start} → {end} ({interval})"
        )
        for i, ticker in enumerate(self.all_tickers, start=1):
            print(f"[UniverseManager] [{i}/{len(self.all_tickers)}] {ticker}")
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
        for ticker in self.all_tickers:
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
    
    def build_membership_mask(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        tickers = self.all_tickers  # tradable only
        mask = pd.DataFrame(False, index=index, columns=tickers)

        for _, row in self.membership_tradable.iterrows():
            sym = row["Symbol"]
            added = row["DateAdded"]
            removed = row["DateRemoved"]

            if sym not in mask.columns:
                continue

            if pd.isna(removed):
                valid = (index >= added)
            else:
                valid = (index >= added) & (index < removed)

            mask.loc[valid, sym] = True

        return mask

    def get_price_matrix(
        self,
        start: str,
        end: str,
        field: str = "Close",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV prices for all tickers in membership, apply membership mask so that
        before DateAdded / after DateRemoved the prices are NaN.
        """
        # 1. Fetch all tickers' data
        data = {}
        for sym in self.all_tickers:
            df = self.data_store.get_ohlcv(sym, start, end, interval=interval)
            if field not in df.columns:
                continue
            data[sym] = df[field]

        prices = pd.DataFrame(data).sort_index()

        # 2. Build membership mask on this date index
        mask = self.build_membership_mask(prices.index)

        # 3. Apply mask: tickers not in index at a given date become NaN
        prices = prices.where(mask)

        return prices
