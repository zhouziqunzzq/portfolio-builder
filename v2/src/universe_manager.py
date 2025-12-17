from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd


@dataclass
class UniverseArtifacts:
    membership: pd.DataFrame
    membership_tradable: pd.DataFrame
    sector_map: pd.DataFrame


class UniverseManager:
    """
    Consolidated Universe Manager for S&P 500 membership and sectors.

    This class replaces scattered scripts and exposes a coherent API for
    production usage. Initially provides thin functionality oriented around
    loading/saving existing artifacts; scraping and enrichment can be added
    step-by-step without breaking the pipeline.
    """

    def __init__(
        self,
        membership_csv: Path,
        sectors_yaml: Optional[Path] = None,
        local_only: bool = False,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.membership_csv = membership_csv
        self.sectors_yaml = sectors_yaml
        # When True, avoid all network calls and only use local artifacts/caches
        self.local_only = bool(local_only)

        self.cached_membership_df: Optional[pd.DataFrame] = None

    # ---- Build methods (to be implemented incrementally) ----
    def build_current_constituents(self) -> pd.DataFrame:
        """Fetch current S&P 500 constituents from Wikipedia and normalize.

        Returns:
            DataFrame with columns: [ticker, name, sector]

        Notes:
            - Ticker normalization converts Wikipedia dot-class tickers (e.g., BRK.B)
              to Yahoo-style hyphen class (e.g., BRK-B), and upper-cases tickers.
            - Sector names are optionally normalized via sectors.yml if provided.
        """
        if self.local_only:
            raise RuntimeError(
                "Local-only mode: build_current_constituents requires network access (Wikipedia)"
            )

        import time
        import re
        import requests

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "MomentumV1.5/1.0 (https://github.com/zhouziqunzzq/portfolio-builder)"
        }

        def _normalize_ticker(sym: str) -> str:
            if sym is None:
                return sym
            s = str(sym).strip().upper()
            # Remove any footnote markers or spaces
            s = re.sub(r"\s+", "", s)
            # Wikipedia uses "." for class shares, Yahoo uses "-"
            s = s.replace(".", "-")
            return s

        # Basic retry logic
        attempts = 3
        backoff = 1.5
        last_exc = None
        html = None
        for i in range(attempts):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                html = resp.text
                break
            except Exception as e:
                last_exc = e
                self.log.warning("Wikipedia fetch attempt %s failed: %s", i + 1, e)
                time.sleep(backoff**i)
        if html is None:
            raise RuntimeError(
                f"Failed to fetch S&P 500 page after {attempts} attempts: {last_exc}"
            )

        from io import StringIO

        tables = pd.read_html(StringIO(html))
        target = None
        for t in tables:
            cols = {str(c).strip(): c for c in t.columns}
            if {"Symbol", "Security", "GICS Sector"}.issubset(cols.keys()):
                target = t
                break
        if target is None:
            raise RuntimeError(
                "Could not locate S&P 500 constituents table on Wikipedia"
            )

        df = target[["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["ticker", "name", "sector"]
        df["ticker"] = df["ticker"].map(_normalize_ticker)
        df["name"] = df["name"].astype(str).str.strip()
        df["sector"] = df["sector"].astype(str).str.strip()

        # Drop any rows without tickers
        df = df[df["ticker"].notna() & (df["ticker"].str.len() > 0)].copy()
        # Deduplicate by ticker, keeping first occurrence
        before = len(df)
        df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        after = len(df)
        if after < before:
            self.log.info(
                "Dropped %d duplicate tickers from Wikipedia list", before - after
            )

        # Apply sector normalization if mapping provided
        df = self.apply_unified_sector_mapping(df)

        self.log.info("Fetched current constituents", extra={"count": len(df)})
        expected_cols = ["ticker", "name", "sector"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns after scraping: {missing}")
        return df

    def build_historical_membership(self) -> pd.DataFrame:
        """Build historical S&P 500 membership ranges using the fja05680/sp500 dataset.

        Returns a DataFrame with columns:
            [ticker, date_added, date_removed, name, sector]

        Notes:
            - date_removed can be NaT for still-active constituents.
            - Sectors are merged from current constituents (live Wikipedia scrape),
              then normalized via sectors.yml if provided; delisted/legacy names may
              remain Unknown at this step.
        """
        if self.local_only:
            raise RuntimeError(
                "Local-only mode: build_historical_membership requires network access (GitHub dataset)"
            )

        import io
        import requests

        TICKER_START_END_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"

        self.log.info("Downloading historical membership from %s", TICKER_START_END_URL)
        resp = requests.get(TICKER_START_END_URL, timeout=30)
        resp.raise_for_status()
        df_hist = pd.read_csv(io.StringIO(resp.text))

        # Robust column detection
        cols = {str(c).strip().lower(): c for c in df_hist.columns}
        symbol_col = cols.get("ticker") or cols.get("symbol") or cols.get("company")
        start_col = cols.get("start_date") or cols.get("start") or cols.get("dateadded")
        end_col = cols.get("end_date") or cols.get("end") or cols.get("dateremoved")

        if not symbol_col or not start_col:
            raise RuntimeError(
                f"Could not infer membership columns; available: {list(df_hist.columns)}"
            )

        # Normalize and rename
        def _norm(sym: str) -> str:
            return str(sym).strip().upper().replace(".", "-")

        df_hist = df_hist.rename(
            columns={
                symbol_col: "ticker",
                start_col: "date_added",
                **({end_col: "date_removed"} if end_col else {}),
            }
        )

        if "date_removed" not in df_hist.columns:
            df_hist["date_removed"] = pd.NaT

        df_hist["ticker"] = df_hist["ticker"].map(_norm)
        df_hist["date_added"] = pd.to_datetime(df_hist["date_added"]).dt.normalize()
        df_hist["date_removed"] = pd.to_datetime(df_hist["date_removed"]).dt.normalize()

        # Merge sector/name from current constituents (live fetch)
        try:
            cur = self.build_current_constituents()[["ticker", "name", "sector"]]
        except Exception as e:
            self.log.warning(
                "Failed to fetch current constituents for sector merge: %s", e
            )
            cur = pd.DataFrame(columns=["ticker", "name", "sector"])

        df = df_hist.merge(cur, on="ticker", how="left")
        if "name" not in df.columns:
            df["name"] = ""
        if "sector" not in df.columns:
            df["sector"] = "Unknown"

        # Normalize sectors if mapping provided
        df = self.apply_unified_sector_mapping(df)

        # Keep a clean column set and sort
        df = (
            df[["ticker", "date_added", "date_removed", "name", "sector"]]
            .sort_values(["date_added", "ticker"])
            .reset_index(drop=True)
        )

        self.log.info("Built historical membership", extra={"rows": len(df)})
        return df

    def enrich_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich sector labels using multiple sources with fallbacks.

        Strategy (in order):
        1) Keep existing sector if it's not marked as unknown.
        2) For tickers with unknown/missing sectors, query Yahoo Finance sector.
        3) If still unknown and we have a company name, try Wikipedia summary heuristics.
        4) Final fallback: infer from company name keywords.

        After enrichment, all sector values are normalized via sectors.yml (if provided).

        Expected columns:
        - ticker (required)
        - name (optional; helps Wikipedia/name-based fallback)
        - sector (optional)
        """
        import time
        from typing import Optional

        # Online sources toggled by local_only
        yf = None  # type: ignore
        requests = None  # type: ignore
        if not self.local_only:
            try:
                import yfinance as yf  # type: ignore
            except Exception as e:
                self.log.warning(
                    "yfinance import failed; skipping Yahoo Finance enrichment: %s", e
                )
                yf = None  # type: ignore
            try:
                import requests  # type: ignore
            except Exception as e:
                self.log.warning(
                    "requests import failed; skipping Wikipedia enrichment: %s", e
                )
                requests = None  # type: ignore
        try:
            import yaml  # type: ignore
        except Exception:
            yaml = None  # type: ignore

        out = df.copy()
        # Ensure expected columns
        if "ticker" not in out.columns:
            raise ValueError("enrich_sectors expects a 'ticker' column")
        if "name" not in out.columns:
            out["name"] = ""
        if "sector" not in out.columns:
            out["sector"] = "Unknown"

        # Determine which labels mean 'unknown' from sectors.yml if available
        unknown_labels = {"Unknown", None, ""}
        if self.sectors_yaml and yaml is not None:
            try:
                mapping = yaml.safe_load(Path(self.sectors_yaml).read_text()) or {}
                unknown_labels |= set(mapping.get("unknown_labels", []))
            except Exception:
                pass

        # Helpers (from reference enrich_sectors.py)
        def normalize_sector_heuristic(raw: Optional[str]) -> str:
            if not raw or not isinstance(raw, str):
                return "Unknown"
            key = raw.strip().lower()
            # Common fallbacks to GICS 11 for non-GICS labels
            local_map = {
                "communication services": "Communication Services",
                "consumer discretionary": "Consumer Discretionary",
                "consumer staples": "Consumer Staples",
                "energy": "Energy",
                "financials": "Financials",
                "health care": "Health Care",
                "industrials": "Industrials",
                "information technology": "Information Technology",
                "materials": "Materials",
                "real estate": "Real Estate",
                "utilities": "Utilities",
                # Yahoo common variants
                "technology": "Information Technology",
                "consumer cyclical": "Consumer Discretionary",
                "consumer cyclicals": "Consumer Discretionary",
                "consumer defensive": "Consumer Staples",
                "basic materials": "Materials",
                "financial services": "Financials",
                "healthcare": "Health Care",
            }
            return local_map.get(key, "Unknown")

        def infer_sector_from_name(name: Optional[str]) -> str:
            if not name or not isinstance(name, str):
                return "Unknown"
            n = name.lower()
            if any(w in n for w in ["bank", "bancorp", "financial", "trust"]):
                return "Financials"
            if any(w in n for w in ["oil", "petroleum", "energy", "gas"]):
                return "Energy"
            if any(w in n for w in ["pharma", "therapeutics", "health", "biotech"]):
                return "Health Care"
            if any(w in n for w in ["software", "technology", "tech", "semiconductor"]):
                return "Information Technology"
            if any(w in n for w in ["telecom", "telecommunications", "communication"]):
                return "Communication Services"
            if any(w in n for w in ["retail", "stores"]):
                return "Consumer Discretionary"
            if any(w in n for w in ["foods", "food", "beverage", "beverages"]):
                return "Consumer Staples"
            if any(w in n for w in ["mining", "chemical", "chemicals", "materials"]):
                return "Materials"
            if "utility" in n or "utilities" in n:
                return "Utilities"
            if any(w in n for w in ["realty", "reit", "property"]):
                return "Real Estate"
            if any(w in n for w in ["aerospace", "industrial", "machinery"]):
                return "Industrials"
            return "Unknown"

        # Wikipedia summary fetch (coarse hints)
        WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
        DEFAULT_HEADERS = {
            "User-Agent": (
                "MomentumV1.5/1.0 (https://github.com/zhouziqunzzq/portfolio-builder)"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "application/json,text/*;q=0.8,*/*;q=0.7",
            "Connection": "keep-alive",
        }

        session = None
        if requests is not None:
            try:
                session = requests.Session()
                session.headers.update(DEFAULT_HEADERS)
            except Exception:
                session = None

        def fetch_wikipedia_sector(company: str) -> Optional[str]:
            if not company or not isinstance(company, str) or session is None:
                return None
            url = WIKI_API.format(company.replace(" ", "_"))
            try:
                resp = session.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json() or {}
                extract = str(data.get("extract", "")).lower()
                if any(
                    w in extract for w in ["software", "technology", "semiconductor"]
                ):
                    return "Information Technology"
                if any(w in extract for w in ["bank", "financial"]):
                    return "Financials"
                if any(
                    w in extract
                    for w in ["pharmaceutical", "biotechnology", "healthcare"]
                ):
                    return "Health Care"
                if any(w in extract for w in ["oil", "gas", "energy"]):
                    return "Energy"
                if any(
                    w in extract
                    for w in ["telecommunications", "communication", "media"]
                ):
                    return "Communication Services"
                if "retail" in extract:
                    return "Consumer Discretionary"
                if any(w in extract for w in ["food", "beverage"]):
                    return "Consumer Staples"
                if any(w in extract for w in ["mining", "chemical", "materials"]):
                    return "Materials"
                if "utility" in extract:
                    return "Utilities"
                if any(w in extract for w in ["real estate", "reit"]):
                    return "Real Estate"
                if any(w in extract for w in ["aerospace", "industrial"]):
                    return "Industrials"
            except Exception:
                return None
            return None

        # Build symbol sets
        out["ticker"] = (
            out["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
        )
        out["sector"] = out["sector"].fillna("Unknown")

        is_unknown = out["sector"].isin(unknown_labels) | (
            out["sector"].astype(str).str.strip() == ""
        )
        unknown_syms = sorted(out.loc[is_unknown, "ticker"].dropna().unique().tolist())

        self.log.info(
            "Sector enrichment: %d tickers with unknown sector", len(unknown_syms)
        )

        enriched: dict[str, str] = {}

        # 1) Yahoo Finance sector
        if yf is not None:
            for sym in unknown_syms:
                try:
                    sec_raw = yf.Ticker(sym).info.get("sector")  # type: ignore[attr-defined]
                except Exception:
                    sec_raw = None
                if sec_raw:
                    # Keep raw first; we'll normalize after merging
                    enriched[sym] = str(sec_raw)
                time.sleep(0.25)

        # 2) Wikipedia summary (for remaining)
        remaining = [s for s in unknown_syms if s not in enriched]
        if remaining and session is not None:
            # Map symbol -> name for wiki probing
            name_by_symbol = (
                out[["ticker", "name"]]
                .drop_duplicates()
                .set_index("ticker")["name"]
                .to_dict()
            )
            for sym in remaining:
                nm = name_by_symbol.get(sym, "")
                if not nm:
                    continue
                sec_raw = fetch_wikipedia_sector(nm)
                if sec_raw:
                    enriched[sym] = sec_raw
                time.sleep(0.2)

        # 3) Fallback: simple name heuristics
        still = [s for s in unknown_syms if s not in enriched]
        if still:
            name_by_symbol = (
                out[["ticker", "name"]]
                .drop_duplicates()
                .set_index("ticker")["name"]
                .to_dict()
            )
            for sym in still:
                guess = infer_sector_from_name(name_by_symbol.get(sym, ""))
                if guess != "Unknown":
                    enriched[sym] = guess

        # Apply enrichment into the DataFrame
        def resolve_sector(row) -> str:
            cur = row.get("sector", "Unknown")
            if cur not in unknown_labels and str(cur).strip() != "":
                return str(cur)
            sym = row.get("ticker")
            val = enriched.get(sym, cur)
            return str(val) if val is not None else str(cur)

        out["sector"] = out.apply(resolve_sector, axis=1)

        # Normalize via sectors.yml (if provided)
        out = self.apply_unified_sector_mapping(out)

        # As a final safety, try local heuristic normalization for unknowns and non-canonical labels
        # Load canonical sector names from sectors.yml if available; otherwise fall back to GICS 11.
        canonical = set()
        try:
            if self.sectors_yaml and yaml is not None:
                mapping = yaml.safe_load(Path(self.sectors_yaml).read_text()) or {}
                canonical = set(mapping.get("sectors", {}).keys())
        except Exception:
            canonical = set()
        if not canonical:
            canonical = {
                "Communication Services",
                "Consumer Discretionary",
                "Consumer Staples",
                "Energy",
                "Financials",
                "Health Care",
                "Industrials",
                "Information Technology",
                "Materials",
                "Real Estate",
                "Utilities",
            }
        # First, fix unknowns
        mask_unknown = out["sector"].isin(unknown_labels) | (
            out["sector"].astype(str).str.strip() == ""
        )
        if mask_unknown.any():
            out.loc[mask_unknown, "sector"] = out.loc[mask_unknown, "sector"].map(
                normalize_sector_heuristic
            )
        # Then, fix non-canonical leftovers using heuristic
        mask_noncanon = ~out["sector"].isin(canonical)
        if mask_noncanon.any():
            out.loc[mask_noncanon, "sector"] = out.loc[mask_noncanon, "sector"].map(
                lambda s: normalize_sector_heuristic(s) or s
            )
        # One more pass through unified mapping to capture aliases
        out = self.apply_unified_sector_mapping(out)

        # Done
        unknown_after = int(
            (
                out["sector"].isin(unknown_labels)
                | (out["sector"].astype(str).str.strip() == "")
            ).sum()
        )
        self.log.info(
            "Sector enrichment complete: unknowns=%d / %d", unknown_after, len(out)
        )
        return out

    def apply_unified_sector_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize sectors using sectors.yml mapping.
        If no sectors_yaml is provided, returns df unchanged.
        """
        if not self.sectors_yaml:
            self.log.warning("No sectors.yml provided; skipping sector normalization")
            return df
        import yaml

        mapping = yaml.safe_load(Path(self.sectors_yaml).read_text()) or {}
        aliases = {}
        for canonical, entry in mapping.get("sectors", {}).items():
            aliases[canonical] = set([canonical] + list(entry.get("aliases", [])))

        def normalize(sector: str) -> str:
            # Treat NaN/None/empty as Unknown so subsequent filtering can work
            if pd.isna(sector) or sector is None or str(sector).strip() == "":
                return "Unknown"
            s = str(sector).strip()
            for canon, al in aliases.items():
                if s in al:
                    return canon
            return s

        out = df.copy()
        if "sector" in out.columns:
            out["sector"] = out["sector"].map(normalize)
        return out

    def filter_unknown_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.sectors_yaml:
            return df
        import yaml

        mapping = yaml.safe_load(Path(self.sectors_yaml).read_text()) or {}
        unknowns = set(mapping.get("unknown_labels", []))
        if "sector" in df.columns:
            return df[~df["sector"].isin(unknowns)].copy()
        return df

    # ---- Persistence ----
    def save_membership_csv(self, df: pd.DataFrame) -> None:
        self.membership_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.membership_csv, index=False)
        self.log.info("Saved membership CSV", extra={"path": str(self.membership_csv)})

    def load_from_membership_csv(self) -> pd.DataFrame:
        if self.cached_membership_df is not None:
            return self.cached_membership_df
        if not self.membership_csv.exists():
            raise FileNotFoundError(f"Membership CSV not found: {self.membership_csv}")
        df = pd.read_csv(self.membership_csv)
        self.log.info(
            "Loaded membership CSV",
            extra={"path": str(self.membership_csv), "rows": len(df)},
        )
        self.cached_membership_df = df
        return df

    # ---- Interfaces for consumers ----
    @property
    def tickers(self) -> List[str]:
        """Return the list of unique tickers in the membership CSV."""
        df = self.load_from_membership_csv()
        if "ticker" not in df.columns:
            raise ValueError("membership CSV must contain 'ticker' column")
        tickers = sorted(
            df["ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .str.replace(".", "-", regex=False)
            .unique()
            .tolist()
        )
        return tickers

    @property
    def sector_map(self) -> Optional[Dict[str, str]]:
        # Build sector map from membership CSV (last known per ticker)
        try:
            mem_df = self.load_from_membership_csv()
            if {"ticker", "sector"}.issubset(mem_df.columns):
                mem_df = (
                    mem_df[["ticker", "sector", "date_added"]].copy()
                    if "date_added" in mem_df.columns
                    else mem_df[["ticker", "sector"]].copy()
                )
                if "date_added" in mem_df.columns:
                    mem_df = mem_df.sort_values(
                        ["ticker", "date_added"]
                    ).drop_duplicates("ticker", keep="last")
                else:
                    mem_df = mem_df.drop_duplicates("ticker", keep="last")
                sector_map = {
                    row["ticker"].upper(): row["sector"] for _, row in mem_df.iterrows()
                }
            else:
                sector_map = None
        except Exception:
            sector_map = None
        return sector_map

    def membership_mask(
        self, start: datetime | str, end: datetime | str
    ) -> pd.DataFrame:
        """Return a mask DataFrame [date x ticker] indicating membership within [start, end].

        Supports two input schemas in membership CSV:
        1) Daily schema with columns [date, ticker, in_sp500]
        2) Range schema with columns [ticker, date_added, date_removed, ...]
           In this case, membership is active for dates >= date_added and
           <= date_removed (or indefinitely if date_removed is NaT).
        """
        df = self.load_from_membership_csv()

        start_dt = pd.to_datetime(start).normalize()
        end_dt = pd.to_datetime(end).normalize()

        # Daily schema path
        if {"date", "ticker", "in_sp500"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            mask = (
                df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
                .pivot_table(
                    index="date", columns="ticker", values="in_sp500", fill_value=0
                )
                .astype(bool)
            )
            return mask

        # Range schema path
        if {"ticker", "date_added"}.issubset(df.columns):
            work = df.copy()
            work["date_added"] = pd.to_datetime(work["date_added"]).dt.normalize()
            if "date_removed" in work.columns:
                work["date_removed"] = pd.to_datetime(
                    work["date_removed"]
                ).dt.normalize()
            else:
                work["date_removed"] = pd.NaT

            # Build events: +1 at added, -1 at removed + 1 day (to include removal date)
            ev_add = work[["ticker", "date_added"]].rename(
                columns={"date_added": "date"}
            )
            ev_add["delta"] = 1
            ev_rem = (
                work[["ticker", "date_removed"]]
                .dropna()
                .rename(columns={"date_removed": "date"})
            )
            if not ev_rem.empty:
                ev_rem["date"] = ev_rem["date"] + pd.Timedelta(days=1)
                ev_rem["delta"] = -1

            events = (
                pd.concat([ev_add, ev_rem], ignore_index=True)
                if not ev_rem.empty
                else ev_add
            )
            # Keep only events that can affect [start, end]
            right_bound = end_dt + pd.Timedelta(days=1)
            events = events[(events["date"] <= right_bound)]

            # Pivot to [date x ticker] deltas
            deltas = events.pivot_table(
                index="date",
                columns="ticker",
                values="delta",
                aggfunc="sum",
                fill_value=0,
            )

            # Compute initial active state at start_dt from ranges that began before start_dt
            # active at start if: date_added <= start_dt and (date_removed is NaT or date_removed >= start_dt)
            cond = (work["date_added"] <= start_dt) & (
                work["date_removed"].isna() | (work["date_removed"] >= start_dt)
            )
            initial_active = cond.groupby(work["ticker"]).any()
            # Convert to int (1 for active else 0)
            initial_active = initial_active.astype(int)

            # Reindex over requested date range and cumulatively sum
            days = pd.date_range(start_dt, end_dt, freq="D")
            deltas = deltas.reindex(days, fill_value=0).sort_index()
            active = deltas.cumsum().astype(int)
            # Add baseline initial activity across all dates
            # Align columns; missing tickers in deltas should also be considered (add zero column)
            initial_active = initial_active.reindex(active.columns, fill_value=0)
            active = active.add(initial_active, axis=1)
            mask = active > 0
            return mask

        raise ValueError(
            "membership CSV must contain either daily [date,ticker,in_sp500] or range [ticker,date_added(,date_removed)] schema"
        )

    def get_price_matrix(
        self,
        price_loader,
        start: datetime | str,
        end: datetime | str,
        tickers: Optional[List[str]] = None,
        field: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        auto_apply_membership_mask: bool = True,
        local_only: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Assemble a price matrix [date x ticker] using the provided loader.

        - price_loader may implement get_ohlcv(...) or load_ohlcv(...).
        - If `field` is None, prefer 'Adjclose' then 'Close'.
        - Index dates are timezone-naive and normalized to midnight.
        Args:
            price_loader: An object providing get_ohlcv(...) or load_ohlcv(...) method.
            tickers: List of ticker symbols to fetch.
            start: Start date (inclusive).
            end: End date (inclusive).
            field: Price field to extract (e.g., 'Adjclose', 'Close'); if None, auto-detect.
            interval: Data interval (e.g., '1d').
            auto_adjust: Whether to auto-adjust prices if supported.
            auto_apply_membership_mask: If True, apply membership mask so that prices that
                fall outside membership dates are set to NaN.
            local_only: If set, overrides instance local_only for this call.
        """

        def _fetch_one(sym: str) -> Optional[pd.Series]:
            try:
                if hasattr(price_loader, "get_ohlcv"):
                    # Pass through local_only if supported by the loader
                    kwargs = dict(
                        ticker=sym,
                        start=start,
                        end=end,
                        interval=interval,
                        auto_adjust=auto_adjust,
                    )
                    try:
                        if local_only is None:
                            # Fall back to instance default
                            kwargs["local_only"] = bool(self.local_only)
                        else:
                            kwargs["local_only"] = bool(local_only)
                    except Exception:
                        pass
                    ohlcv = price_loader.get_ohlcv(**kwargs)
                elif hasattr(price_loader, "load_ohlcv"):
                    ohlcv = price_loader.load_ohlcv(sym, start=start, end=end)
                else:
                    raise AttributeError(
                        "price_loader must provide get_ohlcv or load_ohlcv"
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
                self.log.warning(f"Failed loading price for {sym}: {e}")
                import traceback
                traceback.print_exc()
                return None

        if auto_apply_membership_mask:
            # Union of members active any time in window
            # Note: window includes non-trading days!
            mask = self.membership_mask(start=start, end=end)

        # Default to all tickers if none provided
        if tickers is None:
            tickers = self.tickers

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
            return out
        # Restrict to requested window and sort
        out = out.loc[pd.to_datetime(start) : pd.to_datetime(end)].sort_index()
        # Apply membership mask if requested
        if auto_apply_membership_mask:
            out = out.where(mask, other=pd.NA)
        return out
