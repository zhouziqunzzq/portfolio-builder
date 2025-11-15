#!/usr/bin/env python3
import requests
import pandas as pd
import yfinance as yf
import time
from pathlib import Path
from typing import Optional

# Use a shared Session with a realistic browser User-Agent to reduce HTTP 403
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
}

SESSION = requests.Session()
SESSION.headers.update(DEFAULT_HEADERS)

MEMBERSHIP_PATH = Path("data/universe/sp500_membership_history.csv")
OUT_PATH = Path("data/universe/sp500_membership_history_enriched.csv")

# ---------- Unified sector mapping (to GICS 11) ----------

SECTOR_NORMALIZATION_MAP = {
    # GICS standard (already normalized)
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

    # Yahoo Finance -> /GICS
    "communication services (yf)": "Communication Services",
    "communication services ": "Communication Services",
    "communication services": "Communication Services",
    "technology": "Information Technology",
    "consumer cyclical": "Consumer Discretionary",
    "consumer cyclicals": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "basic materials": "Materials",
    "financial services": "Financials",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "industrials (yf)": "Industrials",
    "energy (yf)": "Energy",
    "real estate (yf)": "Real Estate",
    "utilities (yf)": "Utilities",

    # Some common generic labels -> GICS
    "software": "Information Technology",
    "semiconductors": "Information Technology",
    "semiconductor": "Information Technology",
    "internet": "Communication Services",
    "media": "Communication Services",
    "insurance": "Financials",
    "banks": "Financials",
    "banking": "Financials",
    "oil & gas": "Energy",
    "oil and gas": "Energy",
    "oil & gas e&p": "Energy",
    "aerospace": "Industrials",
    "aerospace & defense": "Industrials",
    "pharmaceuticals": "Health Care",
    "pharmaceutical": "Health Care",
    "biotechnology": "Health Care",
    "biotech": "Health Care",
    "retail": "Consumer Discretionary",
    "food": "Consumer Staples",
    "beverage": "Consumer Staples",
    "beverages": "Consumer Staples",
    "chemicals": "Materials",
    "packaging": "Materials",
    "mining": "Materials",
    "construction": "Industrials",
    "transportation": "Industrials",

    # Finviz-ish / misc
    "consumer goods": "Consumer Staples",
    "conglomerates": "Industrials",
    "services": "Consumer Discretionary",
}


def normalize_sector(raw: str) -> str:
    """Map various sector strings into GICS 11 names."""
    if not raw or not isinstance(raw, str):
        return "Unknown"
    key = raw.strip().lower()
    return SECTOR_NORMALIZATION_MAP.get(key, "Unknown")


def infer_sector_from_name(name: str) -> str:
    """Heuristic fallback from company name when sector unknown."""
    if not name or not isinstance(name, str):
        return "Unknown"
    n = name.lower()

    if "bank" in n or "bancorp" in n or "financial" in n or "trust" in n:
        return "Financials"
    if "oil" in n or "petroleum" in n or "energy" in n or "gas" in n:
        return "Energy"
    if "pharma" in n or "therapeutics" in n or "health" in n or "biotech" in n:
        return "Health Care"
    if "software" in n or "technology" in n or "tech" in n or "semiconductor" in n:
        return "Information Technology"
    if "telecom" in n or "telecommunications" in n or "communication" in n:
        return "Communication Services"
    if "retail" in n or "stores" in n:
        return "Consumer Discretionary"
    if "foods" in n or "food" in n or "beverage" in n or "beverages" in n:
        return "Consumer Staples"
    if "mining" in n or "chemical" in n or "chemicals" in n or "materials" in n:
        return "Materials"
    if "utility" in n or "utilities" in n:
        return "Utilities"
    if "realty" in n or "reit" in n or "property" in n:
        return "Real Estate"
    if "aerospace" in n or "industrial" in n or "machinery" in n:
        return "Industrials"

    return "Unknown"


# ---------- External data fetch helpers ----------

def fetch_yf_sector(ticker):
    try:
        info = yf.Ticker(ticker).info
        sec = info.get("sector")
        if sec:
            return sec
    except Exception:
        pass
    return None


# Simple Wikipedia summary-based heuristic (optional, coarse)
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


def _get_json_with_retries(url: str, retries: int = 3, backoff: float = 0.5) -> Optional[dict]:
    """GET JSON using the shared SESSION with a simple retry/backoff loop."""
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries:
                return None
            time.sleep(backoff * attempt)
    return None


def fetch_wikipedia_sector(company: str) -> Optional[str]:
    try:
        # Try direct page via REST summary API
        url = WIKI_API.format(company.replace(" ", "_"))
        data = _get_json_with_retries(url)
        if not data:
            return None
        extract = data.get("extract", "").lower()
        # Very coarse text-based hints
        if "software" in extract or "technology" in extract or "semiconductor" in extract:
            return "Information Technology"
        if "bank" in extract or "financial" in extract:
            return "Financials"
        if "pharmaceutical" in extract or "biotechnology" in extract or "healthcare" in extract:
            return "Health Care"
        if "oil" in extract or "gas" in extract or "energy" in extract:
            return "Energy"
        if "telecommunications" in extract or "communication" in extract or "media" in extract:
            return "Communication Services"
        if "retail" in extract:
            return "Consumer Discretionary"
        if "food" in extract or "beverage" in extract:
            return "Consumer Staples"
        if "mining" in extract or "chemical" in extract or "materials" in extract:
            return "Materials"
        if "utility" in extract:
            return "Utilities"
        if "real estate" in extract or "reit" in extract:
            return "Real Estate"
        if "aerospace" in extract or "industrial" in extract:
            return "Industrials"
    except Exception:
        pass
    return None


def main():
    df = pd.read_csv(MEMBERSHIP_PATH)

    # Ensure basic columns
    if "GICS Sector" not in df.columns:
        df["GICS Sector"] = "Unknown"
    if "Name" not in df.columns:
        df["Name"] = ""

    df["Symbol"] = df["Symbol"].str.upper()
    df["GICS Sector"] = df["GICS Sector"].fillna("Unknown")

    unknown_syms = df[df["GICS Sector"] == "Unknown"]["Symbol"].unique()
    print(f"Tickers with Unknown sector before enrichment: {len(unknown_syms)}")

    # Map from symbol -> enriched sector
    enriched_sector = {}

    # 1) Try Yahoo Finance
    for sym in unknown_syms:
        print(f"[YF] {sym} ...", end="")
        sec_raw = fetch_yf_sector(sym)
        if sec_raw:
            sec_norm = normalize_sector(sec_raw)
            if sec_norm != "Unknown":
                print(f" -> {sec_raw!r} -> {sec_norm}")
                enriched_sector[sym] = sec_norm
            else:
                print(f" -> {sec_raw!r} (unmapped)")
        else:
            print(" -> none")
        time.sleep(0.4)  # be polite to YF

    remaining = [s for s in unknown_syms if s not in enriched_sector]
    print(f"Remaining Unknowns after YF: {len(remaining)}")

    # 2) Try Wikipedia (using company Name from membership where possible)
    # Build a quick map from Symbol -> Name
    name_by_symbol = (
        df[["Symbol", "Name"]]
        .drop_duplicates()
        .set_index("Symbol")["Name"]
        .to_dict()
    )

    for sym in remaining:
        company_name = name_by_symbol.get(sym, "")
        if not company_name:
            continue

        print(f"[WIKI] {sym} ({company_name}) ...", end="")
        sec_raw = fetch_wikipedia_sector(company_name)
        if sec_raw:
            sec_norm = normalize_sector(sec_raw)
            if sec_norm == "Unknown":
                # Wikipedia function already returns something close to GICS,
                # so if normalize fails, just trust raw mapping.
                sec_norm = sec_raw
            print(f" -> {sec_raw!r} -> {sec_norm}")
            enriched_sector[sym] = sec_norm
        else:
            print(" -> none")
        time.sleep(0.3)

    # 3) Final fallback from company name
    still_remaining = [s for s in unknown_syms if s not in enriched_sector]
    print(f"Remaining Unknowns after WIKI: {len(still_remaining)}")

    for sym in still_remaining:
        company_name = name_by_symbol.get(sym, "")
        if not company_name:
            continue
        sec_guess = infer_sector_from_name(company_name)
        if sec_guess != "Unknown":
            print(f"[NAME] {sym} ({company_name}) -> {sec_guess}")
            enriched_sector[sym] = sec_guess

    # Apply enrichment
    def resolve_sector(row):
        sym = row["Symbol"]
        current = row["GICS Sector"]
        if current != "Unknown" and isinstance(current, str) and current.strip():
            # normalize existing sectors too
            norm = normalize_sector(current)
            return norm if norm != "Unknown" else current
        # else use enriched map
        return enriched_sector.get(sym, current)

    df["GICS Sector"] = df.apply(resolve_sector, axis=1)

    # Save enriched file
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved enriched membership CSV to: {OUT_PATH}")

    # Quick summary
    unknown_after = (df["GICS Sector"] == "Unknown").sum()
    print(f"Rows with Unknown sector AFTER enrichment: {unknown_after} / {len(df)}")


if __name__ == "__main__":
    main()
