#!/usr/bin/env python3
import io
import sys
import requests
import pandas as pd
from pathlib import Path

# Where to save the final membership file
OUT_PATH = Path("data/universe/sp500_membership_history.csv")

# Raw GitHub CSV with first/last membership dates
TICKER_START_END_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)

# Your existing static constituents file (from get_SP500.py)
CURRENT_CONST_PATH = Path("data/universe/sp500_constituents.csv")


def to_yf_symbol(sym: str) -> str:
    """
    Normalize a ticker to yfinance-compatible form.
    Same idea as your get_SP500.py: 'BRK.B' -> 'BRK-B', etc.
    """
    s = str(sym).strip().upper()
    s = s.replace(".", "-")
    return s


def main():
    # --- 1) Download historical ticker start/end from GitHub ---
    print(f"Downloading historical S&P membership from:\n  {TICKER_START_END_URL}")
    resp = requests.get(TICKER_START_END_URL)
    resp.raise_for_status()

    # sp500_ticker_start_end.csv format (from repo docs):
    # typically something like: Symbol,StartDate,EndDate
    # You may want to `print(df.head())` once to confirm column names.
    df_hist = pd.read_csv(io.StringIO(resp.text))

    # Try to detect column names robustly
    cols = {c.lower(): c for c in df_hist.columns}
    # Common names in that repo: "Ticker", "Start", "End" or similar
    symbol_col = cols.get("ticker", None)
    start_col = cols.get("start_date", cols.get("start", None))
    end_col = cols.get("end_date", cols.get("end", None))

    if not symbol_col or not start_col:
        print("Could not infer column names from sp500_ticker_start_end.csv", file=sys.stderr)
        print("Columns found:", list(df_hist.columns), file=sys.stderr)
        sys.exit(1)

    # If there is no explicit end_col, create one (still active)
    if end_col is None:
        df_hist["EndDate"] = pd.NaT
        end_col = "EndDate"

    # Normalize and rename
    df_hist["Symbol"] = df_hist[symbol_col].apply(to_yf_symbol)
    df_hist["DateAdded"] = pd.to_datetime(df_hist[start_col]).dt.normalize()
    df_hist["DateRemoved"] = pd.to_datetime(df_hist[end_col]).dt.normalize()

    df_hist = df_hist[["Symbol", "DateAdded", "DateRemoved"]]

    print("Historical membership sample:")
    print(df_hist.head())

    # --- 2) Load current constituents for GICS Sector / Name (optional) ---
    if not CURRENT_CONST_PATH.exists():
        print(
            f"WARNING: {CURRENT_CONST_PATH} not found. "
            "Sectors will be set to 'Unknown'.",
            file=sys.stderr,
        )
        df_sector = pd.DataFrame(columns=["Symbol", "Name", "GICS Sector"])
    else:
        df_sector = pd.read_csv(CURRENT_CONST_PATH)
        # Expecting columns from your get_SP500.py: Symbol, Name, GICS Sector, GICS Sub-Industry
        # Make sure Symbol is normalized the same way
        if "Symbol" not in df_sector.columns:
            raise ValueError(f"'Symbol' column not found in {CURRENT_CONST_PATH}")
        df_sector["Symbol"] = df_sector["Symbol"].apply(to_yf_symbol)

        # Name may or may not exist; default if missing
        if "Name" not in df_sector.columns:
            df_sector["Name"] = ""

        if "GICS Sector" not in df_sector.columns:
            df_sector["GICS Sector"] = "Unknown"

        df_sector = df_sector[["Symbol", "Name", "GICS Sector"]]

    # --- 3) Merge sectors into the historical membership ---
    df = df_hist.merge(df_sector, on="Symbol", how="left")

    # Fill missing sectors and names where we don't have them (e.g. delisted firms)
    df["Name"] = df["Name"].fillna("")
    df["GICS Sector"] = df["GICS Sector"].fillna("Unknown")

    # Optional: sort by DateAdded then Symbol
    df = df.sort_values(["DateAdded", "Symbol"]).reset_index(drop=True)

    # --- 4) Save to CSV ---
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nDone. Wrote membership CSV to:\n  {OUT_PATH}")
    print("Preview:")
    print(df.head(10))


if __name__ == "__main__":
    main()
