from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Ensure package imports work when run from the repo root or the v2 folder
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine


def test_mds():
    data_root = Path("data/prices")

    mds = MarketDataStore(
        str(data_root), source="yfinance", local_only=False, use_memory_cache=True
    )

    tickers = ["AAPL", "MSFT", "GOOG"]
    start = "2025-10-01"
    end = "2025-11-30"

    for t in tickers:
        print(f"\nFetching daily and weekly bars for {t} from {start} to {end}")
        try:
            df_daily = mds.get_ohlcv(
                t, start, end, interval="1d", auto_adjust=True, local_only=False
            )
            df_weekly = mds.get_ohlcv(
                t, start, end, interval="1wk", auto_adjust=True, local_only=False
            )
            df_weekly_orig = df_weekly.copy() if not df_weekly.empty else df_weekly

            if df_daily.empty and df_weekly.empty:
                print(f"No data returned for {t} (both daily and weekly empty)")
                continue

            if not df_daily.empty:
                # Resample daily close to weekly (week ending Friday) to compare
                try:
                    daily_close = df_daily["Close"].resample("W-FRI").last()
                except Exception:
                    daily_close = df_daily["Close"].resample("W").last()
            else:
                daily_close = pd.Series(dtype="float64")

            if not df_weekly.empty:
                weekly_close = df_weekly["Close"].copy()
                # Reindex weekly bars so the timestamp falls on the Friday of the same week
                idx = weekly_close.index
                days_to_friday = (4 - idx.weekday) % 7
                weekly_close.index = idx + pd.to_timedelta(days_to_friday, unit="d")
                weekly_close = weekly_close.rename("Weekly_Close")
            else:
                weekly_close = pd.Series(dtype="float64")

            comp = pd.concat([daily_close.rename("Daily_Close"), weekly_close], axis=1)

            # Align and show first few rows and differences
            comp = comp.sort_index()
            comp["Diff"] = comp["Daily_Close"] - comp["Weekly_Close"]

            print(f"{t}: comparison (resampled daily vs weekly) — {len(comp)} rows")
            print(comp.head(10).to_string())

            # Debug: inspect specific Friday (2025-11-21)
            target = pd.Timestamp("2025-11-21")
            if target in comp.index:
                print(f"\n--- Debug for {t} week ending {target.date()} ---")
                week_start = target - pd.Timedelta(days=6)
                print("Daily rows for the week:")
                try:
                    print(df_daily.loc[week_start:target].to_string())
                except Exception:
                    print("(no daily rows found for week)")

                print("\nOriginal weekly bars (as returned by yfinance):")
                try:
                    print(df_weekly_orig.to_string())
                except Exception:
                    print("(no original weekly bars)")

                print("\nShifted weekly bar used for comparison:")
                try:
                    print(weekly_close.loc[[target]].to_string())
                except Exception:
                    print("(no shifted weekly bar at target)")

        except Exception as e:
            print(f"Error fetching {t}: {e}")


def test_sig_engine():
    data_root = Path("data/prices")

    mds = MarketDataStore(
        str(data_root), source="yfinance", local_only=False, use_memory_cache=True
    )
    se = SignalEngine(mds)

    ticker = "QQQ"
    start = "2025-01-01"
    end = "2025-11-30"

    last_price = se.get_series(
        ticker,
        "last_price",
        start,
        end,
        interval="1wk",
        price_col="Close",
    )
    print(last_price)


if __name__ == "__main__":
    # test_mds()
    test_sig_engine()
