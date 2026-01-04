from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Ensure package imports work when run from the repo root or the v2 folder
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from src.vec_signal_engine import VectorizedSignalEngine


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

    ticker = "SPY"
    start = "2025-01-01"
    end = "2025-11-30"
    interval = "1wk"

    last_price = se.get_series(
        ticker,
        "last_price",
        start,
        end,
        interval=interval,
        price_col="Close",
    )
    print(last_price)

    ret = se.get_series(
        ticker,
        "ret",
        start,
        end,
        interval=interval,
        window=1,
        price_col="Close",
        buffer_bars=10,
    )
    print(ret)

    log_ret = se.get_series(
        ticker,
        "log_ret",
        start,
        end,
        interval=interval,
        window=1,
        price_col="Close",
        buffer_bars=10,
    )
    print(log_ret)

    ts_mom = se.get_series(
        ticker,
        "ts_mom",
        start,
        end,
        interval=interval,
        window=1,
        price_col="Close",
        buffer_bars=10,
    )
    print(ts_mom)

    vol = se.get_series(
        ticker,
        "vol",
        start,
        end,
        interval=interval,
        window=2,
        price_col="Close",
        buffer_bars=10,
    )
    print(vol)

    adv = se.get_series(
        ticker,
        "adv",
        start,
        end,
        interval=interval,
        window=5,
        buffer_bars=10,
    )
    print(adv)

    median_volume = se.get_series(
        ticker,
        "median_volume",
        start,
        end,
        interval=interval,
        window=5,
        buffer_bars=10,
    )
    print(median_volume)

    spread_mom_self = se.get_series(
        ticker,
        "spread_mom",
        start,
        end,
        interval=interval,
        window=2,
        benchmark="SPY",
        buffer_bars=10,
    )
    print(
        spread_mom_self
    )  # Should be all zeros, as SPY is the benchmark and has no spread to itself

    spread_mom_other = se.get_series(
        ticker,
        "spread_mom",
        start,
        end,
        interval=interval,
        window=2,
        benchmark="GLD",
        buffer_bars=10,
    )
    print(spread_mom_other)


def test_vec_sig_engine():
    data_root = Path("data/prices")

    um = UniverseManager(
        membership_csv=Path("data/sp500_membership.csv"),
        sectors_yaml=Path("config/sectors.yml"),
    )
    mds = MarketDataStore(
        str(data_root), source="yfinance", local_only=False, use_memory_cache=True
    )
    vse = VectorizedSignalEngine(um, mds)

    tickers = ["SPY", "GLD"]
    start = "2025-01-01"
    end = "2025-11-30"
    interval = "1wk"

    price_mat = vse.get_price_matrix(
        tickers,
        start,
        end,
        interval=interval,
        membership_aware=False,
    )
    print(price_mat)

    ret_matrix = vse.get_returns(price_mat)
    print(ret_matrix)

    moms = vse.get_momentum(
        price_mat=price_mat,
        lookbacks=[1, 2, 3],
    )
    for lookback, mom in moms.items():
        print(f"Mom {lookback}:")
        print(mom)

    ts_moms = vse.get_ts_momentum(
        price_mat=price_mat,
        lookbacks=[2, 3],
    )
    for lookback, ts_mom in ts_moms.items():
        print(f"TS Mom {lookback}:")
        print(ts_mom)

    spread_moms = vse.get_spread_momentum(
        price_mat=price_mat,
        lookbacks=[1, 2],
        benchmark="SPY",
        interval=interval,
    )
    for lookback, spread_mom in spread_moms.items():
        print(f"Spread Mom {lookback}:")
        print(spread_mom)

    vol = vse.get_volatility(
        price_mat=price_mat,
        window=2,
        interval=interval,
    )
    print(vol)

    ewm_vol = vse.get_ewm_volatility(
        price_mat=price_mat,
        halflife=4,
        interval=interval,
    )
    print(ewm_vol)


if __name__ == "__main__":
    # test_mds()
    # test_sig_engine()
    test_vec_sig_engine()
