"""
Simple integration test for TrendSleeve (V2).

This test:
    - initializes UniverseManager / MarketDataStore / SignalEngine
    - instantiates TrendSleeve
    - computes target weights for a given date
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine

from src.sleeves.trend.trend_sleeve import TrendSleeve
from src.sleeves.trend.trend_config import TrendConfig


def test_trend_sleeve():
    # -------------------------
    # 1) Universe setup
    # -------------------------
    membership_csv = Path("./data/sp500_membership.csv")
    sectors_yaml = Path("./config/sectors.yml")

    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,
        local_only=False,
    )

    # -------------------------
    # 2) Market data + signals
    # -------------------------
    mds = MarketDataStore(
        data_root=Path("./data/prices"),
        source="yfinance",
        local_only=True,
        use_memory_cache=True,
    )
    signals = SignalEngine(mds)

    # -------------------------
    # 3) Trend Sleeve
    # -------------------------
    cfg = TrendConfig()
    sleeve = TrendSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        config=cfg,
    )

    # -------------------------
    # 4) Test dates
    # -------------------------
    test_dates = [
        "2020-03-31",
        "2022-06-30",
        # "2024-12-31",
        "2025-11-14",
    ]

    for dt in test_dates:
        as_of = pd.Timestamp(dt)
        start_for_signals = as_of - pd.Timedelta(days=600)

        # Trend sleeve does not use regime directly, but we pass a dummy
        regime = "bull"

        weights = sleeve.generate_target_weights_for_date(
            as_of=as_of,
            start_for_signals=start_for_signals,
            regime=regime,
        )

        print("\n============================================================")
        print(f"Trend Sleeve - As of {as_of.date()}")
        print("============================================================")

        if not weights:
            print("No weights returned (empty universe or missing signals).")
            continue

        # Print top weights
        top = sorted(weights.items(), key=lambda x: -x[1])
        for t, w in top[:20]:  # print top 20
            print(f"{t:6s} : {w:.4f}")

        print(f"\nSum of weights = {sum(weights.values()):.6f}")

        # Run liquidity diagnostic
        df_liq = run_liquidity_diagnostic(
            as_of=as_of,
            start=start_for_signals,
            universe=sleeve._get_trend_universe(as_of),
            mds=mds,
            signals=signals,
            cfg=cfg,
        )
        print(df_liq.head())


def run_liquidity_diagnostic(
    as_of: pd.Timestamp,
    start: pd.Timestamp,
    universe: list[str],
    mds,
    signals,
    cfg,
):
    """
    Prints a liquidity diagnostic table for the specified universe.

    Columns:
        ADV20, MedianVol20, LastPrice, PassPrice, PassADV20, PassMedVol20, Eligible
    """

    rows = []
    for t in universe:
        try:
            # --- Last Price ---
            price_ser = signals.get_series(
                t,
                "ts_mom",  # any signal works; we only use underlying price series
                start=start,
                end=as_of,
                window=1,
            )
            if price_ser.empty:
                continue
            df = mds.get_ohlcv(t, start=start, end=as_of)
            if df.empty or "Close" not in df.columns:
                continue
            last_price = float(df["Close"].iloc[-1])

            # --- ADV20 ---
            df_prices = df
            if "Close" in df_prices.columns and "Volume" in df_prices.columns:
                dollar_vol = df_prices["Close"] * df_prices["Volume"]
                adv20 = dollar_vol.rolling(cfg.adv_window).mean().iloc[-1]
            else:
                adv20 = np.nan

            # --- Median Volume 20 ---
            if "Volume" in df_prices.columns:
                medvol20 = (
                    df_prices["Volume"]
                    .rolling(cfg.median_volume_window)
                    .median()
                    .iloc[-1]
                )
            else:
                medvol20 = np.nan

            # --- Pass/fail filters ---
            pass_price = last_price >= cfg.min_price
            pass_adv = (adv20 >= cfg.min_adv20) if pd.notna(adv20) else False
            pass_medv = (
                (medvol20 >= cfg.min_median_volume20) if pd.notna(medvol20) else False
            )

            eligible = pass_price and pass_adv and pass_medv

            rows.append(
                {
                    "ticker": t,
                    "last_price": last_price,
                    "adv20": adv20,
                    "median_vol20": medvol20,
                    "pass_price": pass_price,
                    "pass_adv20": pass_adv,
                    "pass_medvol20": pass_medv,
                    "eligible": eligible,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows).set_index("ticker")
    df = df.sort_values("eligible", ascending=False)

    # --- Summary ---
    total = len(df)
    passed = df["eligible"].sum()

    print("\n========================================================")
    print(f"Liquidity Diagnostic as of {as_of.date()} (Universe={total})")
    print("========================================================")
    print(f"Eligible: {passed}/{total} ({passed/total:.1%})\n")

    print("Minimums:")
    print(f"  Price: {cfg.min_price}")
    print(f"  ADV20: ${cfg.min_adv20:,.0f}")
    print(f"  MedianVol20: {cfg.min_median_volume20:,} shares\n")

    print("Top 20 failing examples:")
    fails = df[df["eligible"] == False].head(20)
    print(fails[["last_price", "adv20", "median_vol20"]])

    print("\nTop 20 passing examples:")
    passes = df[df["eligible"] == True].head(20)
    print(passes[["last_price", "adv20", "median_vol20"]])

    return df


if __name__ == "__main__":
    test_trend_sleeve()
