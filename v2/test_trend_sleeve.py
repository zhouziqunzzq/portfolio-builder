"""
Simple integration test for TrendSleeve (V2).

This test:
    - initializes UniverseManager / MarketDataStore / SignalEngine
    - instantiates TrendSleeve
    - computes target weights for a given date
"""

import pandas as pd
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
        "2025-11-14"
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


if __name__ == "__main__":
    test_trend_sleeve()
