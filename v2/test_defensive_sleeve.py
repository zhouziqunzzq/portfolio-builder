"""
Simple integration test for DefensiveSleeve.

This test:
    - initializes UniverseManager / MarketDataStore / SignalEngine
    - instantiates DefensiveSleeve
    - computes target weights for a given date
"""

import pandas as pd
from pathlib import Path

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from src.sleeves.defensive.defensive_sleeve import DefensiveSleeve
from src.sleeves.defensive.defensive_config import DefensiveConfig


def test_defensive_sleeve():
    # -------------------------
    # 1) Prepare universe
    # -------------------------
    # Use your existing S&P500 membership CSV
    # e.g. ./data/sp500_membership.csv
    membership_csv = Path("./data/sp500_membership.csv")
    sectors_yaml = Path("./config/sectors.yml")  # optional

    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,  # or point to your sectors.yml
        local_only=True,
    )

    # -------------------------
    # 2) Market data + signals
    # -------------------------
    mds = MarketDataStore(
        data_root=Path("./data/prices"),
        source="yfinance",
        local_only=True,
    )

    signals = SignalEngine(mds)

    # -------------------------
    # 3) Defensive Sleeve
    # -------------------------
    cfg = DefensiveConfig()
    sleeve = DefensiveSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        config=cfg,
    )

    # -------------------------
    # 4) Run test date
    # -------------------------
    as_of = pd.Timestamp("2018-08-31")
    # as_of = pd.Timestamp("2022-06-30")
    start_for_signals = as_of - pd.Timedelta(days=500)  # enough history

    # Assume regime is "bull" (or "sideways" / "bear" depending on your config)
    # regime = "bull"
    regime = "bear"

    weights = sleeve.generate_target_weights_for_date(
        as_of=as_of,
        start_for_signals=start_for_signals,
        regime=regime,
    )

    print("\n=== Defensive Sleeve Weights ===")
    print(f"As-of date: {as_of.date()}, Regime: {regime}")
    for t, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"{t:6s} : {w:.4f}")

    print(f"\nSum of weights = {sum(weights.values()):.6f}")


if __name__ == "__main__":
    test_defensive_sleeve()
