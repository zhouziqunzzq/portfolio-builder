"""
Real integration test for MultiSleeveAllocator with both:

    - TrendSleeve (NEW)
    - DefensiveSleeve

The run includes:
    - MarketDataStore (yfinance w/ memory cache)
    - SignalEngine
    - RegimeEngine
    - MultiSleeveAllocator (score-blending)
"""

from pathlib import Path
from datetime import datetime

import pandas as pd

from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from src.regime_engine import RegimeEngine
from src.universe_manager import UniverseManager

from src.sleeves.trend.trend_sleeve import TrendSleeve
from src.sleeves.defensive.defensive_sleeve import DefensiveSleeve

from src.allocator.multi_sleeve_allocator import MultiSleeveAllocator
from src.allocator.multi_sleeve_config import MultiSleeveConfig


# --------------------------------------------------------------------
# Build runtime context
# --------------------------------------------------------------------
def build_runtime():
    # ==== Universe ====
    membership_csv = Path("data/sp500_membership.csv")
    sectors_yaml = Path("config/sectors.yml")

    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,
        local_only=False,
    )

    # ==== Market Data ====
    mds = MarketDataStore(
        data_root="data/prices",
        source="yfinance",
        local_only=True,  # Set to False if you want live fetches
        use_memory_cache=True,  # IMPORTANT: avoid repeated disk loads
    )

    # ==== Signals ====
    signals = SignalEngine(mds)

    # ==== Regime Engine ====
    regime_engine = RegimeEngine(
        signals=signals,
        config=None,
    )

    # ==== Sleeves ====
    defensive = DefensiveSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        config=None,
    )

    trend = TrendSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        config=None,
    )

    # ==== Multi-sleeve allocator ====
    ms_config = MultiSleeveConfig()

    allocator = MultiSleeveAllocator(
        regime_engine=regime_engine,
        sleeves={
            "trend": trend,
            "defensive": defensive,
        },
        config=ms_config,
    )

    return allocator


# --------------------------------------------------------------------
# Run snapshot
# --------------------------------------------------------------------
def run_snapshot(allocator: MultiSleeveAllocator, as_of: str):
    as_of_ts = pd.to_datetime(as_of)
    weights, ctx = allocator.generate_global_target_weights(as_of_ts)

    print("\n" + "=" * 60)
    print(f"Global portfolio as of {as_of_ts.date()}")
    print("=" * 60)

    primary = allocator.last_primary_regime
    scores = allocator.last_regime_scores or {}

    print(f"Primary regime: {primary}")
    if scores:
        print("Regime scores:")
        for r, s in scores.items():
            print(f"  {r:10s}: {s:.3f}")
    else:
        print("Regime scores: <none> (fallback to primary regime only)")

    print("\nTop 20 weights:")
    for t, w in sorted(weights.items(), key=lambda x: -x[1])[:20]:
        print(f"  {t:6s} : {w:.4f}")

    print(f"\nSum of weights = {sum(weights.values()):.6f}")


# --------------------------------------------------------------------
def main():
    allocator = build_runtime()

    dates = [
        "2020-03-31",
        "2022-06-30",
        "2025-11-14",
    ]

    for d in dates:
        run_snapshot(allocator, d)


if __name__ == "__main__":
    main()
