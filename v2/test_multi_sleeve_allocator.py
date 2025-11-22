"""
Real integration test for MultiSleeveAllocator using:

- MarketDataStore (yfinance)
- SignalEngine
- RegimeEngine
- DefensiveSleeve
- MultiSleeveAllocator (with score-based regime blending)

Run from v2 root:

    (.venv) python3 test_multi_sleeve_allocator.py
"""

from pathlib import Path
from datetime import datetime

import pandas as pd

from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from src.regime_engine import RegimeEngine
from src.universe_manager import UniverseManager
from src.sleeves.defensive.defensive_sleeve import DefensiveSleeve
from src.allocator.multi_sleeve_allocator import MultiSleeveAllocator, MultiSleeveConfig


def build_runtime():
    # ---- UniverseManager ----
    # Adjust membership_csv path if your file is named differently.
    membership_csv = Path("data/sp500_membership.csv")
    sectors_yaml = Path("config/sectors.yml")

    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,
        local_only=False,
    )

    # ---- MarketDataStore ----
    # Adjust data_root if needed
    mds = MarketDataStore(
        data_root="data/prices",
        source="yfinance",
        local_only=False,
    )

    # ---- SignalEngine ----
    signals = SignalEngine(mds)

    # ---- RegimeEngine ----
    # Adjust constructor args if your RegimeEngine signature differs
    regime_engine = RegimeEngine(
        signals=signals,
        config=None,  # use default config
    )

    # ---- Defensive Sleeve ----
    defensive = DefensiveSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        config=None,  # use default config
    )

    # ---- Multi-sleeve config & allocator ----
    # For now, only the defensive sleeve is wired. All regimes map to defensive: 1.0
    ms_config = MultiSleeveConfig()

    allocator = MultiSleeveAllocator(
        regime_engine=regime_engine,
        sleeves={"defensive": defensive},
        config=ms_config,
    )

    return allocator


def run_snapshot(allocator: MultiSleeveAllocator, as_of: str):
    as_of_ts = pd.to_datetime(as_of)

    weights = allocator.generate_global_target_weights(as_of_ts)

    print("\n" + "=" * 60)
    print(f"Global portfolio as of {as_of_ts.date()}")
    print("=" * 60)

    # Regime context
    primary = allocator.last_primary_regime
    scores = allocator.last_regime_scores or {}

    print(f"Primary regime: {primary}")
    if scores:
        print("Regime scores:")
        for r, s in scores.items():
            print(f"  {r:10s}: {s:.3f}")
    else:
        print("Regime scores: <none> (fallback to primary regime only)")

    # Weights
    if not weights:
        print("No weights returned (empty portfolio).")
        return

    print("\nTop weights:")
    for t, w in sorted(weights.items(), key=lambda x: -x[1])[:20]:
        print(f"  {t:6s} : {w:.4f}")

    print(f"\nSum of weights = {sum(weights.values()):.6f}")


def main():
    allocator = build_runtime()

    # Test a couple of dates that should hit different regimes / environments
    dates = [
        "2020-03-31",  # Covid crash / high vol
        "2022-06-30",  # 2022 bear market
        "2024-12-31",  # Recent-ish date
    ]

    for d in dates:
        run_snapshot(allocator, d)


if __name__ == "__main__":
    main()
