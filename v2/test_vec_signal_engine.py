"""
Quick sanity tests for VectorizedSignalEngine.

Run from v2 root:

    (.venv) python3 test_vec_signal_engine.py

This will:

  - Build a small price matrix for a few tickers (AAPL/MSFT/GLD)
  - Build an SP500-only membership-aware price matrix
  - Compute vectorized returns, momentum, and volatility
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.vec_signal_engine import VectorizedSignalEngine


def build_runtime():
    # ---- UniverseManager ----
    membership_csv = Path("data/sp500_membership.csv")
    sectors_yaml = Path("config/sectors.yml")

    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,
        local_only=False,
    )

    # ---- MarketDataStore ----
    mds = MarketDataStore(
        data_root="data/prices",
        source="yfinance",
        local_only=True,  # assume prices are cached; avoid network
        use_memory_cache=True,  # if you added this knob
    )

    vse = VectorizedSignalEngine(universe=um, mds=mds)
    return um, mds, vse


def test_basic_price_matrix(vse: VectorizedSignalEngine):
    print("=" * 60)
    print(" BASIC PRICE MATRIX TEST ")
    print("=" * 60)

    tickers = ["AAPL", "MSFT", "GLD"]
    start = "2020-01-01"
    end = "2020-03-31"

    pm = vse.get_price_matrix(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        local_only=True,
        auto_adjust=True,
    )

    print(f"Price matrix shape: {pm.shape}")
    print(pm.head())
    print(pm.tail())


def test_sp500_price_matrix(vse: VectorizedSignalEngine):
    print("\n" + "=" * 60)
    print(" SP500-ONLY PRICE MATRIX TEST ")
    print("=" * 60)

    start = "2020-01-01"
    end = "2020-03-31"

    pm_sp = vse.get_price_matrix_sp500(
        start=start,
        end=end,
        interval="1d",
        local_only=True,
        auto_adjust=True,
    )

    print(f"SP500 matrix shape: {pm_sp.shape}")
    if not pm_sp.empty:
        print("Example rows:")
        print(pm_sp.iloc[:5, :5])

        nan_summary = pm_sp.isna().sum()
        nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)
        print("\nNaN summary (top 5):")
        print(nan_summary.head())


def test_vectorized_signals(vse: VectorizedSignalEngine):
    print("\n" + "=" * 60)
    print(" VECTORIZED SIGNALS TEST (returns / momentum / vol) ")
    print("=" * 60)

    tickers = ["AAPL", "MSFT", "GLD"]
    start = "2020-01-01"
    end = "2020-03-31"

    price_mat = vse.get_price_matrix(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        local_only=True,
        auto_adjust=True,
    )

    if price_mat.empty:
        print("Price matrix is empty; cannot test signals.")
        return

    # ---- Returns ----
    rets = vse.get_returns(price_mat)
    print("\nReturns (head):")
    print(rets.head())
    print("\nReturns (tail):")
    print(rets.tail())

    # ---- Momentum ----
    lookbacks = [21, 63]
    mom_dict = vse.get_momentum(price_mat, lookbacks=lookbacks)
    for w, mom in mom_dict.items():
        print(f"\nMomentum w={w} (tail):")
        print(mom.tail())

    # ---- Volatility ----
    vol_mat = vse.get_volatility(price_mat, window=20, annualize=True)
    print("\nVolatility (tail):")
    print(vol_mat.tail())

    # Simple sanity stats
    print("\nBasic stats:")
    print(f"  Returns mean (per-ticker):\n{rets.mean()}")
    print(f"\n  Vol (per-ticker, last date):\n{vol_mat.iloc[-1]}")


def main():
    _, _, vse = build_runtime()

    test_basic_price_matrix(vse)
    test_sp500_price_matrix(vse)
    test_vectorized_signals(vse)


if __name__ == "__main__":
    main()
