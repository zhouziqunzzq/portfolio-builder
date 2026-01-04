"""
Diagnostic script to compare vectorized vs non-vectorized backtest paths
on actual rebalance dates to identify where discrepancies occur.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from src.vec_signal_engine import VectorizedSignalEngine
from src.sleeves.trend.trend_sleeve import TrendSleeve
from src.sleeves.trend.trend_config import TrendConfig


def main():
    print("="*80)
    print("BACKTEST PATH COMPARISON: VEC VS NON-VEC")
    print("="*80)
    
    # Setup
    membership_csv = Path("./data/sp500_membership.csv")
    sectors_yaml = Path("./config/sectors.yml")
    
    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,
        local_only=True,
    )
    
    mds = MarketDataStore(
        data_root=Path("./data/prices"),
        source="yfinance",
        local_only=True,
        use_memory_cache=True,
    )
    
    signals = SignalEngine(mds)
    vec_engine = VectorizedSignalEngine(universe=um, mds=mds)
    
    cfg = TrendConfig()
    
    # Create two sleeves
    sleeve_non_vec = TrendSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        vec_engine=None,
        config=cfg,
    )
    
    sleeve_vec = TrendSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        vec_engine=vec_engine,
        config=cfg,
    )
    
    # Test rebalance dates (matching your backtest)
    rebalance_dates = pd.date_range(start="2025-06-01", end="2025-12-01", freq="MS")
    rebalance_dates = [pd.Timestamp(d) for d in rebalance_dates]
    
    print(f"\nRebalance dates: {[d.date() for d in rebalance_dates]}")
    
    # Run precompute for vectorized path
    print("\n" + "="*80)
    print("VECTORIZED PATH: Precompute")
    print("="*80)
    
    vec_weights = sleeve_vec.precompute(
        start="2025-06-01",
        end="2025-12-31",
        sample_dates=rebalance_dates,
    )
    
    print(f"\nPrecomputed weights shape: {vec_weights.shape}")
    print(f"Dates in precomputed weights: {vec_weights.index.tolist()}")
    
    # Run non-vec path sequentially
    print("\n" + "="*80)
    print("NON-VECTORIZED PATH: Sequential rebalances")
    print("="*80)
    
    non_vec_results = {}
    for rebal_date in rebalance_dates:
        print(f"\n--- Rebalancing on {rebal_date.date()} ---")
        
        # Call sleeve (non-vectorized path)
        weights = sleeve_non_vec(as_of=rebal_date, cash=100000.0, regime=None)
        
        # Store results
        non_vec_results[rebal_date] = weights
        
        print(f"Allocated {len(weights)} stocks")
        print(f"Total weight: {sum(weights.values()):.6f}")
        top_5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5: {top_5}")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    for rebal_date in rebalance_dates:
        if rebal_date not in vec_weights.index:
            print(f"\n[{rebal_date.date()}] NOT IN VEC WEIGHTS")
            continue
        
        vec_w = vec_weights.loc[rebal_date]
        vec_w = vec_w[vec_w > 0]
        
        non_vec_w = non_vec_results.get(rebal_date, {})
        non_vec_w_series = pd.Series(non_vec_w)
        
        print(f"\n--- {rebal_date.date()} ---")
        print(f"Vec: {len(vec_w)} stocks, sum={vec_w.sum():.6f}")
        print(f"Non-vec: {len(non_vec_w_series)} stocks, sum={non_vec_w_series.sum():.6f}")
        
        # Find tickers in both
        common = set(vec_w.index) & set(non_vec_w_series.index)
        only_vec = set(vec_w.index) - set(non_vec_w_series.index)
        only_non_vec = set(non_vec_w_series.index) - set(vec_w.index)
        
        print(f"Common tickers: {len(common)}")
        if only_vec:
            print(f"Only in vec: {sorted(only_vec)}")
        if only_non_vec:
            print(f"Only in non-vec: {sorted(only_non_vec)}")
        
        if common:
            # Compare weights
            comparison = pd.DataFrame({
                'vec': vec_w.loc[list(common)],
                'non_vec': non_vec_w_series.loc[list(common)],
            })
            comparison['diff'] = comparison['vec'] - comparison['non_vec']
            comparison['abs_diff'] = comparison['diff'].abs()
            comparison = comparison.sort_values('abs_diff', ascending=False)
            
            print(f"\nMax abs diff: {comparison['abs_diff'].max():.6e}")
            print(f"Mean abs diff: {comparison['abs_diff'].mean():.6e}")
            
            if comparison['abs_diff'].max() > 0.001:
                print(f"\n⚠️ SIGNIFICANT DIFFERENCES (>0.001):")
                print(comparison.head(5).to_string())


if __name__ == "__main__":
    main()
