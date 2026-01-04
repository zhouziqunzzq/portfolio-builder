"""
Simple comparison of vec vs non-vec for a single rebalance date.
"""
import pandas as pd
from pathlib import Path

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from src.vec_signal_engine import VectorizedSignalEngine
from src.sleeves.trend.trend_sleeve import TrendSleeve
from src.sleeves.trend.trend_config import TrendConfig


def main():
    print("="*80)
    print("SINGLE DATE COMPARISON: FIRST REBALANCE")
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
    
    # Test first rebalance date
    test_date = pd.Timestamp("2025-06-02")  # First rebalance from your backtest
    
    print("\n" + "="*80)
    print(f"NON-VECTORIZED PATH: {test_date.date()}")
    print("="*80)
    
    # Call the non-vec path
    start_for_signals = test_date - pd.Timedelta(days=600)
    
    # Patch to capture stock scores from non-vec path
    orig_compute_stock_scores = sleeve_non_vec._compute_stock_scores
    non_vec_scored = None
    def capture_scored(sigs):
        nonlocal non_vec_scored
        result = orig_compute_stock_scores(sigs)
        non_vec_scored = result
        return result
    sleeve_non_vec._compute_stock_scores = capture_scored
    
    weights_non_vec = sleeve_non_vec.generate_target_weights_for_date(
        as_of=test_date,
        start_for_signals=start_for_signals,
        regime="bull"
    )
    
    # Show Communication Services stock scores from non-vec path
    if non_vec_scored is not None:
        cs_tickers_nv = [t for t, s in um.sector_map.items() if s == "Communication Services" and t in non_vec_scored.index]
        if cs_tickers_nv:
            print(f"\n[Non-Vec Path] Communication Services stocks ({len(cs_tickers_nv)}):")
            print(f"  Mean: {non_vec_scored.loc[cs_tickers_nv, 'stock_score'].mean()}")
            print(f"  Tickers: {sorted(cs_tickers_nv)}")
    
    print(f"\nNon-vec allocated {len(weights_non_vec)} stocks")
    print(f"Total weight: {sum(weights_non_vec.values()):.6f}")
    sorted_weights = sorted(weights_non_vec.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_weights:
        print(f"  {ticker}: {weight:.4f}")
    
    print("\n" + "="*80)
    print(f"VECTORIZED PATH: {test_date.date()}")
    print("="*80)
    
    # Run precompute for just this date
    vec_weights_mat = sleeve_vec.precompute(
        start="2025-06-02",
        end="2025-06-02",
        sample_dates=[test_date],
    )
    
    if test_date in vec_weights_mat.index:
        vec_weights_series = vec_weights_mat.loc[test_date]
        vec_weights = {t: float(w) for t, w in vec_weights_series.items() if w > 0}
        
        print(f"\nVec allocated {len(vec_weights)} stocks")
        print(f"Total weight: {sum(vec_weights.values()):.6f}")
        sorted_weights = sorted(vec_weights.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_weights:
            print(f"  {ticker}: {weight:.4f}")
        
        # Try to access cached stock scores from vec path
        if hasattr(sleeve_vec, '_cs_stock_score_mat') and sleeve_vec._cs_stock_score_mat is not None:
            vec_stock_scores = sleeve_vec._cs_stock_score_mat
            if test_date in vec_stock_scores.index:
                vec_scores_at_date = vec_stock_scores.loc[test_date].dropna()
                print(f"\n[Vec Path] Found {len(vec_scores_at_date)} stocks with CS scores at {test_date.date()}")
                # Show Communication Services stocks
                cs_tickers = [t for t, s in um.sector_map.items() if s == "Communication Services" and t in vec_scores_at_date.index]
                if cs_tickers:
                    print(f"\nCommunication Services stocks in Vec path ({len(cs_tickers)}):")
                    cs_scores = [(t, vec_scores_at_date[t]) for t in cs_tickers]
                    cs_scores.sort(key=lambda x: x[1], reverse=True)
                    for t, score in cs_scores[:15]:  # Show top 15
                        print(f"  {t}: {score:.6f}")
    else:
        print("ERROR: Date not found in vec weights")
        return
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    common = set(weights_non_vec.keys()) & set(vec_weights.keys())
    only_non_vec = set(weights_non_vec.keys()) - set(vec_weights.keys())
    only_vec = set(vec_weights.keys()) - set(weights_non_vec.keys())
    
    print(f"\nCommon tickers: {len(common)}")
    if only_non_vec:
        print(f"Only in non-vec: {sorted(only_non_vec)}")
    if only_vec:
        print(f"Only in vec: {sorted(only_vec)}")
    
    if common:
        print(f"\nWeight differences:")
        diffs = []
        for ticker in sorted(common):
            nv = weights_non_vec[ticker]
            v = vec_weights[ticker]
            diff = nv - v
            diffs.append((ticker, nv, v, diff, abs(diff)))
        
        diffs.sort(key=lambda x: x[4], reverse=True)
        
        for ticker, nv, v, diff, abs_diff in diffs:
            print(f"  {ticker}: non_vec={nv:.6f}, vec={v:.6f}, diff={diff:+.6f}")
        
        max_diff = max(abs(d[3]) for d in diffs)
        mean_diff = sum(abs(d[3]) for d in diffs) / len(diffs)
        print(f"\nMax abs diff: {max_diff:.6e}")
        print(f"Mean abs diff: {mean_diff:.6e}")


if __name__ == "__main__":
    main()
