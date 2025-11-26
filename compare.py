from pathlib import Path
import pandas as pd

from v1.src.market_data_store import MarketDataStore as MDSv1
from v2.src.market_data_store import MarketDataStore as MDSv2

from v1.src.universe_manager import UniverseManager as UMv1
from v2.src.universe_manager import UniverseManager as UMv2

from v1.src.signal_engine import SignalEngine as SEv1
from v2.src.utils.stats import zscore_matrix_column_wise
from v2.src.vec_signal_engine import VectorizedSignalEngine as SEv2
from v2.src.sleeves.trend.trend_sleeve import TrendSleeve, TrendConfig

from v1.src.sector_weight_engine import SectorWeightEngine as SWEv1

from v1.src.stock_allocator import StockAllocator as SAv1

# Epsilon threshold for comparison: suppress tiny numeric residuals
EPS = 1e-6

def compute_mom2_score(mom2, mom_windows, mom_weights):
    z_mats = {}

    for w in mom_windows:
        mat = mom2[w]
        z = zscore_matrix_column_wise(mat)
        z_mats[f"z_mom_{w}"] = z

    mom_part = None
    for w, wt in zip(mom_windows, mom_weights):
        if mom_part is None:
            mom_part = wt * z_mats[f"z_mom_{w}"]
        else:
            mom_part += wt * z_mats[f"z_mom_{w}"]
    return mom_part


def compute_vol2_score(vol2):
    z_vol = zscore_matrix_column_wise(vol2)
    return -z_vol  # invert: lower vol -> higher score


if __name__ == "__main__":
    # MarketDataStore comparison
    print("Comparing MarketDataStore outputs between V1 and V2...")
    mds1 = MDSv1(data_root="v1/data/prices")
    mds2 = MDSv2(data_root="v2/data/prices")

    test_price_ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    prices1 = mds1.get_ohlcv(
        ticker=test_price_ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        local_only=True,
    )
    prices2 = mds2.get_ohlcv(
        ticker=test_price_ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        local_only=True,
    )
    print(f"Prices for {test_price_ticker} from {start_date} to {end_date}:")
    print("V1:")
    print(prices1)
    print("V2:")
    print(prices2)
    if prices1.equals(prices2):
        print(f"MarketDataStore outputs match for {test_price_ticker}.")
    else:
        print(f"MarketDataStore outputs differ for {test_price_ticker}.")

    # UniverseManager comparison
    print("\nComparing UniverseManager outputs between V1 and V2...")
    um1 = UMv1(
        membership_csv=Path("v1/data/sp500_membership.csv").resolve(),
        sectors_yaml=Path("v1/config/sectors.yml").resolve(),
        local_only=True,
    )
    um2 = UMv2(
        membership_csv=Path("v2/data/sp500_membership.csv").resolve(),
        sectors_yaml=Path("v2/config/sectors.yml").resolve(),
        local_only=True,
    )

    tickers1 = um1.tickers
    tickers2 = um2.tickers
    if tickers1 == tickers2:
        print("UniverseManager tickers match.")
    else:
        print("UniverseManager tickers differ.")
        print("V1 tickers:", tickers1)
        print("V2 tickers:", tickers2)

    sm1 = um1.sector_map
    sm2 = um2.sector_map
    if sm1 == sm2:
        print("UniverseManager sector maps match.")
    else:
        print("UniverseManager sector maps differ.")
        print("V1 sector map:", sm1)
        print("V2 sector map:", sm2)

    start_date = "2005-01-01"
    end_date = "2010-12-31"
    pm1 = um1.get_price_matrix(
        price_loader=mds1,
        start=start_date,
        end=end_date,
        tickers=um1.tickers,
        interval="1d",
        auto_apply_membership_mask=True,
        local_only=True,
    )
    pm2 = um2.get_price_matrix(
        price_loader=mds2,
        start=start_date,
        end=end_date,
        tickers=um2.tickers,
        interval="1d",
        auto_apply_membership_mask=True,
        local_only=True,
    )
    if pm1.equals(pm2):
        print("UniverseManager price matrices match.")
    else:
        print("UniverseManager price matrices differ.")
        print("V1 price matrix:")
        print(pm1)
        print("V2 price matrix:")
        print(pm2)

    # Signals calculation comparison
    print("\nComparing SignalEngine outputs between V1 and V2...")
    se1 = SEv1(
        prices=pm1,
        sector_map=sm1,
    )
    se2 = SEv2(
        universe=um2,
        mds=mds2,
    )
    trend_sleeve = TrendSleeve(
        universe=um2,
        mds=mds2,
        signals=None,
        vec_engine=se2,
        config=TrendConfig(),
    )
    # Momentum signal comparison
    mom_windows = [63, 126, 252]
    mom_weights = [1.0, 1.0, 1.0]
    mom1 = se1.compute_momentum_components(
        windows=mom_windows,
    )
    mom2 = se2.get_momentum(
        pm2,
        lookbacks=mom_windows,
    )
    for w in mom_windows:
        m1 = mom1[w]
        m2 = mom2[w]
        if m1.equals(m2):
            print(f"Momentum signals match for window {w}.")
        else:
            print(f"Momentum signals differ for window {w}.")
            print("V1 momentum signal:")
            print(m1)
            print("V2 momentum signal:")
            print(m2)
    # Volatility signal comparison
    vol_window = 20
    vol1 = se1.compute_volatility(window=vol_window)
    vol2 = se2.get_volatility(pm2, window=vol_window)
    if vol1.equals(vol2):
        print(f"Volatility signals match for window {vol_window}.")
    else:
        print(f"Volatility signals differ for window {vol_window}.")
        print("V1 volatility signal:")
        print(vol1)
        print("V2 volatility signal:")
        print(vol2)
    # Momentum score comparison
    mom_score1 = se1.compute_momentum_score(
        windows=mom_windows,
        weights=mom_weights,
    )
    mom_score2 = compute_mom2_score(
        mom2=mom2,
        mom_windows=mom_windows,
        mom_weights=mom_weights,
    )
    if mom_score1.equals(mom_score2):
        print("Momentum scores match.")
    else:
        print("Momentum scores differ.")
        print("V1 momentum score:")
        print(mom_score1)
        print("V2 momentum score:")
        print(mom_score2)
    # Volatility score comparison
    vol_score1 = se1.compute_vol_score(window=vol_window)
    vol_score2 = compute_vol2_score(
        vol2=vol2,
    )
    if vol_score1.equals(vol_score2):
        print("Volatility scores match.")
    else:
        print("Volatility scores differ.")
        print("V1 volatility score:")
        print(vol_score1)
        print("V2 volatility score:")
        print(vol_score2)
    # Stock score comparison
    vol_weight = 0.5
    stock_score1 = se1.compute_stock_score(
        mom_windows=mom_windows,
        mom_weights=mom_weights,
        vol_window=vol_window,
        vol_weight=vol_weight,
    )
    stock_score2 = trend_sleeve._compute_stock_scores_vectorized(
        price_mat=pm2,
    )
    if stock_score1.equals(stock_score2):
        print("Composite stock scores match.")
    else:
        print("Composite stock scores differ.")
        print("V1 stock scores:")
        print(stock_score1)
        print("V2 stock scores:")
        print(stock_score2)
    # Sector score comparison
    sector_score1 = se1.compute_sector_scores_from_stock_scores(
        stock_score=stock_score1,
        sector_map=sm1,
    )
    sector_score2 = trend_sleeve._compute_sector_scores_vectorized(
        stock_score_mat=stock_score2,
        sector_map=sm2,
    )
    if sector_score1.equals(sector_score2):
        print("Sector scores match.")
    else:
        print("Sector scores differ.")
        print("V1 sector scores:")
        print(sector_score1)
        print("V2 sector scores:")
        print(sector_score2)
    # Sector weights comparison
    # Params from cfg.sectors
    alpha = 1.0
    beta = 0.3
    w_min = 0.00
    w_max = 0.30
    risk_on_frac = 1.0
    risk_off_frac = 0.7
    top_k_sectors = 5
    swe1 = SWEv1(
        sector_scores=sector_score1,
        benchmark_prices=None,
        alpha=alpha,
        w_min=w_min,
        w_max=w_max,
        beta=beta,
        trend_window=200,
        trend_enabled=False,
        risk_on_equity_frac=risk_on_frac,
        risk_off_equity_frac=risk_off_frac,
        top_k_sectors=top_k_sectors,
    )
    sw1_daily = swe1.compute_weights()
    sw2_daily = trend_sleeve._compute_sector_weights_vectorized(
        sector_scores_mat=sector_score2,
    )
    if sw1_daily.equals(sw2_daily):
        print("Sector weights match.")
    else:
        print("Sector weights differ.")
        print("V1 sector weights:")
        print(sw1_daily)
        print("V2 sector weights:")
        print(sw2_daily)
    # Stock allocation comparison
    sa1 = SAv1(
        sector_weights=sw1_daily,
        stock_scores=stock_score1,
        sector_map=sm1,
        stock_vol=None,
        top_k=2,
        weighting_mode="equal",
        preserve_cash=True,
    )
    alloc1 = sa1.compute_stock_weights()
    alloc2 = trend_sleeve._allocate_to_stocks_vectorized(
        price_mat=pm2,
        stock_score_mat=stock_score2,
        sector_weights_mat=sw2_daily,
        vol_mat=None,
        sector_map=sm2,
    )
    # Align both allocation tables to the union of dates and tickers and
    # compare using the EPS threshold to avoid claiming differences due to
    # tiny numerical residuals.
    all_dates = sorted(set(alloc1.index).union(set(alloc2.index)))
    all_tickers = sorted(set(alloc1.columns).union(set(alloc2.columns)))
    a1 = alloc1.reindex(index=all_dates, columns=all_tickers).fillna(0.0)
    a2 = alloc2.reindex(index=all_dates, columns=all_tickers).fillna(0.0)

    overall_maxdiff = float((a1 - a2).abs().to_numpy().max()) if not a1.empty else 0.0
    if overall_maxdiff <= EPS:
        print(f"Final stock allocations match (within EPS={EPS}).")
    else:
        print("Final stock allocations differ. Showing per-date top-10 allocations and diffs:")
        # Iterate over union of dates present in either allocation
        dates = all_dates
        for dt in dates:
            # Safely fetch rows (missing dates -> zeros)
            row1 = a1.loc[dt]
            row2 = a2.loc[dt]

            # Compute per-ticker absolute differences and skip days with no diff
            diff = (row1 - row2).abs()
            maxdiff = float(diff.max()) if not diff.empty else 0.0
            # Skip days where the maximum absolute difference is below the
            # numerical-noise threshold `EPS`.
            if maxdiff <= EPS:
                continue

            top1 = row1.nlargest(10)
            top2 = row2.nlargest(10)

            print(f"Date: {dt}")
            print(" V1 top-10:")
            print(top1)
            print(" V2 top-10:")
            print(top2)

            print(f" Max abs diff on {dt}: {maxdiff:.6f}")
            # Show biggest differing tickers (up to 10), filtering out tiny
            # residuals below `EPS` so output is meaningful.
            biggest = diff[diff > EPS].sort_values(ascending=False).head(10)
            if not biggest.empty:
                print(" Biggest diffs:")
                print(biggest)

            print("-" * 60)
