from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.sleeves.trend.trend_sleeve import TrendSleeve
from src.sleeves.trend.trend_configs import TREND_CONFIG_WEEKLY

from src.market_data_store import MarketDataStore
from src.universe_manager import UniverseManager
from src.signal_engine import SignalEngine
from src.vec_signal_engine import VectorizedSignalEngine
from src.backtest_runner import (
    build_rebalance_schedule,
    shift_rebalance_dates_to_trading_calendar,
)


def build_runtime(local_only: bool = True) -> Dict[str, Any]:
    # Universe
    um = UniverseManager(
        membership_csv=Path("data/sp500_membership.csv"),
        sectors_yaml=Path("config/sectors.yaml"),
        local_only=local_only,
    )

    # Market data store (with in-memory cache enabled for speed)
    mds = MarketDataStore(
        data_root=Path("data/prices"),
        source="yfinance",
        local_only=local_only,
        use_memory_cache=True,
    )

    # Signals
    signals = SignalEngine(
        mds,
        disable_cache_margin=True,
        disable_cache_extension=True,
    )
    vec_engine = VectorizedSignalEngine(um, mds)

    # Sleeves
    trend = TrendSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        vec_engine=vec_engine,
        config=TREND_CONFIG_WEEKLY,  # Use the weekly trend configuration
    )

    runtime = {
        "universe_manager": um,
        "market_data_store": mds,
        "signals": signals,
        "vec_engine": vec_engine,
        "sleeves": {
            "trend": trend,
        },
    }

    return runtime


def main():
    runtime = build_runtime()
    print("Runtime setup completed successfully.")

    test_vec_path(runtime)


def get_closest_date_on_or_before(
    date: pd.Timestamp, dates: pd.DatetimeIndex
) -> pd.Timestamp:
    return dates[dates <= date].max()


def test_vec_path(runtime: Dict[str, Any]):
    START = "2025-01-01"
    END = "2025-11-30"
    start_ts = pd.to_datetime(START)
    end_ts = pd.to_datetime(END)
    warmup_start = start_ts - pd.Timedelta(days=365)

    trend: TrendSleeve = runtime["sleeves"]["trend"]
    print("Testing vec_path method of TrendSleeve...")

    # --------------------------------------------------
    # 1) Price matrix (WITHOUT membership masking to avoid NaN gaps)
    # --------------------------------------------------
    price_mat = trend.um.get_price_matrix(
        price_loader=trend.mds,
        start=warmup_start,
        end=end_ts,
        tickers=trend.um.tickers,
        interval=trend.config.signals_interval,
        auto_apply_membership_mask=False,  # Changed: Don't mask prices
        local_only=getattr(trend.mds, "local_only", False),
    )
    print(f"Price matrix shape: {price_mat.shape}")
    print(f"Price matrix sample:\n{price_mat}")

    # Drop tickers with no data at all
    price_mat = price_mat.dropna(axis=1, how="all")
    if price_mat.empty:
        trend._cached_stock_scores_mat = pd.DataFrame()
        trend._cached_sector_scores_mat = pd.DataFrame()
        print("Price matrix is empty after dropping tickers with no data.")
        return

    # Use a reference ticker to define trading days (to match non-vec SignalEngine behavior)
    # The non-vec SignalEngine uses mds.get_ohlcv() which returns only actual trading days
    # To ensure .shift() operations match exactly, we use the same trading day calendar
    # reference_ticker = "AAPL"  # Use a liquid ticker typically in the universe
    # if reference_ticker not in price_mat.columns:
    #     # Fallback: use the first ticker with the most non-NaN values
    #     non_nan_counts = price_mat.notna().sum()
    #     reference_ticker = non_nan_counts.idxmax()

    # # Get reference ticker's trading days (dates where it has non-NaN prices)
    # ref_dates = price_mat[reference_ticker].dropna().index

    # Normalize tickers to uppercase and align sector_map to columns
    price_mat.columns = [c.upper() for c in price_mat.columns]
    sector_map = trend.um.sector_map
    if sector_map is not None:
        sector_map = {t.upper(): s for t, s in sector_map.items()}
        # Keep sector_map only for tickers present in prices
        sector_map = {t: s for t, s in sector_map.items() if t in price_mat.columns}
    print(f"Price matrix after normalization and alignment:\n{price_mat}")
    # print(f"Sector map: {sector_map}")

    # Get membership mask to apply after signal calculation
    membership_mask = trend.um.membership_mask(
        start=warmup_start.strftime("%Y-%m-%d"),
        end=end_ts.strftime("%Y-%m-%d"),
    )
    # Ensure mask columns match price_mat columns
    if not membership_mask.empty:
        membership_mask = membership_mask.reindex(
            columns=price_mat.columns, fill_value=False
        )
    print(f"Membership mask:\n{membership_mask}")

    # --------------------------------------------------
    # 2) Vectorized stock scores (Date x Ticker)
    # --------------------------------------------------
    stock_score_mat = trend._compute_stock_scores_vectorized(
        price_mat, membership_mask=membership_mask
    )
    if stock_score_mat.empty:
        trend._cached_stock_scores_mat = pd.DataFrame()
        trend._cached_sector_scores_mat = pd.DataFrame()
        print("Stock score matrix is empty after computation.")
        return
    print(f"Stock score matrix shape: {stock_score_mat.shape}")
    print(f"Stock score matrix sample:\n{stock_score_mat}")
    # Membership mask already applied inside _compute_stock_scores_vectorized
    # before z-scoring to ensure cross-sectional rankings are correct

    # --------------------------------------------------
    # 3) Vectorized sector scores (Date x Sector)
    # --------------------------------------------------
    sector_scores_mat = trend._compute_sector_scores_vectorized(
        stock_score_mat, sector_map=sector_map
    )
    if sector_scores_mat.empty:
        trend._cached_stock_scores_mat = pd.DataFrame()
        trend._cached_sector_scores_mat = pd.DataFrame()
        print("Sector score matrix is empty after computation.")
        return
    print(f"Sector score matrix shape: {sector_scores_mat.shape}")
    print(f"Sector score matrix sample:\n{sector_scores_mat}")

    # --------------------------------------------------
    # 4) Cache the computed matrices for use in generate_target_weights_for_date
    # --------------------------------------------------
    # Forward-fill stock scores to daily frequency to handle non-trading days
    # (e.g., 2025-01-01 should use scores from 2024-12-31)
    # stock_score_mat_daily = stock_score_mat.asfreq("D", method="ffill")
    # sector_scores_mat_daily = sector_scores_mat.asfreq("D", method="ffill")

    # Slice to [start, end] (drop warmup portion)
    stock_score_mat_sliced = stock_score_mat.loc[
        (stock_score_mat.index >= warmup_start) & (stock_score_mat.index <= end_ts)
    ]
    sector_scores_mat_sliced = sector_scores_mat.loc[
        (sector_scores_mat.index >= warmup_start) & (sector_scores_mat.index <= end_ts)
    ]

    # Cache the scores for use in generate_target_weights_for_date
    trend._cached_stock_scores_mat = stock_score_mat_sliced.copy()
    trend._cached_sector_scores_mat = sector_scores_mat_sliced.copy()

    # --------------------------------------------------
    # Simulate monthly rebalances from start_ts to end_ts using the precomputed vectorized scores
    # --------------------------------------------------
    print("Simulating monthly rebalances...")
    bench_symbol = "SPY"
    mds: MarketDataStore = runtime["market_data_store"]
    df_bench = mds.get_ohlcv(
        bench_symbol,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=True,
    )
    rebal_schedule = build_rebalance_schedule(start_ts, end_ts, frequency="monthly")
    rebal_schedule = shift_rebalance_dates_to_trading_calendar(
        rebal_schedule, df_bench.index
    )
    print(f"Rebalance schedule: {rebal_schedule}")

    for rebalance_date in rebal_schedule:
        rebalance_date = rebalance_date.normalize()
        print(f"Rebalancing on {rebalance_date.date()}...")
        # Extract from cache
        # IMPORTANT CHANGE: Instead of exact date, use the closest date that's on or before the rebalance date
        stock_scores_mat = trend._cached_stock_scores_mat
        sector_scores_mat = trend._cached_sector_scores_mat
        stock_as_of = get_closest_date_on_or_before(
            rebalance_date, stock_scores_mat.index
        )
        sector_as_of = get_closest_date_on_or_before(
            rebalance_date, sector_scores_mat.index
        )
        if stock_as_of != sector_as_of:
            print(
                f"Warning: Stock and sector scores are out of sync at {rebalance_date}"
            )
        as_of = stock_as_of

        print(f"Using closest date {stock_as_of} for stock and sector scores")
        stock_scores_series = trend._cached_stock_scores_mat.loc[stock_as_of]
        sector_scores = trend._cached_sector_scores_mat.loc[sector_as_of].dropna()

        # Convert to DataFrame format
        stock_scores = pd.DataFrame({"stock_score": stock_scores_series})
        stock_scores = stock_scores[stock_scores["stock_score"].notna()]

        # Get vol data if needed for inverse-vol weighting
        if trend.config.weighting_mode == "inverse-vol":
            if (
                trend._cached_feature_mats is not None
                and "vol" in trend._cached_feature_mats
            ):
                vol_mat = trend._cached_feature_mats["vol"]
                vol_as_of = get_closest_date_on_or_before(rebalance_date, vol_mat.index)
                vol_series = vol_mat.loc[vol_as_of]
                stock_scores["vol"] = vol_series.reindex(stock_scores.index)
                if vol_as_of != stock_as_of:
                    print(f"Warning: Vol data is out of sync at {rebalance_date}")
            else:
                raise ValueError("Vol data not cached for inverse-vol weighting")

        print(f"Stock scores as of {stock_as_of.date()}:\n{stock_scores}")
        print(f"Sector scores as of {sector_as_of.date()}:\n{sector_scores}")

        # ---------- Common path: sector weighting and stock allocation ----------
        if stock_scores.empty or sector_scores.empty:
            return {}

        # Compute sector scores for daily smoothing if needed
        smoothing_freq = getattr(
            trend.config, "sector_smoothing_freq", "rebalance_dates"
        )
        intermediate_sector_scores = None
        if (
            smoothing_freq == "daily"
            and trend.state.last_as_of is not None
            and trend.state.last_sector_weights is not None
        ):
            # Check if we'll need daily interpolation
            gap = (as_of - trend.state.last_as_of).days
            if gap > 0 and gap <= np.ceil(2 * trend._approx_rebalance_days):
                # Compute scores for [last_as_of+1, as_of]
                start_date = trend.state.last_as_of + pd.Timedelta(days=1)
                intermediate_sector_scores = trend._compute_sector_scores_for_range(
                    start_date, as_of, warmup_start
                )

        sector_weights = trend._compute_smoothed_sector_weights(
            as_of, sector_scores, intermediate_sector_scores
        )
        if sector_weights.isna().all() or sector_weights.sum() <= 0:
            return {}
        print(f"Sector weights:\n{sector_weights}")

        stock_weights = trend._allocate_to_stocks(stock_scores, sector_weights)
        if not stock_weights:
            return {}

        total = sum(stock_weights.values())
        if total <= 0:
            return {}

        # Normalize
        stock_weights = {t: w / total for t, w in stock_weights.items()}
        print(f"Stock weights:\n{stock_weights}")


if __name__ == "__main__":
    main()
