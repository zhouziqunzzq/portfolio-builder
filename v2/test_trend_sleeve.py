"""
Simple integration test for TrendSleeve (V2).

This test:
    - initializes UniverseManager / MarketDataStore / SignalEngine
    - instantiates TrendSleeve
    - computes target weights for a given date
    - runs a liquidity diagnostic
    - runs a quick vectorized stock-score sanity check
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine

from src.sleeves.trend.trend_sleeve import TrendSleeve
from src.sleeves.trend.trend_config import TrendConfig
from src.vec_signal_engine import VectorizedSignalEngine


def test_trend_sleeve():
    # -------------------------
    # 1) Universe setup
    # -------------------------
    membership_csv = Path("./data/sp500_membership.csv")
    sectors_yaml = Path("./config/sectors.yml")

    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,
        local_only=True,
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

    # Vectorized engine (used only for the vec sanity check)
    vse = VectorizedSignalEngine(
        universe=um,
        mds=mds,
    )

    # -------------------------
    # 3) Trend Sleeve
    # -------------------------
    cfg = TrendConfig()
    sleeve = TrendSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        vec_engine=vse,
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
        else:
            # Print top weights
            top = sorted(weights.items(), key=lambda x: -x[1])
            for t, w in top[:20]:  # top 20
                print(f"{t:6s} : {w:.4f}")

            print(f"\nSum of weights = {sum(weights.values()):.6f}")

        # -------------------------
        # 5) Liquidity diagnostic
        # -------------------------
        df_liq = run_liquidity_diagnostic(
            as_of=as_of,
            start=start_for_signals,
            universe=sleeve._get_trend_universe(as_of),
            mds=mds,
            signals=signals,
            cfg=cfg,
        )
        # print a small tail just to show structure
        print(df_liq.head())

        # -------------------------
        # 6) Vectorized stock-score sanity check
        # -------------------------
        print("\n[Vectorized stock-score sanity check]")

        trend_universe = sleeve._get_trend_universe(as_of)
        if not trend_universe:
            print("  Trend universe empty; skipping vec test.")
            continue

        price_mat = vse.get_price_matrix(
            tickers=trend_universe,
            start=start_for_signals,
            end=as_of,
            membership_aware=True,
            local_only=True,
        )

        # Drop all-NaN columns (tickers with no data)
        price_mat = price_mat.dropna(axis=1, how="all")
        if price_mat.empty:
            print("  Price matrix empty; skipping vec test.")
            continue
        stock_score_mat = sleeve._compute_stock_scores_vectorized(price_mat)
        if stock_score_mat.empty:
            print("  Score matrix empty; skipping.")
            continue
        print(stock_score_mat.tail())

        # -------------------------
        # 7) Vectorized sector-score test
        # -------------------------
        try:
            if not stock_score_mat.empty:
                sector_scores_mat = sleeve._compute_sector_scores_vectorized(
                    stock_score_mat
                )
                print("\n[Vectorized sector-score test]")
                print(
                    f"Sector scores shape: {getattr(sector_scores_mat, 'shape', None)}"
                )
                print(sector_scores_mat.tail())
        except Exception as e:
            print(f"Vectorized sector-score test failed: {e}")
            return
        
        # -------------------------
        # 8) Vectorized sector-weights test
        # -------------------------
        try:
            if not stock_score_mat.empty:
                sector_weights_mat = sleeve._compute_sector_weights_vectorized(
                    sector_scores_mat
                )
                print("\n[Vectorized sector-weights test]")
                print(
                    f"Sector weights shape: {getattr(sector_weights_mat, 'shape', None)}"
                )
                print(sector_weights_mat.head())
                print(sector_weights_mat.tail())
        except Exception as e:
            print(f"Vectorized sector-weights test failed: {e}")
            return
        
        # -------------------------
        # 9) Vectorized stock allocation test
        # -------------------------
        try:
            if not stock_score_mat.empty:
                stock_alloc_mat = sleeve._allocate_to_stocks_vectorized(
                    price_mat,
                    stock_score_mat,
                    sector_weights_mat,
                )
                print("\n[Vectorized stock-allocation test]")
                print(
                    f"Stock allocations shape: {getattr(stock_alloc_mat, 'shape', None)}"
                )
                print(stock_alloc_mat.head())
                print(stock_alloc_mat.tail())
                # Print the top-10 allocations for the last date
                last_date = stock_alloc_mat.index[-1]
                top_allocs = stock_alloc_mat.loc[last_date].sort_values(ascending=False).head(10)
                print(f"\nTop 10 stock allocations for {last_date.date()}:")
                print(top_allocs)
        except Exception as e:
            print(f"Vectorized stock-allocation test failed: {e}")
            return


def run_liquidity_diagnostic(
    as_of: pd.Timestamp,
    start: pd.Timestamp,
    universe: list[str],
    mds: MarketDataStore,
    signals: SignalEngine,
    cfg: TrendConfig,
) -> pd.DataFrame:
    """
    Prints a liquidity diagnostic table for the specified universe.

    Columns:
        ADV20, MedianVol20, LastPrice, PassPrice, PassADV20, PassMedVol20, Eligible
    """

    # Support both older (adv_window / median_volume_window) and newer
    # (liquidity_window) naming for the rolling windows.
    adv_window = getattr(cfg, "adv_window", getattr(cfg, "liquidity_window", 20))
    mv_window = getattr(
        cfg, "median_volume_window", getattr(cfg, "liquidity_window", 20)
    )

    min_adv = getattr(cfg, "min_adv20", None)
    min_medvol = getattr(cfg, "min_median_volume20", None)
    min_price = getattr(cfg, "min_price", None)

    rows = []
    for t in universe:
        try:
            # --- Prices for liquidity metrics ---
            df = mds.get_ohlcv(t, start=start, end=as_of)
            if df.empty or "Close" not in df.columns:
                continue
            last_price = float(df["Close"].iloc[-1])

            # --- ADV (rolling mean dollar volume) ---
            if "Close" in df.columns and "Volume" in df.columns:
                dollar_vol = df["Close"] * df["Volume"]
                adv = dollar_vol.rolling(adv_window).mean().iloc[-1]
            else:
                adv = np.nan

            # --- Median Volume ---
            if "Volume" in df.columns:
                medvol = df["Volume"].rolling(mv_window).median().iloc[-1]
            else:
                medvol = np.nan

            # --- Pass/fail filters ---
            pass_price = (min_price is None) or (last_price >= min_price)
            pass_adv = (min_adv is None) or (pd.notna(adv) and adv >= min_adv)
            pass_medv = (min_medvol is None) or (
                pd.notna(medvol) and medvol >= min_medvol
            )

            eligible = pass_price and pass_adv and pass_medv

            rows.append(
                {
                    "ticker": t,
                    "last_price": last_price,
                    "adv20": adv,
                    "median_vol20": medvol,
                    "pass_price": pass_price,
                    "pass_adv20": pass_adv,
                    "pass_medvol20": pass_medv,
                    "eligible": eligible,
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")
    df = df.sort_values("eligible", ascending=False)

    # --- Summary ---
    total = len(df)
    passed = int(df["eligible"].sum())

    print("\n========================================================")
    print(f"Liquidity Diagnostic as of {as_of.date()} (Universe={total})")
    print("========================================================")
    if total > 0:
        print(f"Eligible: {passed}/{total} ({passed/total:.1%})\n")
    else:
        print("Eligible: 0/0\n")

    print("Minimums:")
    print(f"  Price: {min_price}")
    print(f"  ADV20: ${min_adv:,.0f}" if min_adv is not None else "  ADV20: (none)")
    print(
        f"  MedianVol20: {min_medvol:,} shares\n"
        if min_medvol is not None
        else "  MedianVol20: (none)\n"
    )

    print("Top 20 failing examples:")
    fails = df[df["eligible"] == False].head(20)
    print(fails[["last_price", "adv20", "median_vol20"]])

    print("\nTop 20 passing examples:")
    passes = df[df["eligible"] == True].head(20)
    print(passes[["last_price", "adv20", "median_vol20"]])

    return df


if __name__ == "__main__":
    test_trend_sleeve()
