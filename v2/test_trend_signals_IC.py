from pathlib import Path
from src.signal_engine import SignalEngine
from src.sleeves.trend.trend_sleeve import TrendSleeve
from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.vec_signal_engine import VectorizedSignalEngine
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as _Path


LOCAL_ONLY = True


def build_runtime() -> Dict[str, object]:
    """
    Build UniverseManager, MarketDataStore, SignalEngine, RegimeEngine,
    DefensiveSleeve, TrendSleeve, MultiSleeveAllocator.
    """
    membership_csv = Path("data/sp500_membership.csv")
    sectors_yaml = Path("config/sectors.yaml")

    # Universe
    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,
        local_only=LOCAL_ONLY,
    )

    # Market data store (with in-memory cache enabled for speed)
    mds = MarketDataStore(
        data_root="data/prices",
        source="yfinance",
        local_only=LOCAL_ONLY,
        use_memory_cache=True,
    )

    # Signals
    signals = SignalEngine(mds)
    vec_engine = VectorizedSignalEngine(um, mds)

    # Sleeves
    # defensive = DefensiveSleeve(
    #     universe=um,
    #     mds=mds,
    #     signals=signals,
    #     config=None,  # default DefensiveConfig
    # )
    trend = TrendSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        vec_engine=vec_engine,
        config=None,  # default TrendConfig
    )

    return {
        "um": um,
        "mds": mds,
        "signals": signals,
        "vec_engine": vec_engine,
        # "regime_engine": regime_engine,
        # "defensive": defensive,
        "trend": trend,
        # "allocator": allocator,
    }


def compute_cross_sectional_ics(
    signal_mat: pd.DataFrame,
    price_mat: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    H: int = 21,
):
    """Compute cross-sectional (per-date) ICs between `signal_mat` and
    forward returns at horizon `H` trading days.

    Returns:
        ic_series (pd.Series): series of per-date IC values indexed by date.
        ic_mean (float): mean IC over the sample.
        ic_std (float): std dev of ICs.
        ic_tstat (float): t-stat of mean IC (mean / (std/sqrt(N))).
    """
    # Compute forward returns
    future_price = price_mat.shift(-H)
    future_ret = future_price / price_mat - 1.0
    # Align to the desired date range and drop the last H rows (no forward price)
    future_ret = future_ret.loc[start:end]
    future_ret = future_ret.iloc[:-H]
    # Align tickers between signals and returns
    future_ret = future_ret.reindex(columns=signal_mat.columns)

    # Align signals to the same dates as future returns
    signal_aligned = signal_mat.loc[future_ret.index]

    # Compute cross-sectional ICs
    ics = []
    for dt in future_ret.index:
        s = signal_aligned.loc[dt]
        r = future_ret.loc[dt]
        mask = s.notna() & r.notna()
        if mask.sum() < 10:
            continue

        ic_t = s[mask].corr(r[mask], method="pearson")
        ics.append((dt, ic_t))

    ic_series = pd.Series(
        data=[v for _, v in ics], index=[dt for dt, _ in ics], name=f"IC_H{H}"
    )

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std > 0 else np.nan

    return ic_series, ic_mean, ic_std, ic_tstat


def plot_ic_series(ic_series: pd.Series, H: int, out_dir: _Path = _Path("data/plots")):
    """Save IC timeseries plot and histogram to `out_dir` and return file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if ic_series.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(10, 4))
    ic_series.plot(ax=ax, marker="o", linestyle="-")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_title(f"Cross-sectional IC over time (H={H})")
    ax.set_ylabel("IC")
    fig.tight_layout()
    timeseries_path = out_dir / f"ic_timeseries_H{H}.png"
    fig.savefig(timeseries_path)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ic_series.plot(kind="hist", bins=30, ax=ax2)
    ax2.set_title(f"IC Distribution (H={H})")
    ax2.set_xlabel("IC")
    fig2.tight_layout()
    hist_path = out_dir / f"ic_hist_H{H}.png"
    fig2.savefig(hist_path)
    plt.close(fig2)

    return timeseries_path, hist_path


def main():
    START = "2007-01-01"
    END = "2008-12-31"

    # Build runtime components
    rt = build_runtime()
    um: UniverseManager = rt["um"]
    mds: MarketDataStore = rt["mds"]
    signals: SignalEngine = rt["signals"]
    vec_engine: VectorizedSignalEngine = rt["vec_engine"]
    trend: TrendSleeve = rt["trend"]

    start = pd.to_datetime(START)
    start_with_warmup = start - pd.Timedelta(days=365)
    end = pd.to_datetime(END)
    price_mat = um.get_price_matrix(
        mds,
        start=start_with_warmup.to_pydatetime(),
        end=end.to_pydatetime(),
        # tickers=["MU"],
        field="Close",
    )
    price_mat = price_mat.dropna(axis=1, how="all")
    price_mat.columns = [c.upper() for c in price_mat.columns]
    print(f"[test_trend_signals_IC] price_mat:\n{price_mat}")

    stock_score_mat = trend._compute_stock_scores_vectorized(
        price_mat=price_mat,
    )
    stock_score_mat = stock_score_mat.loc[start:end]
    # Keep only tickers with non-NaN scores
    stock_score_mat = stock_score_mat.dropna(axis=1, how="any")
    signal_mat = stock_score_mat
    print(f"[test_trend_signals_IC] signal_mat:\n{signal_mat}")
    print(
        f"[test_trend_signals_IC] signal_mat index range: {signal_mat.index.min()} to {signal_mat.index.max()}"
    )

    # Compute ICs for a given horizon H
    for H in [5,21,63]:
        ic_series, ic_mean, ic_std, ic_tstat = compute_cross_sectional_ics(
            signal_mat=signal_mat, price_mat=price_mat, start=start, end=end, H=H
        )
        # print(f"[test_trend_signals_IC] IC series:\n{ic_series}")
        print("[test_trend_signals_IC] IC Stats:")
        print("Test Period:", START, "to", END)
        print("Horizon:", H)
        print("Mean IC:", ic_mean)
        print("Std(IC):", ic_std)
        print("IC t-stat:", ic_tstat)

        # Save plots for IC over time and distribution
        ts_path, hist_path = plot_ic_series(ic_series, H)
        if ts_path and hist_path:
            print(f"Saved IC timeseries to: {ts_path}")
            print(f"Saved IC histogram to: {hist_path}")
        else:
            print(f"No ICs to plot for H={H}")


if __name__ == "__main__":
    main()
