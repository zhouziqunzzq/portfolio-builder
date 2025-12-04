# backtest_v2.py
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from universe_manager import UniverseManager
from market_data_store import MarketDataStore
from signal_engine import SignalEngine
from vec_signal_engine import VectorizedSignalEngine
from regime_engine import RegimeEngine
from sleeves.defensive.defensive_sleeve import DefensiveSleeve
from sleeves.trend.trend_sleeve import TrendSleeve
from allocator.multi_sleeve_allocator import MultiSleeveAllocator
from allocator.multi_sleeve_config import MultiSleeveConfig
from portfolio_backtester import PortfolioBacktester  # updated v2 version


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Sleeve V2 Backtest Runner")

    # Paths
    p.add_argument(
        "--membership-csv",
        default="data/sp500_membership.csv",
        help="Path to S&P500 membership CSV (for UniverseManager)",
    )
    p.add_argument(
        "--sectors-yaml",
        default="config/sectors.yml",
        help="Path to sectors.yml",
    )
    p.add_argument(
        "--data-root",
        default="data/prices",
        help="Root directory for MarketDataStore price cache",
    )

    # Backtest window
    p.add_argument(
        "--start",
        dest="backtest_start",
        default=None,
        help="Backtest start date (YYYY-MM-DD). If omitted, use earliest feasible.",
    )
    p.add_argument(
        "--end",
        dest="backtest_end",
        default=None,
        help="Backtest end date (YYYY-MM-DD). If omitted, use today / latest.",
    )

    p.add_argument(
        "--rebalance-frequency",
        dest="rebalance_frequency",
        choices=["monthly", "bi-weekly", "weekly"],
        default="monthly",
        help=(
            "Rebalance frequency: 'monthly' (business month-end), "
            "'bi-weekly' (every 2 weeks, anchored to Fridays), "
            "or 'weekly' (every Friday)."
        ),
    )

    p.add_argument(
        "--local-only",
        action="store_true",
        help="Disable network calls and use only local caches",
    )

    # Cost & risk-free knobs
    p.add_argument(
        "--initial-equity",
        type=float,
        default=100_000.0,
        help="Initial portfolio value",
    )
    p.add_argument(
        "--cost-per-turnover",
        type=float,
        default=0.001,
        help="Base transaction cost per unit turnover (e.g. 0.001 = 10 bps per full turnover)",
    )
    p.add_argument(
        "--bid-ask-bps-per-side",
        type=float,
        default=0.0,
        help="Bid-ask spread cost per side, in bps (e.g. 5 = 5 bps each side ~= 10 bps round trip)",
    )
    p.add_argument(
        "--rf-annual",
        type=float,
        default=0.03,
        help="Annual risk-free rate used for Sharpe (e.g. 0.03 = 3%)",
    )

    # Plotting controls
    p.add_argument(
        "--plot-all",
        action="store_true",
        help="Generate all standard plots",
    )
    p.add_argument(
        "--plot-equity",
        action="store_true",
        help="Plot equity curve (log)",
    )
    p.add_argument(
        "--plot-drawdown",
        action="store_true",
        help="Plot strategy drawdown",
    )
    p.add_argument(
        "--plot-annual",
        action="store_true",
        help="Plot calendar-year returns vs benchmark",
    )
    p.add_argument(
        "--plot-rolling",
        action="store_true",
        help="Plot rolling 1-year (252-day) Sharpe vs benchmark",
    )
    p.add_argument(
        "--plot-turnover",
        action="store_true",
        help="Plot average monthly turnover time series",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving to disk",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Wiring V2 runtime
# ---------------------------------------------------------------------


def build_runtime(args: argparse.Namespace) -> Dict[str, object]:
    """
    Build UniverseManager, MarketDataStore, SignalEngine, RegimeEngine,
    DefensiveSleeve, TrendSleeve, MultiSleeveAllocator.
    """
    membership_csv = Path(args.membership_csv)
    sectors_yaml = Path(args.sectors_yaml)

    # Universe
    um = UniverseManager(
        membership_csv=membership_csv,
        sectors_yaml=sectors_yaml,
        local_only=bool(args.local_only),
    )

    # Market data store (with in-memory cache enabled for speed)
    mds = MarketDataStore(
        data_root=args.data_root,
        source="yfinance",
        local_only=bool(args.local_only),
        use_memory_cache=True,
    )

    # Signals
    signals = SignalEngine(mds)
    vec_engine = VectorizedSignalEngine(um, mds)

    # Regime engine
    regime_engine = RegimeEngine(
        signals=signals,
        config=None,  # use default RegimeConfig
    )

    # Sleeves
    defensive = DefensiveSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        config=None,  # default DefensiveConfig
    )
    trend = TrendSleeve(
        universe=um,
        mds=mds,
        signals=signals,
        vec_engine=vec_engine,
        config=None,  # default TrendConfig
    )

    # Multi-sleeve configuration (already has defensive + trend defaults)
    ms_config = MultiSleeveConfig()

    allocator = MultiSleeveAllocator(
        regime_engine=regime_engine,
        sleeves={
            "defensive": defensive,
            "trend": trend,
        },
        config=ms_config,
    )

    return {
        "um": um,
        "mds": mds,
        "signals": signals,
        "regime_engine": regime_engine,
        "defensive": defensive,
        "trend": trend,
        "allocator": allocator,
    }


# ---------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------


def build_rebalance_schedule(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str = "monthly",
) -> pd.DatetimeIndex:
    """
    Build a rebalance schedule between `start` and `end` according to `frequency`.

    Supported frequencies:
    - "monthly": business month-end (freq="BME")
    - "weekly": weekly on Fridays (freq="W-FRI")
    - "bi-weekly": every two weeks on Fridays (freq="2W-FRI").
    """
    freq_lc = (frequency or "monthly").lower()

    if freq_lc in ("monthly", "month", "bme"):
        return pd.date_range(start=start, end=end, freq="BME")

    if freq_lc in ("weekly", "week"):
        return pd.date_range(start=start, end=end, freq="W-FRI")

    if freq_lc in ("bi-weekly", "biweekly", "bi_weekly"):
        # Every 2 weeks on Friday
        return pd.date_range(start=start, end=end, freq="2W-FRI")

    # fallback: monthly BME
    return pd.date_range(start=start, end=end, freq="BME")


def generate_target_weights(
    allocator: MultiSleeveAllocator,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rebalance_frequency: str = "monthly",
) -> pd.DataFrame:
    """
    Call MultiSleeveAllocator on each rebalance date and build
    a Date x Ticker matrix of target weights.
    """
    schedule = build_rebalance_schedule(start, end, frequency=rebalance_frequency)

    if schedule.empty:
        raise ValueError(f"No rebalance dates between {start.date()} and {end.date()}")

    all_tickers: set[str] = set()
    rows: Dict[pd.Timestamp, Dict[str, float]] = {}

    for as_of in schedule:
        print(f"[backtest_v2] Generating weights for {as_of.date()}")
        w = allocator.generate_global_target_weights(as_of)
        rows[as_of] = w
        all_tickers.update(w.keys())

    if not all_tickers:
        return pd.DataFrame(index=schedule)

    cols = sorted(all_tickers)
    weights_df = pd.DataFrame(0.0, index=schedule, columns=cols)

    for dt, w in rows.items():
        for t, val in w.items():
            weights_df.at[dt, t] = float(val)

    return weights_df


def get_price_matrix_for_weights(
    um: UniverseManager,
    mds: MarketDataStore,
    weights: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    local_only: bool,
) -> pd.DataFrame:
    """
    Fetch daily Close prices for *exactly* the tickers that appear in the weights matrix.
    """
    tickers = sorted([c.upper() for c in weights.columns])
    if not tickers:
        return pd.DataFrame()

    price_mat = um.get_price_matrix(
        price_loader=mds,
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        field="Close",
        interval="1d",
        # !IMPORTANT - we need prices for all tickers (including non SP500 ones like GLD)
        auto_apply_membership_mask=False,
        local_only=bool(local_only),
    )
    return price_mat


# ---------------------------------------------------------------------
# Plotting helpers (V2-local; no external cfg)
# ---------------------------------------------------------------------


def _import_matplotlib() -> Tuple[Optional[object], Optional[object]]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import dates as mdates  # type: ignore

        return plt, mdates
    except Exception as e:
        print(f"[backtest_v2] Matplotlib import failed; skipping plots. ({e})")
        return None, None


def _ensure_plots_dir() -> Path:
    out = Path("data").joinpath("plots").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_equity_curve(
    strat_eq: pd.Series,
    bench_eq: Optional[pd.Series],
    show: bool = False,
) -> Optional[Path]:
    plt, mdates = _import_matplotlib()
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strat_eq.index, strat_eq.values, label="Strategy")

    if bench_eq is not None and len(bench_eq) > 0:
        try:
            bench_scaled = bench_eq * (strat_eq.iloc[0] / bench_eq.iloc[0])
            ax.plot(
                bench_scaled.index, bench_scaled.values, label="Benchmark", alpha=0.85
            )
        except Exception:
            ax.plot(bench_eq.index, bench_eq.values, label="Benchmark", alpha=0.85)

    ax.set_yscale("log")
    ax.set_title("Equity Curve (Log Scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    if mdates is not None:
        try:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        except Exception:
            pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir()
    start_s = (
        str(pd.to_datetime(strat_eq.index.min()).date()) if len(strat_eq) else "na"
    )
    end_s = str(pd.to_datetime(strat_eq.index.max()).date()) if len(strat_eq) else "na"
    out_path = plots_dir / f"v2_equity_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[backtest_v2] Saved equity plot: {out_path}")
    return out_path


def plot_drawdown(
    strat_eq: pd.Series,
    show: bool = False,
) -> Optional[Path]:
    plt, mdates = _import_matplotlib()
    if plt is None:
        return None

    running_max = strat_eq.cummax()
    dd = strat_eq / running_max - 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dd.index, dd.values, label="Drawdown")
    ax.set_title("Strategy Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", linewidth=0.5)

    if mdates is not None:
        try:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        except Exception:
            pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir()
    start_s = str(pd.to_datetime(dd.index.min()).date()) if len(dd) else "na"
    end_s = str(pd.to_datetime(dd.index.max()).date()) if len(dd) else "na"
    out_path = plots_dir / f"v2_drawdown_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[backtest_v2] Saved drawdown plot: {out_path}")
    return out_path


def plot_calendar_year_returns(
    strat_rets: pd.Series,
    bench_rets: Optional[pd.Series],
    show: bool = False,
) -> Optional[Path]:
    plt, _ = _import_matplotlib()
    if plt is None:
        return None

    strat_yearly = (1 + strat_rets).groupby(strat_rets.index.year).prod() - 1
    bench_yearly = None
    if bench_rets is not None and len(bench_rets) > 0:
        bench_yearly = (1 + bench_rets).groupby(bench_rets.index.year).prod() - 1
        idx = strat_yearly.index.intersection(bench_yearly.index)
        strat_yearly = strat_yearly.loc[idx]
        bench_yearly = bench_yearly.loc[idx]

    years = strat_yearly.index.astype(int)
    x = range(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [i - width / 2 for i in x], strat_yearly.values, width=width, label="Strategy"
    )
    if bench_yearly is not None:
        ax.bar(
            [i + width / 2 for i in x],
            bench_yearly.values,
            width=width,
            label="Benchmark",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(list(years), rotation=45)
    ax.axhline(0, linestyle="--", linewidth=0.5)
    ax.set_title("Calendar-Year Returns")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()

    plots_dir = _ensure_plots_dir()
    start_s = str(years.min()) if len(years) else "na"
    end_s = str(years.max()) if len(years) else "na"
    out_path = plots_dir / f"v2_calendar_returns_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[backtest_v2] Saved calendar-year returns plot: {out_path}")
    return out_path


def plot_rolling_sharpe(
    strat_rets: pd.Series,
    bench_rets: Optional[pd.Series],
    window: int = 252,
    show: bool = False,
) -> Optional[Path]:
    plt, mdates = _import_matplotlib()
    if plt is None:
        return None

    roll_mean_s = strat_rets.rolling(window).mean()
    roll_std_s = strat_rets.rolling(window).std()
    s_sharpe = (roll_mean_s / roll_std_s) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s_sharpe.index, s_sharpe.values, label="Strategy")

    if bench_rets is not None and len(bench_rets) > 0:
        roll_mean_b = bench_rets.rolling(window).mean()
        roll_std_b = bench_rets.rolling(window).std()
        b_sharpe = (roll_mean_b / roll_std_b) * np.sqrt(252)
        ax.plot(b_sharpe.index, b_sharpe.values, label="Benchmark", linestyle="--")

    ax.axhline(0, linestyle="--", linewidth=0.5)
    ax.set_title(f"Rolling {int(window / 21):d}-Month Sharpe ({window}-day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)

    if mdates is not None:
        try:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        except Exception:
            pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir()
    start_s = (
        str(pd.to_datetime(strat_rets.index.min()).date()) if len(strat_rets) else "na"
    )
    end_s = (
        str(pd.to_datetime(strat_rets.index.max()).date()) if len(strat_rets) else "na"
    )
    out_path = plots_dir / f"v2_rolling_sharpe_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[backtest_v2] Saved rolling Sharpe plot: {out_path}")
    return out_path


def plot_monthly_turnover(
    daily_turnover: pd.Series,
    show: bool = False,
) -> Optional[Path]:
    plt, mdates = _import_matplotlib()
    if plt is None:
        return None

    monthly = daily_turnover.resample("ME").mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values)
    ax.set_title("Average Monthly Turnover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover (fraction)")
    ax.grid(True, linestyle="--", linewidth=0.5)

    if mdates is not None:
        try:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        except Exception:
            pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir()
    start_s = str(pd.to_datetime(monthly.index.min()).date()) if len(monthly) else "na"
    end_s = str(pd.to_datetime(monthly.index.max()).date()) if len(monthly) else "na"
    out_path = plots_dir / f"v2_monthly_turnover_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[backtest_v2] Saved monthly turnover plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    # Build runtime stack
    rt = build_runtime(args)
    um: UniverseManager = rt["um"]  # type: ignore
    mds: MarketDataStore = rt["mds"]  # type: ignore
    allocator: MultiSleeveAllocator = rt["allocator"]  # type: ignore

    # Backtest window
    if args.backtest_start:
        start_dt = pd.to_datetime(args.backtest_start)
    else:
        try:
            membership = um.membership_df  # depending on your UM implementation
            start_dt = pd.to_datetime(membership.index.min())
        except Exception:
            start_dt = pd.Timestamp("2015-01-01")

    if args.backtest_end:
        end_dt = pd.to_datetime(args.backtest_end)
    else:
        end_dt = pd.Timestamp.today().normalize()

    if end_dt <= start_dt:
        raise ValueError(
            f"Backtest end {end_dt.date()} must be after start {start_dt.date()}"
        )

    print(
        f"Running V2 backtest from {start_dt.date()} to {end_dt.date()} "
        f"(local_only={args.local_only})"
    )

    # Build rebalance schedule early for precompute (frequency from CLI)
    rebalance_schedule = build_rebalance_schedule(
        start=start_dt, end=end_dt, frequency=args.rebalance_frequency
    )
    if rebalance_schedule.empty:
        print("No rebalance dates in window; aborting.")
        return 0

    # Optional vectorized sleeve precompute to accelerate per-date calls.
    # Only sleeves exposing a `precompute` method (e.g., TrendSleeve) will be used.
    try:
        print("[backtest_v2] Starting sleeve precompute phase...")
        allocator.precompute(
            start=start_dt,
            end=end_dt,
            rebalance_dates=list(rebalance_schedule),
            warmup_buffer=30,
        )
        print("[backtest_v2] Sleeve precompute phase completed.")
    except Exception as e:
        print(
            f"[backtest_v2] Sleeve precompute failed; continuing without cache. ({e})"
        )

    # 1) Generate sleeve-based target weights
    rebalance_target_weights = generate_target_weights(
        allocator=allocator,
        start=start_dt,
        end=end_dt,
        rebalance_frequency=args.rebalance_frequency,
    )

    if rebalance_target_weights.empty:
        print("No weights generated by allocator; aborting.")
        return 0

    print(
        f"Generated {len(rebalance_target_weights)} rebalance points; "
        f"{len(rebalance_target_weights.columns)} unique tickers."
    )

    # 2) Get daily price matrix for all tickers that appear in weights
    price_mat = get_price_matrix_for_weights(
        um=um,
        mds=mds,
        weights=rebalance_target_weights,
        start=start_dt,
        end=end_dt,
        local_only=args.local_only,
    )

    if price_mat.empty:
        print("Price matrix is empty for backtest window; aborting.")
        return 0

    # 3) Backtester
    bt = PortfolioBacktester(
        prices=price_mat,
        weights=rebalance_target_weights,
        trading_days_per_year=252,
        initial_value=float(args.initial_equity),
        cost_per_turnover=float(args.cost_per_turnover),
        bid_ask_bps_per_side=float(args.bid_ask_bps_per_side),
        risk_free_rate_annual=float(args.rf_annual),
    )

    result = bt.run()
    stats = bt.stats(result, auto_warmup=True, warmup_days=0)

    # 4) Benchmark (SPY)
    bench_symbol = "SPY"
    df_bench = mds.get_ohlcv(
        bench_symbol,
        start=start_dt,
        end=end_dt,
        interval="1d",
        auto_adjust=True,
        local_only=bool(args.local_only),
    )
    bench_series = None
    bench_returns = None

    if df_bench is not None and not df_bench.empty:
        price_col = (
            "Adjclose"
            if "Adjclose" in df_bench.columns
            else (
                "Adj Close"
                if "Adj Close" in df_bench.columns
                else ("Close" if "Close" in df_bench.columns else None)
            )
        )
        if price_col:
            bench_series = df_bench[price_col].copy()
            # Align to strategy index
            bench_series = bench_series.reindex(result.index).ffill().bfill()
            bench_returns = bench_series.pct_change().fillna(0.0)

    # 5) Print stats
    def pct(x):
        return f"{x * 100:.2f}%" if x is not None and not pd.isna(x) else "n/a"

    def num(x):
        return f"{x:.2f}" if x is not None and not pd.isna(x) else "n/a"

    eff_start = stats.get("EffectiveStart")
    eff_end = stats.get("EffectiveEnd")

    print("\n================ V2 Backtest Summary ================")
    print(
        f"Effective window : "
        f"{eff_start.date() if eff_start is not None else 'n/a'}"
        f" -> {eff_end.date() if eff_end is not None else 'n/a'}"
    )
    print(f"CAGR              : {pct(stats.get('CAGR'))}")
    print(f"Volatility        : {pct(stats.get('Volatility'))}")
    print(f"Sharpe (excess)   : {num(stats.get('Sharpe'))}")
    print(f"Max Drawdown      : {pct(stats.get('MaxDrawdown'))}")
    print(f"Avg Daily Turnover: {pct(stats.get('AvgDailyTurnover'))}")
    print(f"Final equity      : {result['equity'].iloc[-1]:.2f}")
    print("=====================================================\n")

    # 6) Plotting
    do_all = bool(args.plot_all)
    do_equity = do_all or bool(args.plot_equity)
    do_drawdown = do_all or bool(args.plot_drawdown)
    do_annual = do_all or bool(args.plot_annual)
    do_rolling = do_all or bool(args.plot_rolling)
    do_turnover = do_all or bool(args.plot_turnover)
    show = bool(args.show)

    # Slice strategy series to effective window for plots
    strat_eq = result["equity"]
    strat_rets = result["portfolio_return"]

    if eff_start is not None and eff_end is not None:
        strat_eq = strat_eq.loc[eff_start:eff_end]
        strat_rets = strat_rets.loc[eff_start:eff_end]
        if bench_series is not None:
            bench_series = bench_series.loc[eff_start:eff_end]
        if bench_returns is not None:
            bench_returns = bench_returns.loc[eff_start:eff_end]

    if do_equity:
        plot_equity_curve(strat_eq=strat_eq, bench_eq=bench_series, show=show)

    if do_drawdown:
        plot_drawdown(strat_eq=strat_eq, show=show)

    if do_annual:
        plot_calendar_year_returns(
            strat_rets=strat_rets,
            bench_rets=bench_returns,
            show=show,
        )

    if do_rolling:
        plot_rolling_sharpe(
            strat_rets=strat_rets,
            bench_rets=bench_returns,
            window=252,
            show=show,
        )

    if do_turnover:
        plot_monthly_turnover(
            daily_turnover=result["turnover"],
            show=show,
        )

    # 7) Save daily backtest results to CSV
    out_dir = Path("data").joinpath("backtests").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"backtest_{stamp}.csv"
    result.to_csv(out_path)
    print(f"[backtest_v2] Saved detailed daily backtest result to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
