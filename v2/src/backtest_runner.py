# backtest_v2.py
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

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
from context.rebalance import RebalanceContext
from sleeves.defensive.defensive_sleeve import DefensiveSleeve
from sleeves.trend.trend_sleeve import TrendSleeve
from sleeves.trend.trend_configs import (
    TREND_CONFIG_WEEKLY,
)

# from sleeves.sideways.sideways_sleeve import SidewaysSleeve
# from sleeves.sideways_mr.sideways_mr import SidewaysMRSleeve
# from sleeves.fast_alpha.fast_alpha_sleeve import FastAlphaSleeve
from sleeves.sideways_base.sideways_base_sleeve import SidewaysBaseSleeve
from allocator.multi_sleeve_allocator import MultiSleeveAllocator
from allocator.multi_sleeve_config import MultiSleeveConfig
from portfolio_backtester import PortfolioBacktester, PortfolioBacktesterConfig
from friction_control.hysteresis import apply_weight_hysteresis_matrix
from backtest_plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_calendar_year_returns,
    plot_rolling_sharpe,
    plot_monthly_turnover,
    plot_regime_scores,
    plot_sleeve_weights,
    plot_trend_status,
)


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
        required=True,
        help="Backtest start date (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end",
        dest="backtest_end",
        default=None,
        help="Backtest end date (YYYY-MM-DD). If omitted, use today / latest.",
    )

    p.add_argument(
        "--sample-frequency",
        dest="sample_frequency",
        choices=[
            "monthly",
            "bi-weekly",
            "weekly",
            "semi-monthly",
            "daily",
            "semi-weekly",
        ],
        default="monthly",
        help=(
            "Sample frequency: How often the allocator is called to generate new target weights. "
            "Note that this is separate from each sleeve's internal rebalance frequency. "
            "Options:"
            "'monthly' (business month-start), "
            "'bi-weekly' (every 2 weeks, anchored to Mondays), "
            "'weekly' (every week, anchored to Mondays), "
            "'semi-monthly' (2x per month: business month-start and mid-month Monday), "
            "'daily' (every business day), 'semi-weekly' (twice per week: Monday + mid-week)"
        ),
    )
    p.add_argument(
        "--signal-delay-days",
        type=int,
        default=0,
        help="Number of days to delay signal generation (default: 0); Useful for simulating lagged signals.",
    )

    p.add_argument(
        "--local-only",
        action="store_true",
        help="Disable network calls and use only local caches",
    )
    p.add_argument(
        "--execution-mode",
        choices=["open_to_open"],
        default="open_to_open",
        help=("Execution mode for backtest: 'open_to_open' (default)"),
    )
    p.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip sleeves precomputing before backtest",
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
    should_disable_cache_margin = False
    if args.sample_frequency in ("daily", "semi-weekly"):
        # For high-frequency sampling, disable cache margin to avoid
        # missing signals on tight windows.
        should_disable_cache_margin = True
    signals = SignalEngine(
        mds,
        disable_cache_margin=should_disable_cache_margin,
        disable_cache_extension=True,
    )
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
        # config=TREND_CONFIG_MONTHLY,
        config=TREND_CONFIG_WEEKLY,
        # config=TREND_CONFIG_DAILY,
    )
    # sideways = SidewaysSleeve(
    #     mds=mds,
    #     signals=signals,
    #     config=None,  # default SidewaysConfig
    # )
    # sideways_mr = SidewaysMRSleeve(
    #     mds=mds,
    #     signals=signals,
    #     config=None,  # default SidewaysMRConfig
    # )
    # fast_alpha = FastAlphaSleeve(
    #     universe=um,
    #     mds=mds,
    #     signals=signals,
    #     vec_engine=vec_engine,
    #     config=None,  # default FastAlphaConfig
    # )
    sideways_base = SidewaysBaseSleeve(
        mds=mds,
        signals=signals,
        config=None,  # default SidewaysBaseConfig
    )

    # Multi-sleeve configuration (already has defensive + trend defaults)
    ms_config = MultiSleeveConfig()

    allocator = MultiSleeveAllocator(
        regime_engine=regime_engine,
        sleeves={
            "defensive": defensive,
            "trend": trend,
            # "sideways": sideways,
            # "sideways_mr": sideways_mr,
            # "fast_alpha": fast_alpha,
            "sideways_base": sideways_base,
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
        # "sideways": sideways,
        "allocator": allocator,
    }


# ---------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------


def build_sampling_schedule(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str = "monthly",
) -> pd.DatetimeIndex:
    """
    Build a sampling schedule between `start` and `end` according to `frequency`.

    Supported frequencies:
    - "monthly": business month-start (freq="BMS")
    - "weekly": weekly on Mondays (freq="W-MON")
    - "bi-weekly": every two weeks on Mondays (freq="2W-MON").
    - "semi-monthly": two dates per month: business month-start (BMS)
        and a mid-month Monday (nearest Monday on/after the 15th when possible).
    """
    freq_lc = (frequency or "monthly").lower()

    if freq_lc in ("monthly", "month", "bms"):
        return pd.date_range(start=start, end=end, freq="BMS")

    if freq_lc in ("weekly", "week"):
        return pd.date_range(start=start, end=end, freq="W-MON")

    if freq_lc in ("daily", "day", "business-daily", "bday"):
        # Every business day in the range
        return pd.date_range(start=start, end=end, freq="B")

    if freq_lc in ("bi-weekly", "biweekly", "bi_weekly"):
        return pd.date_range(start=start, end=end, freq="2W-MON")

    if freq_lc in ("semi-weekly", "semi_weekly", "semiweekly"):
        # Twice per week: Mondays and mid-week (Wednesdays). Union and sort.
        mon = pd.date_range(start=start, end=end, freq="W-MON")
        wed = pd.date_range(start=start, end=end, freq="W-WED")
        all_dates = sorted({d.normalize(): d for d in list(mon) + list(wed)}.keys())
        return pd.DatetimeIndex(all_dates)

    if freq_lc in (
        "semi-monthly",
        "semimonthly",
        "semi_monthly",
    ):
        # Build Business Month Start dates
        bms = pd.date_range(start=start, end=end, freq="BMS")

        # For each month in the window, pick a mid-month Monday if possible
        mids = []
        months = pd.date_range(start=start, end=end, freq="MS")
        for m in months:
            year = int(m.year)
            month = int(m.month)
            # mid candidate = 15th
            try:
                mid = pd.Timestamp(year=year, month=month, day=15)
            except Exception:
                # fallback: first of month
                mid = m

            # find Monday on or after the 15th within the same month
            candidate = mid
            last_day = int((m + pd.offsets.MonthEnd(0)).day)
            # Move candidate forward to Monday if possible and within month
            while candidate.weekday() != 0 and candidate.day <= last_day:
                candidate = candidate + pd.Timedelta(days=1)

            if candidate.month != month:
                # couldn't find Monday on/after 15th within month; search backwards
                candidate = mid - pd.Timedelta(days=1)
                while candidate.weekday() != 0 and candidate.day >= 1:
                    candidate = candidate - pd.Timedelta(days=1)

            # Ensure candidate is within global start/end
            if candidate >= pd.to_datetime(start) and candidate <= pd.to_datetime(end):
                mids.append(candidate)

        # Union BMS + mids, sort, unique
        all_dates = sorted({d.normalize(): d for d in list(bms) + mids}.keys())
        return pd.DatetimeIndex(all_dates)

    # fallback: monthly BMS
    return pd.date_range(start=start, end=end, freq="BMS")


def generate_target_weights(
    allocator: MultiSleeveAllocator,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sample_schedule: pd.DatetimeIndex,
    signal_delay_days: Optional[int] = 0,
    initial_equity: float = 100_000.0,
) -> tuple[pd.DataFrame, Dict[pd.Timestamp, Dict[str, Any]]]:
    """
    Call MultiSleeveAllocator on each sample date and build
    a Date x Ticker matrix of target weights.
    """
    schedule = sample_schedule
    if schedule.empty:
        raise ValueError(f"No sample dates between {start.date()} and {end.date()}")

    all_tickers: set[str] = set()
    rows: Dict[pd.Timestamp, Dict[str, float]] = {}
    contexts: Dict[pd.Timestamp, Dict[str, Any]] = {}

    approx_aum = initial_equity
    approx_CAGR = 0.10  # 10% p.a. approximate
    last_as_of: Optional[pd.Timestamp] = None
    for as_of in schedule:
        # use prior day's data to avoid lookahead
        as_of_shifted = as_of - pd.Timedelta(days=1)
        # Apply signal delay if specified
        as_of_shifted -= pd.Timedelta(days=signal_delay_days)
        print(
            f"[backtest_v2] Generating weights for {as_of.date()} (using data as of {as_of_shifted.date()})"
        )
        w, ctx = allocator.generate_global_target_weights(
            as_of_shifted,
            rebalance_ctx=RebalanceContext(
                rebalance_ts=as_of,
                # Use an approximate AUM for now
                # TODO: Update with actual AUM from backtest state
                aum=approx_aum,
            ),
        )
        rows[as_of] = w
        contexts[as_of] = ctx
        all_tickers.update(w.keys())
        # Debug: print weights
        print(
            f"  Weights: {', '.join([f'{t}:{v:.4f}' for t,v in w.items() if v > 0.0])}"
        )
        # Estimate next AUM for context (simple CAGR growth)
        if last_as_of is not None:
            days_diff = (as_of - last_as_of).days
            approx_aum = approx_aum * (1.0 + approx_CAGR) ** (days_diff / 252.0)
            # print(f"  Estimated next AUM: {approx_aum:.2f}")
        last_as_of = as_of

    if not all_tickers:
        return pd.DataFrame(index=schedule), contexts

    cols = sorted(all_tickers)
    weights_df = pd.DataFrame(0.0, index=schedule, columns=cols)

    for dt, w in rows.items():
        for t, val in w.items():
            weights_df.at[dt, t] = float(val)

    return weights_df, contexts


def get_trading_calendar(
    start: pd.Timestamp,
    end: pd.Timestamp,
    mds: MarketDataStore,
    benchmark: str = "SPY",
) -> pd.DatetimeIndex:
    """
    Fetch the trading calendar between `start` and `end` (inclusive)
    using the benchmark ticker's price data from `mds`.
    """
    df_bench = mds.get_ohlcv(
        benchmark,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
    )
    if df_bench.empty:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(sorted(df_bench.index.normalize().unique()))


def shift_dates_to_trading_calendar(
    dates: pd.DatetimeIndex, trading_calendar: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """
    Shift each date in `dates` forward to the next available
    date in `trading_calendar` (both treated as normalized dates).

    If a sample date falls on a non-trading day, the next trading day
    on or after that date is used. If no trading day exists after a
    sample date (i.e., it's after the last trading date), that
    sample date is dropped.
    Returns a deduplicated, ordered `DatetimeIndex` of shifted dates.
    """
    if dates.empty:
        return dates

    # Normalize and sort trading calendar
    trading = pd.DatetimeIndex(sorted(pd.to_datetime(trading_calendar).normalize()))
    if len(trading) == 0:
        return dates

    shifted: list[pd.Timestamp] = []
    for d in dates:
        dn = pd.Timestamp(d).normalize()
        if dn in trading:
            shifted.append(dn)
        else:
            pos = trading.searchsorted(dn)
            if pos < len(trading):
                newd = trading[pos]
                shifted.append(newd)
                print(f"[backtest_v2] Shifted sample date {dn.date()} -> {newd.date()}")
            else:
                # No trading day after this date in the calendar; skip it
                print(
                    f"[backtest_v2] Dropping sample date {dn.date()}: no later trading day available"
                )

    # Deduplicate while preserving order
    seen = set()
    deduped: list[pd.Timestamp] = []
    for d in shifted:
        if d not in seen:
            seen.add(d)
            deduped.append(d)

    return pd.DatetimeIndex(deduped)


# Plot helpers have been moved to `backtest_plots.py` and are imported
# near the top of this file. The original in-file plotting helpers were
# removed to keep this runner focused on orchestration.


def print_backtest_summary(
    stats: Dict[str, Any],
    allocator: MultiSleeveAllocator,
    result: pd.DataFrame,
    eff_start: Optional[pd.Timestamp],
    eff_end: Optional[pd.Timestamp],
) -> None:
    """Print a standardized backtest summary."""

    def pct(x):
        return f"{x * 100:.2f}%" if x is not None and not pd.isna(x) else "n/a"

    def num(x):
        return f"{x:.2f}" if x is not None and not pd.isna(x) else "n/a"

    def money(x):
        return f"${x:,.2f}" if x is not None and not pd.isna(x) else "n/a"

    print("\n================ V2 Backtest Summary ================")
    print(
        f"Effective window : "
        f"{eff_start.date() if eff_start is not None else 'n/a'}"
        f" -> {eff_end.date() if eff_end is not None else 'n/a'}"
    )
    enabled = getattr(allocator, "enabled_sleeves", None)
    enabled_list = ", ".join(sorted(enabled)) if enabled else "n/a"
    print(f"Enabled Sleeves   : {enabled_list}")
    print(f"CAGR              : {pct(stats.get('CAGR'))}")
    print(f"Volatility        : {pct(stats.get('Volatility'))}")
    print(f"Sharpe (excess)   : {num(stats.get('Sharpe'))}")
    print(f"Skewness          : {num(stats.get('Skewness'))}")
    print(f"Kurtosis          : {num(stats.get('Kurtosis'))}")
    print(f"Max Drawdown      : {pct(stats.get('MaxDrawdown'))}")
    peak_dt = stats.get("DDPeakDate")
    trough_dt = stats.get("DDTroughDate")
    recovery_dt = stats.get("DDRecoveryDate")
    days_in_dd = stats.get("DaysInDrawdown")
    print(f"Peak Date         : {peak_dt.date() if peak_dt is not None else 'n/a'}")
    print(f"Trough Date       : {trough_dt.date() if trough_dt is not None else 'n/a'}")
    print(
        f"Recovery Date     : {recovery_dt.date() if recovery_dt is not None else 'n/a'}"
    )
    print(f"Days in Drawdown  : {int(days_in_dd) if days_in_dd is not None else 'n/a'}")
    print(f"Avg Daily Turnover: {pct(stats.get('AvgDailyTurnover'))}")
    print(f"Initial equity     : {money(stats.get('InitialEquity'))}")
    print(f"Total Costs        : {money(stats.get('TotalCost'))}")
    print(f"Final equity       : {money(result['equity'].iloc[-1])}")
    print("=====================================================\n")


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

    # Debug: print universe info
    # all_tickers = list(sorted(allocator.get_universe()))
    # print(f"Universe tickers ({len(all_tickers)}):")
    # print(f"All tickers in universe: {all_tickers}")

    # Backtest window
    if args.backtest_start:
        start_dt = pd.to_datetime(args.backtest_start)
    else:
        raise ValueError("Backtest start date (--start) is required.")
    if args.backtest_end:
        end_dt = pd.to_datetime(args.backtest_end)
    else:
        print(
            f"[backtest_v2] No backtest end date specified; using today ({pd.Timestamp.today().date()})."
        )
        end_dt = pd.Timestamp.today().normalize()
    if end_dt <= start_dt:
        raise ValueError(
            f"Backtest end {end_dt.date()} must be after start {start_dt.date()}"
        )
    print(
        f"Running V2 backtest from {start_dt.date()} to {end_dt.date()} "
        f"(local_only={args.local_only})"
    )

    # Hardcoded benchmark: SPY
    bench_symbol = "SPY"
    # Trading calendar from benchmark prices
    trading_calendar = get_trading_calendar(
        start=start_dt,
        end=end_dt,
        mds=mds,
        benchmark=bench_symbol,
    )

    # Build sample schedule early for precompute (frequency from CLI)
    sample_schedule = build_sampling_schedule(
        start=start_dt, end=end_dt, frequency=args.sample_frequency
    )
    if sample_schedule.empty:
        raise ValueError("No sample dates in backtest window.")
    # Shift sample dates to next trading day if needed given the trading calendar
    if trading_calendar is not None and not trading_calendar.empty:
        orig_count = len(sample_schedule)
        sample_schedule = shift_dates_to_trading_calendar(
            sample_schedule, trading_calendar
        )
        new_count = len(sample_schedule)
        if new_count != orig_count:
            print(
                f"[backtest_v2] Sample schedule adjusted: {orig_count} -> {new_count} dates after shifting to trading calendar"
            )
    else:
        print(
            "[backtest_v2] No trading calendar available; skipping sample date shifting."
        )

    # Shift to prior day for lookahead bias free precompute
    sample_schedule_shifted = sample_schedule - pd.Timedelta(days=1)
    # Apply any signal delay to sample schedule
    sample_schedule_shifted = sample_schedule_shifted - pd.Timedelta(
        days=args.signal_delay_days
    )

    # Optional vectorized sleeve precompute to accelerate per-date calls.
    # Only sleeves exposing a `precompute` method (e.g., TrendSleeve) will be used.
    if args.skip_precompute:
        print("[backtest_v2] Skipping sleeve precompute phase as requested.")
    else:
        try:
            print("[backtest_v2] Starting sleeve precompute phase...")
            allocator.precompute(
                start=start_dt,
                end=end_dt,
                sample_dates=list(sample_schedule_shifted),
                warmup_buffer=30,
            )
            print("[backtest_v2] Sleeve precompute phase completed.")
        except Exception as e:
            print(
                f"[backtest_v2] Sleeve precompute failed; continuing without cache. ({e})"
            )

    # Generate sleeve-based target weights (and collect regime context)
    rebalance_target_weights, rebalance_contexts = generate_target_weights(
        allocator=allocator,
        start=start_dt,
        end=end_dt,
        sample_schedule=sample_schedule,
        signal_delay_days=args.signal_delay_days,
        initial_equity=float(args.initial_equity),
    )
    if rebalance_target_weights.empty:
        print("No weights generated by allocator; aborting.")
        return 1
    print(
        f"Generated {len(rebalance_target_weights)} rebalance points; "
        f"{len(rebalance_target_weights.columns)} unique tickers."
    )

    # Backtester
    exec_mode = getattr(args, "execution_mode", "open_to_open")
    print(f"Backtest execution mode: {exec_mode}")
    bt_cfg = PortfolioBacktesterConfig(
        execution_mode=exec_mode,
        initial_value=float(args.initial_equity),
        cost_per_turnover=float(args.cost_per_turnover),
        bid_ask_bps_per_side=float(args.bid_ask_bps_per_side),
        risk_free_rate_annual=float(args.rf_annual),
    )
    bt = PortfolioBacktester(
        market_data_store=mds,
    )
    # IMPORTANT:
    # Weights are timestamped on the *execution day t*, not the prior day.
    #
    # Backtest flow:
    #   - Signals are computed using data up to t-1 (no lookahead)
    #   - Target weights are generated for date t
    #   - Trades are assumed to execute on day t at execution prices (default to open prices)
    #   - PnL should therefore accrue from t → t+1
    result = bt.run_vec(
        weights=rebalance_target_weights,
        config=bt_cfg,
        start=start_dt,
        end=end_dt,
    )
    stats = bt.stats(result, auto_warmup=True, warmup_days=0, config=bt_cfg)

    # Prepare benchmark series and returns
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

    eff_start = stats.get("EffectiveStart")
    eff_end = stats.get("EffectiveEnd")

    # Print summary
    print_backtest_summary(stats, allocator, result, eff_start, eff_end)

    # Plotting
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

    # Additional plots when doing full plots
    if do_all:
        try:
            plot_regime_scores(rebalance_contexts, show=show)
        except Exception as e:
            print(f"[backtest_v2] Failed to plot regime scores: {e}")
        try:
            plot_sleeve_weights(rebalance_contexts, show=show)
        except Exception as e:
            print(f"[backtest_v2] Failed to plot sleeve weights: {e}")
        try:
            plot_trend_status(rebalance_contexts, show=show)
        except Exception as e:
            print(f"[backtest_v2] Failed to plot trend status: {e}")

    # Save backtest results to CSV
    out_dir = Path("data").joinpath("backtests").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"backtest_{stamp}.csv"
    result.to_csv(out_path)
    print(f"[backtest_v2] Saved detailed backtest result to: {out_path}")

    # Also save summary stats to a separate CSV for easy inspection
    stats_path = out_dir / f"backtest_{stamp}_stats.csv"
    try:
        # Serialize stats as a single-row CSV; convert to DataFrame to handle mixed types
        pd.DataFrame([stats]).to_csv(stats_path, index=False)
        print(f"[backtest_v2] Saved backtest stats to: {stats_path}")
    except Exception as e:
        print(f"[backtest_v2] Failed to save stats CSV: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
