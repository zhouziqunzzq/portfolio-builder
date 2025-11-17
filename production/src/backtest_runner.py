from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd

from .utils.config import load_app_config
from .utils.logging import configure_logging
from .universe_manager import UniverseManager
from .market_data_store import MarketDataStore
from portfolio_backtester import PortfolioBacktester
from .utils.backtest_plotting import (
    plot_equity_curve,
    plot_drawdown,
    plot_calendar_year_returns,
    plot_rolling_sharpe,
    plot_monthly_turnover,
    plot_sector_allocation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Momentum V1.5 Backtest Runner")
    p.add_argument("--strategy", default=str(Path(__file__).resolve().parents[1] / "config" / "strategy.yml"), help="Path to strategy.yml")
    p.add_argument("--sectors", default=str(Path(__file__).resolve().parents[1] / "config" / "sectors.yml"), help="Path to sectors.yml")
    p.add_argument("--local-only", action="store_true", help="Disable network calls and use only local caches/artifacts")
    p.add_argument("--backtest-start", default=None, help="Explicit backtest start date (YYYY-MM-DD); overrides earliest weight date")
    # Plotting controls
    p.add_argument("--plot-all", action="store_true", help="Generate all standard plots")
    p.add_argument("--plot-equity", action="store_true", help="Plot equity curve (log)")
    p.add_argument("--plot-drawdown", action="store_true", help="Plot strategy drawdown")
    p.add_argument("--plot-annual", action="store_true", help="Plot calendar-year returns vs benchmark")
    p.add_argument("--plot-rolling", action="store_true", help="Plot rolling 1-year (252-day) Sharpe vs benchmark")
    p.add_argument("--plot-turnover", action="store_true", help="Plot average monthly turnover time series")
    p.add_argument("--plot-sectors", action="store_true", help="Plot sector allocation stacked area (monthly)")
    p.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving to disk")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    strategy_yaml = Path(args.strategy).resolve()
    sectors_yaml = Path(args.sectors).resolve()

    cfg = load_app_config(strategy_yaml)
    logger = configure_logging(cfg.output_root_path, level=cfg.runtime.log_level, log_to_file=cfg.runtime.save.get("logs", True))
    logger.info("Starting BacktestRunner")

    # Managers
    um = UniverseManager(membership_csv=cfg.membership_csv_path, sectors_yaml=sectors_yaml, local_only=bool(args.__dict__.get("local_only", False)))
    mds = MarketDataStore(data_root=str((cfg.output_root_path / "prices").resolve()))

    try:
        weights_dir = (cfg.output_root_path / "weights").resolve()
        if not weights_dir.exists():
            logger.warning("Weights directory %s does not exist; run stock weight computation first", weights_dir)
            return 0

        stock_weight_files = sorted(weights_dir.glob("stock_weights_monthly_*.csv"))
        if not stock_weight_files:
            logger.warning("No stock_weights_monthly_*.csv found under %s; run stock weight computation first", weights_dir)
            return 0

        latest_sw = stock_weight_files[-1]
        stock_weights_monthly = pd.read_csv(latest_sw, index_col=0)
        stock_weights_monthly.index = pd.to_datetime(stock_weights_monthly.index)

        # Determine date range
        override_start = None
        if getattr(args, "backtest_start", None):
            try:
                override_start = datetime.strptime(str(args.backtest_start), "%Y-%m-%d").date()
            except Exception:
                logger.warning("Invalid --backtest-start=%s (expected YYYY-MM-DD); ignoring", args.backtest_start)

        earliest_weight_date = stock_weights_monthly.index.min().date()
        latest_weight_date = stock_weights_monthly.index.max().date()

        if override_start:
            if override_start > latest_weight_date:
                logger.warning("--backtest-start %s is after last weight date %s; aborting backtest", override_start, latest_weight_date)
                return 0
            elif override_start < earliest_weight_date:
                logger.info("--backtest-start %s precedes earliest weight date %s; using earliest weight date", override_start, earliest_weight_date)
                override_start = earliest_weight_date

        effective_start_date = override_start or earliest_weight_date
        # Trim weights to effective_start_date onward
        stock_weights_monthly = stock_weights_monthly.loc[stock_weights_monthly.index.date >= effective_start_date]
        if stock_weights_monthly.empty:
            logger.warning("No weights remain after applying backtest start date %s", effective_start_date)
            return 0

        start_dt = effective_start_date
        end_dt = stock_weights_monthly.index.max().date()

        tickers = um.tickers
        price_mat = um.get_price_matrix(
            price_loader=mds,
            tickers=tickers,
            start=str(start_dt),
            end=str(end_dt),
            field=None,
            interval="1d",
            local_only=bool(args.__dict__.get("local_only", False)),
        )

        if price_mat.empty:
            logger.warning("Price matrix empty for backtest window [%s..%s]", start_dt, end_dt)
            return 0

        # Align prices to full date range
        price_mat = price_mat.sort_index()

        # Backtester (cost could be exposed via config later)
        bt = PortfolioBacktester(
            prices=price_mat,
            weights=stock_weights_monthly,
            trading_days_per_year=252,
            initial_value=float(getattr(cfg.strategy, "initial_equity", 100_000.0)),
            cost_per_turnover=float(getattr(cfg.strategy, "cost_per_turnover", 0.001)),
        )
        result = bt.run()
        stats = bt.stats(result, auto_warmup=True, warmup_days=0)

        eff_start = stats.get("EffectiveStart")
        eff_end = stats.get("EffectiveEnd")

        # Benchmark (SPY) basic stats
        benchmark = getattr(cfg.sectors.trend_filter, "benchmark", "SPY") if getattr(cfg, "sectors", None) and getattr(cfg.sectors, "trend_filter", None) else "SPY"
        df_bench = mds.get_ohlcv(
            benchmark,
            start=str(start_dt),
            end=str(end_dt),
            interval="1d",
            auto_adjust=True,
            local_only=bool(args.__dict__.get("local_only", False)),
        )
        bench_stats = {}
        bench_series = None
        bench_returns = None
        if df_bench is None or df_bench.empty:
            logger.warning("Benchmark data unavailable for %s; skipping benchmark stats", benchmark)
        else:
            price_col = "Adjclose" if "Adjclose" in df_bench.columns else ("Close" if "Close" in df_bench.columns else None)
            if price_col:
                bench_series = df_bench[price_col].copy()
                if eff_start and eff_end:
                    bench_series = bench_series.reindex(result.index).ffill().bfill()
                    bench_series = bench_series.loc[eff_start:eff_end]
                bench_returns = bench_series.pct_change().fillna(0.0)
                if len(bench_returns) > 0:
                    total_ret = (1 + bench_returns).prod() - 1
                    # Basic CAGR; PortfolioBacktester.stats already guards its own
                    cagr = total_ret ** (252 / len(bench_returns)) - 1
                    vol = bench_returns.std() * (252 ** 0.5)
                    sharpe = (bench_returns.mean() / bench_returns.std() * (252 ** 0.5)) if bench_returns.std() > 0 else float("nan")
                    eq = (1 + bench_returns).cumprod()
                    dd = eq / eq.cummax() - 1
                    max_dd = dd.min()
                    bench_stats = {
                        "CAGR": cagr,
                        "Volatility": vol,
                        "Sharpe": sharpe,
                        "MaxDrawdown": max_dd,
                    }
            else:
                logger.warning("Benchmark %s lacks Close columns for stats", benchmark)

        def pct_fmt(x):
            return f"{x*100:.2f}%" if x is not None and not pd.isna(x) else "n/a"

        def num_fmt(x):
            return f"{x:.2f}" if x is not None and not pd.isna(x) else "n/a"

        logger.info(
            "Backtest stats: start_param=%s effective_window=[%s..%s] CAGR=%s Vol=%s Sharpe=%s MaxDD=%s AvgTurnover=%s FinalEquity=%.2f",
            getattr(args, "backtest_start", None),
            eff_start.date() if eff_start else None,
            eff_end.date() if eff_end else None,
            pct_fmt(stats.get("CAGR")),
            pct_fmt(stats.get("Volatility")),
            num_fmt(stats.get("Sharpe")),
            pct_fmt(stats.get("MaxDrawdown")),
            pct_fmt(stats.get("AvgDailyTurnover")),
            result["equity"].iloc[-1] if len(result) else float("nan"),
        )

        if bench_stats:
            logger.info(
                "Benchmark stats (%s): CAGR=%s Vol=%s Sharpe=%s MaxDD=%s",
                benchmark,
                pct_fmt(bench_stats.get("CAGR")),
                pct_fmt(bench_stats.get("Volatility")),
                num_fmt(bench_stats.get("Sharpe")),
                pct_fmt(bench_stats.get("MaxDrawdown")),
            )

        # -------------------- Plotting (optional) --------------------
        do_all = bool(args.__dict__.get("plot_all", False))
        do_equity = do_all or bool(args.__dict__.get("plot_equity", False))
        do_drawdown = do_all or bool(args.__dict__.get("plot_drawdown", False))
        do_annual = do_all or bool(args.__dict__.get("plot_annual", False))
        do_rolling = do_all or bool(args.__dict__.get("plot_rolling", False))
        do_turnover = do_all or bool(args.__dict__.get("plot_turnover", False))
        do_sectors = do_all or bool(args.__dict__.get("plot_sectors", False))
        show = bool(args.__dict__.get("show", False))

        # Slice strategy series to effective window
        strat_eq = result["equity"]
        strat_rets = result["portfolio_return"]
        if eff_start and eff_end:
            strat_eq = strat_eq.loc[eff_start:eff_end]
            strat_rets = strat_rets.loc[eff_start:eff_end]

        if do_equity:
            plot_equity_curve(logger, cfg, strat_eq=strat_eq, bench_eq=bench_series, show=show)
        if do_drawdown:
            plot_drawdown(logger, cfg, strat_eq=strat_eq, show=show)
        if do_annual:
            plot_calendar_year_returns(logger, cfg, strat_rets=strat_rets, bench_rets=bench_returns, show=show)
        if do_rolling:
            plot_rolling_sharpe(logger, cfg, strat_rets=strat_rets, bench_rets=bench_returns, window=252, show=show)
        if do_turnover:
            plot_monthly_turnover(logger, cfg, daily_turnover=result["turnover"], show=show)

        if do_sectors:
            # Try to locate matching daily sector weights for the same weight snapshot stem
            try:
                weights_dir = (cfg.output_root_path / "weights").resolve()
                # latest_sw: .../stock_weights_monthly_<stem>.csv
                stem_token = latest_sw.stem.replace("stock_weights_monthly_", "")
                daily_sector_path = weights_dir / f"sector_weights_daily_{stem_token}.csv"
                sector_weights_daily = None
                if daily_sector_path.exists():
                    sector_weights_daily = pd.read_csv(daily_sector_path, index_col=0)
                    sector_weights_daily.index = pd.to_datetime(sector_weights_daily.index)
                if sector_weights_daily is not None and not sector_weights_daily.empty:
                    start_w = eff_start.date() if eff_start else None
                    end_w = eff_end.date() if eff_end else None
                    plot_sector_allocation(logger, cfg, sector_weights_daily=sector_weights_daily, start=start_w, end=end_w, show=show)
                else:
                    logger.warning("No matching sector_weights_daily found for stem %s; skipping sector allocation plot", stem_token)
            except Exception as e:
                logger.warning("Failed sector allocation plotting: %s", e)

    except Exception as e:
        logger.exception("Monthly backtest failed: %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
