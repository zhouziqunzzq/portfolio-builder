from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _import_matplotlib(logger):
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import dates as mdates  # type: ignore
        return plt, mdates
    except Exception as e:
        logger.error("Matplotlib is required for plotting. Please install matplotlib. (%s)", e)
        return None, None


def _ensure_plots_dir(cfg) -> Path:
    out = (cfg.output_root_path / "plots").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_equity_curve(logger, cfg, strat_eq: pd.Series, bench_eq: Optional[pd.Series] = None, show: bool = False) -> Optional[Path]:
    plt, mdates = _import_matplotlib(logger)
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strat_eq.index, strat_eq.values, label="Strategy")
    if bench_eq is not None and len(bench_eq) > 0:
        # Normalize bench to strategy initial
        try:
            bench_scaled = bench_eq * (strat_eq.iloc[0] / bench_eq.iloc[0])
            ax.plot(bench_scaled.index, bench_scaled.values, label="Benchmark", alpha=0.85)
        except Exception:
            ax.plot(bench_eq.index, bench_eq.values, label="Benchmark", alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Equity Curve (Log Scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    try:
        if mdates is not None:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    except Exception:
        pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(pd.to_datetime(strat_eq.index.min()).date()) if len(strat_eq) else "na"
    end_s = str(pd.to_datetime(strat_eq.index.max()).date()) if len(strat_eq) else "na"
    out_path = plots_dir / f"backtest_equity_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved equity plot: %s", out_path.name)
    return out_path


def plot_drawdown(logger, cfg, strat_eq: pd.Series, show: bool = False) -> Optional[Path]:
    plt, mdates = _import_matplotlib(logger)
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
    try:
        if mdates is not None:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    except Exception:
        pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(pd.to_datetime(dd.index.min()).date()) if len(dd) else "na"
    end_s = str(pd.to_datetime(dd.index.max()).date()) if len(dd) else "na"
    out_path = plots_dir / f"backtest_drawdown_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved drawdown plot: %s", out_path.name)
    return out_path


def plot_calendar_year_returns(logger, cfg, strat_rets: pd.Series, bench_rets: Optional[pd.Series] = None, show: bool = False) -> Optional[Path]:
    plt, _ = _import_matplotlib(logger)
    if plt is None:
        return None
    strat_yearly = (1 + strat_rets).groupby(strat_rets.index.year).prod() - 1
    bench_yearly = None
    if bench_rets is not None and len(bench_rets) > 0:
        bench_yearly = (1 + bench_rets).groupby(bench_rets.index.year).prod() - 1
        # Align years
        idx = strat_yearly.index.intersection(bench_yearly.index)
        strat_yearly = strat_yearly.loc[idx]
        bench_yearly = bench_yearly.loc[idx]
    years = strat_yearly.index.astype(int)
    x = range(len(years))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], strat_yearly.values, width=width, label="Strategy")
    if bench_yearly is not None:
        ax.bar([i + width/2 for i in x], bench_yearly.values, width=width, label="Benchmark")
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(years), rotation=45)
    ax.axhline(0, linestyle="--", linewidth=0.5)
    ax.set_title("Calendar-Year Returns")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(years.min()) if len(years) else "na"
    end_s = str(years.max()) if len(years) else "na"
    out_path = plots_dir / f"backtest_calendar_returns_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved calendar returns plot: %s", out_path.name)
    return out_path


def plot_rolling_sharpe(logger, cfg, strat_rets: pd.Series, bench_rets: Optional[pd.Series] = None, window: int = 252, show: bool = False) -> Optional[Path]:
    plt, mdates = _import_matplotlib(logger)
    if plt is None:
        return None
    roll_mean_s = strat_rets.rolling(window).mean()
    roll_std_s = strat_rets.rolling(window).std()
    s_sharpe = (roll_mean_s / roll_std_s) * (252 ** 0.5)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s_sharpe.index, s_sharpe.values, label="Strategy")
    if bench_rets is not None and len(bench_rets) > 0:
        roll_mean_b = bench_rets.rolling(window).mean()
        roll_std_b = bench_rets.rolling(window).std()
        b_sharpe = (roll_mean_b / roll_std_b) * (252 ** 0.5)
        ax.plot(b_sharpe.index, b_sharpe.values, label="Benchmark", linestyle="--")
    ax.axhline(0, linestyle="--", linewidth=0.5)
    ax.set_title(f"Rolling {int(window/21):d}-Month Sharpe ({window}-day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    try:
        if mdates is not None:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    except Exception:
        pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(pd.to_datetime(strat_rets.index.min()).date()) if len(strat_rets) else "na"
    end_s = str(pd.to_datetime(strat_rets.index.max()).date()) if len(strat_rets) else "na"
    out_path = plots_dir / f"backtest_rolling_sharpe_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved rolling Sharpe plot: %s", out_path.name)
    return out_path


def plot_monthly_turnover(logger, cfg, daily_turnover: pd.Series, show: bool = False) -> Optional[Path]:
    plt, mdates = _import_matplotlib(logger)
    if plt is None:
        return None
    monthly = daily_turnover.resample("ME").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values)
    ax.set_title("Average Monthly Turnover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover (fraction)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    try:
        if mdates is not None:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    except Exception:
        pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(pd.to_datetime(monthly.index.min()).date()) if len(monthly) else "na"
    end_s = str(pd.to_datetime(monthly.index.max()).date()) if len(monthly) else "na"
    out_path = plots_dir / f"backtest_monthly_turnover_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved monthly turnover plot: %s", out_path.name)
    return out_path


def plot_sector_allocation(logger, cfg, sector_weights_daily: pd.DataFrame, start=None, end=None, show: bool = False) -> Optional[Path]:
    plt, mdates = _import_matplotlib(logger)
    if plt is None:
        return None
    SW = sector_weights_daily.copy()
    if start is not None:
        SW = SW.loc[SW.index >= pd.to_datetime(start)]
    if end is not None:
        SW = SW.loc[SW.index <= pd.to_datetime(end)]
    if SW.empty:
        logger.warning("No sector weights available for plotting in the selected window")
        return None
    monthly = SW.resample("ME").last()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(monthly.index, [monthly[c].values for c in monthly.columns], labels=monthly.columns)
    ax.set_title("Sector Allocation Over Time (Monthly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(ncol=2, fontsize="small", loc="upper left", bbox_to_anchor=(0, -0.12))
    try:
        if mdates is not None:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    except Exception:
        pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(pd.to_datetime(monthly.index.min()).date()) if len(monthly) else "na"
    end_s = str(pd.to_datetime(monthly.index.max()).date()) if len(monthly) else "na"
    out_path = plots_dir / f"backtest_sector_allocation_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved sector allocation plot: %s", out_path.name)
    return out_path
