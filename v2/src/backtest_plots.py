from __future__ import annotations

from typing import Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd


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


def plot_regime_scores(
    contexts: Dict[pd.Timestamp, Dict[str, Any]], show: bool = False
) -> Optional[Path]:
    plt, mdates = _import_matplotlib()
    if plt is None:
        return None

    if not contexts:
        print("[backtest_v2] No regime contexts to plot.")
        return None

    dates = sorted(contexts.keys())
    # collect all regime names
    regime_names = set()
    for c in contexts.values():
        rs = c.get("regime_scores") or {}
        regime_names.update(rs.keys())

    df = pd.DataFrame(index=dates, columns=sorted(regime_names), dtype=float).fillna(
        0.0
    )
    for dt, c in contexts.items():
        rs = c.get("regime_scores") or {}
        for k, v in rs.items():
            df.at[dt, k] = float(v)

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in df.columns:
        ax.plot(df.index, df[col].values, label=col)
    ax.set_title("Regime Soft Scores Over Rebalance Dates")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
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
    start_s = str(dates[0].date()) if dates else "na"
    end_s = str(dates[-1].date()) if dates else "na"
    out_path = plots_dir / f"v2_regime_scores_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[backtest_v2] Saved regime scores plot: {out_path}")
    return out_path


def plot_sleeve_weights(
    contexts: Dict[pd.Timestamp, Dict[str, Any]], show: bool = False
) -> Optional[Path]:
    plt, mdates = _import_matplotlib()
    if plt is None:
        return None

    if not contexts:
        print("[backtest_v2] No sleeve context to plot.")
        return None

    dates = sorted(contexts.keys())
    # collect sleeve names
    sleeve_names = set()
    for c in contexts.values():
        sw = c.get("sleeve_weights") or {}
        sleeve_names.update(sw.keys())

    df = pd.DataFrame(index=dates, columns=sorted(sleeve_names), dtype=float).fillna(
        0.0
    )
    for dt, c in contexts.items():
        sw = c.get("sleeve_weights") or {}
        for k, v in sw.items():
            df.at[dt, k] = float(v)

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in df.columns:
        ax.plot(df.index, df[col].values, label=col)
    ax.set_title("Effective Sleeve Weights Over Rebalance Dates")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
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
    start_s = str(dates[0].date()) if dates else "na"
    end_s = str(dates[-1].date()) if dates else "na"
    out_path = plots_dir / f"v2_sleeve_weights_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[backtest_v2] Saved sleeve weights plot: {out_path}")
    return out_path


def plot_trend_status(
    contexts: Dict[pd.Timestamp, Dict[str, Any]], show: bool = False
) -> Optional[Path]:
    """
    Plot binary trend status (risk-on=1 / risk-off=0) over rebalance dates.

    Expects each context to contain a `trend_status` key with values
    like "risk-on" or "risk-off".
    """
    plt, mdates = _import_matplotlib()
    if plt is None:
        return None

    if not contexts:
        print("[backtest_v2] No regime contexts to plot trend status.")
        return None

    dates = sorted(contexts.keys())
    series = []
    for d in dates:
        c = contexts.get(d) or {}
        ts = c.get("trend_status")
        series.append(1 if ts == "risk-on" else 0)

    idx = pd.DatetimeIndex(dates)
    s = pd.Series(series, index=idx)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(s.index, s.values, where="post", linewidth=2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["risk-off", "risk-on"])
    ax.set_title("Trend Filter Status Over Rebalance Dates")
    ax.set_xlabel("Date")
    ax.set_ylabel("Status")
    ax.grid(True, linestyle="--", linewidth=0.5)

    if mdates is not None:
        try:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        except Exception:
            pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir()
    start_s = str(dates[0].date()) if dates else "na"
    end_s = str(dates[-1].date()) if dates else "na"
    out_path = plots_dir / f"v2_trend_status_{start_s}_{end_s}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[backtest_v2] Saved trend status plot: {out_path}")
    return out_path
