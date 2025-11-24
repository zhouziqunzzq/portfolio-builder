from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

from .utils.config import load_app_config
from .utils.logging import configure_logging


@dataclass
class AnalyzerConfig:
    frequency: str  # "monthly" or "daily"
    top: int
    as_of: Optional[pd.Timestamp]  # None means latest
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Portfolio Analyzer: explore portfolio allocations across time"
    )
    p.add_argument(
        "--strategy",
        default=str(Path(__file__).resolve().parents[1] / "config" / "strategy.yml"),
        help="Path to strategy.yml",
    )
    p.add_argument(
        "--frequency",
        choices=["monthly", "daily"],
        default="monthly",
        help="Weights frequency to analyze",
    )
    p.add_argument(
        "--as-of", default="latest", help="Date to snapshot (YYYY-MM-DD) or 'latest'"
    )
    p.add_argument(
        "--start",
        default=None,
        help="Optional start date YYYY-MM-DD for time-series summaries",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Optional end date YYYY-MM-DD for time-series summaries",
    )
    p.add_argument(
        "--top", type=int, default=20, help="Top N items to show for holdings output"
    )

    # Actions
    p.add_argument(
        "--list-dates",
        action="store_true",
        help="List available dates for selected frequency",
    )
    p.add_argument(
        "--print-sectors",
        action="store_true",
        help="Print sector allocation snapshot for --as-of date",
    )
    p.add_argument(
        "--print-holdings",
        action="store_true",
        help="Print stock allocation snapshot for --as-of date",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print brief time-series summary (positions, cash) for the selected range",
    )

    # Plotting options
    p.add_argument(
        "--plot-stacked",
        action="store_true",
        help="Generate stacked area plot for top-N holdings + Others + Cash",
    )
    p.add_argument(
        "--stacked-top",
        type=int,
        default=15,
        help="Top N tickers to include (stacked area)",
    )
    p.add_argument(
        "--stacked-downsample",
        default=None,
        help="Optional pandas offset alias to resample daily data (e.g., 'W-FRI')",
    )

    p.add_argument(
        "--plot-heatmap",
        action="store_true",
        help="Generate heatmap of weights for top-N tickers",
    )
    p.add_argument(
        "--heatmap-top",
        type=int,
        default=40,
        help="Top N tickers to include in heatmap",
    )
    p.add_argument(
        "--heatmap-cmap",
        default="viridis",
        help="Matplotlib/Seaborn colormap for heatmap",
    )

    p.add_argument(
        "--plot-bump",
        action="store_true",
        help="Generate bump chart of rank trajectories for top-K tickers",
    )
    p.add_argument(
        "--bump-top-k",
        type=int,
        default=10,
        help="Top K tickers to show in rank bump chart",
    )

    p.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving to disk",
    )

    # Specific tickers timeseries plot
    p.add_argument(
        "--plot-tickers",
        action="store_true",
        help="Plot weight time series for specified tickers (comma-separated via --tickers)",
    )
    p.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated list of tickers to plot (e.g. AAPL,MSFT,GOOGL)",
    )

    return p.parse_args()


def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    if isinstance(s, str) and s.lower() == "latest":
        return None
    return pd.to_datetime(s)


def _weights_globs(freq: str) -> Tuple[str, str]:
    if freq == "daily":
        return ("sector_weights_daily_*.csv", "stock_weights_daily_*.csv")
    return ("sector_weights_monthly_*.csv", "stock_weights_monthly_*.csv")


def _pick_asof_index(
    idx: pd.DatetimeIndex, as_of: Optional[pd.Timestamp]
) -> Optional[pd.Timestamp]:
    if idx.empty:
        return None
    if as_of is None:
        return idx[-1]
    as_of = pd.Timestamp(as_of)
    # find last index <= as_of
    loc = idx.get_indexer([as_of], method="pad")[0]
    if loc == -1:
        return None
    return idx[loc]


def load_weights(cfg, frequency: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weights_dir = (cfg.output_root_path / "weights").resolve()
    sector_glob, stock_glob = _weights_globs(frequency)

    # Sector weights
    sec_files = sorted(weights_dir.glob(sector_glob))
    sec_df = pd.DataFrame()
    if sec_files:
        sec_df = pd.read_csv(sec_files[-1], index_col=0)
        if not sec_df.empty:
            try:
                sec_df.index = pd.to_datetime(sec_df.index)
            except Exception:
                pass

    # Stock weights
    stk_files = sorted(weights_dir.glob(stock_glob))
    stk_df = pd.DataFrame()
    if stk_files:
        stk_df = pd.read_csv(stk_files[-1], index_col=0)
        if not stk_df.empty:
            try:
                stk_df.index = pd.to_datetime(stk_df.index)
            except Exception:
                pass

    return sec_df, stk_df


# --------------------------- Plotting helpers ---------------------------
def _import_matplotlib(logger):
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as e:
        logger.error(
            "Matplotlib is required for plotting. Please install matplotlib. (%s)", e
        )
        return None


def _import_seaborn():
    try:
        import seaborn as sns  # type: ignore

        return sns
    except Exception:
        return None


def _pick_top_tickers(
    df: pd.DataFrame, top_n: int = 20, method: str = "avg"
) -> List[str]:
    W = df.fillna(0.0)
    agg = W.mean() if method == "avg" else W.max()
    return list(agg.sort_values(ascending=False).head(top_n).index)


def _transform_topn_others_cash(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    W = df.fillna(0.0)
    top = _pick_top_tickers(W, top_n=top_n, method="avg")
    top_df = W[top].copy()
    others = W.drop(columns=top, errors="ignore").sum(axis=1)
    total = W.sum(axis=1)
    cash = (1.0 - total).clip(lower=0.0)
    out = top_df
    out["Others"] = others
    out["Cash"] = cash
    return out


def _maybe_downsample(df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if not rule:
        return df
    try:
        return df.resample(rule).last()
    except Exception:
        return df


def _ensure_plots_dir(cfg) -> Path:
    out = (cfg.output_root_path / "plots").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_stacked_topn(
    logger,
    cfg,
    df: pd.DataFrame,
    frequency: str,
    top_n: int = 15,
    downsample: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    plt = _import_matplotlib(logger)
    if plt is None:
        return None
    df_use = _maybe_downsample(df, downsample) if frequency == "daily" else df
    trans = _transform_topn_others_cash(df_use, top_n=top_n)
    cols = [c for c in trans.columns if c != "Cash"] + ["Cash"]
    Y = trans[cols].clip(lower=0.0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(Y.index, Y.T.values, labels=cols, alpha=0.9)
    title = f"Portfolio Weights Over Time (Top {top_n} + Others + Cash, {frequency})"
    if downsample:
        title += f" [downsample={downsample}]"
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=3, fontsize="small", loc="upper left", bbox_to_anchor=(0, -0.12))
    # Make x-axis tick density a bit higher than default
    try:
        from matplotlib import dates as mdates  # type: ignore

        if len(Y.index) > 0:
            # Bi-annual ticks (every 6 months) for both monthly and daily
            locator = mdates.MonthLocator(interval=6)
            formatter = mdates.DateFormatter("%Y-%m")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
    except Exception:
        pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(Y.index.min().date()) if len(Y.index) else "na"
    end_s = str(Y.index.max().date()) if len(Y.index) else "na"
    fname = f"stacked_{frequency}_{start_s}_{end_s}_top{top_n}{('_' + downsample) if downsample else ''}.png"
    out_path = plots_dir / fname
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved stacked plot: %s", out_path.name)
    return out_path


def plot_heatmap(
    logger,
    cfg,
    df: pd.DataFrame,
    frequency: str,
    top_n: int = 40,
    cmap: str = "viridis",
    show: bool = False,
) -> Optional[Path]:
    plt = _import_matplotlib(logger)
    sns = _import_seaborn()
    if plt is None:
        return None
    if sns is None:
        logger.error("Seaborn is required for heatmap. Please install seaborn.")
        return None

    top = _pick_top_tickers(df, top_n=top_n, method="max")
    mat = df[top].fillna(0.0).T
    # Convert datetime columns to date-only strings BEFORE plotting so seaborn doesn't include time
    try:
        mat.columns = [
            c.strftime("%Y-%m-%d") if isinstance(c, pd.Timestamp) else str(c)
            for c in mat.columns
        ]
    except Exception:
        mat.columns = [str(c) for c in mat.columns]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(mat, cmap=cmap, ax=ax, cbar_kws={"label": "Weight"})
    ax.set_title(f"Weight Heatmap (Top {top_n}, {frequency})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ticker")
    # Ensure x tick labels are the date-only strings
    try:
        ax.set_xticklabels(list(mat.columns), rotation=45, ha="right")
    except Exception:
        pass
    plt.tight_layout()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(df.index.min().date()) if len(df.index) else "na"
    end_s = str(df.index.max().date()) if len(df.index) else "na"
    out_path = plots_dir / f"heatmap_{frequency}_{start_s}_{end_s}_top{top_n}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved heatmap plot: %s", out_path.name)
    return out_path


def plot_bump_ranks(
    logger, cfg, df: pd.DataFrame, frequency: str, top_k: int = 10, show: bool = False
) -> Optional[Path]:
    plt = _import_matplotlib(logger)
    if plt is None:
        return None
    W = df.fillna(0.0)
    ranks = W.rank(axis=1, method="min", ascending=False)
    keep = ranks.mean().sort_values().head(top_k).index
    R = ranks[keep]
    inv = (R.max().max() + 1) - R
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in inv.columns:
        ax.plot(inv.index, inv[col].values, label=col, linewidth=2)
    ax.set_title(f"Rank Trajectories (Top {top_k}, {frequency})")
    ax.set_ylabel("Higher is better (rank inverted)")
    ax.legend(ncol=3, fontsize="small", loc="upper left", bbox_to_anchor=(0, -0.12))
    # Make x-axis tick density a bit higher than default
    try:
        from matplotlib import dates as mdates  # type: ignore

        if len(R.index) > 0:
            # Bi-annual ticks (every 6 months)
            locator = mdates.MonthLocator(interval=6)
            formatter = mdates.DateFormatter("%Y-%m")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
    except Exception:
        pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(df.index.min().date()) if len(df.index) else "na"
    end_s = str(df.index.max().date()) if len(df.index) else "na"
    out_path = plots_dir / f"bump_{frequency}_{start_s}_{end_s}_top{top_k}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved bump plot: %s", out_path.name)
    return out_path


def plot_tickers_timeseries(
    logger,
    cfg,
    df: pd.DataFrame,
    frequency: str,
    tickers: List[str],
    show: bool = False,
) -> Optional[Path]:
    """Plot raw weight time series for specified tickers.

    Missing tickers are ignored with a warning. Lines are plotted for each ticker; optional legend.
    """
    plt = _import_matplotlib(logger)
    if plt is None:
        return None
    if not tickers:
        logger.warning("No tickers provided for plot_tickers_timeseries")
        return None
    W = df.fillna(0.0).copy()
    # Normalize case: internal columns assumed uppercase
    cols_upper = {c.upper(): c for c in W.columns}
    chosen = []
    for t in tickers:
        tu = t.strip().upper()
        if tu in cols_upper:
            chosen.append(cols_upper[tu])
        else:
            logger.warning("Ticker %s not found in weights; skipping", tu)
    if not chosen:
        logger.warning("None of the requested tickers found; aborting plot")
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in chosen:
        ax.plot(W.index, W[col].values, label=col, linewidth=1.8)
    ax.set_title(f"Ticker Weight Time Series ({frequency})")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, max(0.05, float(W[chosen].max().max()) * 1.1))
    ax.legend(ncol=3, fontsize="small", loc="upper left", bbox_to_anchor=(0, -0.12))
    # Set bi-annual ticks on x-axis
    try:
        from matplotlib import dates as mdates  # type: ignore

        if len(W.index) > 0:
            locator = mdates.MonthLocator(interval=6)
            formatter = mdates.DateFormatter("%Y-%m")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
    except Exception:
        pass
    fig.autofmt_xdate()

    plots_dir = _ensure_plots_dir(cfg)
    start_s = str(W.index.min().date()) if len(W.index) else "na"
    end_s = str(W.index.max().date()) if len(W.index) else "na"
    tickers_tag = "_".join([t.strip().upper() for t in tickers])[:80]
    out_path = plots_dir / f"tickers_{frequency}_{start_s}_{end_s}_{tickers_tag}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info(
        "Saved tickers timeseries plot: %s (tickers=%s)",
        out_path.name,
        ",".join(tickers),
    )
    return out_path


def _format_pct(x: float) -> str:
    return f"{x*100.0:6.2f}%"


def _print_series(
    logger, header: str, s: pd.Series, top: Optional[int] = None, cash: float = 0.0
) -> None:
    logger.info(header)
    if s.empty and cash <= 0.0:
        logger.info("(empty)")
        return
    if top is not None and top > 0:
        s = s.head(top)
    for i, (k, v) in enumerate(s.items(), start=1):
        logger.info("%2d) %-24s %s", i, str(k), _format_pct(float(v)))
    if cash > 1e-12:
        logger.info("    %-24s %s", "Cash", _format_pct(float(cash)))


def _compute_cash(row: pd.Series) -> float:
    total = float(row.fillna(0.0).sum()) if len(row) else 0.0
    return max(0.0, 1.0 - total)


def _timeseries_summary(logger, df: pd.DataFrame, label: str) -> None:
    if df.empty:
        logger.info("No %s weights available for summary", label)
        return
    positions = (df.fillna(0.0) > 0).sum(axis=1)
    cash = 1.0 - df.fillna(0.0).sum(axis=1)
    logger.info("=== %s summary ===", label)
    logger.info(
        "dates: %s .. %s", str(df.index.min().date()), str(df.index.max().date())
    )
    logger.info(
        "avg positions: %.1f | min: %d | max: %d",
        float(positions.mean()),
        int(positions.min()),
        int(positions.max()),
    )
    logger.info(
        "avg cash: %s | min: %s | max: %s",
        _format_pct(float(cash.mean())),
        _format_pct(float(cash.min())),
        _format_pct(float(cash.max())),
    )


def main() -> int:
    args = parse_args()
    cfg = load_app_config(Path(args.strategy).resolve())
    logger = configure_logging(
        cfg.output_root_path,
        level=cfg.runtime.log_level,
        log_to_file=cfg.runtime.save.get("logs", True),
    )

    frequency = args.frequency
    as_of = _parse_date(args.as_of)
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    sec_df, stk_df = load_weights(cfg, frequency)

    # List dates
    if args.list_dates:
        if not sec_df.empty:
            logger.info(
                "Sector dates (%s): %s",
                frequency,
                ", ".join(d.strftime("%Y-%m-%d") for d in sec_df.index),
            )
        else:
            logger.info("No sector weights found for %s", frequency)
        if not stk_df.empty:
            logger.info(
                "Stock dates (%s): %s",
                frequency,
                ", ".join(d.strftime("%Y-%m-%d") for d in stk_df.index),
            )
        else:
            logger.info("No stock weights found for %s", frequency)

    # Snapshot printing
    if args.print_sectors and not sec_df.empty:
        dt = _pick_asof_index(sec_df.index, as_of)
        if dt is None:
            logger.info("No sector data on/before %s", args.as_of)
        else:
            row = sec_df.loc[dt].fillna(0.0)
            s = row[row > 0].sort_values(ascending=False)
            cash = _compute_cash(row)
            _print_series(
                logger,
                f"=== Sector allocation ({frequency.upper()} as of {dt.date()}) ===",
                s,
                None,
                cash,
            )

    if args.print_holdings and not stk_df.empty:
        dt = _pick_asof_index(stk_df.index, as_of)
        if dt is None:
            logger.info("No stock data on/before %s", args.as_of)
        else:
            row = stk_df.loc[dt].fillna(0.0)
            s = row[row > 0].sort_values(ascending=False)
            cash = _compute_cash(row)
            _print_series(
                logger,
                f"=== Stock allocation ({frequency.upper()} as of {dt.date()}) ===",
                s,
                int(args.top),
                cash,
            )

    # Summary over an interval
    if args.summary:
        # Restrict to [start, end] if provided
        if start is not None:
            sec_df = sec_df.loc[sec_df.index >= start]
            stk_df = stk_df.loc[stk_df.index >= start]
        if end is not None:
            sec_df = sec_df.loc[sec_df.index <= end]
            stk_df = stk_df.loc[stk_df.index <= end]
        _timeseries_summary(logger, sec_df, f"Sectors ({frequency})")
        _timeseries_summary(logger, stk_df, f"Stocks ({frequency})")

    # If no explicit actions chosen, default to printing both snapshots at latest
    if not any(
        [args.list_dates, args.print_sectors, args.print_holdings, args.summary]
    ):
        dt_s = _pick_asof_index(sec_df.index, None) if not sec_df.empty else None
        if dt_s is not None:
            row = sec_df.loc[dt_s].fillna(0.0)
            s = row[row > 0].sort_values(ascending=False)
            cash = _compute_cash(row)
            _print_series(
                logger,
                f"=== Sector allocation ({frequency.upper()} as of {dt_s.date()}) ===",
                s,
                None,
                cash,
            )
        dt_h = _pick_asof_index(stk_df.index, None) if not stk_df.empty else None
        if dt_h is not None:
            row = stk_df.loc[dt_h].fillna(0.0)
            s = row[row > 0].sort_values(ascending=False)
            cash = _compute_cash(row)
            _print_series(
                logger,
                f"=== Stock allocation ({frequency.upper()} as of {dt_h.date()}) ===",
                s,
                int(args.top),
                cash,
            )

    # Plotting actions
    if args.plot_stacked and not stk_df.empty:
        plot_stacked_topn(
            logger,
            cfg,
            df=stk_df,
            frequency=frequency,
            top_n=int(args.stacked_top),
            downsample=args.stacked_downsample,
            show=bool(args.show),
        )

    if args.plot_heatmap and not stk_df.empty:
        plot_heatmap(
            logger,
            cfg,
            df=stk_df,
            frequency=frequency,
            top_n=int(args.heatmap_top),
            cmap=str(args.heatmap_cmap),
            show=bool(args.show),
        )

    if args.plot_bump and not stk_df.empty:
        plot_bump_ranks(
            logger,
            cfg,
            df=stk_df,
            frequency=frequency,
            top_k=int(args.bump_top_k),
            show=bool(args.show),
        )

    if args.plot_tickers and not stk_df.empty:
        raw_list = []
        if args.tickers:
            raw_list = [t.strip() for t in str(args.tickers).split(",") if t.strip()]
        plot_tickers_timeseries(
            logger,
            cfg,
            df=stk_df,
            frequency=frequency,
            tickers=raw_list,
            show=bool(args.show),
        )

    return 0
