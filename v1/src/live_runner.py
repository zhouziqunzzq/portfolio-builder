from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from datetime import timedelta
from .signal_engine import SignalEngine

from .utils.config import load_app_config
from .utils.logging import configure_logging
from .universe_manager import UniverseManager
from .market_data_store import MarketDataStore
from .sector_weight_engine import SectorWeightEngine
from .stock_allocator import StockAllocator
 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Momentum V1.5 Production Runner")
    p.add_argument("--date", default="today", help="Run date (YYYY-MM-DD) or 'today'")
    p.add_argument("--strategy", default=str(Path(__file__).resolve().parents[1] / "config" / "strategy.yml"), help="Path to strategy.yml")
    p.add_argument("--sectors", default=str(Path(__file__).resolve().parents[1] / "config" / "sectors.yml"), help="Path to sectors.yml")
    p.add_argument("--local-only", action="store_true", help="Disable network calls and use only local caches/artifacts")

    # Main pipeline actions
    p.add_argument("--regen-universe", action="store_true", help="Rebuild universe artifacts: current constituents snapshot, historical membership, sector enrichment")
    p.add_argument("--update-prices", action="store_true", help="Update OHLCV cache for tickers and compute SPY trend status")
    p.add_argument("--rebalance", action="store_true", help="Force a rebalance run")
    p.add_argument("--rebalance-start", default=None, help="Explicit start date YYYY-MM-DD for rebalance; overrides --signal-start during rebalance")

    # Single-step actions that can be run independently
    p.add_argument("--compute-signals", action="store_true", help="Compute stock scores and sector scores using SignalEngine over window-union membership")
    p.add_argument("--signal-start", default=None, help="Explicit start date YYYY-MM-DD for signal window; overrides automatic lookback")
    p.add_argument("--compute-sector-weights", action="store_true", help="Compute sector weights from sector scores and benchmark using SectorWeightEngine")
    p.add_argument("--compute-stock-weights", action="store_true", help="Allocate sector weights to stocks using StockAllocator")
    p.add_argument("--dump-membership-mask", action="store_true", help="Dump membership mask CSV/Parquet under output_root/masks for archival")

    # Debugging options
    p.add_argument("--mask-summary", action="store_true", help="Print a brief summary of the membership mask for a recent window")
    p.add_argument("--mask-start", default=None, help="Optional start date YYYY-MM-DD for mask summary (defaults to run_date - 30 days)")
    p.add_argument("--mask-end", default=None, help="Optional end date YYYY-MM-DD for mask summary (defaults to run_date)")
  
    
    return p.parse_args()


def resolve_run_date(s: str) -> date:
    if s.lower() == "today":
        return datetime.utcnow().date()
    return datetime.strptime(s, "%Y-%m-%d").date()


def regenerate_universe(um: UniverseManager, cfg, logger) -> None:
    """Full universe regeneration pipeline.

    Steps:
    1. Fetch current constituents snapshot (Wikipedia) and save filtered copy.
    2. Build full historical membership, enrich sectors, optional filtering of unknowns.
    3. Persist final enriched membership CSV to configured path plus raw & enriched diagnostic copies.
    """
    try:
        # 1) Current constituents snapshot
        cur_df = um.build_current_constituents()
        if cur_df is None or cur_df.empty:
            logger.warning("Current constituents snapshot empty or unavailable")
        else:
            if getattr(cfg.universe, "filter_unknown_sectors", False):
                cur_df = um.filter_unknown_sectors(cur_df)
            fname_cur = getattr(cfg.universe, "current_constituents_filename", "current_constituents.csv")
            cur_out = (cfg.output_root_path / fname_cur).resolve()
            cur_out.parent.mkdir(parents=True, exist_ok=True)
            cur_df.to_csv(cur_out, index=False)
            sector_counts = cur_df.get("sector").value_counts().to_dict() if "sector" in cur_df.columns else {}
            logger.info(
                "Saved current constituents snapshot",
                extra={"rows": len(cur_df), "path": str(cur_out), "sector_counts": sector_counts},
            )
    except Exception as e:
        logger.exception("Failed step: current constituents snapshot: %s", e)

    try:
        # 2) Build historical membership
        hist_raw = um.build_historical_membership()
        if hist_raw is None or hist_raw.empty:
            logger.warning("Historical membership build returned empty DataFrame")
            return

        unknown_before = 0
        if "sector" in hist_raw.columns:
            unknown_before = int((hist_raw["sector"].astype(str).str.strip() == "Unknown").sum())

        hist_enriched = um.enrich_sectors(hist_raw)
        unknown_after = 0
        if "sector" in hist_enriched.columns:
            unknown_after = int((hist_enriched["sector"].astype(str).str.strip() == "Unknown").sum())

        # Optional filtering
        if getattr(cfg.universe, "filter_unknown_sectors", False):
            hist_final = um.filter_unknown_sectors(hist_enriched)
        else:
            hist_final = hist_enriched

        # 3) Persist artifacts
        target_path = cfg.membership_csv_path.resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        hist_final.to_csv(target_path, index=False)

        # Resolve raw/enriched filenames (allow explicit overrides in config)
        raw_fn = getattr(cfg.universe, "membership_raw_filename", None)
        enr_fn = getattr(cfg.universe, "membership_enriched_filename", None)

        if raw_fn:
            raw_path = target_path.parent / raw_fn
        else:
            raw_path = target_path.with_name(target_path.stem + "_raw" + target_path.suffix)
        hist_raw.to_csv(raw_path, index=False)

        if enr_fn:
            enriched_path = target_path.parent / enr_fn
        else:
            enriched_path = target_path.with_name(target_path.stem + "_enriched" + target_path.suffix)
        hist_enriched.to_csv(enriched_path, index=False)

        logger.info(
            "Universe regeneration complete",
            extra={
                "membership_path": str(target_path),
                "rows_final": len(hist_final),
                "unknown_before": unknown_before,
                "unknown_after": unknown_after,
                "raw_path": str(raw_path),
                "enriched_path": str(enriched_path),
            },
        )
        # Auto dump membership mask snapshot after regeneration
        try:
            dump_membership_mask(args=argparse.Namespace(mask_start=None, mask_end=None), um=um, cfg=cfg, logger=logger, run_dt=datetime.utcnow().date())
        except Exception as e:
            logger.warning("Auto dump of membership mask failed after regeneration: %s", e)
    except Exception as e:
        logger.exception("Failed step: historical membership/enrichment: %s", e)


def summarize_membership_mask(args: argparse.Namespace, um: UniverseManager, logger) -> None:
    """Print a brief membership mask summary for a requested window.

    Uses --mask-start / --mask-end if provided; otherwise defaults to the last 30 days
    ending at the run date specified by --date.
    """
    try:
        end_dt = resolve_run_date(args.date)
        start_arg = getattr(args, "mask_start", None)
        end_arg = getattr(args, "mask_end", None)
        if start_arg is None:
            start_dt = end_dt - timedelta(days=30)
        else:
            start_dt = datetime.strptime(start_arg, "%Y-%m-%d").date()
        if end_arg is not None:
            end_dt = datetime.strptime(end_arg, "%Y-%m-%d").date()

        mask = um.membership_mask(start=str(start_dt), end=str(end_dt))
        mean_members = float(mask.sum(axis=1).mean()) if not mask.empty else 0.0
        start_s = str(mask.index.min().date()) if not mask.empty else str(start_dt)
        end_s = str(mask.index.max().date()) if not mask.empty else str(end_dt)
        logger.info(
            f"Membership mask summary: window=[{start_s}..{end_s}] days={mask.shape[0]} tickers={mask.shape[1]} avg_members={mean_members:.1f}"
        )
    except Exception as e:
        logger.exception("Failed to compute mask summary: %s", e)


def dump_membership_mask(
    args: argparse.Namespace,
    um: UniverseManager,
    cfg,
    logger,
    run_dt: date | None = None,
) -> None:
    """Dump a membership mask snapshot to output_root/masks.

    If --mask-start/--mask-end are provided, restrict to that window; otherwise
    infer the full historical range from the membership CSV:
      - Daily schema: use [min(date), max(date)].
      - Range schema: use [min(date_added), max(date_removed or run_dt)].
    """
    try:
        # Load membership to infer default window
        mem = um.load_from_membership_csv()
        start_arg = getattr(args, "mask_start", None)
        end_arg = getattr(args, "mask_end", None)

        if start_arg and end_arg:
            start_dt = datetime.strptime(str(start_arg), "%Y-%m-%d").date()
            end_dt = datetime.strptime(str(end_arg), "%Y-%m-%d").date()
        else:
            # Infer from schema
            if {"date", "ticker", "in_sp500"}.issubset(mem.columns):
                start_dt = pd.to_datetime(mem["date"]).min().date()
                end_dt = pd.to_datetime(mem["date"]).max().date()
            elif {"ticker", "date_added"}.issubset(mem.columns):
                start_dt = pd.to_datetime(mem["date_added"]).min().date()
                if "date_removed" in mem.columns:
                    dr = pd.to_datetime(mem["date_removed"]).copy()
                    # Replace NaT with run_dt (or today if not provided)
                    if run_dt is None:
                        run_dt = datetime.utcnow().date()
                    dr = dr.fillna(pd.Timestamp(run_dt))
                    end_dt = dr.max().date()
                else:
                    end_dt = (run_dt or datetime.utcnow().date())
            else:
                raise ValueError("Unsupported membership CSV schema for mask dump")

        # Build and persist mask
        mask = um.membership_mask(start=str(start_dt), end=str(end_dt))
        out_dir = (cfg.output_root_path / "masks").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        fname_stem = f"membership_mask_{start_dt}_{end_dt}"
        csv_path = out_dir / f"{fname_stem}.csv"
        pq_path = out_dir / f"{fname_stem}.parquet"
        mask.astype(int).to_csv(csv_path)
        try:
            mask.astype(bool).to_parquet(pq_path)
        except Exception as e:
            logger.warning("Parquet save failed for membership mask (%s); CSV written at %s", e, csv_path)
        logger.info(
            "Membership mask dumped",
            extra={
                "rows": int(mask.shape[0]),
                "tickers": int(mask.shape[1]),
                "start": str(start_dt),
                "end": str(end_dt),
                "path": str(csv_path.name),
            },
        )
    except Exception as e:
        logger.exception("Failed to dump membership mask: %s", e)


def update_prices_and_trend(
    args: argparse.Namespace,
    um: UniverseManager,
    mds: MarketDataStore,
    cfg,
    logger,
    run_dt: date,
) -> None:
    """Update OHLCV cache for selected tickers and log benchmark trend status.
    """
    # Determine lookback window to cover momentum/vol/trend warmups
    trend_window = int(cfg.sectors.trend_filter.get("window", 200)) if cfg.sectors.trend_filter else 200
    momentum_max = max(cfg.signals.momentum_windows)
    vol_window = int(cfg.signals.vol_window)
    warmup = int(cfg.strategy.warmup_days)
    days_back = max(warmup, momentum_max + vol_window + 10, trend_window + 10, 260)

    start_dt = run_dt - timedelta(days=days_back)
    end_dt = run_dt

    # Use all tickers in the universe manager
    tickers = um.tickers

    # Always include benchmark
    benchmark = cfg.sectors.trend_filter.get("benchmark", "SPY") if cfg.sectors.trend_filter else "SPY"
    if benchmark not in tickers:
        tickers.append(benchmark)

    # Update cache for each ticker
    updated = 0
    for t in tickers:
        try:
            df = mds.get_ohlcv(
                ticker=t,
                start=str(start_dt),
                end=str(end_dt),
                interval="1d",
                auto_adjust=True,
                local_only=bool(args.__dict__.get("local_only", False)),
            )
            if df is not None and not df.empty:
                updated += 1
        except Exception as e:
            logger.warning("Price update failed for %s: %s", t, e)

    logger.info(
        f"Price update complete: tickers={len(tickers)} updated={updated} window_days={days_back}"
    )

    # Compute benchmark trend status
    try:
        df_bench = mds.get_ohlcv(
            benchmark,
            start=str(start_dt),
            end=str(end_dt),
            interval="1d",
            auto_adjust=True,
            local_only=bool(args.__dict__.get("local_only", False)),
        )
        if df_bench is not None and not df_bench.empty:
            price_col = "Adjclose" if "Adjclose" in df_bench.columns else ("Close" if "Close" in df_bench.columns else None)
            if price_col:
                sma = df_bench[price_col].rolling(trend_window).mean()
                trend_on = bool(df_bench[price_col].iloc[-1] > sma.iloc[-1]) if len(sma.dropna()) > 0 else False
                last_price_val = df_bench[price_col].iloc[-1]
                last_price = float(last_price_val) if last_price_val is not None and not pd.isna(last_price_val) else None
                sma_val = sma.iloc[-1] if len(sma) else None
                last_sma = float(sma_val) if sma_val is not None and not pd.isna(sma_val) else None
                last_price_str = f"{last_price:.2f}" if last_price is not None else "NA"
                last_sma_str = f"{last_sma:.2f}" if last_sma is not None else "NA"
                logger.info(
                    f"Trend status: benchmark={benchmark} window={trend_window} last_price={last_price_str} last_sma={last_sma_str} trend_on={trend_on}"
                )
            else:
                logger.warning("Benchmark %s missing price columns for trend calc", benchmark)
        else:
            logger.warning("No data for benchmark %s to compute trend", benchmark)
    except Exception as e:
        logger.exception("Failed computing trend for %s: %s", benchmark, e)


def compute_signals_step(
    args: argparse.Namespace,
    cfg,
    um: UniverseManager,
    mds: MarketDataStore,
    logger,
    run_dt: date,
) -> None:
    """Compute stock and sector signals and save artifacts."""
    try:
        momentum_windows = list(getattr(cfg.signals, "momentum_windows", [63, 126, 252]))
        vol_window = int(getattr(cfg.signals, "vol_window", 20))
        # Prefer 'vol_weight' if present; else fall back to 'vol_penalty'
        vol_weight = float(getattr(cfg.signals, "vol_weight", getattr(cfg.signals, "vol_penalty", 1.0)))
        mom_weights = getattr(cfg.signals, "momentum_weights", None)
        warmup = int(getattr(cfg.strategy, "warmup_days", 30))

        momentum_max = max(momentum_windows) if momentum_windows else 63
        if getattr(args, "signal_start", None):
            try:
                start_dt_candidate = datetime.strptime(str(args.signal_start), "%Y-%m-%d").date()
                if start_dt_candidate >= run_dt:
                    logger.warning("--signal-start %s is not before run date %s; using run_date - 1 day", start_dt_candidate, run_dt)
                    start_dt_candidate = run_dt - timedelta(days=1)
                start_dt = start_dt_candidate
                logger.info("Signal start override applied", extra={"signal_start": str(start_dt)})
            except Exception:
                logger.warning("Invalid --signal-start=%s (expected YYYY-MM-DD); falling back to automatic lookback", args.signal_start)
                days_back = max(warmup, momentum_max + vol_window + 10, 120)
                start_dt = run_dt - timedelta(days=days_back)
        else:
            days_back = max(warmup, momentum_max + vol_window + 10, 120)
            start_dt = run_dt - timedelta(days=days_back)
        end_dt = run_dt

        # Load prices via UniverseManager helper (prefers Adjclose -> Close)
        price_mat = um.get_price_matrix(
            price_loader=mds,
            start=str(start_dt),
            end=str(end_dt),
            field=None,
            interval="1d",
            local_only=bool(args.__dict__.get("local_only", False)),
        )
        if price_mat.empty:
            logger.warning("No price data available for signals window; skipping")
            return

        sector_map = um.sector_map
        eng = SignalEngine(prices=price_mat, sector_map=sector_map)
        stock_score = eng.compute_stock_score(
            mom_windows=momentum_windows,
            mom_weights=mom_weights,
            vol_window=vol_window,
            vol_weight=vol_weight,
        )

        out_dir = (cfg.output_root_path / "signals").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"stock_scores_{end_dt.strftime('%Y-%m-%d')}"
        csv_path = out_dir / f"{fname}.csv"
        pq_path = out_dir / f"{fname}.parquet"
        stock_score.to_csv(csv_path)
        try:
            stock_score.to_parquet(pq_path)
        except Exception as e:
            logger.warning("Parquet save failed (%s); CSV written at %s", e, csv_path)

        nn = int(stock_score.notna().sum().sum())
        total = int(stock_score.size)
        nan_ratio = 1.0 - (nn / total) if total else 0.0
        logger.info(
            f"Signals computed: dates=[{stock_score.index.min().date()}..{stock_score.index.max().date()}] tickers={stock_score.shape[1]} nan_ratio={nan_ratio:.2%} saved={csv_path.name}"
        )

        # Also compute and save sector scores alongside stock scores
        try:
            sector_scores = eng.compute_sector_scores_from_stock_scores(stock_score, sector_map or {})
            ss_dir = (cfg.output_root_path / "signals").resolve()
            ss_dir.mkdir(parents=True, exist_ok=True)
            ss_fname = f"sector_scores_{end_dt.strftime('%Y-%m-%d')}"
            ss_csv = ss_dir / f"{ss_fname}.csv"
            ss_pq = ss_dir / f"{ss_fname}.parquet"
            sector_scores.to_csv(ss_csv)
            try:
                sector_scores.to_parquet(ss_pq)
            except Exception as e:
                logger.warning("Parquet save failed for sector scores (%s); CSV written at %s", e, ss_csv)
            logger.info(
                f"Sector scores computed: sectors={sector_scores.shape[1]} dates=[{sector_scores.index.min().date()}..{sector_scores.index.max().date()}] saved={ss_csv.name}"
            )
        except Exception as e:
            logger.exception("Failed to compute sector scores: %s", e)

        # Also compute and persist per-stock realized volatility (useful for inverse-vol weighting later)
        try:
            stock_vol = eng.compute_volatility(window=vol_window)
            sv_fname = f"stock_vol_{end_dt.strftime('%Y-%m-%d')}"
            sv_csv = out_dir / f"{sv_fname}.csv"
            sv_pq = out_dir / f"{sv_fname}.parquet"
            try:
                stock_vol.to_parquet(sv_pq)
            except Exception:
                # Fallback to CSV if parquet save fails
                stock_vol.to_csv(sv_csv)
            logger.info("Saved stock volatility matrix: %s", sv_pq.name if sv_pq.exists() else sv_csv.name)
        except Exception as e:
            logger.warning("Failed to compute/save stock volatility (%s); continuing", e)
        
    except Exception as e:
        logger.exception("Signal computation failed: %s", e)


def compute_sector_weights_step(
    args: argparse.Namespace,
    cfg,
    mds: MarketDataStore,
    logger,
) -> None:
    """Compute daily and monthly sector weights from latest sector scores."""
    try:
        sig_dir = (cfg.output_root_path / "signals").resolve()
        stock_files = sorted(sig_dir.glob("sector_scores_*.csv"))
        if not stock_files:
            logger.warning("No sector_scores_*.csv found under %s; run --compute-signals first", sig_dir)
            return

        latest = stock_files[-1]
        sector_scores = pd.read_csv(latest, index_col=0)
        sector_scores.index = pd.to_datetime(sector_scores.index)

        # Determine window and benchmark
        start_dt = sector_scores.index.min().date()
        end_dt = sector_scores.index.max().date()
        trend_window = int(cfg.sectors.trend_filter.get("window", 200)) if cfg.sectors.trend_filter else 200
        benchmark = cfg.sectors.trend_filter.get("benchmark", "SPY") if cfg.sectors.trend_filter else "SPY"

        # Load benchmark prices
        df_bench = mds.get_ohlcv(benchmark, start=str(start_dt), end=str(end_dt), interval="1d", auto_adjust=True, local_only=bool(args.__dict__.get("local_only", False)))
        if df_bench is None or df_bench.empty:
            logger.warning("No benchmark data for %s; cannot compute sector weights", benchmark)
            return

        price_col = "Adjclose" if "Adjclose" in df_bench.columns else ("Close" if "Close" in df_bench.columns else None)
        if not price_col:
            logger.warning("Benchmark %s missing Close columns; cannot compute sector weights", benchmark)
            return

        spy_series = df_bench[price_col].reindex(sector_scores.index).ffill().bfill()

        # Params from cfg.sectors
        alpha = float(getattr(cfg.sectors, "smoothing_alpha", 1.0))
        beta = float(getattr(cfg.sectors, "smoothing_beta", 0.3))
        w_min = float(cfg.sectors.weights.get("w_min", 0.03))
        w_max = float(cfg.sectors.weights.get("w_max", 0.30))
        risk_on_frac = float(getattr(cfg.sectors, "risk_on_equity_frac", 1.0))
        risk_off_frac = float(getattr(cfg.sectors, "risk_off_equity_frac", 0.7))
        top_k_sectors = getattr(cfg.sectors, "top_k_sectors", None)

        swe = SectorWeightEngine(
            sector_scores=sector_scores,
            benchmark_prices=spy_series,
            alpha=alpha,
            w_min=w_min,
            w_max=w_max,
            beta=beta,
            trend_window=trend_window,
            risk_on_equity_frac=risk_on_frac,
            risk_off_equity_frac=risk_off_frac,
            top_k_sectors=top_k_sectors,
        )
        sector_weights_daily = swe.compute_weights()

        out_dir = (cfg.output_root_path / "weights").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = latest.stem.replace("sector_scores_", "")
        daily_csv = out_dir / f"sector_weights_daily_{stem}.csv"
        daily_pq = out_dir / f"sector_weights_daily_{stem}.parquet"
        sector_weights_daily.to_csv(daily_csv)
        try:
            sector_weights_daily.to_parquet(daily_pq)
        except Exception as e:
            logger.warning("Parquet save failed for daily weights (%s); CSV written at %s", e, daily_csv)

        # Monthly weights: last business day per month (use 'ME' for month-end to avoid deprecated 'M')
        sector_weights_monthly = sector_weights_daily.resample("ME").last()
        monthly_csv = out_dir / f"sector_weights_monthly_{stem}.csv"
        monthly_pq = out_dir / f"sector_weights_monthly_{stem}.parquet"
        sector_weights_monthly.to_csv(monthly_csv)
        try:
            sector_weights_monthly.to_parquet(monthly_pq)
        except Exception as e:
            logger.warning("Parquet save failed for monthly weights (%s); CSV written at %s", e, monthly_csv)

        logger.info(
            f"Sector weights computed: daily_rows={sector_weights_daily.shape[0]} sectors={sector_weights_daily.shape[1]} saved_daily={daily_csv.name} saved_monthly={monthly_csv.name}"
        )
    except Exception as e:
        logger.exception("Sector weights computation failed: %s", e)


def compute_stock_weights_step(
    args: argparse.Namespace,
    cfg,
    um: UniverseManager,
    mds: MarketDataStore,
    logger,
) -> None:
    """Compute stock-level weights from latest monthly sector weights and stock scores."""
    try:
        weights_dir = (cfg.output_root_path / "weights").resolve()
        signals_dir = (cfg.output_root_path / "signals").resolve()

        # Find latest monthly sector weights and stock scores
        monthly_weights = sorted(weights_dir.glob("sector_weights_monthly_*.csv"))
        stock_scores_files = sorted(signals_dir.glob("stock_scores_*.csv"))

        if not monthly_weights:
            logger.warning("No sector_weights_monthly_*.csv found under %s; run --compute-sector-weights first", weights_dir)
            return
        if not stock_scores_files:
            logger.warning("No stock_scores_*.csv found under %s; run --compute-signals first", signals_dir)
            return

        w_latest = monthly_weights[-1]
        s_latest = stock_scores_files[-1]

        sector_weights_monthly = pd.read_csv(w_latest, index_col=0)
        sector_weights_monthly.index = pd.to_datetime(sector_weights_monthly.index)

        # Try to load matching daily sector weights for the same stem
        stem = w_latest.stem.replace("sector_weights_monthly_", "")
        daily_sector_path = (weights_dir / f"sector_weights_daily_{stem}.csv")
        sector_weights_daily = None
        if daily_sector_path.exists():
            try:
                sector_weights_daily = pd.read_csv(daily_sector_path, index_col=0)
                sector_weights_daily.index = pd.to_datetime(sector_weights_daily.index)
            except Exception as e:
                logger.warning("Failed to load daily sector weights %s (%s)", daily_sector_path.name, e)

        stock_scores = pd.read_csv(s_latest, index_col=0)
        stock_scores.index = pd.to_datetime(stock_scores.index)

        sector_map = um.sector_map

        # Optional: compute stock vol for inverse-vol weighting if requested
        weighting_mode_cfg = getattr(cfg.stocks, "weighting", "equal-weight")
        weighting_mode = "equal" if weighting_mode_cfg == "equal-weight" else "inverse_vol"

        stock_vol = None
        if weighting_mode == "inverse_vol":
            try:
                # Attempt to load cached stock volatility computed during the signals step
                stem = s_latest.stem.replace("stock_scores_", "")
                sv_pq = (signals_dir / f"stock_vol_{stem}.parquet")
                sv_csv = (signals_dir / f"stock_vol_{stem}.csv")
                if sv_pq.exists():
                    try:
                        stock_vol = pd.read_parquet(sv_pq)
                        stock_vol.index = pd.to_datetime(stock_vol.index)
                        logger.info("Loaded cached stock volatility from %s", sv_pq.name)
                    except Exception as e:
                        logger.warning("Failed to read cached parquet stock_vol (%s); will attempt CSV or recompute", e)
                        stock_vol = None
                if stock_vol is None and sv_csv.exists():
                    try:
                        stock_vol = pd.read_csv(sv_csv, index_col=0)
                        stock_vol.index = pd.to_datetime(stock_vol.index)
                        logger.info("Loaded cached stock volatility from %s", sv_csv.name)
                    except Exception as e:
                        logger.warning("Failed to read cached CSV stock_vol (%s); will recompute", e)
                        stock_vol = None

                if stock_vol is None:
                    # Determine window from config and load prices for the needed range (+warmup)
                    vol_window = int(getattr(cfg.signals, "vol_window", 20))
                    warmup_days = max(30, vol_window + 10)
                    start_dt = (sector_weights_monthly.index.min() - pd.Timedelta(days=warmup_days)).date()
                    end_dt = sector_weights_monthly.index.max().date()

                    # Use columns present in stock_scores as our universe tickers
                    tickers = [t.strip().upper() for t in stock_scores.columns]

                    price_mat = um.get_price_matrix(
                        price_loader=mds,
                        tickers=tickers,
                        start=str(start_dt),
                        end=str(end_dt),
                        field=None,
                        interval="1d",
                        local_only=bool(args.__dict__.get("local_only", False)),
                    )
                    # Align to entire date range and compute vol
                    price_mat = price_mat.reindex(pd.date_range(price_mat.index.min(), price_mat.index.max(), freq="D")).ffill()
                    eng = SignalEngine(prices=price_mat, sector_map=sector_map)
                    stock_vol = eng.compute_volatility(window=vol_window)
            except Exception as e:
                logger.warning("Failed to compute/load stock vol for inverse-vol weighting (%s); falling back to equal-weight", e)
                weighting_mode = "equal"
                stock_vol = None

        allocator = StockAllocator(
            sector_weights=sector_weights_monthly,
            stock_scores=stock_scores,
            sector_map=sector_map,
            stock_vol=stock_vol,
            top_k=int(getattr(cfg.stocks, "top_k_per_sector", 2)),
            weighting_mode=weighting_mode,
            preserve_cash=True,
        )

        stock_weights_monthly = allocator.compute_stock_weights()

        out_dir = weights_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        sw_csv = out_dir / f"stock_weights_monthly_{stem}.csv"
        sw_pq = out_dir / f"stock_weights_monthly_{stem}.parquet"
        stock_weights_monthly.to_csv(sw_csv)
        try:
            stock_weights_monthly.to_parquet(sw_pq)
        except Exception as e:
            logger.warning("Parquet save failed for stock weights (%s); CSV written at %s", e, sw_csv)

        # Some quick stats
        total_cash = 1.0 - stock_weights_monthly.sum(axis=1)
        avg_cash = float(total_cash.mean()) if len(total_cash) else 0.0
        logger.info(
            f"Stock weights computed: months={stock_weights_monthly.shape[0]} tickers={stock_weights_monthly.shape[1]} top_k={getattr(cfg.stocks, 'top_k_per_sector', 2)} mode={weighting_mode} avg_cash={avg_cash:.2%} saved={sw_csv.name}"
        )

        # Optionally compute and save DAILY stock weights if we have daily sector weights
        if sector_weights_daily is not None and not sector_weights_daily.empty:
            try:
                allocator_daily = StockAllocator(
                    sector_weights=sector_weights_daily,
                    stock_scores=stock_scores,
                    sector_map=sector_map,
                    stock_vol=stock_vol,
                    top_k=int(getattr(cfg.stocks, "top_k_per_sector", 2)),
                    weighting_mode=weighting_mode,
                    preserve_cash=True,
                )
                stock_weights_daily = allocator_daily.compute_stock_weights()

                swd_csv = out_dir / f"stock_weights_daily_{stem}.csv"
                swd_pq = out_dir / f"stock_weights_daily_{stem}.parquet"
                stock_weights_daily.to_csv(swd_csv)
                try:
                    stock_weights_daily.to_parquet(swd_pq)
                except Exception as e:
                    logger.warning("Parquet save failed for daily stock weights (%s); CSV written at %s", e, swd_csv)

                logger.info(
                    f"Daily stock weights computed: days={stock_weights_daily.shape[0]} tickers={stock_weights_daily.shape[1]} saved={swd_csv.name}"
                )
            except Exception as e:
                logger.warning("Failed to compute daily stock weights (%s)", e)
    except Exception as e:
        logger.exception("Stock weights computation failed: %s", e)


def rebalance_pipeline(
    args: argparse.Namespace,
    cfg,
    um: UniverseManager,
    mds: MarketDataStore,
    logger,
    run_dt: date,
) -> None:
    """Rebalance orchestration: signals -> sector weights -> stock weights."""
    # Determine effective start date override
    effective_start = getattr(args, "rebalance_start", None) or getattr(args, "signal_start", None)
    # Log the start date at the beginning of rebalance
    logger.info(
        "Rebalance starting",
        extra={
            "run_date": str(run_dt),
            "start": str(effective_start) if effective_start else "(auto-lookback)",
        },
    )

    # Temporarily override signal_start if rebalance_start provided
    original_signal_start = getattr(args, "signal_start", None)
    if getattr(args, "rebalance_start", None):
        setattr(args, "signal_start", getattr(args, "rebalance_start"))

    try:
        compute_signals_step(args=args, cfg=cfg, um=um, mds=mds, logger=logger, run_dt=run_dt)
    finally:
        # Restore original signal_start
        setattr(args, "signal_start", original_signal_start)
    compute_sector_weights_step(args=args, cfg=cfg, mds=mds, logger=logger)
    compute_stock_weights_step(args=args, cfg=cfg, um=um, mds=mds, logger=logger)




def main() -> int:
    args = parse_args()

    strategy_yaml = Path(args.strategy).resolve()
    sectors_yaml = Path(args.sectors).resolve()

    cfg = load_app_config(strategy_yaml)

    logger = configure_logging(cfg.output_root_path, level=cfg.runtime.log_level, log_to_file=cfg.runtime.save.get("logs", True))
    logger.info("Starting LiveRunner", extra={"date": str(args.date), "rebalance": args.rebalance, "regen_universe": args.__dict__.get("regen_universe")})

    run_dt = resolve_run_date(args.date)

    # Universe Manager
    um = UniverseManager(membership_csv=cfg.membership_csv_path, sectors_yaml=sectors_yaml, local_only=bool(args.__dict__.get("local_only", False)))

    if args.__dict__.get("regen_universe"):
        regenerate_universe(um=um, cfg=cfg, logger=logger)

    # Optional: print mask summary via helper
    if args.__dict__.get("mask_summary"):
        summarize_membership_mask(args=args, um=um, logger=logger)

    # Market data
    mds = MarketDataStore(data_root=str((cfg.output_root_path / "prices").resolve()))

    # Optional: update prices and compute SPY trend
    if args.__dict__.get("update_prices"):
        update_prices_and_trend(args=args, um=um, mds=mds, cfg=cfg, logger=logger, run_dt=run_dt)

    # Compute signals via SignalEngine using window-union membership
    if args.__dict__.get("compute_signals"):
        compute_signals_step(args=args, cfg=cfg, um=um, mds=mds, logger=logger, run_dt=run_dt)

    # Compute sector weights using SectorWeightEngine
    if args.__dict__.get("compute_sector_weights"):
        compute_sector_weights_step(args=args, cfg=cfg, mds=mds, logger=logger)

    # Compute stock weights using StockAllocator
    if args.__dict__.get("compute_stock_weights"):
        compute_stock_weights_step(args=args, cfg=cfg, um=um, mds=mds, logger=logger)

    # Dump membership mask (archival)
    if args.__dict__.get("dump_membership_mask"):
        dump_membership_mask(args=args, um=um, cfg=cfg, logger=logger, run_dt=run_dt)

    # Full rebalance pipeline
    if args.__dict__.get("rebalance"):
        rebalance_pipeline(args=args, cfg=cfg, um=um, mds=mds, logger=logger, run_dt=run_dt)

    # Printing of portfolio has moved to the dedicated explorer (production/explore_portfolio.py)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
