from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
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
from portfolio_backtester import PortfolioBacktester


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Momentum V1.5 Production Runner")
    p.add_argument("--date", default="today", help="Run date (YYYY-MM-DD) or 'today'")
    p.add_argument("--rebalance", action="store_true", help="Force a rebalance run")
    p.add_argument("--regen-universe", action="store_true", help="Rebuild and save universe membership")
    p.add_argument("--strategy", default=str(Path(__file__).resolve().parents[1] / "config" / "strategy.yml"), help="Path to strategy.yml")
    p.add_argument("--sectors", default=str(Path(__file__).resolve().parents[1] / "config" / "sectors.yml"), help="Path to sectors.yml")
    p.add_argument("--local-only", action="store_true", help="Disable network calls and use only local caches/artifacts")

    p.add_argument("--dump-current-constituents", action="store_true", help="Fetch Wikipedia constituents and save CSV")
    p.add_argument("--build-membership", action="store_true", help="Build historical membership CSV and save to configured path")
    p.add_argument("--enrich-membership", action="store_true", help="Enrich sectors for existing membership CSV and write an enriched copy")
    p.add_argument("--mask-summary", action="store_true", help="Print a brief summary of the membership mask for a recent window")
    p.add_argument("--mask-start", default=None, help="Optional start date YYYY-MM-DD for mask summary (defaults to run_date - 30 days)")
    p.add_argument("--mask-end", default=None, help="Optional end date YYYY-MM-DD for mask summary (defaults to run_date)")
    p.add_argument("--update-prices", action="store_true", help="Update OHLCV cache for tickers and compute SPY trend status")
    p.add_argument("--tickers-source", choices=["current", "mask"], default="current", help="Source of tickers for price update: current (Wikipedia) or mask (from membership CSV)")
    p.add_argument("--mask-days", type=int, default=30, help="When --tickers-source=mask, number of days before run_date to build the mask")
    p.add_argument("--price-tickers", default=None, help="Comma-separated explicit tickers to update (overrides --tickers-source)")
    p.add_argument("--compute-signals", action="store_true", help="Compute stock scores and sector scores using SignalEngine over window-union membership")
    p.add_argument("--signal-start", default=None, help="Explicit start date YYYY-MM-DD for signal window; overrides automatic lookback")
    p.add_argument("--compute-sector-weights", action="store_true", help="Compute sector weights from sector scores and benchmark using SectorWeightEngine")
    p.add_argument("--compute-stock-weights", action="store_true", help="Allocate sector weights to stocks using StockAllocator")
    p.add_argument("--backtest-monthly", action="store_true", help="Run integrated monthly backtest using latest stock weights")
    p.add_argument("--backtest-start", default=None, help="Explicit backtest start date (YYYY-MM-DD); overrides earliest weight date")
    return p.parse_args()


def resolve_run_date(s: str) -> date:
    if s.lower() == "today":
        return datetime.utcnow().date()
    return datetime.strptime(s, "%Y-%m-%d").date()


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
        logger.warning("Universe regeneration requested but not implemented; expecting existing CSV at %s", cfg.membership_csv_path)

    # Optional: dump current Wikipedia constituents for inspection
    if args.__dict__.get("dump_current_constituents"):
        try:
            df_cur = um.build_current_constituents()
            if cfg.universe.filter_unknown_sectors:
                df_cur = um.filter_unknown_sectors(df_cur)
            out_path = (cfg.output_root_path / "current_constituents.csv").resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_cur.to_csv(out_path, index=False)
            # Log a brief sector summary
            sector_counts = df_cur["sector"].value_counts().to_dict() if "sector" in df_cur.columns else {}
            logger.info("Saved current constituents", extra={"rows": len(df_cur), "path": str(out_path), "sector_counts": sector_counts})
        except Exception as e:
            logger.exception("Failed to dump current constituents: %s", e)

    # Optional: build historical membership and save to configured path
    if args.__dict__.get("build_membership"):
        try:
            df_mem_raw = um.build_historical_membership()
            # Enrich sectors BEFORE filtering unknowns, so enrichment isn't a no-op
            unknown_before = int((df_mem_raw.get("sector", pd.Series([])).astype(str).str.strip() == "Unknown").sum()) if "sector" in df_mem_raw.columns else 0
            df_mem_enriched = um.enrich_sectors(df_mem_raw)
            unknown_after = int((df_mem_enriched.get("sector", pd.Series([])).astype(str).str.strip() == "Unknown").sum()) if "sector" in df_mem_enriched.columns else 0

            # Optionally filter unknowns after enrichment
            if cfg.universe.filter_unknown_sectors:
                df_mem_final = um.filter_unknown_sectors(df_mem_enriched)
            else:
                df_mem_final = df_mem_enriched

            # Ensure directory exists and save the enriched (final) membership
            cfg.membership_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_mem_final.to_csv(cfg.membership_csv_path, index=False)
            logger.info(
                f"Saved historical membership (enriched): path={cfg.membership_csv_path} rows={len(df_mem_final)} unknown_before={unknown_before} unknown_after={unknown_after}"
            )

            # Optionally also save the raw pre-enrichment membership alongside for diagnostics
            raw_path = cfg.membership_csv_path.with_name(cfg.membership_csv_path.stem + "_raw" + cfg.membership_csv_path.suffix)
            df_mem_raw.to_csv(raw_path, index=False)
            logger.info("Saved raw (pre-enrichment) membership copy", extra={"path": str(raw_path), "rows": len(df_mem_raw)})
        except Exception as e:
            logger.exception("Failed to build membership: %s", e)

    # Optional: enrich sectors on existing membership and save a sibling file
    if args.__dict__.get("enrich_membership"):
        try:
            df_mem = um.load_from_membership_csv()
            # Count unknowns before
            unknown_before = int((df_mem.get("sector", pd.Series([])).astype(str).str.strip() == "Unknown").sum()) if "sector" in df_mem.columns else 0
            df_enriched = um.enrich_sectors(df_mem)
            unknown_after = int((df_enriched.get("sector", pd.Series([])).astype(str).str.strip() == "Unknown").sum()) if "sector" in df_enriched.columns else 0
            out_path = cfg.membership_csv_path.with_name(cfg.membership_csv_path.stem + "_enriched" + cfg.membership_csv_path.suffix)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_enriched.to_csv(out_path, index=False)
            logger.info(
                f"Saved enriched membership: path={out_path} rows={len(df_enriched)} unknown_before={unknown_before} unknown_after={unknown_after}"
            )
        except Exception as e:
            logger.exception("Failed to enrich membership sectors: %s", e)

    # Optional: print mask summary for a small window
    if args.__dict__.get("mask_summary"):
        try:
            end_dt = resolve_run_date(args.date)
            start_arg = args.__dict__.get("mask_start")
            end_arg = args.__dict__.get("mask_end")
            if start_arg is None:
                start_dt = end_dt - timedelta(days=30)
            else:
                start_dt = datetime.strptime(start_arg, "%Y-%m-%d").date()
            if end_arg is not None:
                end_dt = datetime.strptime(end_arg, "%Y-%m-%d").date()

            mask = um.membership_mask(start=str(start_dt), end=str(end_dt))
            # Summary: dates, columns, mean members per day
            mean_members = float(mask.sum(axis=1).mean()) if not mask.empty else 0.0
            start_s = str(mask.index.min().date()) if not mask.empty else str(start_dt)
            end_s = str(mask.index.max().date()) if not mask.empty else str(end_dt)
            logger.info(
                f"Membership mask summary: window=[{start_s}..{end_s}] days={mask.shape[0]} tickers={mask.shape[1]} avg_members={mean_members:.1f}"
            )
        except Exception as e:
            logger.exception("Failed to compute mask summary: %s", e)

    # Market data
    mds = MarketDataStore(data_root=str((cfg.output_root_path / "prices").resolve()))

    # Optional: update prices and compute SPY trend
    if args.__dict__.get("update_prices"):
        # Determine lookback window to cover momentum/vol/trend warmups
        trend_window = int(cfg.sectors.trend_filter.get("window", 200)) if cfg.sectors.trend_filter else 200
        momentum_max = max(cfg.signals.momentum_windows)
        vol_window = int(cfg.signals.vol_window)
        warmup = int(cfg.strategy.warmup_days)
        days_back = max(warmup, momentum_max + vol_window + 10, trend_window + 10, 260)

        start_dt = run_dt - timedelta(days=days_back)
        end_dt = run_dt

        # Choose tickers
        tickers = []
        try:
            if args.price_tickers:
                tickers = [t.strip().upper() for t in str(args.price_tickers).split(",") if t.strip()]
            elif args.tickers_source == "mask" or bool(args.__dict__.get("local_only", False)):
                mask_start = run_dt - timedelta(days=int(args.mask_days))
                mask = um.membership_mask(start=str(mask_start), end=str(end_dt))
                tickers = list(mask.columns)
            else:
                cur = um.build_current_constituents()
                tickers = cur["ticker"].dropna().astype(str).tolist()
        except Exception as e:
            logger.exception("Failed to determine tickers from %s: %s", args.tickers_source, e)
            tickers = []

        # Always include benchmark
        benchmark = cfg.sectors.trend_filter.get("benchmark", "SPY") if cfg.sectors.trend_filter else "SPY"
        if benchmark not in tickers:
            tickers.append(benchmark)

        # Update cache for each ticker
        updated = 0
        for t in tickers:
            try:
                df = mds.get_ohlcv(ticker=t, start=str(start_dt), end=str(end_dt), interval="1d", auto_adjust=True, local_only=bool(args.__dict__.get("local_only", False)))
                if not df.empty:
                    updated += 1
            except Exception as e:
                logger.warning("Price update failed for %s: %s", t, e)

        logger.info(
            f"Price update complete: tickers={len(tickers)} updated={updated} window_days={days_back}"
        )

        # Compute SPY trend status
        try:
            df_spy = mds.get_ohlcv(benchmark, start=str(start_dt), end=str(end_dt), interval="1d", auto_adjust=True, local_only=bool(args.__dict__.get("local_only", False)))
            if not df_spy.empty:
                # Prefer Adjclose; fall back to Close
                price_col = "Adjclose" if "Adjclose" in df_spy.columns else ("Close" if "Close" in df_spy.columns else None)
                if price_col:
                    sma = df_spy[price_col].rolling(trend_window).mean()
                    trend_on = bool(df_spy[price_col].iloc[-1] > sma.iloc[-1]) if len(sma.dropna()) > 0 else False
                    last_price_val = df_spy[price_col].iloc[-1]
                    last_price = float(last_price_val) if not pd.isna(last_price_val) else None
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

    # Compute signals via SignalEngine using window-union membership
    if args.__dict__.get("compute_signals"):
        try:
            momentum_windows = list(getattr(cfg.signals, "momentum_windows", [63, 126, 252]))
            vol_window = int(getattr(cfg.signals, "vol_window", 20))
            # Prefer 'vol_weight' if present; else fall back to 'vol_penalty'
            vol_weight = float(getattr(cfg.signals, "vol_weight", getattr(cfg.signals, "vol_penalty", 1.0)))
            mom_weights = getattr(cfg.signals, "momentum_weights", None)
            warmup = int(getattr(cfg.strategy, "warmup_days", 30))

            momentum_max = max(momentum_windows) if momentum_windows else 63
            if args.signal_start:
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
            else:
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
        except Exception as e:
            logger.exception("Signal computation failed: %s", e)

    # Compute sector weights using SectorWeightEngine
    if args.__dict__.get("compute_sector_weights"):
        try:
            sig_dir = (cfg.output_root_path / "signals").resolve()
            stock_files = sorted(sig_dir.glob("sector_scores_*.csv"))
            if not stock_files:
                logger.warning("No sector_scores_*.csv found under %s; run --compute-signals first", sig_dir)
            else:
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
                else:
                    price_col = "Adjclose" if "Adjclose" in df_bench.columns else ("Close" if "Close" in df_bench.columns else None)
                    if not price_col:
                        logger.warning("Benchmark %s missing Close columns; cannot compute sector weights", benchmark)
                    else:
                        spy_series = df_bench[price_col].reindex(sector_scores.index).ffill().bfill()

                        # Params from cfg.sectors
                        alpha = float(getattr(cfg.sectors, "smoothing_alpha", 1.0))
                        beta = float(getattr(cfg.sectors, "smoothing_beta", 0.3))
                        w_min = float(cfg.sectors.weights.get("w_min", 0.03))
                        w_max = float(cfg.sectors.weights.get("w_max", 0.30))
                        risk_on_frac = float(getattr(cfg.sectors, "risk_on_equity_frac", 1.0))
                        risk_off_frac = float(getattr(cfg.sectors, "risk_off_equity_frac", 0.7))

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

    # Compute stock weights using StockAllocator
    if args.__dict__.get("compute_stock_weights"):
        try:
            weights_dir = (cfg.output_root_path / "weights").resolve()
            signals_dir = (cfg.output_root_path / "signals").resolve()

            # Find latest monthly sector weights and stock scores
            monthly_weights = sorted(weights_dir.glob("sector_weights_monthly_*.csv"))
            stock_scores_files = sorted(signals_dir.glob("stock_scores_*.csv"))

            if not monthly_weights:
                logger.warning("No sector_weights_monthly_*.csv found under %s; run --compute-sector-weights first", weights_dir)
            elif not stock_scores_files:
                logger.warning("No stock_scores_*.csv found under %s; run --compute-signals first", signals_dir)
            else:
                w_latest = monthly_weights[-1]
                s_latest = stock_scores_files[-1]

                sector_weights_monthly = pd.read_csv(w_latest, index_col=0)
                sector_weights_monthly.index = pd.to_datetime(sector_weights_monthly.index)

                stock_scores = pd.read_csv(s_latest, index_col=0)
                stock_scores.index = pd.to_datetime(stock_scores.index)

                # Build sector map from membership CSV (last known per ticker)
                try:
                    mem_df = um.load_from_membership_csv()
                    if {"ticker", "sector"}.issubset(mem_df.columns):
                        mem_df = (
                            mem_df[["ticker", "sector", "date_added"]].copy()
                            if "date_added" in mem_df.columns
                            else mem_df[["ticker", "sector"]].copy()
                        )
                        if "date_added" in mem_df.columns:
                            mem_df = mem_df.sort_values(["ticker", "date_added"]).drop_duplicates("ticker", keep="last")
                        else:
                            mem_df = mem_df.drop_duplicates("ticker", keep="last")
                        sector_map = {row["ticker"].upper(): row["sector"] for _, row in mem_df.iterrows()}
                    else:
                        sector_map = {}
                except Exception:
                    sector_map = {}

                # Optional: compute stock vol for inverse-vol weighting if requested
                weighting_mode_cfg = getattr(cfg.stocks, "weighting", "equal-weight")
                weighting_mode = "equal" if weighting_mode_cfg == "equal-weight" else "inverse_vol"

                stock_vol = None
                if weighting_mode == "inverse_vol":
                    try:
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
                        logger.warning("Failed to compute stock vol for inverse-vol weighting (%s); falling back to equal-weight", e)
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
                stem = w_latest.stem.replace("sector_weights_monthly_", "")
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
        except Exception as e:
            logger.exception("Stock weights computation failed: %s", e)

    # Integrated monthly backtest (simple stats, no plots)
    if args.__dict__.get("backtest_monthly"):
        try:
            weights_dir = (cfg.output_root_path / "weights").resolve()
            if not weights_dir.exists():
                logger.warning("Weights directory %s does not exist; run --compute-stock-weights first", weights_dir)
            else:
                stock_weight_files = sorted(weights_dir.glob("stock_weights_monthly_*.csv"))
                if not stock_weight_files:
                    logger.warning("No stock_weights_monthly_*.csv found under %s; run --compute-stock-weights first", weights_dir)
                else:
                    latest_sw = stock_weight_files[-1]
                    stock_weights_monthly = pd.read_csv(latest_sw, index_col=0)
                    stock_weights_monthly.index = pd.to_datetime(stock_weights_monthly.index)

                    # Determine date range (extend backwards for warmup)
                    warmup_days = int(getattr(cfg.strategy, "warmup_days", 30))
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
                            override_start = None
                        elif override_start < earliest_weight_date:
                            logger.info("--backtest-start %s precedes earliest weight date %s; using earliest weight date", override_start, earliest_weight_date)
                            override_start = earliest_weight_date

                    effective_start_date = override_start or earliest_weight_date
                    # Trim weights to effective_start_date onward
                    stock_weights_monthly = stock_weights_monthly.loc[stock_weights_monthly.index.date >= effective_start_date]
                    if stock_weights_monthly.empty:
                        logger.warning("No weights remain after applying backtest start date %s", effective_start_date)
                        return 0

                    start_dt = (effective_start_date - timedelta(days=warmup_days))
                    end_dt = stock_weights_monthly.index.max().date()

                    # Universe tickers based on weight columns
                    tickers = [t.strip().upper() for t in stock_weights_monthly.columns]
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
                    else:
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
                        if df_bench is None or df_bench.empty:
                            logger.warning("Benchmark data unavailable for %s; skipping benchmark stats", benchmark)
                        else:
                            price_col = "Adjclose" if "Adjclose" in df_bench.columns else ("Close" if "Close" in df_bench.columns else None)
                            if price_col:
                                bench_series = df_bench[price_col].copy()
                                if eff_start and eff_end:
                                    bench_series = bench_series.reindex(result.index).ffill().bfill()
                                    bench_series = bench_series.loc[eff_start:eff_end]
                                returns = bench_series.pct_change().fillna(0.0)
                                if len(returns) > 0:
                                    total_ret = (1 + returns).prod() - 1
                                    cagr = total_ret ** (252 / len(returns)) - 1
                                    vol = returns.std() * (252 ** 0.5)
                                    sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else float("nan")
                                    eq = (1 + returns).cumprod()
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
        except Exception as e:
            logger.exception("Monthly backtest failed: %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
