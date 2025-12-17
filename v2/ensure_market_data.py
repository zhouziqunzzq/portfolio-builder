#!/usr/bin/env python
from __future__ import annotations

"""Ensure market data coverage for a list of tickers.

Usage example:
	python v2/ensure_market_data.py \
		--data-root data/prices \
		--start 2018-01-01 --end 2025-11-24 \
		--tickers AAPL,MSFT,SPY,XLK \
		--interval 1d

	python v2/ensure_market_data.py \
		--data-root data/prices \
		--start 2018-01-01 --end 2025-11-24 \
		--use-defensive-etfs \
		--interval 1d

When --use-defensive-etfs is provided, the script ignores --tickers and instead
loads all ETF tickers defined in DefensiveConfig.asset_class_for_etf.

This script:
  1) Instantiates MarketDataStore
  2) Calls get_ohlcv for each ticker to fill gaps (unless --local-only)
  3) Computes coverage statistics within the requested window:
		earliest_available: first date in data intersecting [start,end]
		latest_available:   last date in data intersecting [start,end]
		coverage_rate:      (# rows in range) / (# expected business days in range)
  4) Prints a summary table.
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.sleeves.defensive.defensive_config import DefensiveConfig
from src.sleeves.sideways.sideways_config import SidewaysConfig
from src.sleeves.sideways_mr.sideways_mr_config import SidewaysMRConfig

from src.market_data_store import MarketDataStore
from src.universe_manager import UniverseManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensure market data coverage for tickers")
    p.add_argument(
        "--data-root", default="data/prices", help="Root for cached OHLCV data"
    )
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--tickers",
        help="Comma-separated list of tickers (e.g. AAPL,MSFT,SPY). Optional if --use-defensive-etfs set",
    )
    p.add_argument(
        "--use-defensive-etfs",
        action="store_true",
        help="Use all defensive sleeve ETF tickers from DefensiveConfig (overrides --tickers)",
    )
    p.add_argument(
        "--use-sideways-tickers",
        action="store_true",
        help="Use all tickers defined in SidewaysConfig (overrides --tickers)",
    )
    p.add_argument(
        "--use-sideways-mr-tickers",
        action="store_true",
        help="Use all tickers defined in SidewaysMRConfig (overrides --tickers)",
    )
    p.add_argument(
        "--use-universe",
        action="store_true",
        help="Use all tickers from UniverseManager (overrides --tickers)",
    )
    p.add_argument(
        "--universe-membership-csv",
        default="data/sp500_membership.csv",
        help="Path to universe membership CSV (used with --use-universe)",
    )
    p.add_argument("--interval", default="1d", help="Interval (default: 1d)")
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Do not fetch online; only use local cache",
    )
    p.add_argument(
        "--no-auto-adjust",
        action="store_true",
        help="Disable auto_adjust when fetching",
    )
    return p.parse_args()


def expected_row_count(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> int:
    if interval == "1d":
        # Business days between start & end inclusive
        return len(pd.date_range(start=start, end=end, freq="B"))
    # For other intervals, rely on actual returned rows (can't easily infer)
    return max(1, (end - start).days)


def main() -> int:
    args = parse_args()

    start_dt = pd.to_datetime(args.start).normalize()
    end_dt = pd.to_datetime(args.end).normalize()
    if end_dt < start_dt:
        raise ValueError("End date must be >= start date")

    if not args.tickers and not (
        args.use_defensive_etfs
        or args.use_universe
        or args.use_sideways_tickers
        or args.use_sideways_mr_tickers
    ):
        raise ValueError(
            "Must supply --tickers unless --use-defensive-etfs or --use-universe is set"
        )

    tickers: list[str] = []
    if args.use_universe:
        um = UniverseManager(membership_csv=Path(args.universe_membership_csv))
        tickers.extend(um.tickers)
    if args.use_defensive_etfs:
        cfg = DefensiveConfig()
        # Use all ETF tickers defined in asset_class_for_etf
        tickers.extend(sorted({k.upper() for k in cfg.asset_class_for_etf.keys()}))
    if args.use_sideways_tickers:
        scfg = SidewaysConfig()
        tickers.extend(sorted({t.upper() for t in scfg.tickers}))
    if args.use_sideways_mr_tickers:
        scfg = SidewaysMRConfig()
        tickers.extend(scfg.get_universe(include_benchmarks=True))
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if not tickers:
            raise ValueError("No tickers parsed from --tickers")

    mds = MarketDataStore(
        data_root=args.data_root,
        source="yfinance",
        local_only=bool(args.local_only),
        use_memory_cache=True,
    )

    interval = args.interval
    auto_adjust = not args.no_auto_adjust

    rows = []
    for t in tickers:
        try:
            df = mds.get_ohlcv(
                ticker=t,
                start=start_dt,
                end=end_dt,
                interval=interval,
                auto_adjust=auto_adjust,
                local_only=bool(args.local_only),
            )
            if df is None or df.empty:
                rows.append(
                    {
                        "ticker": t,
                        "earliest": None,
                        "latest": None,
                        "rows": 0,
                        "expected": expected_row_count(start_dt, end_dt, interval),
                        "coverage": 0.0,
                    }
                )
                continue

            # Clip to requested window (already done by get_ohlcv but safe)
            df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
            earliest = df.index.min()
            latest = df.index.max()
            count_rows = len(df)
            expected_rows = expected_row_count(start_dt, end_dt, interval)
            coverage = (count_rows / expected_rows) if expected_rows > 0 else 0.0

            rows.append(
                {
                    "ticker": t,
                    "earliest": earliest.date() if earliest is not None else None,
                    "latest": latest.date() if latest is not None else None,
                    "rows": count_rows,
                    "expected": expected_rows,
                    "coverage": coverage,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "ticker": t,
                    "earliest": None,
                    "latest": None,
                    "rows": 0,
                    "expected": expected_row_count(start_dt, end_dt, interval),
                    "coverage": 0.0,
                    "error": str(e),
                }
            )

    if not rows:
        print("No results.")
        return 0

    # Build summary DataFrame
    summary = pd.DataFrame(rows).set_index("ticker")

    print("\n=== Market Data Coverage Summary ===")
    print(
        f"Window: {start_dt.date()} -> {end_dt.date()} (interval={interval}, auto_adjust={auto_adjust})"
    )
    print("Ticker  Earliest    Latest      Rows  Expected  Coverage")
    for t, r in summary.iterrows():
        earliest = r["earliest"] or "-"
        latest = r["latest"] or "-"
        rows_c = int(r["rows"])
        exp = int(r["expected"])
        cov_pct = f"{(r['coverage']*100):5.1f}%"
        print(
            f"{t:6s}  {earliest!s:10s}  {latest!s:10s}  {rows_c:5d}  {exp:8d}  {cov_pct}"
        )
    print("====================================\n")

    # Optionally write a CSV summary next to data-root
    # try:
    # 	out_dir = Path(args.data_root).resolve()
    # 	out_path = out_dir / f"coverage_{start_dt.date()}_{end_dt.date()}.csv"
    # 	summary.to_csv(out_path)
    # 	print(f"Saved coverage summary to {out_path}")
    # except Exception:
    # 	pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
