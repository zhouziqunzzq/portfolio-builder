#!/usr/bin/env python3
"""Agent-friendly CLI for portfolio-builder v2.

Usage:
  python3 agent-v2-cli.py <sub-command> [options]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Make v2/src importable
_ROOT = Path(__file__).resolve().parents[2]  # .../v2
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from configs import AppConfig
from runtime_manager import RuntimeManager, RuntimeManagerOptions
from market_data_store import MarketDataStore
from universe_manager import UniverseManager
from sleeves.defensive.defensive_config import DefensiveConfig
from sleeves.sideways.sideways_config import SidewaysConfig
from sleeves.sideways_mr.sideways_mr_config import SidewaysMRConfig
from sleeves.sideways_base.sideways_base_config import SidewaysBaseConfig


def _parse_date(s: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid date: {s!r} ({e})")


def _calc_lookback_days(cfg) -> int:
    # Use the max window among regime config parameters.
    # Add a small buffer; then convert to calendar days with ~2x factor.
    rcfg = cfg.regime_engine
    windows = [
        rcfg.vol_window,
        rcfg.mom_window,
        rcfg.fast_ma,
        rcfg.slow_ma,
        rcfg.dd_lookback,
        rcfg.vol_norm_window,
        rcfg.ewm_vol_halflife or 0,
    ]
    max_window = max(int(w) for w in windows if w is not None)
    # 2x to account for weekends/holidays
    return max_window * 2 + 5


def _resolve_regime_lookback_days(args: argparse.Namespace, app_cfg) -> int:
    # Priority: CLI flag > file config > allocator prod config > legacy calc
    if getattr(args, "lookback_days", None) is not None:
        return int(args.lookback_days)

    # File-based config for skill-level overrides
    cfg_path = Path(args.lookback_config)
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            val = payload.get("lookback_days")
            if val is not None:
                return int(val)
        except Exception:
            pass

    # Mirror production allocator setting when available
    try:
        val = app_cfg.multi_sleeve_allocator.regime_lookback_days
        if val is not None:
            return int(val)
    except Exception:
        pass

    # Fallback to old behavior
    return _calc_lookback_days(app_cfg)


def _regime_command(args: argparse.Namespace) -> int:
    app_cfg = AppConfig.load_from_yaml(Path(args.config))

    # Build runtime manager (local-only by default for speed)
    rm = RuntimeManager.from_app_config(
        app_cfg,
        options=RuntimeManagerOptions(local_only=True, use_memory_cache=True),
    )
    regime_engine = rm["regime_engine"]

    target_dt = _parse_date(args.date)
    lookback_days = _resolve_regime_lookback_days(args, app_cfg)
    start_dt = target_dt - timedelta(days=lookback_days)

    df = regime_engine.get_regime_frame(start_dt, target_dt)
    if df.empty:
        raise SystemExit(f"No regime data available for {args.date}")

    # Take the last available row at or before target date
    df = df.loc[:target_dt]
    if df.empty:
        raise SystemExit(f"No regime data available on or before {args.date}")

    row = df.iloc[-1]
    out = {
        "date": row.name.strftime("%Y-%m-%d"),
        "lookback_days": int(lookback_days),
        "primary_regime": row.get("primary_regime"),
        "scores": {
            "bull": float(row.get("bull", 0.0)),
            "correction": float(row.get("correction", 0.0)),
            "bear": float(row.get("bear", 0.0)),
            "crisis": float(row.get("crisis", 0.0)),
            "sideways": float(row.get("sideways", 0.0)),
        },
    }

    if args.pretty:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(json.dumps(out))
    return 0


def _expected_row_count(start: pd.Timestamp, end: pd.Timestamp, interval: str) -> int:
    if interval == "1d":
        return len(pd.date_range(start=start, end=end, freq="B"))
    return max(1, (end - start).days)


def _market_data_command(args: argparse.Namespace) -> int:
    start_dt = _parse_date(args.start).normalize()
    end_dt = _parse_date(args.end).normalize()
    if end_dt < start_dt:
        raise SystemExit("End date must be >= start date")

    if not args.tickers and not (
        args.use_defensive_etfs or args.use_universe or args.use_sideways_tickers
    ):
        raise SystemExit(
            "Must supply --tickers unless any of the following are set:"
            " --use-defensive-etfs, --use-universe, --use-sideways-tickers"
        )

    tickers: list[str] = []
    if args.use_universe:
        um = UniverseManager(
            membership_csv=Path(args.universe_membership_csv),
            current_constituents_csv=Path(args.universe_constituents_csv),
        )
        tickers.extend(um.get_tickers(current_only=bool(args.use_current_constituents)))
    if args.use_defensive_etfs:
        cfg = DefensiveConfig()
        tickers.extend(sorted({k.upper() for k in cfg.asset_class_for_etf.keys()}))
    if args.use_sideways_tickers:
        scfg = SidewaysConfig()
        tickers.extend(sorted({t.upper() for t in scfg.tickers}))
        scfg = SidewaysMRConfig()
        tickers.extend(scfg.get_universe(include_benchmarks=True))
        scfg = SidewaysBaseConfig()
        tickers.extend(sorted({t.upper() for t in scfg.sideways_etfs}))
        tickers = sorted(set(tickers))
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if not tickers:
            raise SystemExit("No tickers parsed from --tickers")

    mds = MarketDataStore(
        data_root=args.data_root,
        source="yfinance",
        local_only=bool(args.local_only),
        use_memory_cache=True,
    )

    interval = args.interval
    auto_adjust = not args.no_auto_adjust

    rows = []
    tot = len(tickers)
    for i, t in enumerate(tickers, start=1):
        print(f"[{i}/{tot}] Processing ticker: {t}...")
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
                        "expected": _expected_row_count(start_dt, end_dt, interval),
                        "coverage": 0.0,
                    }
                )
                continue

            df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
            earliest = df.index.min()
            latest = df.index.max()
            count_rows = len(df)
            expected_rows = _expected_row_count(start_dt, end_dt, interval)
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
                    "expected": _expected_row_count(start_dt, end_dt, interval),
                    "coverage": 0.0,
                    "error": str(e),
                }
            )

    if not rows:
        print("No results.")
        return 0

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

    return 0


def _load_weights_json(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("weights JSON must be an object")
    weights = payload.get("weights")
    if not isinstance(weights, list) or not weights:
        raise SystemExit("weights JSON must contain non-empty 'weights' list")
    norm: list[dict[str, Any]] = []
    for row in weights:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        try:
            weight = float(row.get("weight"))
        except Exception:
            continue
        norm.append({"ticker": ticker, "weight": weight})
    if not norm:
        raise SystemExit("no valid ticker/weight rows found in weights JSON")
    as_of_month = payload.get("as_of_month")
    return (str(as_of_month) if as_of_month is not None else None, norm)


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _calc_one_ticker_metrics(
    mds: MarketDataStore, ticker: str, start_dt: pd.Timestamp, as_of_dt: pd.Timestamp
) -> dict[str, Any]:
    df = mds.get_ohlcv(
        ticker=ticker,
        start=start_dt,
        end=as_of_dt,
        interval="1d",
        auto_adjust=True,
        local_only=False,
    )
    if df is None or df.empty:
        return {"ticker": ticker, "status": "missing"}

    close_col = "Adjclose" if "Adjclose" in df.columns else ("Close" if "Close" in df.columns else None)
    if close_col is None:
        return {"ticker": ticker, "status": "missing"}

    closes = df[close_col].dropna()
    closes = closes[closes.index <= as_of_dt]
    if len(closes) < 2:
        return {"ticker": ticker, "status": "missing"}

    c0 = float(closes.iloc[-2])
    c1 = float(closes.iloc[-1])
    ret = (c1 / c0) - 1.0 if c0 else None

    sma20 = _safe_float(closes.tail(20).mean()) if len(closes) >= 20 else None
    sma50 = _safe_float(closes.tail(50).mean()) if len(closes) >= 50 else None
    last_close = _safe_float(c1)
    rsi14 = None
    if len(closes) >= 15:
        delta = closes.diff().dropna()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.tail(14).mean()
        avg_loss = losses.tail(14).mean()
        if avg_loss == 0 and avg_gain > 0:
            rsi14 = 100.0
        elif avg_loss == 0 and avg_gain == 0:
            rsi14 = 50.0
        else:
            rs = avg_gain / avg_loss
            rsi14 = 100.0 - (100.0 / (1.0 + rs))

    trend_bits: list[str] = []
    if last_close is not None and sma20 is not None:
        trend_bits.append("above 20D MA" if last_close > sma20 else "below 20D MA")
    if last_close is not None and sma50 is not None:
        trend_bits.append("above 50D MA" if last_close > sma50 else "below 50D MA")
    if rsi14 is not None:
        if rsi14 >= 70:
            trend_bits.append("RSI14 overbought")
        elif rsi14 <= 30:
            trend_bits.append("RSI14 oversold")
        else:
            trend_bits.append("RSI14 neutral")
    trend = ", ".join(trend_bits) if trend_bits else "insufficient MA history"

    return {
        "ticker": ticker,
        "status": "ok",
        "return_pct": ret * 100.0 if ret is not None else None,
        "last_close": last_close,
        "sma20": sma20,
        "sma50": sma50,
        "rsi14": _safe_float(rsi14),
        "trend": trend,
    }


def _postmarket_metrics_command(args: argparse.Namespace) -> int:
    as_of_dt = _parse_date(args.as_of).normalize()
    lookback_start = as_of_dt - timedelta(days=int(args.lookback_days))
    _, weights = _load_weights_json(Path(args.weights_json))

    mds = MarketDataStore(
        data_root=args.data_root,
        source="yfinance",
        local_only=False,
        use_memory_cache=True,
    )

    per_ticker: list[dict[str, Any]] = []
    for row in weights:
        ticker = row["ticker"]
        weight = float(row["weight"])
        m = _calc_one_ticker_metrics(mds, ticker, lookback_start, as_of_dt)
        m["weight"] = weight
        ret_pct = m.get("return_pct")
        if ret_pct is None:
            m["contribution_bp"] = None
        else:
            m["contribution_bp"] = weight * (ret_pct / 100.0) * 10000.0
        per_ticker.append(m)

    usable = [x for x in per_ticker if x.get("contribution_bp") is not None]
    portfolio_day_move_estimate = None
    if usable:
        total_ret = sum(float(x["weight"]) * (float(x["return_pct"]) / 100.0) for x in usable)
        portfolio_day_move_estimate = total_ret * 100.0

    usable_sorted = sorted(usable, key=lambda x: float(x["contribution_bp"]), reverse=True)
    top_n = int(args.top_n)
    top_contributors = [
        {
            "ticker": x["ticker"],
            "contribution_bp": round(float(x["contribution_bp"]), 3),
            "return_pct": round(float(x["return_pct"]), 4),
            "weight": float(x["weight"]),
        }
        for x in usable_sorted[:top_n]
    ]
    top_detractors = [
        {
            "ticker": x["ticker"],
            "contribution_bp": round(float(x["contribution_bp"]), 3),
            "return_pct": round(float(x["return_pct"]), 4),
            "weight": float(x["weight"]),
        }
        for x in sorted(usable, key=lambda x: float(x["contribution_bp"]))[:top_n]
    ]

    top_weight_names = sorted(per_ticker, key=lambda x: float(x["weight"]), reverse=True)[:top_n]
    risk_technical_notes = []
    for x in top_weight_names:
        note = {
            "ticker": x["ticker"],
            "weight": float(x["weight"]),
            "status": x.get("status"),
            "trend": x.get("trend"),
            "return_pct": x.get("return_pct"),
            "rsi14": x.get("rsi14"),
        }
        risk_technical_notes.append(note)

    out = {
        "as_of": as_of_dt.strftime("%Y-%m-%d"),
        "weights_count": len(weights),
        "portfolio_day_move_estimate_pct": round(portfolio_day_move_estimate, 6)
        if portfolio_day_move_estimate is not None
        else None,
        "top_contributors": top_contributors,
        "top_detractors": top_detractors,
        "risk_technical_notes": risk_technical_notes,
        "missing_tickers": [x["ticker"] for x in per_ticker if x.get("status") != "ok"],
    }
    if args.pretty:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(json.dumps(out))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agent-friendly CLI for portfolio-builder v2")
    p.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "app.yml"),
        help="Path to v2 app config YAML file",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # regime subcommand
    pr = sub.add_parser("regime", help="Get regime score for a specific date")
    pr.add_argument("date", help="Target date (YYYY-MM-DD or any pandas-parseable date)")
    pr.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Override calendar lookback days used to compute regime context",
    )
    pr.add_argument(
        "--lookback-config",
        default=os.path.expanduser(
            "~/.openclaw/workspace/config/portfolio-builder-v2-regime.json"
        ),
        help="JSON config path for regime lookback (expects: {\"lookback_days\": <int>})",
    )
    pr.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    pr.set_defaults(func=_regime_command)

    # market-data subcommand
    pm = sub.add_parser(
        "market-data", help="Download/ensure market data coverage for tickers"
    )
    pm.add_argument("--data-root", default="data/prices", help="Root for cached OHLCV")
    pm.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    pm.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    pm.add_argument(
        "--tickers",
        help="Comma-separated tickers (e.g. AAPL,MSFT,SPY). Optional if using a preset set",
    )
    pm.add_argument(
        "--use-defensive-etfs",
        action="store_true",
        help="Use all defensive sleeve ETF tickers",
    )
    pm.add_argument(
        "--use-sideways-tickers",
        action="store_true",
        help="Use all sideways sleeve tickers",
    )
    pm.add_argument(
        "--use-universe",
        action="store_true",
        help="Use all tickers from UniverseManager",
    )
    pm.add_argument(
        "--use-current-constituents",
        action="store_true",
        help="When --use-universe is set, use only current constituents",
    )
    pm.add_argument(
        "--universe-membership-csv",
        default="data/sp500_membership.csv",
        help="Universe membership CSV (used with --use-universe)",
    )
    pm.add_argument(
        "--universe-constituents-csv",
        default="data/current_sp500_constituents.csv",
        help="Universe current constituents CSV (used with --use-universe)",
    )
    pm.add_argument("--interval", default="1d", help="Interval (default: 1d)")
    pm.add_argument(
        "--local-only", action="store_true", help="Do not fetch online; cache only"
    )
    pm.add_argument(
        "--no-auto-adjust", action="store_true", help="Disable auto_adjust when fetching"
    )
    pm.set_defaults(func=_market_data_command)

    # postmarket-metrics subcommand
    pp = sub.add_parser(
        "postmarket-metrics",
        help="Compute deterministic postmarket metrics from structured weights JSON",
    )
    pp.add_argument("--weights-json", required=True, help="Path to structured weights JSON")
    pp.add_argument("--as-of", required=True, help="As-of date (YYYY-MM-DD)")
    pp.add_argument("--data-root", default="data/prices", help="Root for cached OHLCV")
    pp.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="Calendar lookback window for moving-average context (default: 120)",
    )
    pp.add_argument("--top-n", type=int, default=5, help="Top N contributors/detractors")
    pp.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    pp.set_defaults(func=_postmarket_metrics_command)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
