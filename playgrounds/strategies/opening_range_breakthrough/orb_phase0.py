#!/usr/bin/env python3
"""
Phase 0 ORB sketch (standalone, one file).

Goal: Structural sanity checks (NOT production-realistic):
- directional asymmetry (we do long-only here; short can be added)
- volatility gating impact (ATR-based)
- time stop necessity
- touch vs close-confirmed entry comparison (with OHLC caveats)

Data:
- Intraday: yfinance 15m bars (limited history)
- Daily: yfinance 1d bars for ATR(14) gate

WARNING:
- 15m OHLC cannot resolve intrabar ordering (TP vs SL hit within same bar).
  We use a conservative tie-breaker: if both TP and SL are "hit" in the same bar,
  assume SL hit first (worst-case for longs).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Config + utilities
# ----------------------------


@dataclass
class ORBConfig:
    interval: str = "15m"
    intraday_period: str = "60d"  # yfinance 15m usually limited; keep modest
    daily_period: str = "2y"  # for ATR gate
    or_start: str = "09:30"
    or_end: str = "09:45"  # opening range window (first 15m)
    time_stop: str = "11:00"  # exit by this time if TP/SL not hit; or "CLOSE"
    entry_mode: str = "touch"  # "touch" or "close"
    tp_mult: float = 2.0  # TP = entry + tp_mult * OR_range
    sl_mode: str = "or_low"  # currently only "or_low"
    atr_len: int = 14
    vol_gate_mult: float = 0.0  # require OR_range >= vol_gate_mult * ATR; 0 disables
    min_or_range_pct: float = 0.0  # require OR_range / entry >= pct; 0 disables
    max_days_per_symbol: Optional[int] = None  # limit for speed/debug
    tiebreak: str = "worst"  # "worst", "best", "mid"


def parse_hhmm(s: str) -> Tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def ensure_tz(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Ensure DateTimeIndex is tz-aware in NY time."""
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        # yfinance sometimes returns tz-aware; sometimes naive in UTC.
        # We'll assume naive means UTC, then convert to NY.
        df = df.copy()
        df.index = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        df = df.copy()
        df.index = df.index.tz_convert(tz)
    return df


def compute_atr(daily: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR using Wilder's TR definition, simple rolling mean (good enough for Phase 0)."""
    if daily.empty:
        return pd.Series(dtype=float)
    h = daily["High"]
    l = daily["Low"]
    c = daily["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(
        axis=1
    )
    atr = tr.rolling(n, min_periods=n).mean()
    return atr


# ----------------------------
# ORB simulation core
# ----------------------------


@dataclass
class Trade:
    symbol: str
    day: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_px: float
    exit_px: float
    or_high: float
    or_low: float
    tp_px: float
    sl_px: float
    outcome: str  # "TP", "SL", "TIME", "NO_TRADE"
    r_mult: float  # PnL / (entry_px - sl_px) for long


def simulate_orb_for_symbol(sym: str, cfg: ORBConfig) -> pd.DataFrame:
    # Fetch intraday 15m
    intraday = yf.download(
        sym,
        interval=cfg.interval,
        period=cfg.intraday_period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    intraday = intraday.rename(columns=str.title)
    intraday = intraday[["Open", "High", "Low", "Close", "Volume"]].dropna()
    intraday = ensure_tz(intraday, "America/New_York")

    if intraday.empty:
        return pd.DataFrame()

    # Fetch daily for ATR gate
    daily = yf.download(
        sym,
        interval="1d",
        period=cfg.daily_period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    daily = daily.rename(columns=str.title)
    daily = daily[["Open", "High", "Low", "Close", "Volume"]].dropna()
    # Daily index is date-like; treat as NY date
    daily.index = pd.to_datetime(daily.index).tz_localize("America/New_York")

    atr = compute_atr(daily, cfg.atr_len)

    # Group intraday by NY date
    intraday["date"] = intraday.index.date
    by_day = intraday.groupby("date", sort=True)

    trades: List[Trade] = []
    or_hh, or_mm = parse_hhmm(cfg.or_start)
    oe_hh, oe_mm = parse_hhmm(cfg.or_end)

    ts_hh, ts_mm = (None, None)
    use_close_stop = False
    if cfg.time_stop.upper() == "CLOSE":
        use_close_stop = True
    else:
        ts_hh, ts_mm = parse_hhmm(cfg.time_stop)

    days_processed = 0

    for d, df_day in by_day:
        if (
            cfg.max_days_per_symbol is not None
            and days_processed >= cfg.max_days_per_symbol
        ):
            break
        days_processed += 1

        df_day = df_day.copy()
        # Only regular hours slice for Phase 0
        # (yfinance 15m typically includes only RTH, but be explicit)
        day_start = pd.Timestamp(d).tz_localize("America/New_York")
        df_day = df_day[
            (df_day.index >= day_start + pd.Timedelta(hours=9, minutes=30))
            & (df_day.index <= day_start + pd.Timedelta(hours=16, minutes=0))
        ]

        if df_day.empty:
            continue

        # Identify OR bar (09:30-09:45). With 15m, that's the first bar at 09:30.
        or_bar_time = day_start + pd.Timedelta(hours=or_hh, minutes=or_mm)
        or_bar = df_day[df_day.index == or_bar_time]
        if or_bar.empty:
            # If missing, skip (timestamp misalignment is common with yfinance)
            continue

        or_high = float(or_bar["High"].iloc[0])
        or_low = float(or_bar["Low"].iloc[0])
        or_range = or_high - or_low
        if not np.isfinite(or_range) or or_range <= 0:
            continue

        # ATR gate using daily ATR at that date (or nearest previous)
        day_ts = pd.Timestamp(d).tz_localize("America/New_York")
        atr_val = (
            atr.loc[:day_ts].dropna().iloc[-1]
            if atr.loc[:day_ts].dropna().shape[0]
            else np.nan
        )

        # Entry scan starts AFTER OR bar
        df_after = df_day[df_day.index > or_bar_time]
        if df_after.empty:
            continue

        entry_time = None
        entry_px = None

        if cfg.entry_mode == "touch":
            # Enter when high touches/exceeds OR high; assume fill at OR high (optimistic-ish)
            hit = df_after[df_after["High"] >= or_high]
            if not hit.empty:
                entry_time = hit.index[0]
                entry_px = or_high
        elif cfg.entry_mode == "close":
            # Enter on first bar close above OR high; fill at that bar close (more conservative)
            hit = df_after[df_after["Close"] > or_high]
            if not hit.empty:
                entry_time = hit.index[0]
                entry_px = float(hit["Close"].iloc[0])
        else:
            raise ValueError("entry_mode must be 'touch' or 'close'")

        if entry_time is None:
            continue

        # Vol gates (applied at entry)
        if cfg.min_or_range_pct > 0:
            if (or_range / entry_px) < cfg.min_or_range_pct:
                continue

        if cfg.vol_gate_mult > 0 and np.isfinite(atr_val):
            if or_range < cfg.vol_gate_mult * float(atr_val):
                continue

        sl_px = or_low  # cfg.sl_mode currently fixed
        risk = entry_px - sl_px
        if risk <= 0:
            continue

        tp_px = entry_px + cfg.tp_mult * or_range

        # Determine time stop cutoff
        if use_close_stop:
            cutoff_time = df_day.index.max()  # last bar of day
        else:
            cutoff_time = day_start + pd.Timedelta(hours=ts_hh, minutes=ts_mm)
            cutoff_time = min(cutoff_time, df_day.index.max())

        # Walk forward bar-by-bar until cutoff
        df_fwd = df_day[(df_day.index >= entry_time) & (df_day.index <= cutoff_time)]
        if df_fwd.empty:
            continue

        outcome = "TIME"
        exit_time = df_fwd.index[-1]
        exit_px = float(df_fwd["Close"].iloc[-1])

        for t, row in df_fwd.iterrows():
            hi = float(row["High"])
            lo = float(row["Low"])

            hit_tp = hi >= tp_px
            hit_sl = lo <= sl_px

            if hit_tp and hit_sl:
                if cfg.tiebreak == "worst":
                    # pessimistic: SL first
                    outcome = "SL"
                    exit_time = t
                    exit_px = sl_px

                elif cfg.tiebreak == "best":
                    # optimistic: TP first
                    outcome = "TP"
                    exit_time = t
                    exit_px = tp_px

                elif cfg.tiebreak == "mid":
                    # neutral: exit at bar close, count as TIME
                    outcome = "TIME"
                    exit_time = t
                    exit_px = float(row["Close"])

                else:
                    raise ValueError(f"Unknown tiebreak mode: {cfg.tiebreak}")

                break

            elif hit_sl:
                outcome = "SL"
                exit_time = t
                exit_px = sl_px
                break

            elif hit_tp:
                outcome = "TP"
                exit_time = t
                exit_px = tp_px
                break

        pnl = exit_px - entry_px
        r_mult = pnl / risk

        trades.append(
            Trade(
                symbol=sym,
                day=day_ts,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_px=entry_px,
                exit_px=exit_px,
                or_high=or_high,
                or_low=or_low,
                tp_px=tp_px,
                sl_px=sl_px,
                outcome=outcome,
                r_mult=r_mult,
            )
        )

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame([t.__dict__ for t in trades])
    # Add a simple vol regime label via ATR percentile (per symbol)
    # (Phase 0 only; don't overthink)
    df["atr14"] = np.nan
    if not atr.empty:
        atr_map = atr.rename("atr14").to_frame()
        atr_map["date"] = atr_map.index.date
        atr_by_date = atr_map.dropna().groupby("date")["atr14"].last()
        df["atr14"] = df["day"].dt.date.map(atr_by_date)

    if df["atr14"].notna().any():
        qs = df["atr14"].quantile([0.33, 0.66]).values
        low_q, mid_q = qs[0], qs[1]

        def regime(a):
            if not np.isfinite(a):
                return "unknown"
            if a <= low_q:
                return "low_vol"
            if a <= mid_q:
                return "mid_vol"
            return "high_vol"

        df["vol_regime"] = df["atr14"].apply(regime)
    else:
        df["vol_regime"] = "unknown"

    return df


def summarize(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades.")
        return

    n = len(trades)
    win = (trades["r_mult"] > 0).mean()
    tp_rate = (trades["outcome"] == "TP").mean()
    sl_rate = (trades["outcome"] == "SL").mean()
    time_rate = (trades["outcome"] == "TIME").mean()

    print("\n================ ORB Phase 0 Summary ================")
    print(f"Trades: {n}")
    print(f"Win rate (R>0): {win:.2%}")
    print(f"Outcome rates: TP={tp_rate:.2%}  SL={sl_rate:.2%}  TIME={time_rate:.2%}")
    print(
        f"R-multiple: mean={trades['r_mult'].mean():.3f}  median={trades['r_mult'].median():.3f}"
    )
    print(
        f"R-multiple: p10={trades['r_mult'].quantile(0.10):.3f}  p90={trades['r_mult'].quantile(0.90):.3f}"
    )
    print(f"% trades >= +1R: {(trades['r_mult'] >= 1.0).mean():.2%}")
    print(f"% trades >= +2R: {(trades['r_mult'] >= 2.0).mean():.2%}")
    print("=====================================================\n")

    # Slice by volatility regime (rough)
    print("By ATR(14) regime (per-symbol terciles):")
    grp = trades.groupby("vol_regime")["r_mult"].agg(["count", "mean", "median"])
    print(grp.to_string())
    print()

    # Slice by symbol
    print("By symbol:")
    grp2 = trades.groupby("symbol")["r_mult"].agg(["count", "mean", "median"])
    print(grp2.sort_values("mean", ascending=False).to_string())
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD",
        help="Comma-separated tickers",
    )
    ap.add_argument(
        "--entry",
        type=str,
        default="touch",
        choices=["touch", "close"],
        help="Entry mode: touch OR high, or close-confirm above OR high",
    )
    ap.add_argument("--tp", type=float, default=2.0, help="TP multiple of OR range")
    ap.add_argument(
        "--time-stop", type=str, default="11:00", help='e.g. "11:00" NY time or "CLOSE"'
    )
    ap.add_argument(
        "--vol-gate",
        type=float,
        default=0.0,
        help="Require OR_range >= vol_gate * ATR14 (0 disables)",
    )
    ap.add_argument(
        "--min-or-pct",
        type=float,
        default=0.0,
        help="Require OR_range/entry >= this pct (0 disables)",
    )
    ap.add_argument(
        "--days",
        type=int,
        default=60,
        help="Intraday period days (yfinance uses '60d' etc.)",
    )
    ap.add_argument(
        "--max-days-per-symbol", type=int, default=None, help="Debug cap per symbol"
    )
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    ap.add_argument(
        "--tiebreak", type=str, default="worst", help="Tie-breaker: worst, best, mid"
    )
    args = ap.parse_args()

    cfg = ORBConfig(
        intraday_period=f"{args.days}d",
        entry_mode=args.entry,
        tp_mult=args.tp,
        time_stop=args.time_stop,
        vol_gate_mult=args.vol_gate,
        min_or_range_pct=args.min_or_pct,
        max_days_per_symbol=args.max_days_per_symbol,
        tiebreak=args.tiebreak,
    )

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    all_trades = []
    for sym in syms:
        df = simulate_orb_for_symbol(sym, cfg)
        if not df.empty:
            all_trades.append(df)

    if not all_trades:
        print(
            "No trades across symbols. (Possible causes: yfinance 15m timestamp gaps, market holidays, etc.)"
        )
        return

    trades = pd.concat(all_trades, ignore_index=True)
    summarize(trades)

    if args.out:
        trades.to_csv(args.out, index=False)
        print(f"Wrote trades to {args.out}")


if __name__ == "__main__":
    main()
