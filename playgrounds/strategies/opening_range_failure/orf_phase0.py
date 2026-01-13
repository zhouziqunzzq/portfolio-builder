#!/usr/bin/env python3
"""
Opening Range Failure (ORF) — Phase 0 Findings
==============================================

Context
-------
This script explores ORF (Opening Range Failure) as an *intraday context signal*,
NOT as a tradable alpha. ORF was tested after ORB (Opening Range Breakout) failed
robustly as a standalone strategy at bar-level abstraction.

All experiments use coarse intraday data (15m OHLC, yfinance) and are designed
to answer STRUCTURAL questions, not optimize PnL.

Definitions
-----------
Opening Range (OR):
- First 15 minutes of RTH (09:30-09:45 NY time)
- OR_high, OR_low, OR_range

OR Failure (ORF):
- Price breaks beyond OR boundary (up or down) after 09:45
- But by a fixed early cutoff (e.g. 10:30), price closes back inside the OR

We further characterize failures by:
1) DEPTH: how far price pushed beyond OR boundary (normalized by OR_range)
2) SPEED: how quickly price returned inside OR (bars from attempt → failure)

Key Empirical Findings
---------------------
1) ORF is COMMON (~55-60% of days)
   → This is not a rare-event alpha; it is a day-type descriptor.

2) ORF is ASYMMETRIC:
   - ORF-up days are mostly flat / low-conviction
   - ORF-down days show mild negative drift on average

3) Depth matters more than direction:
   - SHALLOW failures (small excursion beyond OR) are BAD for trend:
       * lower returns
       * higher chop
       * fragile market structure
   - DEEP failures (large excursion beyond OR) are NOT bearish:
       * often positive day-level drift
       * lower realized volatility
       * interpreted as “liquidity test + resolution”

4) Speed is critical:
   - FAST failures (resolved within ~30 minutes of attempt) dominate and are stable
   - SLOW failures are rare, noisy, and should be ignored

Final Stable Buckets (FAST only)
--------------------------------
Using depth threshold deep_k ≈ 0.5 x OR_range and fast_bars ≈ 2 (30 min):

- down_shallow_fast  → worst bucket
    * negative mean & median day returns
    * choppy, fragile environment
    * strong RISK-OFF context

- up_deep_fast OR down_deep_fast → best buckets
    * positive mean & median day returns
    * lower volatility
    * indicates early liquidity resolution
    * RISK-ON / confidence-affirming context

- all other cases → neutral

Distilled Signal (Recommended)
------------------------------
Define a daily intraday context variable:

    ORF_CONTEXT ∈ { -1, 0, +1 }

Rules:
- ORF_CONTEXT = -1 if down_shallow_fast
- ORF_CONTEXT = +1 if (up_deep_fast OR down_deep_fast)
- ORF_CONTEXT =  0 otherwise

Usage Guidance (IMPORTANT)
--------------------------
- ORF_CONTEXT is NOT an alpha.
- It should NEVER select assets or directions.
- It SHOULD be used as a CONTEXT / CONFIDENCE modifier.

Best integration points in V2:
1) Regime confidence scaling (risk-on vs risk-off weighting)
2) Volatility-based throttling of new exposure
3) Rebalance aggressiveness / exposure delay logic
4) Soft drawdown recovery gating

Explicit Non-Uses:
- No direct trading
- No intraday execution
- No parameter optimization beyond coarse thresholds

Status
------
Phase 0 COMPLETE.
ORF (depth x speed) qualifies as a robust intraday-derived CONTEXT FEATURE
and is a candidate for integration into V2 regime / exposure control logic.
"""


from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import List


@dataclass
class ORFConfig:
    interval: str = "15m"
    period: str = "60d"
    or_start: str = "09:30"
    or_end: str = "09:45"
    failure_check_time: str = "10:30"  # when we judge success vs failure
    # depth threshold in units of OR_range
    # e.g. 0.5 means "price pushed >= 0.5 * OR_range beyond the boundary"
    deep_k: float = 0.5
    # speed threshold (in bars) for "fast" failure
    # With 15m bars, 2 bars = 30 minutes after OR bar.
    fast_bars: int = 2


def ensure_ny_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df = df.tz_convert("America/New_York")
    return df


def hhmm(ts: str):
    h, m = ts.split(":")
    return int(h), int(m)


def run_orf(symbol: str, cfg: ORFConfig) -> pd.DataFrame:
    df = yf.download(
        symbol,
        interval=cfg.interval,
        period=cfg.period,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = ensure_ny_tz(df)

    df["date"] = df.index.date
    out = []

    or_h, or_m = hhmm(cfg.or_start)
    chk_h, chk_m = hhmm(cfg.failure_check_time)

    for d, day in df.groupby("date"):
        day_start = pd.Timestamp(d).tz_localize("America/New_York")

        rth = day[
            (day.index >= day_start + pd.Timedelta(hours=9, minutes=30))
            & (day.index <= day_start + pd.Timedelta(hours=16))
        ]

        if rth.empty:
            continue

        or_bar_time = day_start + pd.Timedelta(hours=or_h, minutes=or_m)
        or_bar = rth[rth.index == or_bar_time]
        if or_bar.empty:
            continue

        ORH = float(or_bar["High"].iloc[0])
        ORL = float(or_bar["Low"].iloc[0])
        ORR = ORH - ORL
        if ORR <= 0:
            continue

        after_or = rth[rth.index > or_bar_time]
        if after_or.empty:
            continue

        check_time = day_start + pd.Timedelta(hours=chk_h, minutes=chk_m)
        upto_check = after_or[after_or.index <= check_time]
        if upto_check.empty:
            continue

        # --- Detect attempts ---
        broke_up = (upto_check["High"] > ORH).any()
        broke_down = (upto_check["Low"] < ORL).any()
        max_hi = float(upto_check["High"].max())
        min_lo = float(upto_check["Low"].min())

        # excursion beyond the boundary, normalized by OR_range
        up_excursion_k = max(0.0, (max_hi - ORH) / ORR)
        down_excursion_k = max(0.0, (ORL - min_lo) / ORR)

        close_at_check = float(upto_check.iloc[-1]["Close"])

        orf_up = broke_up and (close_at_check <= ORH)
        orf_down = broke_down and (close_at_check >= ORL)

        # shallow vs deep failure
        orf_up_deep = bool(orf_up and (up_excursion_k >= cfg.deep_k))
        orf_up_shallow = bool(orf_up and (0.0 < up_excursion_k < cfg.deep_k))

        orf_down_deep = bool(orf_down and (down_excursion_k >= cfg.deep_k))
        orf_down_shallow = bool(orf_down and (0.0 < down_excursion_k < cfg.deep_k))

        # --- Time-to-failure (up/down) ---
        # Define "attempt time" = first bar where boundary is breached (High > ORH or Low < ORL)
        # Define "failure time" = first bar AFTER attempt where Close returns inside (<= ORH for up, >= ORL for down)
        # If it never returns inside by check_time, treat as no failure (NaN).

        def time_to_failure_bars_upto_check(direction: str) -> float:
            if direction == "up":
                attempt_mask = upto_check["High"] > ORH
                if not attempt_mask.any():
                    return np.nan
                attempt_time = upto_check.index[attempt_mask.argmax()]  # first True
                after_attempt = upto_check[upto_check.index >= attempt_time]
                fail_mask = after_attempt["Close"] <= ORH
                if not fail_mask.any():
                    return np.nan
                fail_time = after_attempt.index[fail_mask.argmax()]
            else:  # "down"
                attempt_mask = upto_check["Low"] < ORL
                if not attempt_mask.any():
                    return np.nan
                attempt_time = upto_check.index[attempt_mask.argmax()]
                after_attempt = upto_check[upto_check.index >= attempt_time]
                fail_mask = after_attempt["Close"] >= ORL
                if not fail_mask.any():
                    return np.nan
                fail_time = after_attempt.index[fail_mask.argmax()]

            # number of 15m bars from attempt to failure inclusive/exclusive?
            # Use difference in bar index positions (0 means same bar close is already inside).
            # With 15m bars, this is coarse but consistent.
            pos_attempt = upto_check.index.get_loc(attempt_time)
            pos_fail = upto_check.index.get_loc(fail_time)
            return float(pos_fail - pos_attempt)

        up_ttf_bars = time_to_failure_bars_upto_check("up")
        dn_ttf_bars = time_to_failure_bars_upto_check("down")

        # Convert to minutes (15m bars)
        bar_minutes = 15.0
        up_ttf_min = up_ttf_bars * bar_minutes if np.isfinite(up_ttf_bars) else np.nan
        dn_ttf_min = dn_ttf_bars * bar_minutes if np.isfinite(dn_ttf_bars) else np.nan

        # fast vs slow (only meaningful if ORF happened)
        orf_up_fast = bool(
            orf_up and np.isfinite(up_ttf_bars) and (up_ttf_bars <= cfg.fast_bars)
        )
        orf_up_slow = bool(
            orf_up and np.isfinite(up_ttf_bars) and (up_ttf_bars > cfg.fast_bars)
        )

        orf_down_fast = bool(
            orf_down and np.isfinite(dn_ttf_bars) and (dn_ttf_bars <= cfg.fast_bars)
        )
        orf_down_slow = bool(
            orf_down and np.isfinite(dn_ttf_bars) and (dn_ttf_bars > cfg.fast_bars)
        )

        # Combine depth × speed buckets (most useful view)
        # Example bucket names: up_shallow_fast, up_deep_slow, etc.
        up_bucket = "none"
        if orf_up:
            depth = "deep" if up_excursion_k >= cfg.deep_k else "shallow"
            speed = (
                "fast"
                if (np.isfinite(up_ttf_bars) and up_ttf_bars <= cfg.fast_bars)
                else "slow"
            )
            up_bucket = f"up_{depth}_{speed}"

        dn_bucket = "none"
        if orf_down:
            depth = "deep" if down_excursion_k >= cfg.deep_k else "shallow"
            speed = (
                "fast"
                if (np.isfinite(dn_ttf_bars) and dn_ttf_bars <= cfg.fast_bars)
                else "slow"
            )
            dn_bucket = f"down_{depth}_{speed}"

        day_open = float(rth.iloc[0]["Open"])
        day_close = float(rth.iloc[-1]["Close"])
        day_ret = (day_close / day_open) - 1.0

        out.append(
            {
                "symbol": symbol,
                "date": pd.Timestamp(d),
                "orf_up": orf_up,
                "orf_down": orf_down,
                # depth metrics
                "up_excursion_k": up_excursion_k,
                "down_excursion_k": down_excursion_k,
                # shallow/deep flags
                "orf_up_shallow": orf_up_shallow,
                "orf_up_deep": orf_up_deep,
                "orf_down_shallow": orf_down_shallow,
                "orf_down_deep": orf_down_deep,
                "or_range_pct": ORR / ORH,
                "day_ret": day_ret,
                "abs_day_ret": abs(day_ret),
                # NEW: time-to-failure
                "up_ttf_bars": up_ttf_bars,
                "dn_ttf_bars": dn_ttf_bars,
                "up_ttf_min": up_ttf_min,
                "dn_ttf_min": dn_ttf_min,
                # NEW: speed flags + buckets
                "orf_up_fast": orf_up_fast,
                "orf_up_slow": orf_up_slow,
                "orf_down_fast": orf_down_fast,
                "orf_down_slow": orf_down_slow,
                "up_bucket": up_bucket,
                "dn_bucket": dn_bucket,
            }
        )

    return pd.DataFrame(out)


def summarize(df: pd.DataFrame):
    print("\n========== ORF Phase 0 Summary ==========")
    print(f"Days: {len(df)}")

    def show_group(name: str, mask: pd.Series):
        sub = df[mask]
        if sub.empty:
            print(f"\n{name}: no occurrences")
            return
        print(f"\n{name}:")
        print(f"  freq: {len(sub) / len(df):.2%}")
        print(f"  mean day ret: {sub['day_ret'].mean():.3%}")
        print(f"  median day ret: {sub['day_ret'].median():.3%}")
        print(f"  mean |ret|: {sub['abs_day_ret'].mean():.3%}")

        # show average depth if present
        if "up_excursion_k" in sub.columns and "down_excursion_k" in sub.columns:
            print(f"  mean up_exc_k: {sub['up_excursion_k'].mean():.3f}")
            print(f"  mean dn_exc_k: {sub['down_excursion_k'].mean():.3f}")

    show_group("orf_up (all)", df["orf_up"])
    if "orf_up_shallow" in df.columns:
        show_group("orf_up_shallow", df["orf_up_shallow"])
        show_group("orf_up_deep", df["orf_up_deep"])

    show_group("orf_down (all)", df["orf_down"])
    if "orf_down_shallow" in df.columns:
        show_group("orf_down_shallow", df["orf_down_shallow"])
        show_group("orf_down_deep", df["orf_down_deep"])

    base = df[~(df["orf_up"] | df["orf_down"])]
    print("\nNon-ORF days:")
    print(f"  mean day ret: {base['day_ret'].mean():.3%}")
    print(f"  mean |ret|: {base['abs_day_ret'].mean():.3%}")

    # bucket summary (depth × speed)
    def bucket_summary(col: str, title: str):
        if col not in df.columns:
            return
        print(f"\n{title}:")
        g = (
            df[df[col] != "none"]
            .groupby(col)["day_ret"]
            .agg(["count", "mean", "median"])
            .sort_values("count", ascending=False)
        )
        if g.empty:
            print("  (none)")
        else:
            print(g.to_string())

    bucket_summary("up_bucket", "Up ORF buckets (depth x speed)")
    bucket_summary("dn_bucket", "Down ORF buckets (depth x speed)")

    print("========================================\n")


if __name__ == "__main__":
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META"]
    cfg = ORFConfig()

    all_days: List[pd.DataFrame] = []
    for s in symbols:
        df = run_orf(s, cfg)
        if not df.empty:
            all_days.append(df)

    if not all_days:
        print("No data.")
    else:
        res = pd.concat(all_days, ignore_index=True)
        summarize(res)

"""
Example run using 15m intervals and 60d period:
========== ORF Phase 0 Summary ==========
Days: 420

orf_up (all):
  freq: 29.76%
  mean day ret: -0.017%
  median day ret: 0.185%
  mean |ret|: 0.793%
  mean up_exc_k: 0.272
  mean dn_exc_k: 0.180

orf_up_shallow:
  freq: 26.19%
  mean day ret: -0.047%
  median day ret: 0.173%
  mean |ret|: 0.822%
  mean up_exc_k: 0.201
  mean dn_exc_k: 0.184

orf_up_deep:
  freq: 3.57%
  mean day ret: 0.206%
  median day ret: 0.347%
  mean |ret|: 0.582%
  mean up_exc_k: 0.797
  mean dn_exc_k: 0.154

orf_down (all):
  freq: 27.86%
  mean day ret: -0.146%
  median day ret: -0.119%
  mean |ret|: 0.883%
  mean up_exc_k: 0.235
  mean dn_exc_k: 0.291

orf_down_shallow:
  freq: 22.38%
  mean day ret: -0.220%
  median day ret: -0.166%
  mean |ret|: 0.904%
  mean up_exc_k: 0.253
  mean dn_exc_k: 0.176

orf_down_deep:
  freq: 5.48%
  mean day ret: 0.155%
  median day ret: 0.295%
  mean |ret|: 0.796%
  mean up_exc_k: 0.163
  mean dn_exc_k: 0.757

Non-ORF days:
  mean day ret: -0.187%
  mean |ret|: 1.020%

Up ORF buckets (depth x speed):
                 count      mean    median
up_bucket                                 
up_shallow_fast    106 -0.000800  0.001665
up_deep_fast        12  0.001780  0.002892
up_shallow_slow      4  0.008258  0.009525
up_deep_slow         3  0.003196  0.003466

Down ORF buckets (depth x speed):
                   count      mean    median
dn_bucket                                   
down_shallow_fast     94 -0.002199 -0.001662
down_deep_fast        20  0.003107  0.003626
down_deep_slow         3 -0.008848 -0.005663
========================================
"""
