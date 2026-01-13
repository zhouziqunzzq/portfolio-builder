"""
MVP Bear Trap Strategy (15-minute bars) — playground script

Data source: yfinance
Timeframe: 15m only (no L2 / no tick data)
Market: US equities (best on liquid names)

Bear Trap (failed breakdown) — simple rule:
- Define intraday SUPPORT as the session opening range low (first N bars, default=2 -> first 30 min).
- If price breaks below SUPPORT by >= break_pct (default 0.10%) and later closes back ABOVE SUPPORT
  within max_reclaim_bars (default 6 bars = 90 min), enter LONG at reclaim bar close.
- Stop: lowest low during the breakdown episode minus stop_buffer_pct.
- Target: intraday VWAP at entry time (simple, robust), OR optional R-multiple target.
- Exit: stop / target / end-of-day liquidation.

This is intentionally simple and “structure-first”, not speed-first.

Notes:
- yfinance intraday data availability is limited (often ~60 days back for many tickers).
- If you want longer history, you’ll need a paid provider (Polygon, Alpaca SIP, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Config
# -----------------------------
@dataclass
class BearTrapConfig:
    ticker: str = "SPY"
    period: str = "60d"  # yfinance intraday limitation (typically)
    interval: str = "15m"

    # Session definition
    session_tz: str = "America/New_York"
    rth_start: str = "09:30"
    rth_end: str = "16:00"

    # Support level = opening range low of first N bars
    opening_range_bars: int = 2  # 2 * 15m = first 30 minutes

    # Trap definition
    break_pct: float = 0.001  # 0.10% below support qualifies as breakdown
    max_reclaim_bars: int = (
        6  # must reclaim within 6 bars (90 min) after first breakdown
    )

    # Risk/exit
    stop_buffer_pct: float = 0.0005  # 0.05% below breakdown low
    target_mode: str = "vwap"  # "vwap" or "r_multiple"
    target_r: float = 2.0  # only used if target_mode="r_multiple"
    runner_frac: float = 0.5  # fraction of shares for runner (0.0-1.0)
    runner_r: float = 2.0  # runner target in R (beyond entry)
    min_profit_bps: float = 2.0  # minimum profit for VWAP target to be valid

    # Trading constraints
    one_trade_per_day: bool = True
    fee_per_trade: float = 0.0  # simple flat fee (optional)
    slippage_bps: float = 0.0  # apply to entries/exits (optional), in basis points

    # Position sizing
    initial_cash: float = 100_000.0
    risk_fraction: float = (
        0.01  # risk 1% of equity per trade (position size via stop distance)
    )


# -----------------------------
# Helpers
# -----------------------------
def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)


def _in_rth(ts: pd.Timestamp, cfg: BearTrapConfig) -> bool:
    """Check if timestamp is within regular trading hours."""
    ts_local = ts.tz_convert(cfg.session_tz)
    h, m = ts_local.hour, ts_local.minute
    sh, sm = _parse_hhmm(cfg.rth_start)
    eh, em = _parse_hhmm(cfg.rth_end)
    start_ok = (h > sh) or (h == sh and m >= sm)
    end_ok = (h < eh) or (h == eh and m < em)
    return start_ok and end_ok


def _apply_slippage(price: float, bps: float, side: str) -> float:
    """
    Apply a simple slippage model.
    side: "buy" => pay up, "sell" => receive less.
    """
    if bps <= 0:
        return price
    factor = 1.0 + (bps / 10_000.0) if side == "buy" else 1.0 - (bps / 10_000.0)
    return price * factor


def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP computed intraday per session:
      vwap_t = sum(price_typical * volume) / sum(volume)
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = typical * df["Volume"].fillna(0.0)

    # group by local session date (NY)
    local_dt = df.index.tz_convert("America/New_York")
    session_date = local_dt.date

    pv_cum = pv.groupby(session_date).cumsum()
    vol_cum = df["Volume"].fillna(0.0).groupby(session_date).cumsum().replace(0, np.nan)
    return pv_cum / vol_cum


# -----------------------------
# Strategy logic
# -----------------------------
@dataclass
class Trade:
    date: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry: float
    exit: float
    shares: int
    pnl: float
    r_multiple: float
    reason: str


def load_data(cfg: BearTrapConfig) -> pd.DataFrame:
    # IMPORTANT: yfinance sometimes returns naive timestamps; we force UTC then convert.
    df = yf.download(
        cfg.ticker,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=False,
        prepost=False,  # keep it simple: regular session only
        progress=False,
        multi_level_index=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            "No data returned from yfinance. Try a different ticker/period/interval."
        )

    # Ensure standard columns
    needed = {"Open", "High", "Low", "Close", "Volume"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns from yfinance data: {missing}")

    # Localize index
    if df.index.tz is None:
        # yfinance intraday is typically in UTC; safest assumption is UTC if naive.
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Filter to RTH just in case
    df = df[df.index.map(lambda ts: _in_rth(ts, cfg))].copy()
    if df.empty:
        raise RuntimeError(
            "After filtering to RTH, no rows remain. Check timezone/session settings."
        )
    return df


def backtest_bear_trap_15m(
    df: pd.DataFrame, cfg: BearTrapConfig
) -> Tuple[pd.DataFrame, List[Trade]]:
    df = df.copy()
    df["vwap"] = compute_intraday_vwap(df)
    # print(f"vwap: {df['vwap'].head(20)}")

    # session grouping
    local_dt = df.index.tz_convert(cfg.session_tz)
    df["session_date"] = local_dt.date

    # cash = cfg.initial_cash
    equity = cfg.initial_cash

    trades: List[Trade] = []
    equity_curve = []

    for session_date, day in df.groupby("session_date"):
        day = day.copy()
        day = day.sort_index()

        # Define opening range (first N bars)
        if len(day) < cfg.opening_range_bars + 1:
            # not enough bars for signal
            # record equity snapshot at EOD
            equity_curve.append((day.index[-1], equity))
            continue

        opening = day.iloc[: cfg.opening_range_bars]
        support = float(opening["Low"].min())

        # Find first breakdown below support by break_pct
        breakdown_level = support * (1.0 - cfg.break_pct)
        below = day["Low"] <= breakdown_level

        if not below.any():
            equity_curve.append((day.index[-1], equity))
            continue

        first_break_idx = int(np.argmax(below.values))  # first True index
        # Search for reclaim within next max_reclaim_bars
        reclaim_window = day.iloc[
            first_break_idx : first_break_idx + cfg.max_reclaim_bars + 1
        ]
        reclaim = reclaim_window["Close"] > support

        if not reclaim.any():
            equity_curve.append((day.index[-1], equity))
            continue

        reclaim_pos = int(np.argmax(reclaim.values))
        entry_bar = reclaim_window.iloc[reclaim_pos]
        entry_ts = reclaim_window.index[reclaim_pos]
        entry_price_raw = float(entry_bar["Close"])
        entry_price = _apply_slippage(entry_price_raw, cfg.slippage_bps, side="buy")

        # Stop: lowest low from first breakdown up to entry bar, minus buffer
        breakdown_slice = day.iloc[first_break_idx : day.index.get_loc(entry_ts) + 1]
        breakdown_low = float(breakdown_slice["Low"].min())
        stop_price = breakdown_low * (1.0 - cfg.stop_buffer_pct)

        # Risk per share
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            equity_curve.append((day.index[-1], equity))
            continue

        # Position size: risk_fraction of equity
        risk_budget = equity * cfg.risk_fraction
        shares = int(math.floor(risk_budget / risk_per_share))
        if shares <= 0:
            equity_curve.append((day.index[-1], equity))
            continue

        # -----------------------------
        # Determine targets (VWAP + runner)
        # -----------------------------
        if cfg.target_mode == "vwap":
            vwap_price = float(day.loc[entry_ts, "vwap"])
            if not np.isfinite(vwap_price):
                # If VWAP is NaN, fall back to pure R-multiple (single target)
                vwap_price = entry_price + cfg.target_r * risk_per_share

            # Enforce VWAP target must be above entry by at least min_profit_bps
            min_target = entry_price * (1.0 + cfg.min_profit_bps / 10_000.0)
            if vwap_price <= min_target:
                equity_curve.append((day.index[-1], equity))
                continue

            target_vwap = vwap_price

            # Runner target: at least runner_r * R above entry,
            # and also at least as high as VWAP (so runner isn't redundant / worse).
            runner_target = max(
                entry_price + cfg.runner_r * risk_per_share,
                target_vwap,
            )

        elif cfg.target_mode == "r_multiple":
            # Keep your old behavior: one target only (no runner logic)
            target_vwap = entry_price + cfg.target_r * risk_per_share
            runner_target = target_vwap
        else:
            raise ValueError("target_mode must be 'vwap' or 'r_multiple'")

        # Split shares into VWAP tranche + runner tranche
        runner_shares = int(math.floor(shares * cfg.runner_frac))
        runner_shares = max(0, min(shares, runner_shares))
        vwap_shares = shares - runner_shares

        # -----------------------------
        # Simulate forward bar-by-bar exits
        # -----------------------------
        exit_ts = day.index[-1]
        exit_price = float(day.iloc[-1]["Close"])
        reason = "eod"

        # Track tranche exits
        vwap_filled = vwap_shares == 0
        runner_filled = runner_shares == 0

        vwap_exit_px = None
        vwap_exit_ts = None

        runner_exit_px = None
        runner_exit_ts = None

        # Start from next bar since entry is at close of entry_ts
        forward = day.loc[entry_ts:].iloc[1:]

        for ts, row in forward.iterrows():
            # 1) Stop has priority: flatten everything
            if row["Low"] <= stop_price:
                stop_px = _apply_slippage(stop_price, cfg.slippage_bps, side="sell")
                exit_ts = ts
                exit_price = stop_px
                reason = "stop"

                if not vwap_filled:
                    vwap_exit_ts, vwap_exit_px = ts, stop_px
                    vwap_filled = True
                if not runner_filled:
                    runner_exit_ts, runner_exit_px = ts, stop_px
                    runner_filled = True
                break

            # 2) VWAP tranche
            if (not vwap_filled) and (row["High"] >= target_vwap):
                px = _apply_slippage(target_vwap, cfg.slippage_bps, side="sell")
                vwap_exit_ts, vwap_exit_px = ts, px
                vwap_filled = True

            # 3) Runner tranche
            if (not runner_filled) and (row["High"] >= runner_target):
                px = _apply_slippage(runner_target, cfg.slippage_bps, side="sell")
                runner_exit_ts, runner_exit_px = ts, px
                runner_filled = True

            # If both tranches done, we can stop scanning
            if vwap_filled and runner_filled:
                exit_ts = max(vwap_exit_ts, runner_exit_ts)
                # exit_price is not used for PnL now; keep something consistent
                exit_price = (
                    runner_exit_px if runner_exit_ts >= vwap_exit_ts else vwap_exit_px
                )
                reason = "vwap+runner"
                break

        # If still open at EOD, close remaining tranches at close
        if not vwap_filled or not runner_filled:
            eod_ts = day.index[-1]
            eod_px = _apply_slippage(
                float(day.iloc[-1]["Close"]), cfg.slippage_bps, side="sell"
            )

            if not vwap_filled:
                vwap_exit_ts, vwap_exit_px = eod_ts, eod_px
                vwap_filled = True
            if not runner_filled:
                runner_exit_ts, runner_exit_px = eod_ts, eod_px
                runner_filled = True

            exit_ts = eod_ts
            exit_price = eod_px
            reason = "eod"

        # -----------------------------
        # PnL and equity update (tranche-based)
        # -----------------------------
        gross_pnl = 0.0
        if vwap_shares > 0:
            gross_pnl += (float(vwap_exit_px) - entry_price) * vwap_shares
        if runner_shares > 0:
            gross_pnl += (float(runner_exit_px) - entry_price) * runner_shares

        # Fees: entry once, exit once if both tranches exit same time; otherwise count both exits.
        exit_legs = 0
        if vwap_shares > 0:
            exit_legs += 1
        if runner_shares > 0 and (runner_exit_ts != vwap_exit_ts):
            exit_legs += 1

        net_pnl = gross_pnl - cfg.fee_per_trade * (1 + exit_legs)
        equity += net_pnl

        # Realized R multiple uses total risk budget (risk_per_share * total shares)
        denom = risk_per_share * shares
        r_mult = net_pnl / denom if denom > 0 else np.nan

        trades.append(
            Trade(
                date=str(session_date),
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                entry=entry_price,
                exit=exit_price,
                shares=shares,
                pnl=net_pnl,
                r_multiple=float(r_mult),
                reason=reason,
            )
        )

        # One trade per day handling (we already do only one by design here)
        equity_curve.append((day.index[-1], equity))

    eq = (
        pd.DataFrame(equity_curve, columns=["ts", "equity"])
        .set_index("ts")
        .sort_index()
    )
    return eq, trades


def summarize(trades: List[Trade], eq: pd.DataFrame, cfg: BearTrapConfig) -> None:
    print(f"Ticker: {cfg.ticker} | Period: {cfg.period} | Interval: {cfg.interval}")
    print(f"Trades: {len(trades)}")
    if not trades:
        print(
            "No trades generated. Try relaxing break_pct/max_reclaim_bars or changing ticker."
        )
        return

    tdf = pd.DataFrame([t.__dict__ for t in trades])
    win = (tdf["pnl"] > 0).mean()
    avg_r = tdf["r_multiple"].mean()
    med_r = tdf["r_multiple"].median()
    total = tdf["pnl"].sum()
    # Localize timestamps to cfg.session_tz for display
    tdf["entry_ts"] = tdf["entry_ts"].dt.tz_convert(cfg.session_tz)
    tdf["exit_ts"] = tdf["exit_ts"].dt.tz_convert(cfg.session_tz)

    # Simple drawdown on equity curve
    peak = eq["equity"].cummax()
    dd = eq["equity"] / peak - 1.0
    mdd = dd.min()

    print(f"Win rate: {win:.2%}")
    print(f"Avg R: {avg_r:.2f} | Median R: {med_r:.2f}")
    print(f"Total PnL: ${total:,.2f}")
    print(f"Final equity: ${eq['equity'].iloc[-1]:,.2f}")
    print(f"Max drawdown (equity curve): {mdd:.2%}")
    print("\nRecent trades:")
    print(
        tdf.tail(10)[
            [
                "date",
                "entry_ts",
                "exit_ts",
                "entry",
                "exit",
                "shares",
                "pnl",
                "r_multiple",
                "reason",
            ]
        ].to_string(index=False)
    )


def main():
    cfg = BearTrapConfig(
        ticker="SPY",
        period="60d",
        interval="15m",
        opening_range_bars=2,
        break_pct=0.001,
        max_reclaim_bars=6,
        stop_buffer_pct=0.0005,
        target_mode="vwap",  # "vwap" or "r_multiple"
        runner_frac=0.5,
        runner_r=2.0,
        min_profit_bps=2.0,
        target_r=2.0,
        slippage_bps=2.0,
        fee_per_trade=0.0,
        risk_fraction=0.01,
    )

    df = load_data(cfg)
    eq, trades = backtest_bear_trap_15m(df, cfg)
    summarize(trades, eq, cfg)


if __name__ == "__main__":
    main()
