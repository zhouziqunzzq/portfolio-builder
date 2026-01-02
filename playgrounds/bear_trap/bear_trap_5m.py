"""
Bear Trap MVP (15m structure + 5m execution) — standalone playground script

Data source: yfinance
Structure timeframe: 15-minute (support level / opening range)
Execution timeframe: 5-minute (entry, stop, exits)
No L2 / no tick data. Designed to avoid latency dependence.

Core idea (Bear Trap / Failed Breakdown):
1) Build 15m "structure" bars from 5m data (resample).
2) Define SUPPORT = low of the first N 15m bars (opening range).
3) On 5m bars:
   - Breakdown: low <= support * (1 - break_pct)
   - Reclaim: close > support within max_reclaim_bars_5m
   -> Enter long at reclaim bar CLOSE
4) Stop: fixed structural stop = min(low) from first breakdown to entry, minus stop_buffer_pct
5) Exit: tranche-based
   - Take-profit tranche at VWAP (5m intraday VWAP)
   - Runner tranche at max(VWAP, entry + runner_r * R)
   - Stop applies to BOTH tranches (kept fixed; no post-VWAP stop relaxation)
   - Start exit checks from the NEXT 5m bar (no same-bar fantasy fills)
   - Liquidate remaining at EOD close

Notes:
- yfinance intraday availability is limited (often ~60 days for 5m).
- For serious research, use Polygon/Alpaca SIP/etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Config
# -----------------------------
@dataclass
class BearTrapConfig:
    ticker: str = "SPY"
    period: str = "60d"
    interval_exec: str = "5m"  # execution bars from yfinance (5m)
    structure_tf: str = "15min"  # pandas resample rule for structure bars

    session_tz: str = "America/New_York"
    rth_start: str = "09:30"
    rth_end: str = "16:00"

    opening_range_bars_15m: int = 2  # first 2x15m = first 30 minutes
    break_pct: float = 0.001  # 0.10% breakdown threshold below support
    max_reclaim_bars_5m: int = 18  # 18x5m = 90 minutes window to reclaim

    stop_buffer_pct: float = 0.0005  # 0.05% below breakdown-low

    # Targets / tranche logic
    target_mode: str = "vwap"  # "vwap" or "r_multiple"
    target_r: float = 2.0  # used if target_mode="r_multiple" (single target)
    runner_frac: float = 0.5  # runner fraction of shares (0..1)
    runner_r: float = 2.0  # runner target in R (beyond entry)
    min_profit_bps: float = 2.0  # require VWAP target >= entry*(1+min_profit_bps)
    entry_cutoff_hhmm: str = "12:30"  # no new entries after this time

    # Trading costs
    slippage_bps: float = 2.0  # applied to entry and exits
    fee_per_trade: float = 0.0  # flat fee per leg (optional)

    # Position sizing
    initial_cash: float = 100_000.0
    risk_fraction: float = 0.01  # risk 1% of equity per trade

    one_trade_per_day: bool = True


# -----------------------------
# Data structures
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


# -----------------------------
# Helpers
# -----------------------------
def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)


def _in_rth(ts_utc: pd.Timestamp, cfg: BearTrapConfig) -> bool:
    ts_local = ts_utc.tz_convert(cfg.session_tz)
    h, m = ts_local.hour, ts_local.minute
    sh, sm = _parse_hhmm(cfg.rth_start)
    eh, em = _parse_hhmm(cfg.rth_end)
    start_ok = (h > sh) or (h == sh and m >= sm)
    end_ok = (h < eh) or (h == eh and m < em)
    return start_ok and end_ok


def _apply_slippage(price: float, bps: float, side: str) -> float:
    if bps <= 0:
        return price
    factor = 1.0 + (bps / 10_000.0) if side == "buy" else 1.0 - (bps / 10_000.0)
    return price * factor


def compute_intraday_vwap(exec_df: pd.DataFrame, cfg: BearTrapConfig) -> pd.Series:
    """
    VWAP per session using execution bars.
    vwap_t = sum(typical_price * volume) / sum(volume)
    """
    typical = (exec_df["High"] + exec_df["Low"] + exec_df["Close"]) / 3.0
    pv = typical * exec_df["Volume"].fillna(0.0)

    local_dt = exec_df.index.tz_convert(cfg.session_tz)
    session_date = local_dt.date

    pv_cum = pv.groupby(session_date).cumsum()
    vol_cum = (
        exec_df["Volume"].fillna(0.0).groupby(session_date).cumsum().replace(0, np.nan)
    )
    return pv_cum / vol_cum


def resample_to_structure(exec_day: pd.DataFrame, cfg: BearTrapConfig) -> pd.DataFrame:
    """
    Resample a single session's 5m bars into 15m structure bars.
    We resample in NY-local time for clean session alignment, then convert back to UTC.
    """
    day_local = exec_day.copy()
    day_local.index = day_local.index.tz_convert(cfg.session_tz)

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    # label='right' gives bar timestamp at the bar end
    struct = (
        day_local.resample(cfg.structure_tf, label="right", closed="right")
        .agg(agg)
        .dropna(subset=["Open", "High", "Low", "Close"])
    )

    struct.index = struct.index.tz_convert("UTC")
    return struct


# -----------------------------
# I/O
# -----------------------------
def load_exec_data(cfg: BearTrapConfig) -> pd.DataFrame:
    df = yf.download(
        cfg.ticker,
        period=cfg.period,
        interval=cfg.interval_exec,
        auto_adjust=False,
        prepost=False,
        progress=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        raise RuntimeError(
            "No data returned from yfinance. Try a different ticker/period/interval."
        )

    needed = {"Open", "High", "Low", "Close", "Volume"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns from yfinance data: {missing}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Filter to RTH
    df = df[df.index.map(lambda ts: _in_rth(ts, cfg))].copy()
    if df.empty:
        raise RuntimeError(
            "After filtering to RTH, no rows remain. Check timezone/session settings."
        )

    # Add session_date
    local_dt = df.index.tz_convert(cfg.session_tz)
    df["session_date"] = local_dt.date
    return df


# -----------------------------
# Backtest
# -----------------------------
def backtest_bear_trap_15m_structure_5m_exec(
    exec_df: pd.DataFrame, cfg: BearTrapConfig
) -> Tuple[pd.DataFrame, List[Trade]]:
    df = exec_df.copy()
    df["vwap"] = compute_intraday_vwap(df, cfg)

    equity = cfg.initial_cash
    trades: List[Trade] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []

    for session_date, day in df.groupby("session_date"):
        day = day.drop(columns=["session_date"]).sort_index()

        if len(day) < 10:  # guard
            equity_curve.append((day.index[-1], equity))
            continue

        # Build 15m structure bars from 5m
        struct = resample_to_structure(day, cfg)
        if len(struct) < cfg.opening_range_bars_15m + 1:
            equity_curve.append((day.index[-1], equity))
            continue

        opening_struct = struct.iloc[: cfg.opening_range_bars_15m]
        support = float(opening_struct["Low"].min())

        breakdown_level = support * (1.0 - cfg.break_pct)
        below = day["Low"] <= breakdown_level
        if not below.any():
            equity_curve.append((day.index[-1], equity))
            continue

        first_break_idx = int(np.argmax(below.values))
        reclaim_window = day.iloc[
            first_break_idx : first_break_idx + cfg.max_reclaim_bars_5m + 1
        ]
        reclaim = reclaim_window["Close"] > support
        if not reclaim.any():
            equity_curve.append((day.index[-1], equity))
            continue

        # Find reclaim with "reclaim + hold" confirmation:
        # - reclaim bar: Close > support
        # - confirm bar: Low > support (never loses the level)
        # Enter on confirm bar close.
        reclaim_idx = None
        for i in range(len(reclaim_window) - 1):
            reclaim_bar = reclaim_window.iloc[i]
            confirm_bar = reclaim_window.iloc[i + 1]

            if (reclaim_bar["Close"] > support) and (confirm_bar["Low"] > support):
                reclaim_idx = i + 1  # enter on confirm bar
                break

        if reclaim_idx is None:
            continue

        entry_ts = reclaim_window.index[reclaim_idx]
        # Optional: avoid late-day traps that don't have time to reach targets
        entry_local = entry_ts.tz_convert(cfg.session_tz)
        entry_cutoff_hh, entry_cutoff_mm = _parse_hhmm(cfg.entry_cutoff_hhmm)
        if (entry_local.hour > entry_cutoff_hh) or (
            entry_local.hour == entry_cutoff_hh and entry_local.minute > entry_cutoff_mm
        ):
            continue
        entry_price_raw = float(reclaim_window.iloc[reclaim_idx]["Close"])
        entry_price = _apply_slippage(entry_price_raw, cfg.slippage_bps, side="buy")

        # Stop: min low from first breakdown up to entry, minus buffer
        breakdown_slice = day.iloc[first_break_idx : day.index.get_loc(entry_ts) + 1]
        breakdown_low = float(breakdown_slice["Low"].min())
        stop_price = breakdown_low * (1.0 - cfg.stop_buffer_pct)

        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            equity_curve.append((day.index[-1], equity))
            continue

        risk_budget = equity * cfg.risk_fraction
        shares = int(math.floor(risk_budget / risk_per_share))
        if shares <= 0:
            equity_curve.append((day.index[-1], equity))
            continue

        # Targets
        if cfg.target_mode == "vwap":
            vwap_price = float(day.loc[entry_ts, "vwap"])
            if not np.isfinite(vwap_price):
                vwap_price = entry_price + cfg.target_r * risk_per_share

            min_target = entry_price * (1.0 + cfg.min_profit_bps / 10_000.0)
            if vwap_price <= min_target:
                equity_curve.append((day.index[-1], equity))
                continue

            target_vwap = vwap_price
            runner_target = max(
                target_vwap,
                entry_price + cfg.runner_r * risk_per_share,
            )
        elif cfg.target_mode == "r_multiple":
            target_vwap = entry_price + cfg.target_r * risk_per_share
            runner_target = target_vwap
        else:
            raise ValueError("target_mode must be 'vwap' or 'r_multiple'")

        runner_shares = int(math.floor(shares * cfg.runner_frac))
        runner_shares = max(0, min(shares, runner_shares))
        vwap_shares = shares - runner_shares

        # Exit simulation (5m bars), starting next bar
        forward = day.loc[entry_ts:].iloc[1:]
        exit_ts = day.index[-1]
        exit_price = _apply_slippage(
            float(day.iloc[-1]["Close"]), cfg.slippage_bps, side="sell"
        )
        reason = "eod"

        vwap_filled = vwap_shares == 0
        runner_filled = runner_shares == 0

        vwap_exit_ts: Optional[pd.Timestamp] = None
        vwap_exit_px: Optional[float] = None

        runner_exit_ts: Optional[pd.Timestamp] = None
        runner_exit_px: Optional[float] = None

        for ts, row in forward.iterrows():
            # Stop priority (fixed stop for both tranches)
            if row["Low"] <= stop_price:
                stop_px = _apply_slippage(stop_price, cfg.slippage_bps, side="sell")
                reason = "stop"
                exit_ts = ts
                exit_price = stop_px

                if not vwap_filled:
                    vwap_exit_ts, vwap_exit_px = ts, stop_px
                    vwap_filled = True
                if not runner_filled:
                    runner_exit_ts, runner_exit_px = ts, stop_px
                    runner_filled = True
                break

            # VWAP tranche
            if (not vwap_filled) and (row["High"] >= target_vwap):
                px = _apply_slippage(target_vwap, cfg.slippage_bps, side="sell")
                vwap_exit_ts, vwap_exit_px = ts, px
                vwap_filled = True

            # Runner tranche
            if (not runner_filled) and (row["High"] >= runner_target):
                px = _apply_slippage(runner_target, cfg.slippage_bps, side="sell")
                runner_exit_ts, runner_exit_px = ts, px
                runner_filled = True

            if vwap_filled and runner_filled:
                reason = "vwap+runner"
                exit_ts = max(vwap_exit_ts, runner_exit_ts)  # type: ignore[arg-type]
                # pick the exit price of whichever tranche exited last (for display only)
                if runner_exit_ts >= vwap_exit_ts:  # type: ignore[operator]
                    exit_price = float(runner_exit_px)
                else:
                    exit_price = float(vwap_exit_px)
                break

        # EOD close for any remaining tranche
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

        # PnL (tranche-based)
        gross_pnl = 0.0
        if vwap_shares > 0:
            gross_pnl += (float(vwap_exit_px) - entry_price) * vwap_shares
        if runner_shares > 0:
            gross_pnl += (float(runner_exit_px) - entry_price) * runner_shares

        # Fee legs: entry once, exits = 1 or 2 depending on whether tranches exit at different times
        exit_legs = 0
        if vwap_shares > 0:
            exit_legs += 1
        if runner_shares > 0 and (runner_exit_ts != vwap_exit_ts):
            exit_legs += 1
        net_pnl = gross_pnl - cfg.fee_per_trade * (1 + exit_legs)
        equity += net_pnl

        denom = risk_per_share * shares
        r_mult = net_pnl / denom if denom > 0 else np.nan

        trades.append(
            Trade(
                date=str(session_date),
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                entry=entry_price,
                exit=float(exit_price),
                shares=shares,
                pnl=float(net_pnl),
                r_multiple=float(r_mult),
                reason=reason,
            )
        )

        equity_curve.append((day.index[-1], equity))

        if cfg.one_trade_per_day:
            # by construction we only take one, but keep the intent explicit
            pass

    eq = (
        pd.DataFrame(equity_curve, columns=["ts", "equity"])
        .set_index("ts")
        .sort_index()
    )
    return eq, trades


# -----------------------------
# Reporting
# -----------------------------
def summarize(trades: List[Trade], eq: pd.DataFrame, cfg: BearTrapConfig) -> None:
    print(
        f"Ticker: {cfg.ticker} | Period: {cfg.period} | "
        f"Structure: 15m | Exec: {cfg.interval_exec}"
    )
    print(f"Trades: {len(trades)}")
    if not trades:
        print(
            "No trades generated. Try relaxing break_pct/max_reclaim_bars_5m or changing ticker."
        )
        return

    tdf = pd.DataFrame([t.__dict__ for t in trades])
    win = (tdf["pnl"] > 0).mean()
    avg_r = tdf["r_multiple"].mean()
    med_r = tdf["r_multiple"].median()
    total = tdf["pnl"].sum()

    peak = eq["equity"].cummax()
    dd = eq["equity"] / peak - 1.0
    mdd = dd.min()

    print(f"Win rate: {win:.2%}")
    print(f"Avg R: {avg_r:.2f} | Median R: {med_r:.2f}")
    print(f"Total PnL: ${total:,.2f}")
    print(f"Final equity: ${eq['equity'].iloc[-1]:,.2f}")
    print(f"Max drawdown (equity curve): {mdd:.2%}")
    print("\nRecent trades:")
    cols = [
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
    print(tdf.tail(10)[cols].to_string(index=False))


def main():
    cfg = BearTrapConfig(
        ticker="QQQ",
        period="60d",
        interval_exec="5m",
        opening_range_bars_15m=2,  # first 30 minutes (15m bars)
        break_pct=0.001,
        max_reclaim_bars_5m=18,  # 90 minutes
        stop_buffer_pct=0.0005,
        target_mode="vwap",
        runner_frac=0.5,
        runner_r=2.0,
        min_profit_bps=2.0,
        slippage_bps=2.0,
        fee_per_trade=0.0,
        risk_fraction=0.01,
        entry_cutoff_hhmm="12:30",
    )

    exec_df = load_exec_data(cfg)
    eq, trades = backtest_bear_trap_15m_structure_5m_exec(exec_df, cfg)
    summarize(trades, eq, cfg)


if __name__ == "__main__":
    main()
