#!/usr/bin/env python3
"""
Standalone playground backtest: Long-only RSI dip strategy on ETFs (daily bars, yfinance)

Implements changes A–D:

A) Baseline + exposure stats
   - Computes Buy & Hold (per ticker) equity curve (using Adj Close).
   - Prints: % time in market, average exposure, median exposure, # entry signals, etc.

B) Fix reporting: record SIGNAL RSI correctly
   - Stores signal_date + signal_rsi (the day RSI dipped below threshold).
   - Entry occurs next day OPEN; trade log includes both signal + entry info.

C) Improve exits (beyond blunt TP/SL/max-hold)
   - Priority order per day:
       1) Intraday SL/TP using daily Low/High (with ambiguity handling)
       2) RSI mean-reversion exit at CLOSE if RSI >= exit_rsi (optional)
       3) Max-hold exit at CLOSE (optional)
   - TP/SL are still configurable; RSI exit can be enabled/disabled.

D) Entry threshold + optional scaling-in
   - entry_rsi threshold configurable (default 25).
   - Optional tiered sizing by "how oversold" the signal RSI is:
       - Example (default):
           RSI < 20 -> 100% notional
           RSI < 25 -> 60% notional
           RSI < 30 -> 30% notional
     (Only one tier applies; deepest tier wins.)
   - If tiers are disabled, uses a single fixed position fraction.

Notes / limitations:
- Daily OHLC only. TP/SL uses High/Low; when both hit same day, we assume
  stop-first by default (conservative). Fills at exact levels (still optimistic vs gaps).
- One position per ticker at a time (no pyramiding).
- Cash earns 0%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit(
        "Missing dependency: yfinance. Install with: pip install yfinance"
    ) from e


# -----------------------------
# Config
# -----------------------------


@dataclass
class Config:
    tickers: List[str] = None
    start: str = "2005-01-01"
    end: Optional[str] = None  # None -> today

    # RSI
    rsi_period: int = 14
    entry_rsi: float = 25.0

    # D) Position sizing
    # If use_size_tiers=True, size is determined by the first tier that matches (deepest-first).
    # Each tier is (rsi_threshold, position_fraction_of_equity).
    use_size_tiers: bool = True
    size_tiers: List[Tuple[float, float]] = None  # e.g. [(20,1.0),(25,0.6),(30,0.3)]
    default_position_fraction: float = 1.0  # used if tiers disabled or no tier matches

    # Risk controls (TP/SL intraday on High/Low)
    take_profit: Optional[float] = 0.10  # +10%; None disables
    stop_loss: Optional[float] = 0.05  # -5%;  None disables

    # C) Exit improvements
    use_rsi_exit: bool = True
    exit_rsi: float = 50.0  # exit at close when RSI >= exit_rsi
    max_hold_days: Optional[int] = 10  # exit at close when reached; None disables

    # Intraday TP/SL ambiguity handling when both hit same day
    stop_first_if_both_hit: bool = True  # conservative default

    # Portfolio / execution
    initial_equity: float = 100_000.0
    commission_per_trade: float = 0.0  # flat $ per fill
    slippage_bps: float = 0.0  # bps applied to fill price (e.g. 2 = 0.02%)

    # Output / debug
    verbose: bool = True


def _default_config() -> Config:
    cfg = Config(tickers=["QQQ"])
    cfg.size_tiers = [(20.0, 1.00), (25.0, 0.60), (30.0, 0.30)]
    return cfg


# -----------------------------
# Indicators
# -----------------------------


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, returns RSI in [0, 100]."""
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# -----------------------------
# Data
# -----------------------------


def fetch_daily_ohlc(
    tickers: List[str], start: str, end: Optional[str]
) -> Dict[str, pd.DataFrame]:
    """Fetch daily OHLCV from yfinance. Returns dict[ticker] -> DataFrame."""
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        multi_level_index=True,
    )

    out: Dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
        df = data[tickers[0]].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        out[tickers[0]] = df
        return out

    for t in tickers:
        df = data[t].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        out[t] = df
    return out


# -----------------------------
# Backtest engine
# -----------------------------


@dataclass
class Position:
    ticker: str
    signal_date: pd.Timestamp
    signal_rsi: float
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    intended_fraction: float
    hold_days: int = 0


@dataclass
class Trade:
    ticker: str
    signal_date: pd.Timestamp
    signal_rsi: float
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: float
    intended_fraction: float
    reason: str
    pnl: float
    ret: float
    hold_days: int


def apply_slippage(price: float, bps: float, is_buy: bool) -> float:
    if bps <= 0:
        return price
    mult = 1.0 + (bps / 10_000.0) if is_buy else 1.0 - (bps / 10_000.0)
    return price * mult


def choose_position_fraction(cfg: Config, signal_rsi: float) -> float:
    """D) Determine position fraction based on tiers or default."""
    if not math.isfinite(signal_rsi):
        return cfg.default_position_fraction

    if cfg.use_size_tiers and cfg.size_tiers:
        # Expect tiers like [(20,1.0),(25,0.6),(30,0.3)] with increasing thresholds
        # We want deepest-first: sort ascending threshold and check in that order.
        tiers = sorted(cfg.size_tiers, key=lambda x: x[0])
        for thr, frac in tiers:
            if signal_rsi < thr:
                return float(frac)

        # If no tier matches, size = 0 (or default). We'll use default_position_fraction.
        return cfg.default_position_fraction

    return cfg.default_position_fraction


def compute_metrics_from_equity(equity: pd.Series) -> Dict[str, float]:
    rets = equity.pct_change().fillna(0.0)
    ann = 252.0
    n = len(equity)
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (ann / max(n, 1)) - 1.0
    vol = rets.std(ddof=0) * math.sqrt(ann)
    sharpe = (rets.mean() * ann) / (rets.std(ddof=0) * math.sqrt(ann) + 1e-12)
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = dd.min()
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }


def backtest(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Trade], pd.DataFrame]:
    """
    Returns:
      strategy_df: index date, columns [equity, returns, exposure]
      buyhold_df : index date, columns like [BH_QQQ, ...] (Adj Close based)
      trades     : list of Trade
      pos_value_df: index date, columns per ticker position value
    """
    assert cfg.tickers and len(cfg.tickers) > 0

    raw = fetch_daily_ohlc(cfg.tickers, cfg.start, cfg.end)

    # Unified calendar: intersection to keep things aligned cleanly.
    calendars = [df.index for df in raw.values()]
    common_index = calendars[0]
    for idx in calendars[1:]:
        common_index = common_index.intersection(idx)
    common_index = common_index.sort_values()

    frames: Dict[str, pd.DataFrame] = {}
    for t, df in raw.items():
        d = df.reindex(common_index).copy()
        d["RSI"] = rsi_wilder(d["Close"], cfg.rsi_period)
        frames[t] = d

    # A) Buy & Hold baseline (Adj Close based)
    buyhold_cols = {}
    for t in cfg.tickers:
        adj = (
            frames[t]["Adj Close"]
            if "Adj Close" in frames[t].columns
            else frames[t]["Close"]
        )
        adj = adj.astype(float)
        adj0 = float(adj.iloc[0])
        buyhold_cols[f"BH_{t}"] = cfg.initial_equity * (adj / adj0)
    buyhold_df = pd.DataFrame(buyhold_cols, index=common_index)

    cash = cfg.initial_equity
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []

    # pending_entries[ticker] = (entry_dt, signal_dt, signal_rsi, intended_fraction)
    pending_entries: Dict[str, Tuple[pd.Timestamp, pd.Timestamp, float, float]] = {}

    equity_records = []
    pos_value_records = []

    # Track exposures
    last_equity_close = cfg.initial_equity

    for i, dt in enumerate(common_index):
        # 1) Execute pending entries at today's OPEN
        entering = []
        for t, (entry_dt, signal_dt, signal_rsi, frac) in list(pending_entries.items()):
            if entry_dt == dt and t not in positions:
                entering.append((t, signal_dt, signal_rsi, frac))
                del pending_entries[t]

        if entering:
            # Allocate notional based on last close equity and intended fractions.
            total_desired = sum(
                last_equity_close * frac for (_, _, _, frac) in entering
            )
            if total_desired > 0:
                scale = min(1.0, cash / total_desired) if total_desired > 1e-12 else 0.0
            else:
                scale = 0.0

            for t, signal_dt, signal_rsi, frac in entering:
                o = float(frames[t].loc[dt, "Open"])
                if not math.isfinite(o) or o <= 0:
                    continue

                desired_notional = last_equity_close * frac * scale
                if desired_notional <= 0:
                    continue

                fill = apply_slippage(o, cfg.slippage_bps, is_buy=True)
                shares = desired_notional / fill
                cost = shares * fill + cfg.commission_per_trade

                if cost > cash:
                    shares = max((cash - cfg.commission_per_trade) / fill, 0.0)
                    cost = shares * fill + cfg.commission_per_trade

                if shares <= 0:
                    continue

                cash -= cost
                positions[t] = Position(
                    ticker=t,
                    signal_date=signal_dt,
                    signal_rsi=float(signal_rsi),
                    entry_date=dt,
                    entry_price=float(fill),
                    shares=float(shares),
                    intended_fraction=float(frac),
                    hold_days=0,
                )

        # 2) Evaluate exits for open positions
        exit_tickers = []
        for t, pos in positions.items():
            row = frames[t].loc[dt]
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            rsi_val = row["RSI"]

            pos.hold_days += 1

            tp_hit = False
            sl_hit = False
            tp_price = None
            sl_price = None

            if cfg.take_profit is not None:
                tp_price = pos.entry_price * (1.0 + cfg.take_profit)
                tp_hit = math.isfinite(high) and high >= tp_price

            if cfg.stop_loss is not None:
                sl_price = pos.entry_price * (1.0 - cfg.stop_loss)
                sl_hit = math.isfinite(low) and low <= sl_price

            reason = None
            exit_px = None
            exit_at_close = False

            # 2.1 Intraday SL/TP
            if tp_hit and sl_hit:
                if cfg.stop_first_if_both_hit:
                    reason = "SL (both hit)"
                    exit_px = sl_price
                else:
                    reason = "TP (both hit)"
                    exit_px = tp_price
            elif sl_hit:
                reason = "SL"
                exit_px = sl_price
            elif tp_hit:
                reason = "TP"
                exit_px = tp_price
            else:
                # 2.2 RSI exit at close
                if (
                    cfg.use_rsi_exit
                    and pd.notna(rsi_val)
                    and float(rsi_val) >= cfg.exit_rsi
                ):
                    reason = "RSIExit"
                    exit_px = close
                    exit_at_close = True
                # 2.3 Max hold at close
                elif (
                    cfg.max_hold_days is not None and pos.hold_days >= cfg.max_hold_days
                ):
                    reason = "MaxHold"
                    exit_px = close
                    exit_at_close = True

            if (
                reason is not None
                and exit_px is not None
                and math.isfinite(exit_px)
                and exit_px > 0
            ):
                fill = apply_slippage(float(exit_px), cfg.slippage_bps, is_buy=False)
                proceeds = pos.shares * fill - cfg.commission_per_trade
                cash += proceeds

                # pnl includes both commissions (entry + exit) roughly
                pnl = (
                    fill - pos.entry_price
                ) * pos.shares - 2 * cfg.commission_per_trade
                ret = (fill / pos.entry_price) - 1.0

                trades.append(
                    Trade(
                        ticker=t,
                        signal_date=pos.signal_date,
                        signal_rsi=pos.signal_rsi,
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        exit_date=dt,
                        exit_price=fill,
                        shares=pos.shares,
                        intended_fraction=pos.intended_fraction,
                        reason=reason,
                        pnl=float(pnl),
                        ret=float(ret),
                        hold_days=int(pos.hold_days),
                    )
                )
                exit_tickers.append(t)

        for t in exit_tickers:
            del positions[t]

        # 3) Generate new signals at today's CLOSE -> schedule next day OPEN
        for t in cfg.tickers:
            if t in positions:
                continue
            rsi_val = frames[t].loc[dt, "RSI"]
            if pd.notna(rsi_val) and float(rsi_val) < cfg.entry_rsi:
                if i + 1 < len(common_index):
                    signal_rsi = float(rsi_val)
                    frac = choose_position_fraction(cfg, signal_rsi)
                    pending_entries[t] = (common_index[i + 1], dt, signal_rsi, frac)

        # 4) Mark-to-market at CLOSE
        pos_values = {}
        equity = cash
        invested = 0.0
        for t, pos in positions.items():
            c = float(frames[t].loc[dt, "Close"])
            v = pos.shares * c
            pos_values[t] = v
            invested += v
            equity += v

        exposure = 0.0 if equity <= 0 else invested / equity
        equity_records.append((dt, equity, exposure))
        pos_value_records.append((dt, pos_values))

        last_equity_close = equity

    strategy_df = pd.DataFrame(
        equity_records, columns=["date", "equity", "exposure"]
    ).set_index("date")
    strategy_df["returns"] = strategy_df["equity"].pct_change().fillna(0.0)

    pos_value_df = pd.DataFrame(
        {dt: vals for dt, vals in pos_value_records}
    ).T.sort_index()
    for t in cfg.tickers:
        if t not in pos_value_df.columns:
            pos_value_df[t] = 0.0
    pos_value_df = pos_value_df.fillna(0.0)

    return strategy_df, buyhold_df, trades, pos_value_df


# -----------------------------
# Reporting
# -----------------------------


def print_summary(
    cfg: Config,
    strategy_df: pd.DataFrame,
    buyhold_df: pd.DataFrame,
    trades: List[Trade],
) -> None:
    sm = compute_metrics_from_equity(strategy_df["equity"])

    # A) Exposure stats
    exposure = strategy_df["exposure"]
    time_in_mkt = float((exposure > 1e-9).mean())
    avg_exposure = float(exposure.mean())
    med_exposure = float(exposure.median())

    # Buy & Hold metrics (for each ticker)
    bh_metrics = {}
    for col in buyhold_df.columns:
        bh_metrics[col] = compute_metrics_from_equity(buyhold_df[col])

    # Trades DF
    tdf = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()

    # Signal count
    n_trades = len(trades)

    print("\n================ RSI Dip Strategy Backtest ================")
    print(f"Tickers                 : {cfg.tickers}")
    print(
        f"Date range              : {strategy_df.index[0].date()} -> {strategy_df.index[-1].date()}"
    )
    print(f"RSI period / entry      : {cfg.rsi_period} / < {cfg.entry_rsi}")
    print(f"Exit: TP / SL           : {cfg.take_profit} / {cfg.stop_loss}")
    print(
        f"Exit: RSI at close      : {'ON' if cfg.use_rsi_exit else 'OFF'} (>= {cfg.exit_rsi})"
    )
    print(f"Exit: Max hold days     : {cfg.max_hold_days}")
    print(f"Sizing tiers            : {'ON' if cfg.use_size_tiers else 'OFF'}")
    if cfg.use_size_tiers:
        print(f"  tiers (rsi<thr -> frac): {cfg.size_tiers}")
    print(f"Slippage (bps)          : {cfg.slippage_bps}")
    print(f"Commission ($/fill)     : {cfg.commission_per_trade}")
    print("-----------------------------------------------------------")
    print(f"Strategy final equity   : ${sm['final_equity']:,.2f}")
    print(f"Strategy total return   : {sm['total_return']*100:,.2f}%")
    print(f"Strategy CAGR (approx)  : {sm['cagr']*100:,.2f}%")
    print(f"Strategy ann. vol       : {sm['ann_vol']*100:,.2f}%")
    print(f"Strategy Sharpe (rf=0)  : {sm['sharpe']:,.2f}")
    print(f"Strategy max drawdown   : {sm['max_dd']*100:,.2f}%")
    print("-----------------------------------------------------------")
    print(f"Time in market          : {time_in_mkt*100:,.2f}%")
    print(f"Avg exposure            : {avg_exposure*100:,.2f}%")
    print(f"Median exposure         : {med_exposure*100:,.2f}%")
    print(f"Trades                  : {n_trades}")

    # Buy & hold summary
    print("\n--- Buy & Hold baseline (Adj Close) ---")
    for col, m in bh_metrics.items():
        print(
            f"{col}: Final=${m['final_equity']:,.2f}  "
            f"TR={m['total_return']*100:,.2f}%  CAGR={m['cagr']*100:,.2f}%  "
            f"Vol={m['ann_vol']*100:,.2f}%  Sharpe={m['sharpe']:,.2f}  "
            f"MaxDD={m['max_dd']*100:,.2f}%"
        )

    if trades:
        win_rate = float((tdf["pnl"] > 0).mean())
        avg_ret = float(tdf["ret"].mean())
        med_ret = float(tdf["ret"].median())
        avg_hold = float(tdf["hold_days"].mean())

        print("\n--- Trade stats ---")
        print(f"Win rate                : {win_rate*100:,.1f}%")
        print(f"Avg trade return        : {avg_ret*100:,.2f}%")
        print(f"Median trade return     : {med_ret*100:,.2f}%")
        print(f"Avg hold (days)         : {avg_hold:,.2f}")

        by_reason = (
            tdf.groupby("reason")["pnl"]
            .agg(["count", "mean", "sum"])
            .sort_values("count", ascending=False)
        )
        print("\nExit reason breakdown (count/mean/sum pnl):")
        print(by_reason.to_string(float_format=lambda x: f"{x:,.2f}"))

        # last 10 trades, with signal fields
        out = tdf.copy()
        out["signal_date"] = pd.to_datetime(out["signal_date"]).dt.date
        out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.date
        out["exit_date"] = pd.to_datetime(out["exit_date"]).dt.date
        cols = [
            "ticker",
            "signal_date",
            "signal_rsi",
            "entry_date",
            "exit_date",
            "intended_fraction",
            "entry_price",
            "exit_price",
            "hold_days",
            "reason",
            "ret",
            "pnl",
        ]
        print("\nLast 10 trades:")
        print(
            out[cols]
            .tail(10)
            .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
        )

    print("===========================================================\n")


# -----------------------------
# Main
# -----------------------------


def main():
    cfg = _default_config()

    cfg.tickers = ["QQQ"]

    # Edit knobs freely:
    cfg.start = "2005-01-01"
    cfg.end = None

    # Entry
    cfg.entry_rsi = 30.0
    cfg.rsi_period = 14

    # D) sizing
    cfg.use_size_tiers = True
    cfg.size_tiers = [(20.0, 1.00), (25.0, 0.70), (30.0, 0.40), (35.0, 0.20)]
    cfg.default_position_fraction = (
        0.0  # if no tier matches, stay out (useful if entry_rsi < last tier)
    )

    # Risk controls
    cfg.take_profit = 0.10
    cfg.stop_loss = 0.15

    # C) exit upgrades
    cfg.use_rsi_exit = True
    cfg.exit_rsi = 50.0
    cfg.max_hold_days = 10

    # Execution frictions
    cfg.slippage_bps = 0.0
    cfg.commission_per_trade = 0.0

    strategy_df, buyhold_df, trades, pos_value_df = backtest(cfg)
    print_summary(cfg, strategy_df, buyhold_df, trades)

    # Save artifacts
    # strategy_df.to_csv("strategy_equity.csv")
    # buyhold_df.to_csv("buyhold_equity.csv")
    # pos_value_df.to_csv("daily_position_values.csv")
    # if trades:
    #     pd.DataFrame([t.__dict__ for t in trades]).to_csv("trades.csv", index=False)
    # print("Wrote:")
    # print("  strategy_equity.csv (equity, returns, exposure)")
    # print("  buyhold_equity.csv  (Adj Close buy&hold baselines)")
    # print("  daily_position_values.csv")
    # print("  trades.csv (if any)")


if __name__ == "__main__":
    main()
