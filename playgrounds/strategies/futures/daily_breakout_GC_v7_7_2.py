"""
Daily Breakout Futures Playground (v7.7.2) — GC=F via yfinance (daily bars)

v7.7.2 = v7.7.1 + trend_mode knob (MINIMAL CHANGES ONLY)

Trend modes:
- "off"  : ignore trend filter completely (equivalent to trend_filter_enabled=False)
- "gate" : v7.7.1 behavior (long only if C1>SMA1, short only if C1<SMA1)
- "bias" : do NOT gate entries; instead scale sizing risk when trading against trend:
           - long while C1<=SMA1 => risk_fraction *= trend_against_long_risk_mult
           - short while C1>=SMA1 => risk_fraction *= trend_against_short_risk_mult

Everything else is preserved EXACTLY from v7.7.1:
- MTM equity (Cash + Unrealized)
- Gap-aware stop-entry fills
- No-lookahead trailing stop using prior-day HH1/LL1/ATR1, clamped to prior close
- Same-day enter-and-stop handling
- Margin + leverage constraints
- Slippage model + stress multiplier (TR1/ATR1)
- Commission per side (GC vs MGC)
- Roll-aware "avoid entries" window (approx roll proxy)
- RAW + ELIGIBLE trigger diagnostics (eligible includes trend filter + roll-block + flat + ready)
- Full stats printout + last days/trades + equity plot

Default params remain aligned with v7.7.1 mainline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Set
import math

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    symbols: List[str] = None
    start: str = "2006-01-01"
    end: Optional[str] = None

    enable_long: bool = True
    enable_short: bool = True

    atr_n: int = 14
    stop_atr_mult: float = 2.0

    trail_atr_mult: float = 3.0
    trail_lookback: int = 22

    # Risk & sizing
    risk_fraction_per_trade: float = 0.03
    initial_equity: float = 100_000.0

    # Debug sizing knobs
    fixed_contracts: Optional[int] = None
    enforce_constraints_on_fixed: bool = True

    # DEBUG-ONLY floor (OFF by default)
    min_contracts_on_signal: int = 0
    enforce_constraints_on_min: bool = True

    # Contract specs (GC vs MGC)
    gc_multiplier: float = 100.0
    mgc_multiplier: float = 10.0
    gc_commission_per_side: float = 2.5
    mgc_commission_per_side: float = 1.25

    # Costs (points)
    base_slippage_points: float = 0.2
    slip_k_sqrt: float = 0.002

    # Robustness switches
    entry_slip_multiplier: float = 1.0
    entry_slip_multiplier_on_breakout_day: float = 2.0
    exit_slip_multiplier: float = 1.0

    # Stress slippage (no-lookahead: uses TR1/ATR1)
    stress_mode: bool = True
    stress_tr_atr_threshold: float = 1.8
    stress_slip_multiplier: float = 1.5

    # Margin / leverage constraints
    max_leverage: float = 2.0
    margin_rate: float = 0.07
    margin_buffer: float = 0.80

    # Approx roll-aware trading
    roll_active_months: Tuple[int, ...] = (2, 4, 6, 8, 10, 12)
    roll_bdays_before: int = 5
    roll_window_bdays: int = 2
    roll_force_flat: bool = False
    roll_avoid_entries: bool = True
    roll_avoid_window_only: bool = True

    # Trend filter
    trend_filter_enabled: bool = True
    trend_sma_window: int = 200

    # NEW (v7.7.2): trend_mode knob + bias multipliers
    # "off" | "gate" | "bias"
    trend_mode: str = "gate"
    trend_against_long_risk_mult: float = 0.25
    trend_against_short_risk_mult: float = 0.25


# ----------------------------
# Diagnostics
# ----------------------------
@dataclass
class Diagnostics:
    # Raw triggers (may include days while in position)
    trigger_any_raw: int = 0
    trigger_long_raw: int = 0
    trigger_short_raw: int = 0
    trigger_both_raw: int = 0
    both_triggers_skip_raw: int = 0

    # Eligible triggers (ready + flat + not roll-blocked + passes trend filter)
    trigger_any_eligible: int = 0
    trigger_long_eligible: int = 0
    trigger_short_eligible: int = 0
    trigger_both_eligible: int = 0
    both_triggers_skip_eligible: int = 0

    # Roll stats
    roll_force_flat_exits: int = 0
    roll_entry_blocks: int = 0

    # Sizing stats
    entries_taken: int = 0
    size0_by_risk: int = 0
    size0_by_constraints: int = 0
    forced_min_used: int = 0

    # Contract selection stats
    used_gc_entries: int = 0
    used_mgc_entries: int = 0
    no_trade_size0_after_mgc: int = 0

    # Distributions
    desired_risk_values: List[int] = field(default_factory=list)  # chosen instrument
    capped_values: List[int] = field(default_factory=list)  # chosen instrument
    entry_contracts: List[int] = field(default_factory=list)
    entry_kind: List[str] = field(default_factory=list)  # "GC" or "MGC"

    # Stress stats
    stressed_days: int = 0
    stressed_execs: int = 0


# ----------------------------
# Helpers
# ----------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    tr = true_range(df["High"], df["Low"], df["Close"])
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else np.nan


def points_to_dollars(points: float, contracts: int, multiplier: float) -> float:
    return points * multiplier * contracts


def load_ohlc_from_yfinance(
    symbol: str, start: str, end: Optional[str]
) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if "Open" in set(lvl0) and symbol in set(lvl1):
            df = df.xs(symbol, axis=1, level=1)
        elif "Open" in set(lvl1) and symbol in set(lvl0):
            df = df.xs(symbol, axis=1, level=0)
        else:
            df.columns = [c[0] for c in df.columns]

    need = ["Open", "High", "Low", "Close"]
    for col in need:
        if col not in df.columns:
            raise RuntimeError(f"{symbol}: missing {col}. Columns={list(df.columns)}")

    df = df[["Open", "High", "Low", "Close"]].copy()
    for col in need:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=need)
    return df


def cap_contracts_by_constraints(
    desired: int, equity: float, price: float, multiplier: float, cfg: Config
) -> int:
    if desired <= 0:
        return 0
    notional_per_contract = price * multiplier
    if notional_per_contract <= 0:
        return 0

    max_by_leverage = int(
        math.floor((equity * cfg.max_leverage) / notional_per_contract)
    )

    margin_per_contract = notional_per_contract * cfg.margin_rate
    max_margin_dollars = equity * cfg.margin_buffer
    max_by_margin = int(
        math.floor(max_margin_dollars / max(margin_per_contract, 1e-12))
    )

    return max(0, min(desired, max_by_leverage, max_by_margin))


def _first_business_day(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def build_roll_dates(
    index: pd.DatetimeIndex, cfg: Config
) -> Tuple[Set[pd.Timestamp], Set[pd.Timestamp]]:
    if index.empty:
        return set(), set()

    start_year = index.min().year
    end_year = index.max().year

    roll_dates = []
    for y in range(start_year, end_year + 1):
        for m in cfg.roll_active_months:
            fbd = _first_business_day(y, m)
            roll_date = fbd - pd.tseries.offsets.BDay(cfg.roll_bdays_before)
            roll_dates.append(pd.Timestamp(roll_date.date()))

    roll_dates_set = set(roll_dates)

    window_dates_set: Set[pd.Timestamp] = set()
    for rd in roll_dates_set:
        if cfg.roll_window_bdays <= 0:
            window_dates_set.add(rd)
            continue
        if cfg.roll_avoid_window_only:
            lo = rd - pd.tseries.offsets.BDay(cfg.roll_window_bdays)
            hi = rd + pd.tseries.offsets.BDay(cfg.roll_window_bdays)
            window = pd.bdate_range(lo, hi)
        else:
            hi = rd + pd.tseries.offsets.BDay(cfg.roll_window_bdays)
            window = pd.bdate_range(rd, hi)
        window_dates_set.update(pd.Timestamp(d.date()) for d in window)

    idx_dates = set(pd.Timestamp(d.date()) for d in index)
    return (roll_dates_set & idx_dates), (window_dates_set & idx_dates)


def _summ_int_series(xs: List[int]) -> Dict[str, float]:
    if not xs:
        return {"count": 0}
    arr = np.array(xs, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _entry_size_hist(entry_contracts: List[int]) -> Dict[str, float]:
    if not entry_contracts:
        return {"count": 0}
    n = len(entry_contracts)
    c1 = sum(1 for x in entry_contracts if x == 1)
    c2 = sum(1 for x in entry_contracts if x == 2)
    c3 = sum(1 for x in entry_contracts if x == 3)
    c4p = sum(1 for x in entry_contracts if x >= 4)
    return {
        "count": n,
        "pct_1": c1 / n,
        "pct_2": c2 / n,
        "pct_3": c3 / n,
        "pct_4p": c4p / n,
    }


# ----------------------------
# Position
# ----------------------------
@dataclass
class Position:
    side: int
    entry_price: float
    contracts: int
    hard_stop: float
    trail_stop: float
    entry_date: pd.Timestamp

    multiplier: float
    commission_per_side: float
    kind: str  # "GC" or "MGC"


# ----------------------------
# Stats
# ----------------------------
def compute_stats(
    res: pd.DataFrame, trades_df: pd.DataFrame, initial_equity: float
) -> Dict[str, float]:
    eq = res["Equity"].astype(float)
    ret = eq.pct_change().fillna(0.0)

    peak = eq.cummax()
    dd = (eq / peak) - 1.0

    total_return = float(eq.iloc[-1] / initial_equity - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / initial_equity) ** (1.0 / years) - 1.0

    ann_vol = float(ret.std(ddof=0) * math.sqrt(252))
    sharpe = float(safe_div(ret.mean() * 252, ret.std(ddof=0) * math.sqrt(252)))

    n_trades = int(len(trades_df))
    win_rate = float((trades_df["GrossPnL"] > 0).mean()) if n_trades > 0 else np.nan
    gross_profit = (
        float(trades_df.loc[trades_df["GrossPnL"] > 0, "GrossPnL"].sum())
        if n_trades > 0
        else 0.0
    )
    gross_loss = (
        float(-trades_df.loc[trades_df["GrossPnL"] < 0, "GrossPnL"].sum())
        if n_trades > 0
        else 0.0
    )
    profit_factor = (
        float(safe_div(gross_profit, gross_loss)) if gross_loss > 0 else np.inf
    )

    return {
        "FinalEquity": float(eq.iloc[-1]),
        "TotalReturn": total_return,
        "CAGR": float(cagr),
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": float(dd.min()),
        "Trades": n_trades,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
    }


# ----------------------------
# Backtest (v7.7.2 = v7.7.1 + trend_mode)
# ----------------------------
def backtest_one(
    df: pd.DataFrame, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Diagnostics]:
    df = df.copy()

    # Indicators
    df["TR"] = true_range(df["High"], df["Low"], df["Close"])
    df["ATR"] = atr(df, cfg.atr_n)

    # Prior-day refs (no lookahead)
    df["TR1"] = df["TR"].shift(1)
    df["ATR1"] = df["ATR"].shift(1)

    df["HH1"] = (
        df["High"]
        .rolling(cfg.trail_lookback, min_periods=cfg.trail_lookback)
        .max()
        .shift(1)
    )
    df["LL1"] = (
        df["Low"]
        .rolling(cfg.trail_lookback, min_periods=cfg.trail_lookback)
        .min()
        .shift(1)
    )

    df["YH"] = df["High"].shift(1)
    df["YL"] = df["Low"].shift(1)
    df["C1"] = df["Close"].shift(1)

    # Trend SMA (shifted)
    # NOTE: trend_mode="off" will behave like disabled (no SMA needed).
    trend_mode = (cfg.trend_mode or "gate").lower().strip()
    trend_active = cfg.trend_filter_enabled and (trend_mode != "off")

    if trend_active:
        df["SMA"] = (
            df["Close"]
            .rolling(cfg.trend_sma_window, min_periods=cfg.trend_sma_window)
            .mean()
        )
        df["SMA1"] = df["SMA"].shift(1)
    else:
        df["SMA1"] = np.nan

    roll_dates, roll_window_dates = build_roll_dates(df.index, cfg)

    cash = cfg.initial_equity
    pos: Optional[Position] = None

    records: List[Dict] = []
    trades: List[Dict] = []
    diag = Diagnostics()

    EPS = 1e-6

    def unrealized_at(close_price: float) -> float:
        if pos is None:
            return 0.0
        pnl_points = (close_price - pos.entry_price) * pos.side
        return points_to_dollars(pnl_points, pos.contracts, pos.multiplier)

    def clamp_trail_on_entry(
        side: int, entry: float, c_prev: float, trail_init: float
    ) -> float:
        if side == +1:
            return min(trail_init, entry - EPS, c_prev - EPS)
        else:
            return max(trail_init, entry + EPS, c_prev + EPS)

    def update_trail(
        side: int, c_prev: float, candidate: float, current_trail: float
    ) -> float:
        if side == +1:
            cand = min(candidate, c_prev - EPS)
            return max(current_trail, cand)
        else:
            cand = max(candidate, c_prev + EPS)
            return min(current_trail, cand)

    def is_stressed(tr1: float, atr1: float) -> bool:
        if not cfg.stress_mode:
            return False
        if atr1 <= 0 or not np.isfinite(atr1) or not np.isfinite(tr1):
            return False
        return (tr1 / atr1) >= cfg.stress_tr_atr_threshold

    def slip_points(contracts: int, stressed: bool, mult: float) -> float:
        base = cfg.base_slippage_points + (
            cfg.slip_k_sqrt * math.sqrt(max(contracts, 0))
        )
        if stressed:
            return base * mult * cfg.stress_slip_multiplier
        return base * mult

    # v7.7.2 change: risk_fraction passed in explicitly (so bias mode can scale it)
    def risk_desired_contracts(
        equity: float,
        entry_price: float,
        hard_stop: float,
        multiplier: float,
        risk_fraction: float,
    ) -> int:
        risk_per_contract = points_to_dollars(
            abs(entry_price - hard_stop), 1, multiplier
        )
        return int(math.floor((equity * risk_fraction) / max(risk_per_contract, 1e-9)))

    def decide_contracts_and_kind(
        equity: float,
        entry_price: float,
        hard_stop: float,
        risk_fraction: float,
    ) -> Tuple[int, str, float, float, Optional[str], int, int]:
        """
        Returns:
          (contracts_final, kind, multiplier, commission_per_side, note_override,
           desired_chosen, capped_chosen)

        IMPORTANT: desired/capped returned reflect the CHOSEN instrument (GC or MGC),
        so diagnostics match actual trading decisions.
        """
        # Fixed contracts override (debug)
        if cfg.fixed_contracts is not None:
            desired = max(int(cfg.fixed_contracts), 0)
            kind = "GC"
            multiplier = cfg.gc_multiplier
            comm = cfg.gc_commission_per_side
            capped = (
                cap_contracts_by_constraints(
                    desired, equity, entry_price, multiplier, cfg
                )
                if cfg.enforce_constraints_on_fixed
                else desired
            )
            return capped, kind, multiplier, comm, None, desired, capped

        # Try GC first
        desired_gc = risk_desired_contracts(
            equity, entry_price, hard_stop, cfg.gc_multiplier, risk_fraction
        )
        if desired_gc <= 0:
            diag.size0_by_risk += 1

        capped_gc = (
            cap_contracts_by_constraints(
                desired_gc, equity, entry_price, cfg.gc_multiplier, cfg
            )
            if desired_gc > 0
            else 0
        )
        if capped_gc > 0:
            return (
                capped_gc,
                "GC",
                cfg.gc_multiplier,
                cfg.gc_commission_per_side,
                None,
                desired_gc,
                capped_gc,
            )

        # Then try MGC
        desired_mgc = risk_desired_contracts(
            equity, entry_price, hard_stop, cfg.mgc_multiplier, risk_fraction
        )
        capped_mgc = (
            cap_contracts_by_constraints(
                desired_mgc, equity, entry_price, cfg.mgc_multiplier, cfg
            )
            if desired_mgc > 0
            else 0
        )
        if capped_mgc > 0:
            return (
                capped_mgc,
                "MGC",
                cfg.mgc_multiplier,
                cfg.mgc_commission_per_side,
                None,
                desired_mgc,
                capped_mgc,
            )

        diag.no_trade_size0_after_mgc += 1

        # Debug-only min floor (OFF by default)
        if cfg.min_contracts_on_signal and cfg.min_contracts_on_signal > 0:
            desired_min = int(cfg.min_contracts_on_signal)
            forced = (
                cap_contracts_by_constraints(
                    desired_min, equity, entry_price, cfg.gc_multiplier, cfg
                )
                if cfg.enforce_constraints_on_min
                else desired_min
            )
            if forced > 0:
                diag.forced_min_used += 1
                return (
                    forced,
                    "GC",
                    cfg.gc_multiplier,
                    cfg.gc_commission_per_side,
                    "forced_min",
                    desired_min,
                    forced,
                )

        return 0, "GC", cfg.gc_multiplier, cfg.gc_commission_per_side, None, 0, 0

    # Iterate
    for date, row in df.iterrows():
        date_d = pd.Timestamp(date.date())

        o, h, l, c = (
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
        )
        yh, yl, atr1, hh1, ll1, c1 = (
            row["YH"],
            row["YL"],
            row["ATR1"],
            row["HH1"],
            row["LL1"],
            row["C1"],
        )
        tr1 = row["TR1"]
        sma1 = row["SMA1"]

        base_ready = pd.notna([yh, yl, atr1, hh1, ll1, c1, tr1]).all()
        trend_ready = (not trend_active) or pd.notna(sma1)
        ready = bool(base_ready and trend_ready)

        note = ""

        # Stress (prior-day TR/ATR)
        stressed = False
        if ready:
            stressed = is_stressed(float(tr1), float(atr1))
            if stressed:
                diag.stressed_days += 1

        # MTM pre-actions
        unrl = unrealized_at(c)
        equity = cash + unrl

        if not ready:
            records.append(
                dict(
                    Date=date,
                    Cash=cash,
                    Unrealized=unrl,
                    Equity=equity,
                    Position=0 if pos is None else pos.side,
                    Contracts=0 if pos is None else pos.contracts,
                    Kind="" if pos is None else pos.kind,
                    Entry=np.nan if pos is None else pos.entry_price,
                    HardStop=np.nan if pos is None else pos.hard_stop,
                    TrailStop=np.nan if pos is None else pos.trail_stop,
                    Note="warmup",
                )
            )
            continue

        yh, yl, atr1, hh1, ll1, c1, tr1 = map(float, (yh, yl, atr1, hh1, ll1, c1, tr1))
        sma1 = float(sma1) if trend_active else float("nan")

        roll_block = cfg.roll_avoid_entries and (date_d in roll_window_dates)

        # Trend gates (v7.7.1 baseline), but now controlled by trend_mode.
        # In "gate": same behavior as v7.7.1.
        # In "bias": do not gate; instead adjust risk_fraction for sizing when against trend.
        trend_long_ok = True
        trend_short_ok = True
        long_against = False
        short_against = False

        if trend_active:
            trend_long_ok = c1 > sma1
            trend_short_ok = c1 < sma1
            long_against = not trend_long_ok  # long when C1<=SMA1
            short_against = not trend_short_ok  # short when C1>=SMA1

        # ----------------------------
        # RAW triggers (pure breakout)
        # ----------------------------
        long_raw = cfg.enable_long and (h >= yh)
        short_raw = cfg.enable_short and (l <= yl)

        if long_raw or short_raw:
            diag.trigger_any_raw += 1
        if long_raw:
            diag.trigger_long_raw += 1
        if short_raw:
            diag.trigger_short_raw += 1
        if long_raw and short_raw:
            diag.trigger_both_raw += 1

        # ----------------------------
        # Roll forced flat
        # ----------------------------
        if cfg.roll_force_flat and pos is not None and (date_d in roll_dates):
            diag.roll_force_flat_exits += 1
            if stressed:
                diag.stressed_execs += 1

            slip = slip_points(pos.contracts, stressed, cfg.exit_slip_multiplier)
            if pos.side == +1:
                exit_price = o - slip
                gross = points_to_dollars(
                    exit_price - pos.entry_price, pos.contracts, pos.multiplier
                )
            else:
                exit_price = o + slip
                gross = points_to_dollars(
                    pos.entry_price - exit_price, pos.contracts, pos.multiplier
                )

            cash += gross - pos.contracts * pos.commission_per_side

            trades.append(
                dict(
                    EntryDate=pos.entry_date,
                    ExitDate=date,
                    Side="LONG" if pos.side == +1 else "SHORT",
                    Entry=pos.entry_price,
                    Exit=exit_price,
                    Contracts=pos.contracts,
                    Kind=pos.kind,
                    GrossPnL=gross,
                    NetPnL=gross - pos.contracts * pos.commission_per_side,
                    Note="roll_force_flat",
                )
            )
            pos = None
            note = "roll_force_flat_exit"

        # ----------------------------
        # Exit logic
        # ----------------------------
        if pos is not None:
            if pos.side == +1:
                stop_level = max(pos.hard_stop, pos.trail_stop)
                if l <= stop_level:
                    if stressed:
                        diag.stressed_execs += 1
                    slip = slip_points(
                        pos.contracts, stressed, cfg.exit_slip_multiplier
                    )
                    exit_price = stop_level - slip
                    gross = points_to_dollars(
                        exit_price - pos.entry_price, pos.contracts, pos.multiplier
                    )
                    cash += gross - pos.contracts * pos.commission_per_side
                    trades.append(
                        dict(
                            EntryDate=pos.entry_date,
                            ExitDate=date,
                            Side="LONG",
                            Entry=pos.entry_price,
                            Exit=exit_price,
                            Contracts=pos.contracts,
                            Kind=pos.kind,
                            GrossPnL=gross,
                            NetPnL=gross - pos.contracts * pos.commission_per_side,
                            Note="stop",
                        )
                    )
                    pos = None
                    note = "stop"
            else:
                stop_level = min(pos.hard_stop, pos.trail_stop)
                if h >= stop_level:
                    if stressed:
                        diag.stressed_execs += 1
                    slip = slip_points(
                        pos.contracts, stressed, cfg.exit_slip_multiplier
                    )
                    exit_price = stop_level + slip
                    gross = points_to_dollars(
                        pos.entry_price - exit_price, pos.contracts, pos.multiplier
                    )
                    cash += gross - pos.contracts * pos.commission_per_side
                    trades.append(
                        dict(
                            EntryDate=pos.entry_date,
                            ExitDate=date,
                            Side="SHORT",
                            Entry=pos.entry_price,
                            Exit=exit_price,
                            Contracts=pos.contracts,
                            Kind=pos.kind,
                            GrossPnL=gross,
                            NetPnL=gross - pos.contracts * pos.commission_per_side,
                            Note="stop",
                        )
                    )
                    pos = None
                    note = "stop"

        # MTM update after exits
        unrl = unrealized_at(c)
        equity = cash + unrl

        # ----------------------------
        # Entry logic (v7.7.1 fixes preserved)
        # ----------------------------
        if pos is None:
            # 1) BOTH-trigger skip happens BEFORE trend gating
            if long_raw and short_raw:
                diag.both_triggers_skip_raw += 1
                if not roll_block:
                    diag.both_triggers_skip_eligible += 1
                note = "both_triggers_skip"
                # No entry.

            else:
                # 2) Apply trend gating / bias now (v7.7.2 minimal addition)
                if trend_mode == "gate":
                    long_trigger = long_raw and (
                        trend_long_ok if trend_active else True
                    )
                    short_trigger = short_raw and (
                        trend_short_ok if trend_active else True
                    )
                    eff_risk_long = cfg.risk_fraction_per_trade
                    eff_risk_short = cfg.risk_fraction_per_trade
                elif trend_mode == "bias":
                    # no gating; only bias sizing when against trend
                    long_trigger = long_raw
                    short_trigger = short_raw
                    eff_risk_long = cfg.risk_fraction_per_trade
                    eff_risk_short = cfg.risk_fraction_per_trade
                    if trend_active:
                        if long_against:
                            eff_risk_long *= cfg.trend_against_long_risk_mult
                        if short_against:
                            eff_risk_short *= cfg.trend_against_short_risk_mult
                elif trend_mode == "off":
                    long_trigger = long_raw
                    short_trigger = short_raw
                    eff_risk_long = cfg.risk_fraction_per_trade
                    eff_risk_short = cfg.risk_fraction_per_trade
                else:
                    raise ValueError(
                        f"Unknown trend_mode={cfg.trend_mode!r} (use off|gate|bias)"
                    )

                # Eligible trigger counts
                if not roll_block:
                    if long_trigger or short_trigger:
                        diag.trigger_any_eligible += 1
                    if long_trigger:
                        diag.trigger_long_eligible += 1
                    if short_trigger:
                        diag.trigger_short_eligible += 1
                    if long_trigger and short_trigger:
                        diag.trigger_both_eligible += 1

                if long_trigger:
                    if roll_block:
                        diag.roll_entry_blocks += 1
                        note = "roll_block_long_entry"
                    else:
                        entry_raw = max(yh, o)
                        hard_stop = entry_raw - cfg.stop_atr_mult * atr1
                        trail_raw = hh1 - cfg.trail_atr_mult * atr1
                        trail_stop = clamp_trail_on_entry(+1, entry_raw, c1, trail_raw)

                        (
                            contracts,
                            kind,
                            mult,
                            comm,
                            forced_note,
                            desired_chosen,
                            capped_chosen,
                        ) = decide_contracts_and_kind(
                            equity, entry_raw, hard_stop, eff_risk_long
                        )

                        # Diagnostics: chosen sizing distribution
                        diag.desired_risk_values.append(int(desired_chosen))
                        diag.capped_values.append(int(capped_chosen))

                        if contracts <= 0:
                            note = "no_size_long"
                        else:
                            diag.entries_taken += 1
                            diag.entry_contracts.append(int(contracts))
                            diag.entry_kind.append(kind)
                            if kind == "GC":
                                diag.used_gc_entries += 1
                            else:
                                diag.used_mgc_entries += 1

                            if stressed:
                                diag.stressed_execs += 1
                            slip = slip_points(
                                contracts,
                                stressed,
                                cfg.entry_slip_multiplier
                                * cfg.entry_slip_multiplier_on_breakout_day,
                            )
                            entry = entry_raw + slip
                            cash -= contracts * comm

                            stop_level = max(hard_stop, trail_stop)
                            if l <= stop_level:
                                if stressed:
                                    diag.stressed_execs += 1
                                slip_exit = slip_points(
                                    contracts, stressed, cfg.exit_slip_multiplier
                                )
                                exit_price = stop_level - slip_exit
                                gross = points_to_dollars(
                                    exit_price - entry, contracts, mult
                                )
                                cash += gross - contracts * comm
                                trades.append(
                                    dict(
                                        EntryDate=date,
                                        ExitDate=date,
                                        Side="LONG",
                                        Entry=entry,
                                        Exit=exit_price,
                                        Contracts=contracts,
                                        Kind=kind,
                                        GrossPnL=gross,
                                        NetPnL=gross - contracts * comm,
                                        Note="enter_and_stop_same_day",
                                    )
                                )
                                note = "enter_long_and_stop_same_day"
                            else:
                                pos = Position(
                                    +1,
                                    entry,
                                    contracts,
                                    hard_stop,
                                    trail_stop,
                                    date,
                                    mult,
                                    comm,
                                    kind,
                                )
                                note = (
                                    forced_note or f"enter_long({kind}) @ {entry:.2f}"
                                )

                elif short_trigger:
                    if roll_block:
                        diag.roll_entry_blocks += 1
                        note = "roll_block_short_entry"
                    else:
                        entry_raw = min(yl, o)
                        hard_stop = entry_raw + cfg.stop_atr_mult * atr1
                        trail_raw = ll1 + cfg.trail_atr_mult * atr1
                        trail_stop = clamp_trail_on_entry(-1, entry_raw, c1, trail_raw)

                        (
                            contracts,
                            kind,
                            mult,
                            comm,
                            forced_note,
                            desired_chosen,
                            capped_chosen,
                        ) = decide_contracts_and_kind(
                            equity, entry_raw, hard_stop, eff_risk_short
                        )

                        diag.desired_risk_values.append(int(desired_chosen))
                        diag.capped_values.append(int(capped_chosen))

                        if contracts <= 0:
                            note = "no_size_short"
                        else:
                            diag.entries_taken += 1
                            diag.entry_contracts.append(int(contracts))
                            diag.entry_kind.append(kind)
                            if kind == "GC":
                                diag.used_gc_entries += 1
                            else:
                                diag.used_mgc_entries += 1

                            if stressed:
                                diag.stressed_execs += 1
                            slip = slip_points(
                                contracts,
                                stressed,
                                cfg.entry_slip_multiplier
                                * cfg.entry_slip_multiplier_on_breakout_day,
                            )
                            entry = entry_raw - slip
                            cash -= contracts * comm

                            stop_level = min(hard_stop, trail_stop)
                            if h >= stop_level:
                                if stressed:
                                    diag.stressed_execs += 1
                                slip_exit = slip_points(
                                    contracts, stressed, cfg.exit_slip_multiplier
                                )
                                exit_price = stop_level + slip_exit
                                gross = points_to_dollars(
                                    entry - exit_price, contracts, mult
                                )
                                cash += gross - contracts * comm
                                trades.append(
                                    dict(
                                        EntryDate=date,
                                        ExitDate=date,
                                        Side="SHORT",
                                        Entry=entry,
                                        Exit=exit_price,
                                        Contracts=contracts,
                                        Kind=kind,
                                        GrossPnL=gross,
                                        NetPnL=gross - contracts * comm,
                                        Note="enter_and_stop_same_day",
                                    )
                                )
                                note = "enter_short_and_stop_same_day"
                            else:
                                pos = Position(
                                    -1,
                                    entry,
                                    contracts,
                                    hard_stop,
                                    trail_stop,
                                    date,
                                    mult,
                                    comm,
                                    kind,
                                )
                                note = (
                                    forced_note or f"enter_short({kind}) @ {entry:.2f}"
                                )

                else:
                    # No eligible trigger (could be due to trend gate in "gate" mode)
                    pass

        # Trail update at EOD (no lookahead; uses prior-day HH1/LL1/ATR1 already)
        if pos is not None:
            if pos.side == +1:
                candidate = hh1 - cfg.trail_atr_mult * atr1
                pos.trail_stop = update_trail(+1, c1, candidate, pos.trail_stop)
            else:
                candidate = ll1 + cfg.trail_atr_mult * atr1
                pos.trail_stop = update_trail(-1, c1, candidate, pos.trail_stop)

        # Record MTM
        unrl = unrealized_at(c)
        equity = cash + unrl

        records.append(
            dict(
                Date=date,
                Cash=cash,
                Unrealized=unrl,
                Equity=equity,
                Position=0 if pos is None else pos.side,
                Contracts=0 if pos is None else pos.contracts,
                Kind="" if pos is None else pos.kind,
                Entry=np.nan if pos is None else pos.entry_price,
                HardStop=np.nan if pos is None else pos.hard_stop,
                TrailStop=np.nan if pos is None else pos.trail_stop,
                Note=note,
            )
        )

    res = pd.DataFrame.from_records(records).set_index("Date")
    trades_df = pd.DataFrame(trades)
    stats = compute_stats(res, trades_df, cfg.initial_equity)
    return res, trades_df, stats, diag


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = Config(
        symbols=["GC=F"],
        start="2006-01-01",
        end=None,
        # Mainline defaults
        risk_fraction_per_trade=0.03,
        fixed_contracts=None,
        min_contracts_on_signal=0,
        max_leverage=2.0,
        margin_rate=0.07,
        margin_buffer=0.80,
        base_slippage_points=0.2,
        slip_k_sqrt=0.002,
        gc_commission_per_side=2.5,
        mgc_commission_per_side=1.25,
        entry_slip_multiplier=1.0,
        entry_slip_multiplier_on_breakout_day=2.0,
        stress_mode=True,
        stress_tr_atr_threshold=1.8,
        stress_slip_multiplier=1.5,
        roll_force_flat=False,
        roll_avoid_entries=True,
        roll_bdays_before=5,
        roll_window_bdays=2,
        trend_filter_enabled=True,
        trend_sma_window=200,
        # v7.7.2
        trend_mode="bias",  # off | gate | bias
        trend_against_long_risk_mult=0.25,
        trend_against_short_risk_mult=0.25,
    )

    print("========== Daily Breakout Playground (v7.7.2) ==========")
    print(f"Symbols: {cfg.symbols}")
    print(f"Window : {cfg.start} -> {cfg.end or 'latest'}")
    print(
        f"Sizing : fixed_contracts={cfg.fixed_contracts}, risk_fraction={cfg.risk_fraction_per_trade}, "
        f"min_contracts_on_signal={cfg.min_contracts_on_signal} (debug-only)"
    )
    print(
        f"Constraints: max_leverage={cfg.max_leverage}, margin_rate={cfg.margin_rate}, margin_buffer={cfg.margin_buffer}"
    )
    print(
        f"Costs: base_slip={cfg.base_slippage_points}, slip_k_sqrt={cfg.slip_k_sqrt}, "
        f"GC_comm/side={cfg.gc_commission_per_side}, MGC_comm/side={cfg.mgc_commission_per_side}"
    )
    print(
        f"Robustness: entry_breakout_mult={cfg.entry_slip_multiplier_on_breakout_day}, "
        f"stress_mode={cfg.stress_mode} (TR1/ATR1>={cfg.stress_tr_atr_threshold} => x{cfg.stress_slip_multiplier})"
    )
    print(
        f"Roll-aware: force_flat={cfg.roll_force_flat}, avoid_entries={cfg.roll_avoid_entries}, "
        f"roll_bdays_before={cfg.roll_bdays_before}, roll_window_bdays={cfg.roll_window_bdays}"
    )
    print(
        f"Trend filter: enabled={cfg.trend_filter_enabled}, SMA_window={cfg.trend_sma_window}, trend_mode={cfg.trend_mode}"
    )
    if (cfg.trend_mode or "").lower().strip() == "bias":
        print(
            f"  bias multipliers: against_long={cfg.trend_against_long_risk_mult}, "
            f"against_short={cfg.trend_against_short_risk_mult}"
        )
    print()

    for sym in cfg.symbols:
        df = load_ohlc_from_yfinance(sym, cfg.start, cfg.end)
        print(
            f"[{sym}] Data sanity: first close={float(df['Close'].iloc[0]):.4f} @ {df.index[0]} | "
            f"last close={float(df['Close'].iloc[-1]):.4f} @ {df.index[-1]}"
        )

        res, trades_df, stats, diag = backtest_one(df, cfg)

        print(f"\n[{sym}] Stats")
        for k, v in stats.items():
            if isinstance(v, float):
                if k in {
                    "TotalReturn",
                    "CAGR",
                    "AnnVol",
                    "Sharpe",
                    "MaxDrawdown",
                    "WinRate",
                }:
                    print(f"{k:>12}: {v:>10.4f}")
                else:
                    print(f"{k:>12}: {v:>12.2f}")
            else:
                print(f"{k:>12}: {v}")

        print("\nTrigger diagnostics (RAW, not entry-eligible, no trend/roll gating):")
        print(f"  trigger_any_days      : {diag.trigger_any_raw}")
        print(f"  trigger_long_days     : {diag.trigger_long_raw}")
        print(f"  trigger_short_days    : {diag.trigger_short_raw}")
        print(f"  trigger_both_days     : {diag.trigger_both_raw}")
        print(f"  both_triggers_skip    : {diag.both_triggers_skip_raw}")

        print(
            "\nTrigger diagnostics (ELIGIBLE = ready & flat & not roll-blocked & passes trend filter):"
        )
        print(f"  eligible_any_days     : {diag.trigger_any_eligible}")
        print(f"  eligible_long_days    : {diag.trigger_long_eligible}")
        print(f"  eligible_short_days   : {diag.trigger_short_eligible}")
        print(f"  eligible_both_days    : {diag.trigger_both_eligible}")
        print(f"  eligible_both_skip    : {diag.both_triggers_skip_eligible}")

        print("\nRoll diagnostics:")
        print(f"  roll_force_flat_exits : {diag.roll_force_flat_exits}")
        print(f"  roll_entry_blocks     : {diag.roll_entry_blocks}")

        print("\nSlippage stress diagnostics:")
        print(f"  stressed_days         : {diag.stressed_days}")
        print(f"  stressed_execs        : {diag.stressed_execs}")

        print("\nSizing + contract selection diagnostics:")
        print(f"  size0_by_risk         : {diag.size0_by_risk}")
        print(f"  size0_by_constraints  : {diag.size0_by_constraints}")
        print(f"  forced_min_used       : {diag.forced_min_used}")
        print(f"  entries_taken         : {diag.entries_taken}")
        print(f"  used_GC_entries       : {diag.used_gc_entries}")
        print(f"  used_MGC_entries      : {diag.used_mgc_entries}")
        print(f"  size0_after_MGC       : {diag.no_trade_size0_after_mgc}")

        desired_stats = _summ_int_series(diag.desired_risk_values)
        capped_stats = _summ_int_series(diag.capped_values)
        entry_hist = _entry_size_hist(diag.entry_contracts)

        print("\nSizing distribution (CHOSEN instrument, on entry-attempt days):")
        print(f"  desired_risk  : {desired_stats}")
        print(f"  capped        : {capped_stats}")

        print("\nEntry contracts histogram:")
        if entry_hist.get("count", 0) > 0:
            print(f"  entries={entry_hist['count']}")
            print(
                f"  pct(1)={entry_hist['pct_1']:.3f}  pct(2)={entry_hist['pct_2']:.3f}  "
                f"pct(3)={entry_hist['pct_3']:.3f}  pct(4+)={entry_hist['pct_4p']:.3f}"
            )
            vc = pd.Series(diag.entry_contracts).value_counts().sort_index()
            top = vc.sort_values(ascending=False).head(10)
            print("  top_sizes (contracts: count):")
            for k, v in top.items():
                print(f"    {int(k):>3d}: {int(v)}")

            if diag.entry_kind:
                kind_counts = pd.Series(diag.entry_kind).value_counts()
                print("\n  entry_kind counts:")
                for k, v in kind_counts.items():
                    print(f"    {k}: {int(v)}")
        else:
            print("  (no entries)")

        print("\nLast 5 days:")
        print(
            res[
                [
                    "Cash",
                    "Unrealized",
                    "Equity",
                    "Position",
                    "Contracts",
                    "Kind",
                    "Entry",
                    "HardStop",
                    "TrailStop",
                    "Note",
                ]
            ].tail(5)
        )

        print("\nLast 5 trades:")
        if not trades_df.empty:
            cols = [
                "EntryDate",
                "ExitDate",
                "Side",
                "Kind",
                "Entry",
                "Exit",
                "Contracts",
                "GrossPnL",
                "NetPnL",
                "Note",
            ]
            print(trades_df[cols].tail(5).to_string(index=False))
        else:
            print("(no trades)")

        # Plot
        try:
            import matplotlib.pyplot as plt

            res["Equity"].plot(
                title=f"{sym} Equity (MTM) - v7.7.2 - trend_mode={cfg.trend_mode}"
            )
            plt.show()
        except Exception as e:
            print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
