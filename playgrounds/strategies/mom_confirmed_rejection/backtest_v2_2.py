"""
v0.2.2 Playground backtest (closer to ICT): 30m HTF + 5m LTF using yfinance (~60 trading days)

What changed vs v0.2.1:
- Adds configurable "modes" to keep closer to ICT:
  1) entry_mode:
       - "pullback" (ICT-ish): limit entry on pullback into confirmation candle range
       - "market_after_confirm": enter at first 5m bar open after confirmation close (for diagnostics)
  2) no_overnight:
       - If True: force flat by end of day (no holding overnight)
  3) momentum_gate_mode:
       - "off" / "loose" / "strict"
- Keeps:
  - NY timezone + RTH filter
  - HTF rejection + confirmation rules
  - Late confirmation cutoff (default 15:00; set to 14:00 if you want)
  - Intrabar ambiguity fix: manage begins on NEXT 5m bar after entry fill
  - Minimum risk filter via ATR(5m)

Defaults below are set to "closer to ICT":
- entry_mode="pullback"
- no_overnight=True
- momentum_gate_mode="loose"
- latest_confirm_time="15:00" (adjust if you prefer)

Notes on "closer to ICT":
- ICT-style traders often avoid holding overnight for intraday setups.
- Pullback entry is more aligned with "FVG/imbalance retrace" behavior.
- Momentum gate is kept but loosened so it doesn't kill the sample.

DISCLAIMER:
- This is still an OHLCV-only approximation. No VWAP, no true IFVG detection, no orderflow.
- yfinance 5m history is limited; treat results as exploratory only.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import yfinance as yf

NY_TZ = "America/New_York"


# ---------------------------
# Params
# ---------------------------


@dataclass
class Params:
    ticker: str = "QQQ"

    # yfinance 5m window
    yf_period: str = "70d"
    keep_trading_days: int = 60

    # Regular trading hours
    rth_start: str = "09:30"
    rth_end: str = "16:00"

    # HTF time filter (NY local time)
    latest_confirm_time: str = "15:00"  # HH:MM, inclusive

    # HTF (30m) rules
    reject_close_loc_min: float = 0.55
    confirm_min_body_frac: float = 0.10
    confirm_close_loc_min: float = 0.55

    # Modes (closer to ICT)
    entry_mode: str = "pullback"  # "pullback" or "market_after_confirm"
    no_overnight: bool = True  # force flat by end of entry day
    momentum_gate_mode: str = "loose"  # "off" / "loose" / "strict"

    # Momentum gate params (used depending on mode)
    # loose defaults: first hour after confirm close, mild thresholds
    impulse_bars_loose: int = 12
    min_impulse_ret_loose: float = 0.0004
    min_impulse_eff_loose: float = 0.30

    # strict defaults: first 30 min after confirm close, stronger thresholds
    impulse_bars_strict: int = 6
    min_impulse_ret_strict: float = 0.0008
    min_impulse_eff_strict: float = 0.40

    # Entry / risk / exits
    entry_pullback_frac: float = 0.30  # pullback into confirmation candle range
    entry_window_bars: int = 24  # 2 hours after confirm close to get filled
    min_rr: float = 2.0
    max_hold_bars: int = 78  # ~1 RTH session (6.5 hours)
    risk_per_trade: float = 0.01  # toy sizing

    # Minimum risk filter via ATR(5m)
    atr_len: int = 14
    min_risk_atr_mult: float = 1.0  # require risk >= 1.0 * ATR_5m at entry

    debug: bool = False


# ---------------------------
# Data helpers
# ---------------------------


def _ensure_utc_tz(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out = df.copy()
    out.index = idx
    return out


def download_5m(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="5m",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = _ensure_utc_tz(df).tz_convert(NY_TZ)

    # Weekdays only
    df = df[df.index.dayofweek < 5]
    return df


def filter_rth(df_ny: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df_ny.between_time(start, end, inclusive="left")


def trim_to_last_trading_days(df_ny_rth: pd.DataFrame, keep_days: int) -> pd.DataFrame:
    if df_ny_rth.empty:
        return df_ny_rth
    dates = pd.Index(pd.to_datetime(df_ny_rth.index.date)).unique()
    if len(dates) <= keep_days:
        return df_ny_rth
    keep = set(dates[-keep_days:])
    return df_ny_rth[df_ny_rth.index.date.astype("O").isin(keep)]


# ---------------------------
# Resample 5m -> 30m (aligned to 09:30)
# ---------------------------


def make_30m_bins(index_ny: pd.DatetimeIndex) -> pd.DatetimeIndex:
    day = index_ny.normalize()
    session_start = day + pd.Timedelta(hours=9, minutes=30)
    delta = index_ny - session_start
    delta = delta.where(delta >= pd.Timedelta(0), pd.Timedelta(0))

    bin_width = pd.Timedelta(minutes=30)
    k = (delta // bin_width).astype("int64")
    bin_end = session_start + (k + 1) * bin_width
    return pd.DatetimeIndex(bin_end)


def resample_to_30m(df_5m_rth: pd.DataFrame) -> pd.DataFrame:
    if df_5m_rth.empty:
        return df_5m_rth
    df = df_5m_rth.copy()
    df["BinEnd"] = make_30m_bins(df.index)

    g = df.groupby("BinEnd", sort=True)
    out = pd.DataFrame(
        {
            "Open": g["Open"].first(),
            "High": g["High"].max(),
            "Low": g["Low"].min(),
            "Close": g["Close"].last(),
            "Volume": g["Volume"].sum(),
        }
    ).dropna()

    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        out.index = idx.tz_localize(NY_TZ)
    else:
        out.index = idx.tz_convert(NY_TZ)
    return out


# ---------------------------
# Features
# ---------------------------


def close_location(h: float, l: float, c: float) -> float:
    rng = max(h - l, 1e-12)
    return (c - l) / rng


def body_fraction(o: float, h: float, l: float, c: float) -> float:
    rng = max(h - l, 1e-12)
    return abs(c - o) / rng


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def ltf_momentum_ok(ltf: pd.DataFrame, n: int, min_ret: float, min_eff: float) -> bool:
    if ltf.empty or len(ltf) < n:
        return False
    bars = ltf.iloc[:n]
    ret = (bars["Close"].iloc[-1] / bars["Open"].iloc[0]) - 1.0
    moves = bars["Close"].diff().fillna(0.0).abs()
    net = abs(bars["Close"].iloc[-1] - bars["Open"].iloc[0])
    eff = (net / moves.sum()) if moves.sum() > 1e-12 else 0.0
    return (ret >= min_ret) and (eff >= min_eff)


def get_gate_params(p: Params) -> tuple[int, float, float]:
    if p.momentum_gate_mode == "strict":
        return p.impulse_bars_strict, p.min_impulse_ret_strict, p.min_impulse_eff_strict
    # loose
    return p.impulse_bars_loose, p.min_impulse_ret_loose, p.min_impulse_eff_loose


# ---------------------------
# Backtest
# ---------------------------


def backtest(p: Params) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df5 = download_5m(p.ticker, p.yf_period)
    if df5.empty:
        print("No 5m data returned from yfinance.")
        return pd.DataFrame(), pd.DataFrame(), {}

    df5 = filter_rth(df5, p.rth_start, p.rth_end)
    df5 = trim_to_last_trading_days(df5, p.keep_trading_days)

    df5 = df5.copy()
    df5["ATR"] = compute_atr(df5, p.atr_len)

    htf = resample_to_30m(df5)
    if htf.empty:
        print("No 30m HTF bars after resampling.")
        return pd.DataFrame(), pd.DataFrame(), {}

    latest_time = pd.Timestamp(p.latest_confirm_time).time()

    counters = {
        "scanned_htf_bars": 0,
        "pass_reject": 0,
        "pass_confirm": 0,
        "pass_timegate": 0,
        "pass_ltf_mom": 0,
        "pass_entry": 0,
        "pass_min_risk": 0,
        "trades": 0,
    }

    trades = []
    equity = 1.0
    eq_curve = []
    next_allowed_time = None

    # Precompute session end per date for no-overnight rule
    # We'll treat the last available 5m bar timestamp per date as the EOD timestamp.
    eod_by_date = df5.groupby(df5.index.date).apply(lambda x: x.index.max()).to_dict()

    for i in range(1, len(htf) - 2):
        t_rej_end = htf.index[i]
        t_conf_end = htf.index[i + 1]

        if next_allowed_time is not None and t_rej_end < next_allowed_time:
            continue

        counters["scanned_htf_bars"] += 1

        prev = htf.iloc[i - 1]
        rej = htf.iloc[i]
        conf = htf.iloc[i + 1]

        # --- Reject (30m) ---
        cond_reject = (rej["Low"] < prev["Low"]) and (rej["Close"] > prev["Low"])
        rej_loc = close_location(rej["High"], rej["Low"], rej["Close"])
        if not (cond_reject and rej_loc >= p.reject_close_loc_min):
            continue
        counters["pass_reject"] += 1

        # --- Confirm (30m) ---
        cond_confirm = conf["Close"] > rej["High"]
        conf_body = body_fraction(
            conf["Open"], conf["High"], conf["Low"], conf["Close"]
        )
        conf_loc = close_location(conf["High"], conf["Low"], conf["Close"])
        if not (
            cond_confirm
            and conf_body >= p.confirm_min_body_frac
            and conf_loc >= p.confirm_close_loc_min
        ):
            continue
        counters["pass_confirm"] += 1

        # --- Time gate ---
        if t_conf_end.time() > latest_time:
            continue
        counters["pass_timegate"] += 1

        # LTF after confirmation close (no lookahead)
        ltf_after_conf = df5[df5.index > t_conf_end]
        if ltf_after_conf.empty:
            continue

        # --- Momentum gate ---
        if p.momentum_gate_mode != "off":
            n, min_ret, min_eff = get_gate_params(p)
            if not ltf_momentum_ok(ltf_after_conf, n, min_ret, min_eff):
                if p.debug:
                    print(
                        f"{t_conf_end} - LTF momentum gate fail ({p.momentum_gate_mode})"
                    )
                continue
        counters["pass_ltf_mom"] += 1

        # --- Entry ---
        entry_time = None
        entry_price = None

        if p.entry_mode == "market_after_confirm":
            # First 5m bar open after confirmation close
            first_bar = ltf_after_conf.iloc[:1]
            if first_bar.empty:
                continue
            entry_time = first_bar.index[0]
            entry_price = float(first_bar["Open"].iloc[0])
        else:
            # ICT-ish pullback into confirmation candle range
            conf_rng = max(conf["High"] - conf["Low"], 1e-12)
            entry_level = float(conf["Low"] + p.entry_pullback_frac * conf_rng)

            entry_scan = ltf_after_conf.iloc[: p.entry_window_bars]
            for ts, row in entry_scan.iterrows():
                if float(row["Low"]) <= entry_level:
                    entry_time = ts
                    entry_price = entry_level
                    break

        if entry_time is None:
            continue
        counters["pass_entry"] += 1

        # Stop at HTF rejection low
        stop = float(rej["Low"])
        risk = entry_price - stop
        if risk <= 0:
            continue

        # Minimum risk filter (ATR on 5m at/just before entry)
        atr_val = df5.loc[:entry_time, "ATR"].iloc[-1]
        if pd.isna(atr_val):
            continue
        if risk < p.min_risk_atr_mult * float(atr_val):
            if p.debug:
                print(
                    f"{entry_time} - min risk fail: risk={risk:.4f}, atr={float(atr_val):.4f}"
                )
            continue
        counters["pass_min_risk"] += 1

        target = entry_price + p.min_rr * risk

        # --- Manage (intrabar ambiguity fix): start from NEXT 5m bar after entry ---
        manage = df5[df5.index > entry_time]

        # no_overnight: cap manage window to end of entry day
        if p.no_overnight:
            eod = eod_by_date.get(entry_time.date())
            if eod is not None:
                manage = manage[manage.index <= eod]

        manage = manage.iloc[: p.max_hold_bars]
        if manage.empty:
            continue

        exit_price = None
        exit_time = None
        exit_reason = None

        for ts, row in manage.iterrows():
            lo = float(row["Low"])
            hi = float(row["High"])

            if lo <= stop:
                exit_price = stop
                exit_time = ts
                exit_reason = "stop"
                break
            if hi >= target:
                exit_price = target
                exit_time = ts
                exit_reason = "target"
                break

        if exit_price is None:
            exit_time = manage.index[-1]
            exit_price = float(manage["Close"].iloc[-1])
            exit_reason = "time"

        r_mult = (exit_price - entry_price) / risk

        equity *= 1.0 + p.risk_per_trade * r_mult
        eq_curve.append({"time": exit_time, "equity": equity})

        trades.append(
            {
                "htf_reject_end": t_rej_end,
                "htf_confirm_end": t_conf_end,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry": float(entry_price),
                "stop": float(stop),
                "target": float(target),
                "ATR_5m": float(atr_val),
                "risk": float(risk),
                "R": float(r_mult),
                "exit": float(exit_price),
                "exit_reason": exit_reason,
                "entry_mode": p.entry_mode,
                "mom_mode": p.momentum_gate_mode,
                "no_overnight": p.no_overnight,
            }
        )

        counters["trades"] += 1
        next_allowed_time = exit_time

    trades_df = pd.DataFrame(trades)
    eq_df = (
        pd.DataFrame(eq_curve).set_index("time").sort_index()
        if eq_curve
        else pd.DataFrame()
    )
    return trades_df, eq_df, counters


# ---------------------------
# Reporting
# ---------------------------


def summarize(trades: pd.DataFrame, eq: pd.DataFrame, counters: dict):
    if not counters:
        return

    print("\n=== Funnel ===")
    for k in [
        "scanned_htf_bars",
        "pass_reject",
        "pass_confirm",
        "pass_timegate",
        "pass_ltf_mom",
        "pass_entry",
        "pass_min_risk",
        "trades",
    ]:
        print(f"{k:>18}: {counters.get(k, 0)}")

    if trades.empty:
        print("\nNo trades.")
        return

    win_rate = (trades["R"] > 0).mean()
    avg_r = trades["R"].mean()
    med_r = trades["R"].median()

    print("\n=== Results ===")
    print("Trades:", len(trades))
    print("Win rate:", round(float(win_rate), 3))
    print("Avg R:", round(float(avg_r), 3))
    print("Median R:", round(float(med_r), 3))
    print("\nExit reasons:")
    print(trades["exit_reason"].value_counts())

    if not eq.empty:
        total = eq["equity"].iloc[-1]
        print("\nEquity multiple (toy sizing):", round(float(total), 3))


if __name__ == "__main__":
    # "Closer to ICT" defaults:
    p = Params(
        ticker="DBX",
        yf_period="60d",
        keep_trading_days=60,
        latest_confirm_time="15:00",  # you can set back to "14:00" if desired
        reject_close_loc_min=0.55,
        confirm_min_body_frac=0.10,
        confirm_close_loc_min=0.55,
        entry_mode="pullback",  # ICT-ish
        no_overnight=True,  # ICT-ish for intraday setups
        momentum_gate_mode="loose",  # keep but don't strangle sample
        entry_pullback_frac=0.30,
        entry_window_bars=24,
        min_rr=2.0,
        max_hold_bars=78,
        atr_len=14,
        min_risk_atr_mult=1.0,
        debug=True,
    )

    trades, eq, counters = backtest(p)
    summarize(trades, eq, counters)

    if not trades.empty:
        print("\nSample trades:")
        print(trades.head(25).to_string(index=False))
