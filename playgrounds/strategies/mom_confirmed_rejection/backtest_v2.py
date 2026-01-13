"""
v0.2 Playground backtest: 30m HTF + 5m LTF using yfinance (~60 trading days)

Goal:
- Approximate the "mom-confirmed rejection" setup in an intraday, testable way.
- Use ONLY yfinance 5m data (limited history), resample to 30m for HTF.
- No look-ahead: HTF confirmation must CLOSE before we start looking for LTF entry.

Setup (bullish only; add shorts later if desired):
HTF (30m):
  Rejection bar at t:
    - Low[t] < Low[t-1] AND Close[t] > Low[t-1]
    - Close-location strength: (Close-Low)/(High-Low) >= reject_close_loc_min

  Confirmation bar at t+1:
    - Close[t+1] > High[t]
    - Body fraction >= confirm_min_body_frac
    - Close-location >= confirm_close_loc_min

LTF (5m) after confirmation bar CLOSE:
  - Optional momentum check on first N 5m bars (ret + efficiency)
  - Entry: limit pullback into confirmation range:
      entry_level = conf_low + entry_pullback_frac * (conf_high - conf_low)
      fill if any LTF Low <= entry_level within entry_window_bars
  - Stop: rejection_low
  - Target: fixed R (entry + min_rr * risk)
  - Exit: stop first, then target, else time exit after max_hold_bars

Notes:
- yfinance 5m gives ~60 calendar days (often ~40-60 trading days). We'll keep ~60 trading days max.
- We filter to regular trading hours (RTH): 09:30–16:00 America/New_York.
- Resampling to 30m is done from the 5m feed to avoid dataset alignment issues.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd
import yfinance as yf

NY_TZ = "America/New_York"


@dataclass
class Params:
    ticker: str = "QQQ"

    # Data window (yfinance 5m is limited; period works best)
    yf_period: str = "70d"  # download a bit more, then trim to ~60 trading days
    keep_trading_days: int = 60

    # Session filter
    rth_start: str = "09:30"
    rth_end: str = "16:00"

    # HTF (30m) signal rules
    reject_close_loc_min: float = 0.55
    confirm_min_body_frac: float = 0.10
    confirm_close_loc_min: float = 0.55

    # LTF (5m) momentum confirmation (optional but recommended)
    use_ltf_momentum_gate: bool = True
    impulse_bars: int = 6  # first 6x5m = first 30 minutes after confirm close
    min_impulse_ret: float = 0.0008  # 0.08% move
    min_impulse_eff: float = 0.40

    # Entry / risk / exit
    entry_pullback_frac: float = (
        0.30  # pullback into upper-ish part of confirmation bar range
    )
    entry_window_bars: int = 24  # 24x5m = 2 hours to get filled
    min_rr: float = 2.0  # start with 2R for intraday realism
    max_hold_bars: int = 78  # 78x5m = 6.5 hours (~1 RTH session)
    risk_per_trade: float = 0.01  # toy sizing for equity curve

    debug: bool = False


def _ensure_utc_tz(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = df.copy()
    df.index = idx
    return df


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
    df = _ensure_utc_tz(df)

    # Convert to NY timezone for clean RTH slicing
    df = df.tz_convert(NY_TZ)

    # Keep only weekdays (yfinance sometimes includes odd timestamps)
    df = df[df.index.dayofweek < 5]

    return df


def filter_rth(df_ny: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # between_time works on tz-aware indices
    return df_ny.between_time(start, end, inclusive="left")


def trim_to_last_trading_days(df_ny_rth: pd.DataFrame, keep_days: int) -> pd.DataFrame:
    if df_ny_rth.empty:
        return df_ny_rth

    dates = pd.Index(pd.to_datetime(df_ny_rth.index.date)).unique()
    if len(dates) <= keep_days:
        return df_ny_rth

    keep = set(dates[-keep_days:])
    return df_ny_rth[df_ny_rth.index.date.astype("O").isin(keep)]


def make_30m_bins(index_ny: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Assign each 5m bar to a 30m "bar end" timestamp aligned to session start 09:30.

    Example:
      09:30..09:55 -> 10:00 label (right edge)
      10:00..10:25 -> 10:30 label
    """
    # session start per day
    day = index_ny.normalize()
    session_start = day + pd.Timedelta(hours=9, minutes=30)

    delta = index_ny - session_start
    bin_width = pd.Timedelta(minutes=30)

    # For safety: clamp negative deltas (shouldn't happen if RTH sliced correctly)
    delta = delta.where(delta >= pd.Timedelta(0), pd.Timedelta(0))

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

    out.index = (
        pd.DatetimeIndex(out.index).tz_localize(None).tz_localize(NY_TZ)
    )  # keep tz-aware NY
    return out


def wick_and_body_fracs(
    o: float, h: float, l: float, c: float
) -> tuple[float, float, float]:
    rng = max(h - l, 1e-12)
    body = abs(c - o) / rng
    lower = (min(o, c) - l) / rng
    upper = (h - max(o, c)) / rng
    return lower, upper, body


def close_location(h: float, l: float, c: float) -> float:
    rng = max(h - l, 1e-12)
    return (c - l) / rng


def ltf_momentum_ok(ltf: pd.DataFrame, n: int, min_ret: float, min_eff: float) -> bool:
    if ltf.empty or len(ltf) < n:
        return False
    bars = ltf.iloc[:n]
    ret = (bars["Close"].iloc[-1] / bars["Open"].iloc[0]) - 1.0
    moves = bars["Close"].diff().fillna(0.0).abs()
    net = abs(bars["Close"].iloc[-1] - bars["Open"].iloc[0])
    eff = (net / moves.sum()) if moves.sum() > 1e-12 else 0.0
    return (ret >= min_ret) and (eff >= min_eff)


def backtest(p: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    df5 = download_5m(p.ticker, p.yf_period)
    if df5.empty:
        print("No 5m data returned from yfinance.")
        return pd.DataFrame(), pd.DataFrame()

    df5_rth = filter_rth(df5, p.rth_start, p.rth_end)
    df5_rth = trim_to_last_trading_days(df5_rth, p.keep_trading_days)

    htf = resample_to_30m(df5_rth)
    if htf.empty:
        print("No 30m HTF bars after resampling.")
        return pd.DataFrame(), pd.DataFrame()

    # Prepare for LTF slicing by time
    # df5_rth is tz-aware NY; we'll use it directly.
    trades = []
    equity = 1.0
    eq_curve = []

    # Simple non-overlap: once in a trade, skip signal scanning until exit_time
    next_allowed_time = None

    htf_idx = htf.index
    for i in range(1, len(htf) - 2):
        t_rej_end = htf_idx[i]
        t_conf_end = htf_idx[i + 1]

        if next_allowed_time is not None and t_rej_end < next_allowed_time:
            continue

        prev = htf.iloc[i - 1]
        rej = htf.iloc[i]
        conf = htf.iloc[i + 1]

        # --- HTF rejection (30m) ---
        cond_reject = (rej["Low"] < prev["Low"]) and (rej["Close"] > prev["Low"])
        rej_loc = close_location(rej["High"], rej["Low"], rej["Close"])
        cond_rej_strength = rej_loc >= p.reject_close_loc_min

        if not (cond_reject and cond_rej_strength):
            continue

        # --- HTF confirmation (30m) ---
        cond_confirm = conf["Close"] > rej["High"]
        _, _, conf_body = wick_and_body_fracs(
            conf["Open"], conf["High"], conf["Low"], conf["Close"]
        )
        conf_loc = close_location(conf["High"], conf["Low"], conf["Close"])
        if not (
            cond_confirm
            and conf_body >= p.confirm_min_body_frac
            and conf_loc >= p.confirm_close_loc_min
        ):
            continue

        # No look-ahead: trade decisions happen AFTER confirmation bar closes (t_conf_end)
        # LTF window begins right after confirmation close.
        ltf_after_conf = df5_rth[df5_rth.index > t_conf_end]

        # Optional LTF momentum gate: check first N bars after confirmation close
        if p.use_ltf_momentum_gate:
            if not ltf_momentum_ok(
                ltf_after_conf,
                p.impulse_bars,
                p.min_impulse_ret,
                p.min_impulse_eff,
            ):
                if p.debug:
                    print(f"{t_conf_end} - LTF momentum gate fail")
                continue

        # --- Entry level: pullback into confirmation range ---
        conf_rng = max(conf["High"] - conf["Low"], 1e-12)
        entry_level = conf["Low"] + p.entry_pullback_frac * conf_rng

        # Find fill within entry_window_bars
        entry_scan = ltf_after_conf.iloc[: p.entry_window_bars]
        fill_row = None
        for ts, row in entry_scan.iterrows():
            if row["Low"] <= entry_level:
                fill_row = (ts, row)
                break
        if fill_row is None:
            if p.debug:
                print(f"{t_conf_end} - no fill within entry window")
            continue

        entry_time, _ = fill_row
        entry = float(entry_level)

        # Stop at rejection low
        stop = float(rej["Low"])
        risk = entry - stop
        if risk <= 0:
            continue

        # Fixed R target
        target = entry + p.min_rr * risk

        # --- Manage trade on LTF ---
        # manage = df5_rth[df5_rth.index >= entry_time].iloc[: p.max_hold_bars]
        # manage from the NEXT bar after entry to avoid same-bar ambiguity
        manage = df5_rth[df5_rth.index > entry_time].iloc[: p.max_hold_bars]
        if manage.empty:
            continue

        exit_price = None
        exit_time = None
        exit_reason = None

        for ts, row in manage.iterrows():
            lo = float(row["Low"])
            hi = float(row["High"])

            # Conservative ordering: stop first, then target
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
            # time exit at last bar close
            last_ts = manage.index[-1]
            exit_price = float(manage["Close"].iloc[-1])
            exit_time = last_ts
            exit_reason = "time"

        r_mult = (exit_price - entry) / risk

        # Toy equity update: risk_per_trade * R
        equity *= 1.0 + p.risk_per_trade * r_mult
        eq_curve.append({"time": exit_time, "equity": equity})

        trades.append(
            {
                "htf_reject_end": t_rej_end,
                "htf_confirm_end": t_conf_end,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry": entry,
                "stop": stop,
                "target": target,
                "R": float(r_mult),
                "exit": float(exit_price),
                "exit_reason": exit_reason,
            }
        )

        next_allowed_time = exit_time  # prevent overlapping positions

    trades_df = pd.DataFrame(trades)
    eq_df = (
        pd.DataFrame(eq_curve).set_index("time").sort_index()
        if eq_curve
        else pd.DataFrame()
    )
    return trades_df, eq_df


def summarize(trades: pd.DataFrame, eq: pd.DataFrame):
    if trades.empty:
        print("No trades.")
        return

    win_rate = (trades["R"] > 0).mean()
    avg_r = trades["R"].mean()
    med_r = trades["R"].median()

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
    p = Params(
        ticker="GOOGL",
        yf_period="60d",
        keep_trading_days=60,
        reject_close_loc_min=0.55,
        confirm_min_body_frac=0.10,
        confirm_close_loc_min=0.55,
        use_ltf_momentum_gate=True,
        impulse_bars=6,
        min_impulse_ret=0.0008,
        min_impulse_eff=0.40,
        entry_pullback_frac=0.30,
        entry_window_bars=24,
        min_rr=2.0,
        max_hold_bars=78,
        risk_per_trade=0.01,
        debug=True,
    )

    trades, eq = backtest(p)
    summarize(trades, eq)

    if not trades.empty:
        print("\nSample trades:")
        print(trades.head(20).to_string(index=False))
