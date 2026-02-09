"""How to run:
python playgrounds/strategies/logistic_regression/lr_v1.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================
# Config
# =============================
TICKER = "QQQ"
PERIOD = "60d"
INTERVAL = "5m"
TF_FAST = "5min"
TF_SLOW = "15min"
USE_RTH_ONLY = True
RTH_START = "09:30"
RTH_END = "16:00"
TIMEZONE = "US/Eastern"

K_HORIZON = 4
TRAIN_DAYS = 45
VALID_DAYS = 15

COST_BPS = 2.0
HURDLE_BPS = 2.0

ENTER_THRESHOLD = 0.80
REENTER_THRESHOLD = 0.78
COOLDOWN_MINUTES = 15
MAX_TRADES_PER_DAY = 4
DAILY_LOSS_LIMIT_BPS = 50.0

STOP_LOSS_BPS = 20.0
TAKE_PROFIT_BPS = 40.0

CLASS_WEIGHT_BALANCED = True
SAVE_ARTIFACTS = True
ARTIFACTS_DIR = "artifacts"

RANDOM_STATE = 42


@dataclass
class SplitInfo:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def load_data(ticker: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        interval=INTERVAL,
        period=PERIOD,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if raw.empty:
        raise ValueError("No data returned from yfinance.")

    df = raw.copy()
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex from yfinance.")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df.tz_convert(TIMEZONE)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df.dropna()

    if USE_RTH_ONLY:
        df = df.between_time(RTH_START, RTH_END)

    # Remove any rows with zero volume to avoid stale bars
    df = df[df["volume"] > 0]

    if df.empty:
        raise ValueError("Data empty after cleaning.")

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ema_fast = _ema(series, 12)
    ema_slow = _ema(series, 26)
    macd = ema_fast - ema_slow
    signal = _ema(macd, 9)
    return macd, signal


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def _interval_minutes(interval: str) -> int:
    interval = interval.strip().lower()
    if interval.endswith("min"):
        return int(interval[:-3])
    if interval.endswith("m"):
        return int(interval[:-1])
    raise ValueError(f"Unsupported interval format: {interval}")


def _resample_features(df: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    # Resample with right label/closed to ensure bar is complete at timestamp
    ohlcv = df.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    ohlcv = ohlcv.dropna()

    ema_fast = _ema(ohlcv["close"], 10)
    ema_slow = _ema(ohlcv["close"], 20)
    macd, _ = _macd(ohlcv["close"])

    feats = pd.DataFrame(
        {
            f"{prefix}_ret_1": ohlcv["close"].pct_change(1),
            f"{prefix}_vol_5": ohlcv["close"].pct_change().rolling(5).std(),
            f"{prefix}_ema_dist": (ema_fast - ema_slow) / ohlcv["close"],
            f"{prefix}_rsi_14": _rsi(ohlcv["close"], 14),
            f"{prefix}_macd": macd,
        },
        index=ohlcv.index,
    )

    # Forward-fill onto base-interval timestamps so only last completed bar is used.
    feats = feats.reindex(df.index, method="ffill")
    return feats


def _daily_features(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("1D", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    daily = daily.dropna()

    sma20 = daily["close"].rolling(20).mean()
    daily_vol = daily["close"].pct_change().rolling(20).std()

    feats = pd.DataFrame(
        {
            "daily_trend": (daily["close"] - sma20) / daily["close"],
            "daily_vol_20": daily_vol,
        },
        index=daily.index,
    )
    # Shift by one day to avoid using same-day information intraday.
    feats = feats.shift(1)
    feats = feats.reindex(df.index, method="ffill")
    return feats


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    returns = df["close"].pct_change()
    features["ret_1"] = returns
    features["ret_3"] = df["close"].pct_change(3)
    features["ret_5"] = df["close"].pct_change(5)
    features["ret_10"] = df["close"].pct_change(10)

    features["vol_5"] = returns.rolling(5).std()
    features["vol_20"] = returns.rolling(20).std()

    features["vol_z_20"] = _zscore(df["volume"], 20)
    features["vol_med_ratio_50"] = df["volume"] / df["volume"].rolling(50).median()

    ema20 = _ema(df["close"], 20)
    ema50 = _ema(df["close"], 50)
    features["ema20_dist"] = (df["close"] - ema20) / df["close"]
    features["ema50_dist"] = (df["close"] - ema50) / df["close"]

    features["rsi_14"] = _rsi(df["close"], 14)
    macd, macd_signal = _macd(df["close"])
    features["macd"] = macd
    features["macd_signal"] = macd_signal

    tr = _true_range(df)
    features["atr_14"] = tr.rolling(14).mean()
    features["atr_14_pct"] = features["atr_14"] / df["close"]

    if _interval_minutes(TF_FAST) != _interval_minutes(INTERVAL):
        features = features.join(_resample_features(df, TF_FAST, "tf_fast"))
    features = features.join(_resample_features(df, TF_SLOW, "tf_slow"))
    features = features.join(_daily_features(df))

    # Time-of-day features (sin/cos), use local exchange time
    minutes = df.index.hour * 60 + df.index.minute
    minutes_in_day = 24 * 60
    features["tod_sin"] = np.sin(2 * np.pi * minutes / minutes_in_day)
    features["tod_cos"] = np.cos(2 * np.pi * minutes / minutes_in_day)
    features["dow"] = df.index.dayofweek.astype(float)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()

    return features


def make_labels(df: pd.DataFrame, k: int) -> pd.Series:
    # Entry at next bar open, exit at k bars later open.
    entry = df["open"].shift(-1)
    exit_price = df["open"].shift(-(k + 1))
    gross_ret = (exit_price - entry) / entry
    net_ret = gross_ret - (COST_BPS / 10000.0)

    hurdle = HURDLE_BPS / 10000.0
    label = (net_ret > hurdle).astype(int)
    label.name = "label"
    return label


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    last_ts = df.index.max()
    val_start = last_ts - pd.Timedelta(days=VALID_DAYS) + pd.Timedelta(minutes=1)
    train_end = val_start - pd.Timedelta(minutes=1)
    train_start = train_end - pd.Timedelta(days=TRAIN_DAYS) + pd.Timedelta(minutes=1)

    train = df[(df.index >= train_start) & (df.index <= train_end)]
    val = df[df.index >= val_start]

    if train.empty or val.empty:
        raise ValueError("Train or validation split is empty. Adjust days or data.")

    info = SplitInfo(
        train_start=train.index.min(),
        train_end=train.index.max(),
        val_start=val.index.min(),
        val_end=val.index.max(),
    )
    return train, val, info


def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    class_weight = "balanced" if CLASS_WEIGHT_BALANCED else None
    model = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _decile_stats(probs: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"prob": probs, "y": y_true})
    df = df.dropna()
    df["decile"] = pd.qcut(df["prob"], 10, labels=False, duplicates="drop")
    stats = df.groupby("decile").agg(win_rate=("y", "mean"), count=("y", "size"))
    return stats


def eval_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    probs = model.predict_proba(X)[:, 1]
    metrics = {
        "auc": roc_auc_score(y, probs),
        "log_loss": log_loss(y, probs),
    }
    return metrics


def _compute_max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def backtest(
    price_df: pd.DataFrame,
    probs: pd.Series,
    k: int,
    cost_bps: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    trades: List[Dict[str, object]] = []
    in_position = False
    entry_time = None
    entry_price = None
    cooldown_until = None

    daily_trade_count: Dict[pd.Timestamp, int] = {}
    daily_pnl: Dict[pd.Timestamp, float] = {}

    idx = price_df.index
    entry_idx = None

    for i in range(len(idx)):
        ts = idx[i]
        day = ts.normalize()
        next_day = idx[i + 1].normalize() if i + 1 < len(idx) else None
        is_last_bar = next_day is None or next_day != day

        if day not in daily_trade_count:
            daily_trade_count[day] = 0
            daily_pnl[day] = 0.0

        if daily_pnl[day] <= -(DAILY_LOSS_LIMIT_BPS / 10000.0):
            continue

        if in_position:
            if i >= len(idx) - 1:
                continue

            bars_held = i - entry_idx
            exit_due = bars_held >= k

            # Stop-loss / take-profit based on current close
            current_ret = (price_df.loc[ts, "close"] - entry_price) / entry_price
            stop_hit = current_ret <= -(STOP_LOSS_BPS / 10000.0)
            take_hit = current_ret >= (TAKE_PROFIT_BPS / 10000.0)

            if exit_due or stop_hit or take_hit or is_last_bar:
                # Force an end-of-session exit at the close to avoid overnights.
                if exit_due and not is_last_bar and not stop_hit and not take_hit:
                    exit_price = price_df.loc[ts, "open"]
                else:
                    exit_price = price_df.loc[ts, "close"]
                net_ret = (exit_price - entry_price) / entry_price
                net_ret -= cost_bps / 10000.0

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "net_return": net_ret,
                        "hold_bars": bars_held,
                    }
                )

                daily_trade_count[day] += 1
                daily_pnl[day] += net_ret
                in_position = False
                cooldown_until = ts + pd.Timedelta(minutes=COOLDOWN_MINUTES)
            continue

        if cooldown_until is not None and ts < cooldown_until:
            continue

        if daily_trade_count[day] >= MAX_TRADES_PER_DAY:
            continue

        if i >= len(idx) - (k + 2):
            continue

        p = probs.iloc[i]
        threshold = (
            ENTER_THRESHOLD if daily_trade_count[day] == 0 else REENTER_THRESHOLD
        )

        if p >= threshold and idx[i + 1].normalize() == day:
            entry_idx = i + 1
            entry_time = idx[entry_idx]
            entry_price = price_df.loc[entry_time, "open"]
            in_position = True

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {
            "total_return": 0.0,
            "avg_trade_return": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }
        return trades_df, metrics

    equity = (1.0 + trades_df["net_return"]).cumprod()
    trade_rets = trades_df["net_return"]

    avg_trade = trade_rets.mean()
    win_rate = (trade_rets > 0).mean()
    total_return = equity.iloc[-1] - 1.0
    max_dd = _compute_max_drawdown(equity)

    # Sharpe from per-trade returns (simple approximation)
    if trade_rets.std() == 0:
        sharpe = 0.0
    else:
        sharpe = math.sqrt(252) * trade_rets.mean() / trade_rets.std()

    metrics = {
        "total_return": total_return,
        "avg_trade_return": avg_trade,
        "win_rate": win_rate,
        "num_trades": int(len(trades_df)),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }
    return trades_df, metrics


def _save_artifacts(
    features: pd.DataFrame,
    labels: pd.Series,
    trades: pd.DataFrame,
    split_info: SplitInfo,
) -> None:
    if not SAVE_ARTIFACTS:
        return

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    features.to_csv(os.path.join(ARTIFACTS_DIR, "features.csv"))
    labels.to_csv(os.path.join(ARTIFACTS_DIR, "labels.csv"))
    trades.to_csv(os.path.join(ARTIFACTS_DIR, "trades.csv"), index=False)

    split_path = os.path.join(ARTIFACTS_DIR, "split.txt")
    with open(split_path, "w", encoding="utf-8") as f:
        f.write(f"train_start: {split_info.train_start}\n")
        f.write(f"train_end: {split_info.train_end}\n")
        f.write(f"val_start: {split_info.val_start}\n")
        f.write(f"val_end: {split_info.val_end}\n")


def _print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n{title}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:>20s}: {value: .6f}")
        else:
            print(f"  {key:>20s}: {value}")


def main() -> None:
    df = load_data(TICKER)
    features = make_features(df)
    labels = make_labels(df, K_HORIZON)

    # Align features and labels, drop any rows that lack future labels.
    dataset = features.join(labels, how="inner")
    dataset = dataset.dropna()

    train_df, val_df, split_info = split_train_val(dataset)
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_val = val_df.drop(columns=["label"])
    y_val = val_df["label"]

    price_df = df.loc[dataset.index, ["open", "close"]]
    price_val = price_df.loc[val_df.index]

    print("Train/Validation split:")
    print(f"  Train: {split_info.train_start} -> {split_info.train_end}")
    print(f"  Valid: {split_info.val_start} -> {split_info.val_end}")

    model = train_model(X_train, y_train)
    train_metrics = eval_model(model, X_train, y_train)
    val_metrics = eval_model(model, X_val, y_val)

    probs_val = pd.Series(model.predict_proba(X_val)[:, 1], index=X_val.index)

    deciles = _decile_stats(probs_val.values, y_val.values)

    trades, bt_metrics = backtest(price_val, probs_val, K_HORIZON, COST_BPS)
    trades_2x, bt_metrics_2x = backtest(
        price_val, probs_val, K_HORIZON, COST_BPS * 2
    )

    _print_metrics("Model metrics (train)", train_metrics)
    _print_metrics("Model metrics (validation)", val_metrics)

    print("\nCalibration (validation, deciles):")
    print(deciles)

    _print_metrics("Backtest metrics (validation)", bt_metrics)
    _print_metrics("Backtest metrics (validation, 2x cost)", bt_metrics_2x)

    # Top coefficients by absolute weight
    coef = model.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {"feature": X_train.columns, "weight": coef, "abs_weight": np.abs(coef)}
    ).sort_values("abs_weight", ascending=False)

    print("\nTop coefficients:")
    print(coef_df.head(15).to_string(index=False))

    _save_artifacts(features, labels, trades, split_info)


if __name__ == "__main__":
    main()
