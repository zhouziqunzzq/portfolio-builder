from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[2]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from market_data_store import MarketDataStore
from signal_engine import SignalEngine
from .sideways_mr_config import SidewaysMRConfig


@dataclass
class SidewaysMRState:
    last_rebalance: Optional[pd.Timestamp] = None
    last_weights: Optional[Dict[str, float]] = None

    # Per-ticker gate state (hysteresis)
    in_sideways: Dict[str, bool] = None

    # Per-ticker hard-off streak counter
    gate_off_streak: Dict[str, int] = None

    # Per-ticker position state
    positions: Dict[str, bool] = None

    # Per-ticker hold age (in trading days since entry)
    hold_days: Dict[str, int] = None

    # Per-ticker last exit timestamp (for cooldown enforcement)
    last_exit_dates: Dict[str, pd.Timestamp] = None


class SidewaysMRSleeve:
    """
    Daily Mean-Reversion Sideways Sleeve (spread MR, long-only)

    Spread definition:
      spread(t) = log(P_ticker(t)) - log(P_benchmark(t))

    Gate (per ticker):
      - spread bandwidth (BB-like) is small  (quiet / range-bound)
      - spread slope (annualized) is small   (no sustained drift)
      - persistence over gate_window
      - hysteresis: enter >= gate_enter, exit <= gate_exit

    Trade rule (daily):
      - If gate is ON and spread_z <= -entry_z => enter/hold long
      - Exit when spread_z >= -exit_z (reverted) OR max_hold_days reached
      - Optional: exit immediately if gate turns OFF (kill-switch)
    """

    def __init__(
        self,
        mds: MarketDataStore,
        signals: SignalEngine,
        config: Optional[SidewaysMRConfig] = None,
    ):
        self.mds = mds
        self.signals = signals
        self.config = config or SidewaysMRConfig()
        self.state = SidewaysMRState(
            in_sideways={}, gate_off_streak={}, positions={}, hold_days={}
        )

    # ------------------------------------------------------------------
    # Universe + liquidity
    # ------------------------------------------------------------------

    def get_universe(self, as_of: Optional[datetime | str] = None) -> List[str]:
        tickers = sorted(set([t.upper() for t in self.config.tickers]))
        return tickers

    def _apply_liquidity_filters(
        self, tickers: List[str], end: pd.Timestamp
    ) -> List[str]:
        cfg = self.config
        keep: List[str] = []
        for t in tickers:
            try:
                start = end - pd.Timedelta(days=max(90, cfg.min_adv_window + 10))
                px = self.signals.get_series(
                    ticker=t,
                    signal="last_price",
                    start=start,
                    end=end,
                    interval="1d",
                    price_col="Close",
                )
                if px.empty or px.iloc[-1] < cfg.min_price:
                    # print(
                    #     f"[SidewaysMRSleeve] liquidity filter: price below min for {t}"
                    # )
                    continue

                adv = self.signals.get_series(
                    ticker=t,
                    signal="adv",
                    start=start,
                    end=end,
                    interval="1d",
                    window=cfg.min_adv_window,
                    price_col="Close",
                )
                if (
                    adv.empty
                    or not np.isfinite(adv.iloc[-1])
                    or adv.iloc[-1] < cfg.min_adv
                ):
                    # print(f"[SidewaysMRSleeve] liquidity filter: ADV below min for {t}")
                    continue

                keep.append(t)
            except Exception as e:
                print(f"[SidewaysMRSleeve] liquidity filter failed for {t}: {e}")
                import traceback

                traceback.print_exc()
                continue
        return keep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_target_weights_for_date(
        self,
        as_of: datetime | str,
        start_for_signals: datetime | str,  # interface parity
        regime: str = "sideways",
    ) -> Dict[str, float]:
        as_of = pd.to_datetime(as_of).normalize()

        # 1) get universe and apply liquidity filters
        orig_universe = self.get_universe(as_of=as_of)
        universe = self._apply_liquidity_filters(orig_universe, as_of)
        if not universe:
            return {}

        # 2) compute snapshot of signals
        snap = self._compute_snapshot(universe, as_of)
        if snap.empty:
            return {}

        # 3) update gate hysteresis state per ticker
        self._update_gate_state(snap)

        # 4) allocate daily MR with position state + holds
        dropped_by_liquidity = set(orig_universe) - set(universe)
        weights = self._allocate_daily_mr(
            snap, dropped_by_liquidity=dropped_by_liquidity, as_of=as_of
        )

        self.state.last_rebalance = as_of
        self.state.last_weights = weights
        return weights

    def warmup_to_date(
        self,
        as_of: datetime | str,
        buffer_days: int = 20,
    ) -> None:
        """
        Warm up internal signals (esp. gate scores) up to as_of date.
        """
        as_of = pd.to_datetime(as_of).normalize()
        warmup_window = min(
            self.config.gate_window,
            self.config.bw_rank_window,
        )
        lookback_days = warmup_window + 90
        trading_cal = self._get_trading_calendar(
            start=as_of - pd.Timedelta(days=lookback_days),
            end=as_of,
        )
        trading_cal = trading_cal.sort_values()
        warmup_dates = trading_cal[
            (trading_cal >= (as_of - pd.Timedelta(days=warmup_window + buffer_days)))
            & (trading_cal <= as_of)
        ]

        for d in warmup_dates:
            universe = self.get_universe(as_of=d)
            universe = self._apply_liquidity_filters(universe, d)
            if not universe:
                continue
            _ = self._compute_snapshot(universe, d)
            print(f"[SidewaysMRSleeve] warming up signal for {d.date()}")

    def precompute(
        self,
        start: datetime | str,
        end: datetime | str,
        rebalance_dates: Optional[List[datetime | str]] = None,
        warmup_buffer: Optional[int] = None,
    ) -> pd.DataFrame:
        if rebalance_dates is None or len(rebalance_dates) == 0:
            print("[SidewaysMRSleeve] precompute: no rebalance dates provided")
            return pd.DataFrame()
        first_rebal_date = pd.to_datetime(rebalance_dates[0]).normalize()
        self.warmup_to_date(first_rebal_date)
        print(f"[SidewaysMRSleeve] precompute: warmed up to {first_rebal_date.date()}")
        return pd.DataFrame()  # no vectorized precompute for this sleeve

    # ------------------------------------------------------------------
    # Snapshot: build spread series locally (MVP)
    # ------------------------------------------------------------------

    def _get_benchmark_for(self, ticker: str) -> str:
        cfg = self.config
        return cfg.benchmark_by_ticker.get(ticker, cfg.benchmark)

    def _get_trading_calendar(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ticker: str = "SPY",
    ) -> pd.DatetimeIndex | None:
        df = self.mds.get_ohlcv(
            ticker=ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        return df.index

    def _zscore_last(self, s: pd.Series, window: int) -> float:
        if s is None or s.empty:
            return np.nan
        if len(s) < window:
            return np.nan
        x = s.iloc[-window:]
        mu = float(x.mean())
        sd = float(x.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            return np.nan
        return float((x.iloc[-1] - mu) / sd)

    def _bb_bandwidth_last(self, s: pd.Series, window: int, k: float = 2.0) -> float:
        """
        Bandwidth proxy on spread:
          bw = (upper - lower) / (abs(mid) + eps)
             = 2 * k * sd
        """
        if s is None or s.empty or len(s) < window:
            return np.nan
        x = s.iloc[-window:]
        sd = float(x.std(ddof=0))
        if not np.isfinite(sd):
            return np.nan
        return float(2.0 * k * sd)  # absolute width, stable

    def _bw_rank_last(self, t: str, bw_now: float) -> float:
        """
        Percentile rank of current bandwidth within historical bandwidths.
        """
        cfg = self.config
        hist = self._get_scalar_hist(t, "bw")
        if not np.isfinite(bw_now):
            return np.nan
        # need enough history to be meaningful
        if len(hist) < max(20, min(cfg.bw_rank_window, cfg.gate_window)):
            # print(f"[SidewaysMRSleeve] not enough history for bw rank on {t}: {len(hist)} points")
            return np.nan
        # print(f"[SidewaysMRSleeve] bw rank history for {t}: {len(hist)} points")
        # use last bw_rank_window points
        x = np.array(hist[-cfg.bw_rank_window :], dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 20:
            return np.nan
        # percentile rank of bw_now within x
        rank = float((x <= bw_now).mean())
        return rank

    def _trend_slope_ann_last(self, s: pd.Series, window: int) -> float:
        """
        Simple OLS slope on spread (not log price), annualized.
        """
        if s is None or s.empty or len(s) < window:
            return np.nan
        y = s.iloc[-window:].values.astype(float)
        if not np.all(np.isfinite(y)):
            return np.nan
        x = np.arange(len(y), dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = float((x * x).sum())
        if denom <= 0:
            return np.nan
        slope_per_day = float((x * y).sum() / denom)
        # annualize as linear approximation on spread level
        return float(slope_per_day * 252.0)

    def _slope_t_last(self, spread: pd.Series, window: int) -> float:
        """
        t-statistic of slope on spread over given window.
        """
        y = self._get_spread_tail(spread, window)
        if len(y) < window:
            return np.nan
        x = np.arange(window, dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = float((x * x).sum())
        if denom <= 0:
            return np.nan
        slope_per_day = float((x * y).sum() / denom)

        # residual std
        y_hat = slope_per_day * x
        resid = y - y_hat
        s2 = float((resid * resid).sum() / max(1, window - 2))
        se = (s2**0.5) / (denom**0.5)  # standard error of slope
        if not np.isfinite(se) or se <= 0:
            return np.nan
        return float(abs(slope_per_day) / se)

    def _realized_vol_last(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp, window: int
    ) -> float:
        vol = self.signals.get_series(
            ticker=ticker,
            signal="vol",
            start=start,
            end=end,
            interval="1d",
            window=window,
            price_col="Close",
            annualize=True,
        )
        return vol.iloc[-1] if vol is not None and not vol.empty else np.nan

    def _get_log_spread_series(
        self, ticker: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> Tuple[pd.Series, pd.Series]:
        bench = self._get_benchmark_for(ticker)

        if bench == self.config.benchmark:
            print(f"[SidewaysMRSleeve] using default benchmark {bench} for {ticker}")
        else:
            # Check if the preferred benchmark has price data over the desired period
            px_bench = self.signals.get_series(
                ticker=bench,
                signal="last_price",
                start=start,
                end=end,
                interval="1d",
                price_col="Close",
            )
            max_window = max(
                self.config.spread_window,
                self.config.trend_slope_window,
                self.config.gate_window,
            )
            if px_bench is None or px_bench.empty or len(px_bench) < max_window:
                # Fallback to default benchmark if preferred benchmark data is insufficient
                # print(
                #     f"[SidewaysMRSleeve] fallback to default benchmark {self.config.benchmark} for {ticker} due to insufficient data for {bench}"
                # )
                bench = self.config.benchmark

        beta_hedged_spread = pd.Series()
        if self.config.use_beta_hedged_spread:
            beta_hedged_spread = self.signals.get_series(
                ticker=ticker,
                signal="log_spread_beta_hedged",
                start=start,
                end=end,
                interval="1d",
                price_col="Close",
                benchmark=bench,
                hedge_window=self.config.hedge_window,
            )
        raw_spread = self.signals.get_series(
            ticker=ticker,
            signal="log_spread",
            start=start,
            end=end,
            interval="1d",
            price_col="Close",
            benchmark=bench,
        )
        return (beta_hedged_spread, raw_spread)

    def _push_scalar_hist(self, t: str, key: str, val: float, maxlen: int):
        if not hasattr(self.state, "_scalar_hist"):
            self.state._scalar_hist = {}
        d = self.state._scalar_hist.setdefault(t, {})
        arr = d.get(key, [])
        if np.isfinite(val):
            arr.append(float(val))
        if len(arr) > maxlen:
            arr = arr[-maxlen:]
        d[key] = arr

    def _get_scalar_hist(self, t: str, key: str) -> list[float]:
        if not hasattr(self.state, "_scalar_hist"):
            return []
        return self.state._scalar_hist.get(t, {}).get(key, [])

    def _get_spread_tail(self, spread: pd.Series, n: int) -> np.ndarray:
        if spread is None or spread.empty or len(spread) < n:
            return np.array([])
        x = spread.iloc[-n:].astype(float).values
        return x[np.isfinite(x)]

    def _compute_snapshot(self, tickers: List[str], end: pd.Timestamp) -> pd.DataFrame:
        cfg = self.config
        buffer = pd.Timedelta(days=cfg.signals_extra_buffer_days)

        lookback = max(cfg.spread_window, cfg.trend_slope_window) + cfg.gate_window + 5
        start = end - pd.Timedelta(days=lookback) - buffer

        rows = []
        for t in tickers:
            try:
                hedged_spread, raw_spread = self._get_log_spread_series(
                    t, start=start, end=end
                )
                spread = hedged_spread if cfg.use_beta_hedged_spread else raw_spread

                # Gate ingredients
                bw = self._bb_bandwidth_last(spread, window=cfg.spread_window, k=2.0)
                self._push_scalar_hist(
                    t, "bw", bw, maxlen=max(cfg.bw_rank_window, cfg.gate_window) + 5
                )

                # bandwidth slope: pct change over small horizon
                bw_slope = np.nan
                if (
                    spread is not None
                    and not spread.empty
                    and len(spread) >= cfg.spread_window + 6
                ):
                    # compute two bandwidth points spaced by 5 days
                    bw_now = self._bb_bandwidth_last(
                        spread, window=cfg.spread_window, k=2.0
                    )
                    bw_prev = self._bb_bandwidth_last(
                        spread.iloc[:-5], window=cfg.spread_window, k=2.0
                    )
                    if (
                        np.isfinite(bw_now)
                        and np.isfinite(bw_prev)
                        and abs(bw_prev) > 1e-9
                    ):
                        bw_slope = (bw_now / bw_prev) - 1.0

                # Always use raw spread for trend slope
                slope_ann = self._trend_slope_ann_last(
                    raw_spread, window=cfg.trend_slope_window
                )

                gate_score = self._gate_score_from_scalar_history(
                    bw=bw,
                    bw_slope=bw_slope,
                    slope_ann=slope_ann,
                    spread=spread,
                    t=t,
                )
                # print(
                #     f"[SidewaysMRSleeve] {t} as_of {end.date()}: bw={bw:.4f}, bw_slope={bw_slope:.4f}, slope_ann={slope_ann:.4f}, gate_score={gate_score:.4f}"
                # )

                # MR signal
                spread_z = self._zscore_last(spread, window=cfg.spread_window)

                # Optional sizing
                vol_val = np.nan
                if cfg.use_inverse_vol:
                    vol_val = self._realized_vol_last(
                        t,
                        start=end - pd.Timedelta(days=cfg.vol_window + 10),
                        end=end,
                        window=cfg.vol_window,
                    )

                rows.append(
                    {
                        "ticker": t,
                        "gate_score": gate_score,
                        "spread_z": spread_z,
                        "vol": vol_val,
                    }
                )
            except Exception as e:
                print(f"[SidewaysMRSleeve] warning - failed signals for {t}: {e}")
                import traceback

                traceback.print_exc()
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")
        df[["gate_score", "spread_z", "vol"]] = df[
            ["gate_score", "spread_z", "vol"]
        ].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["spread_z"])  # need trade signal
        return df

    # ------------------------------------------------------------------
    # Gate scoring with persistence (MVP: store scalar history in state)
    # ------------------------------------------------------------------

    def _soft_score(self, x: float, soft: float, hard: float) -> float:
        """
        Map a metric where "smaller is better" into [0,1].
        x <= soft -> 1
        x >= hard -> 0
        linear in between
        """
        if not np.isfinite(x):
            return 0.0
        if hard <= soft:
            return 0.0
        if x <= soft:
            return 1.0
        if x >= hard:
            return 0.0
        return float((hard - x) / (hard - soft))

    def _gate_score_from_scalar_history(
        self, bw: float, bw_slope: float, slope_ann: float, spread: pd.Series, t: str
    ) -> float:
        cfg = self.config

        if not hasattr(self.state, "_gate_hist"):
            self.state._gate_hist = {}
        hist: List[float] = self.state._gate_hist.get(t, [])

        # --- bw score ---
        if cfg.use_bw_rank:
            bw_rank = self._bw_rank_last(t, bw_now=bw)
            # smaller rank = tighter = better
            bw_score = self._soft_score(
                bw_rank,
                soft=cfg.bw_rank_enter,  # full score when <= enter
                hard=cfg.bw_rank_exit,  # 0 score when >= exit (add this knob)
            )
        else:
            bw_score = self._soft_score(
                bw, soft=cfg.bw_thresh, hard=cfg.bw_thresh_hard
            )  # add bw_thresh_hard

        # --- slope score ---
        hard_block = False
        if cfg.use_slope_t:
            st = self._slope_t_last(spread, window=cfg.slope_t_window)
            slope_score = self._soft_score(
                st, soft=cfg.slope_t_soft, hard=cfg.slope_t_hard
            )
            if cfg.slope_t_hard_block and np.isfinite(st) and (st >= cfg.slope_t_hard):
                hard_block = True
        else:
            # if you're still supporting slope_ann path, make it soft too
            slope_score = self._soft_score(
                abs(slope_ann),
                soft=cfg.slope_ann_thresh,
                hard=cfg.slope_ann_thresh_hard,  # add this knob
            )

        # --- bw_slope score ---
        # bw_slope <= max is good; above max decays to 0
        if not np.isfinite(bw_slope):
            bw_slope_score = 1.0
        else:
            bw_slope_score = self._soft_score(
                bw_slope,
                soft=cfg.bw_slope_max,  # full score at/under max
                hard=cfg.bw_slope_hard,  # add bw_slope_hard (e.g. 0.60)
            )

        # Combine (weighted average)
        w_bw = float(getattr(cfg, "gate_w_bw", 0.45))
        w_sl = float(getattr(cfg, "gate_w_slope", 0.35))
        w_bs = float(getattr(cfg, "gate_w_bw_slope", 0.20))
        w_sum = w_bw + w_sl + w_bs
        if w_sum <= 0:
            w_bw, w_sl, w_bs, w_sum = 1.0, 0.0, 0.0, 1.0

        score_today = (
            w_bw * bw_score + w_sl * slope_score + w_bs * bw_slope_score
        ) / w_sum

        if hard_block:
            score_today = 0.0  # only kill it when slope is truly extreme

        hist.append(float(score_today))
        if len(hist) > cfg.gate_window:
            hist = hist[-cfg.gate_window :]
        self.state._gate_hist[t] = hist

        if len(hist) < cfg.gate_window:
            return np.nan
        return float(np.mean(hist))

    def _update_gate_state(self, snap: pd.DataFrame) -> None:
        cfg = self.config
        if self.state.in_sideways is None:
            self.state.in_sideways = {}

        for t, row in snap.iterrows():
            score = float(row["gate_score"]) if pd.notna(row["gate_score"]) else np.nan
            prev = bool(self.state.in_sideways.get(t, False))
            now = prev

            if np.isfinite(score):
                if (not prev) and (score >= cfg.gate_enter):
                    print(f"[SidewaysMRSleeve] gate ON for {t}, gate_score={score:.4f}")
                    now = True
                elif prev and (score <= cfg.gate_exit):
                    print(
                        f"[SidewaysMRSleeve] gate OFF for {t}, gate_score={score:.4f}"
                    )
                    now = False

            self.state.in_sideways[t] = now

    # ------------------------------------------------------------------
    # Allocation: daily MR with min/max hold, long-only
    # ------------------------------------------------------------------

    def _allocate_daily_mr(
        self,
        snap: pd.DataFrame,
        dropped_by_liquidity: Optional[Set[str]] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        cfg = self.config
        if self.state.positions is None:
            self.state.positions = {}
        if self.state.hold_days is None:
            self.state.hold_days = {}
        if self.state.last_exit_dates is None:
            self.state.last_exit_dates = {}
        if self.state.gate_off_streak is None:
            self.state.gate_off_streak = {}

        # normalize as_of
        if as_of is None:
            as_of = pd.Timestamp.now()
        as_of = pd.to_datetime(as_of).normalize()

        max_pos = int(getattr(cfg, "max_positions", 0) or 0)
        if max_pos <= 0:
            max_pos = None

        # ----- 0) age all existing holds by 1 day (called daily) -----
        held = {t for t, v in self.state.positions.items() if v}
        for t in held:
            self.state.hold_days[t] = int(self.state.hold_days.get(t, 0)) + 1

        # ----- 1) process exits -----
        for t in list(held):
            # hard exit if removed by liquidity filters
            if t not in snap.index:
                if dropped_by_liquidity and (t in dropped_by_liquidity):
                    print(
                        f"[SidewaysMRSleeve] exiting {t} due to liquidity filter drop"
                    )
                    self.state.positions[t] = False
                    self.state.hold_days[t] = 0
                    # record exit timestamp for cooldown
                    self.state.last_exit_dates[t] = pd.to_datetime(as_of)
                    self.state.gate_off_streak[t] = 0
                    held.remove(t)
                    continue
                # otherwise skip (missing signal for this ticker)
                print(f"[SidewaysMRSleeve] warning: missing signal for held ticker {t}")
                continue

            z = float(snap["spread_z"].get(t, np.nan))
            score = float(snap["gate_score"].get(t, np.nan))
            entry_gate_on = bool(self.state.in_sideways.get(t, False))
            # Holding gate: allow hold as long as score >= gate_hold (or entry gate still ON)
            hold_gate_on = entry_gate_on or (
                np.isfinite(score) and score >= cfg.gate_hold
            )
            age = int(self.state.hold_days.get(t, 0))

            # Hard-off detection with confirmation days
            hard_off = False
            if np.isfinite(score) and (score <= float(cfg.gate_hard_off)):
                self.state.gate_off_streak[t] = (
                    int(self.state.gate_off_streak.get(t, 0)) + 1
                )
            else:
                self.state.gate_off_streak[t] = 0

            if int(self.state.gate_off_streak.get(t, 0)) >= int(
                cfg.gate_off_confirm_days
            ):
                hard_off = True

            # Hard exit policy
            hard_exit = False
            if bool(cfg.exit_on_gate_off):
                if bool(getattr(cfg, "exit_on_gate_off_hard_only", False)):
                    hard_exit = hard_off
                else:
                    # legacy behavior: exit when holding gate is off
                    hard_exit = not hold_gate_on

            can_exit = age >= max(1, cfg.min_hold_days)
            reverted = np.isfinite(z) and (z >= -cfg.exit_z)
            timed_out = age >= cfg.max_hold_days

            if can_exit and (reverted or timed_out or hard_exit):
                print(
                    f"[SidewaysMRSleeve] exiting {t}: age={age}, z={z:.4f}, "
                    f"gate_score={score:.4f}, entry_gate_on={entry_gate_on}, hold_gate_on={hold_gate_on}, "
                    f"reverted={reverted}, timed_out={timed_out}, hard_exit={hard_exit}"
                )
                self.state.positions[t] = False
                self.state.hold_days[t] = 0
                # record exit timestamp for cooldown
                self.state.last_exit_dates[t] = pd.to_datetime(as_of)
                self.state.gate_off_streak[t] = 0
                held.remove(t)

        # ----- 2) choose entries -----
        entries = []
        for t, row in snap.iterrows():
            if t in held:
                continue

            gate_on = bool(self.state.in_sideways.get(t, False))
            score = float(row.get("gate_score", np.nan))
            if not gate_on:
                continue
            if np.isfinite(score) and score < cfg.gate_entry_min:
                print(f"[SidewaysMRSleeve] skipping entry {t} due to low gate score {score:.4f}")
                continue

            z = float(row["spread_z"])
            if not np.isfinite(z):
                continue

            if z <= -cfg.entry_z:
                # enforce cooldown after exit
                last_exit = self.state.last_exit_dates.get(t)
                if (
                    last_exit is not None
                    and cfg.cooldown_days
                    and cfg.cooldown_days > 0
                ):
                    # compute days difference using normalized timestamps
                    try:
                        days_since = (
                            as_of - pd.to_datetime(last_exit).normalize()
                        ).days
                    except Exception:
                        days_since = int(
                            (as_of - pd.Timestamp(last_exit).normalize()).days
                        )
                    if days_since < int(cfg.cooldown_days):
                        # still cooling down; skip entry
                        print(
                            f"[SidewaysMRSleeve] skipping {t} due to cooldown ({days_since}<{cfg.cooldown_days})"
                        )
                        continue

                strength = max(0.0, -z)
                invvol = 1.0
                if cfg.use_inverse_vol:
                    v = float(row["vol"])
                    if np.isfinite(v) and v > 0:
                        invvol = 1.0 / v
                entries.append((t, strength * invvol))

        if max_pos is not None:
            slots = max(0, max_pos - len(held))
            if slots > 0 and entries:
                # select top entries by score
                entries = sorted(entries, key=lambda x: x[1], reverse=True)[:slots]
            else:
                entries = []

        for t, s in entries:
            print(
                f"[SidewaysMRSleeve] entering {t}, z={snap.at[t, 'spread_z']:.4f}, entry_score={s:.4f}"
            )
            self.state.positions[t] = True
            self.state.hold_days[t] = 0
            self.state.gate_off_streak[t] = 0
            held.add(t)

        # ----- 3) allocate weights among held -----
        if not held:
            return {}

        scores = {}
        for t in held:
            z = float(snap["spread_z"].get(t, np.nan))
            strength = max(0.0, -z)

            invvol = 1.0
            if cfg.use_inverse_vol:
                v = float(snap["vol"].get(t, np.nan))
                if np.isfinite(v) and v > 0:
                    invvol = 1.0 / v

            scores[t] = strength * invvol

        s = (
            pd.Series(scores, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        # print(f"[SidewaysMRSleeve] raw scores:\n{s}")
        if s.sum() <= 0:
            s[:] = 1.0

        w = (s / s.sum()) * float(cfg.sleeve_gross_cap)

        # per-asset cap then renormalize to gross cap
        w = w.clip(upper=float(cfg.w_max_per_asset))
        if w.sum() > 0:
            w = w * (float(cfg.sleeve_gross_cap) / float(w.sum()))

        # keep state consistent
        for t in list(self.state.positions.keys()):
            self.state.positions[t] = t in w.index

        return {t: float(x) for t, x in w.items()}
