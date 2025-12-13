from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
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
from .sideways_config import SidewaysConfig


@dataclass
class SidewaysState:
    last_rebalance: Optional[pd.Timestamp] = None
    last_weights: Optional[Dict[str, float]] = None
    # Per-ticker gate state (hysteresis)
    in_sideways: Dict[str, bool] = None
    # Per-ticker position state
    positions: Dict[str, bool] = None


class SidewaysSleeve:
    """
    Sideways Sleeve (BB mean reversion, long-only)

    1) Sideways gate per ticker:
       - bb_bandwidth(window=bb_window,k=bb_k) < bw_thresh
       - abs(trend_slope(window=bb_window, use_log_price=True)) < slope_thresh
       - persistence: rolling mean over gate_window
       - hysteresis: enter >= gate_enter, exit <= gate_exit

    2) Trade rule:
       - if in_sideways[ticker] and bb_z(window=bb_window,k=bb_k) <= -entry_z:
           allocate long weight (mean reversion)
       - weight by strength (more oversold -> more weight), optionally inverse-vol
    """

    def __init__(
        self,
        mds: MarketDataStore,
        signals: SignalEngine,
        config: Optional[SidewaysConfig] = None,
    ):
        self.mds = mds
        self.signals = signals
        self.config = config or SidewaysConfig()
        self.state = SidewaysState(in_sideways={})

    # ------------------------------------------------------------------
    # Universe + liquidity (MVP)
    # ------------------------------------------------------------------

    def get_universe(self, as_of: Optional[datetime | str] = None) -> List[str]:
        # MVP: static list
        return sorted(set([t.upper() for t in self.config.tickers]))

    def _apply_liquidity_filters(
        self, tickers: List[str], end: pd.Timestamp
    ) -> List[str]:
        cfg = self.config
        keep: List[str] = []
        for t in tickers:
            try:
                df = self.mds.get_ohlcv(
                    ticker=t,
                    start=end - pd.Timedelta(days=90),
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                )
                if df is None or df.empty:
                    print(f"[SidewaysSleeve] no price data for {t} up to {end.date()}")
                    continue
                if df["Close"].iloc[-1] < cfg.min_price:
                    print(
                        f"[SidewaysSleeve] {t} price {df['Close'].iloc[-1]:.2f} below min {cfg.min_price}"
                    )
                    continue
                if "Volume" not in df.columns:
                    print(f"[SidewaysSleeve] {t} has no Volume data for ADV filter")
                    continue
                adv = df["Volume"].rolling(cfg.min_adv_window).mean().iloc[-1]
                if not np.isfinite(adv) or adv < cfg.min_adv:
                    print(f"[SidewaysSleeve] {t} ADV {adv:.0f} below min {cfg.min_adv}")
                    continue
                keep.append(t)
            except Exception:
                continue
        return keep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_target_weights_for_date(
        self,
        as_of: datetime | str,
        start_for_signals: (
            datetime | str
        ),  # kept for interface parity; we infer lookbacks internally
        regime: str = "sideways",
    ) -> Dict[str, float]:
        as_of = pd.to_datetime(as_of).normalize()

        orig_universe = self.get_universe(as_of=as_of)
        universe = self._apply_liquidity_filters(orig_universe, as_of)
        # print(f"SidewaysSleeve: universe tickers: {universe}")
        if not universe:
            return {}

        snap = self._compute_snapshot(universe, as_of)
        # print(f"SidewaysSleeve: snapshot at {as_of.date()}:\n{snap}")
        if snap.empty:
            return {}

        # 1) update hysteresis gate state per ticker
        self._update_gate_state(snap)
        # print(
        #     f"SidewaysSleeve: in_sideways state at {as_of.date()}: {self.state.in_sideways}"
        # )

        # 2) choose candidates: only those currently in-sideways AND oversold
        # determine which tickers were dropped by liquidity/universe filters
        dropped_by_liquidity = set(orig_universe) - set(universe)
        weights = self._allocate_mean_reversion(
            snap, dropped_by_liquidity=dropped_by_liquidity
        )

        self.state.last_rebalance = as_of
        self.state.last_weights = weights
        return weights

    # ------------------------------------------------------------------
    # Snapshot computation (non-vec)
    # ------------------------------------------------------------------

    def _compute_snapshot(self, tickers: List[str], end: pd.Timestamp) -> pd.DataFrame:
        cfg = self.config
        buffer = pd.Timedelta(days=cfg.signals_extra_buffer_days)

        rows = []
        for t in tickers:
            try:
                lookback = max(cfg.bb_window, cfg.trend_slope_window) + cfg.gate_window
                start_gate = end - pd.Timedelta(days=lookback) - buffer

                # --- gate inputs ---
                bb_bw = self.signals.get_series(
                    t,
                    "bb_bandwidth",
                    start=start_gate,
                    end=end,
                    window=cfg.bb_window,
                    k=cfg.bb_k,
                )
                # print(
                #     f"SidewaysSleeve: {t} bb_bandwidth at {bb_bw.index[-1].date()} = {bb_bw.iloc[-1]:.4f}"
                # )
                bb_bw_slope = bb_bw.pct_change(cfg.bw_slope_window)
                # print(
                #     f"SidewaysSleeve: {t} bb_bandwidth_slope at {bb_bw_slope.index[-1].date()} = {bb_bw_slope.iloc[-1]:.6f}"
                # )
                slope = self.signals.get_series(
                    t,
                    "trend_slope",
                    start=start_gate,
                    end=end,
                    window=cfg.trend_slope_window,
                    use_log_price=True,
                )
                # print(
                #     f"SidewaysSleeve: {t} trend_slope at {slope.index[-1].date()} = {slope.iloc[-1]:.6f}"
                # )

                gate_score = self._gate_score_from_series(
                    bb_bw, bb_bw_slope, slope
                )  # scalar

                # --- trade signal ---
                bb_z_s = self.signals.get_series(
                    t,
                    "bb_z",
                    start=start_gate,
                    end=end,
                    window=cfg.bb_window,
                )
                bb_z = (
                    bb_z_s.iloc[-1]
                    if (bb_z_s is not None and not bb_z_s.empty)
                    else np.nan
                )
                # print(
                #     f"SidewaysSleeve: {t} bb_z at {bb_z_s.index[-1].date()} = {bb_z:.4f}"
                # )

                # --- optional sizing ---
                vol_val = np.nan
                if cfg.use_inverse_vol:
                    start_vol = end - pd.Timedelta(days=cfg.vol_window) - buffer
                    vol_s = self.signals.get_series(
                        t, "vol", start=start_vol, end=end, window=cfg.vol_window
                    )
                    vol_val = (
                        vol_s.iloc[-1]
                        if (vol_s is not None and not vol_s.empty)
                        else np.nan
                    )

                rows.append(
                    {
                        "ticker": t,
                        "gate_score": gate_score,  # float
                        "bb_z": bb_z,  # float
                        "vol": vol_val,  # float (or nan)
                    }
                )

            except Exception as e:
                print(
                    f"SidewaysSleeve: warning - failed to compute signals for {t}. Exception: {e}"
                )
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")

        # sanitize numeric columns only (all are numeric now)
        df[["gate_score", "bb_z", "vol"]] = df[["gate_score", "bb_z", "vol"]].replace(
            [np.inf, -np.inf], np.nan
        )

        # require trade signal at least; gate_score can be nan (means "no update")
        df = df.dropna(subset=["bb_z"])
        return df

    # ------------------------------------------------------------------
    # Gate state update (persistence + hysteresis)
    # ------------------------------------------------------------------

    def _gate_score_from_series(
        self, bb_bw: pd.Series, bb_bw_slope: pd.Series, slope: pd.Series
    ) -> float:
        cfg = self.config

        # Align + clean
        s = pd.concat(
            [bb_bw.rename("bw"), bb_bw_slope.rename("bw_slope"), slope.rename("slope")],
            axis=1,
            join="inner",
        ).dropna()
        if s.empty:
            return np.nan
        # Annualize slope
        # Note: Assuming daily bars!!!
        slope_ann = np.expm1(s["slope"] * 252.0)

        cond = (
            (s["bw"] < cfg.bw_thresh)
            & (slope_ann.abs() < cfg.slope_ann_thresh)
            & (s["bw_slope"] <= cfg.bw_slope_max)
        )

        bw_ok = s["bw"] < cfg.bw_thresh
        slope_ok = slope_ann.abs() < cfg.slope_ann_thresh
        bwsl_ok = s["bw_slope"] <= cfg.bw_slope_max
        # print(
        #     "gate hit rates:",
        #     float(bw_ok.mean()),
        #     float(slope_ok.mean()),
        #     float(bwsl_ok.mean()),
        #     "all:",
        #     float((bw_ok & slope_ok & bwsl_ok).mean()),
        # )

        # Require full history for gate window
        score = cond.rolling(window=cfg.gate_window, min_periods=cfg.gate_window).mean()
        if score.empty:
            return np.nan
        return float(score.iloc[-1])

    def _update_gate_state(self, snap: pd.DataFrame) -> None:
        cfg = self.config
        if self.state.in_sideways is None:
            self.state.in_sideways = {}

        for t, row in snap.iterrows():
            score = float(row["gate_score"]) if pd.notna(row["gate_score"]) else np.nan

            prev = bool(self.state.in_sideways.get(t, False))
            now = prev

            # Only update state if we have a valid score
            if np.isfinite(score):
                if (not prev) and (score >= cfg.gate_enter):
                    now = True
                elif prev and (score <= cfg.gate_exit):
                    now = False

            self.state.in_sideways[t] = now

        # Optional: drop series columns to avoid accidental downstream use
        # (keeps memory smaller if you store snap)
        # snap.drop(columns=["bb_bw_series", "slope_series"], inplace=True)

    # ------------------------------------------------------------------
    # Allocation: long-only BB mean reversion
    # ------------------------------------------------------------------

    def _allocate_mean_reversion(
        self, snap: pd.DataFrame, dropped_by_liquidity: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        cfg = self.config
        if self.state.positions is None:
            self.state.positions = {}

        max_pos = int(getattr(cfg, "max_positions", 0) or 0)
        if max_pos <= 0:
            max_pos = None  # treat as unlimited

        # ---------- 1) update exits for currently held ----------
        held = {t for t, v in self.state.positions.items() if v}

        for t in list(held):
            # If a held ticker is missing from the snapshot it may have been
            # removed by liquidity/universe filters. Only treat as a hard exit
            # when it's present in `dropped_by_liquidity` (explicitly removed).
            if t not in snap.index:
                if dropped_by_liquidity and (t in dropped_by_liquidity):
                    self.state.positions[t] = False
                    held.remove(t)
                    print(
                        f"[SidewaysSleeve] HARD EXIT {t} - removed by liquidity filters; closing position"
                    )
                    continue
                # Not in snapshot but not dropped by liquidity filters: warn and skip
                print(
                    f"[SidewaysSleeve] WARNING: Held ticker {t} missing from signals snapshot (not dropped by liquidity), skipping exit check"
                )
                continue

            z = float(snap["bb_z"].get(t, np.nan))
            if not np.isfinite(z):
                print(f"[SidewaysSleeve] WARNING: Non-finite z for {t}: {z}")
            gate_on = bool(self.state.in_sideways.get(t, False))

            done = np.isfinite(z) and (z >= -cfg.exit_z)
            hard_exit = bool(getattr(cfg, "exit_on_gate_off", False)) and (not gate_on)

            if done or hard_exit:
                self.state.positions[t] = False
                held.remove(t)
                print(
                    f"[SidewaysSleeve] EXIT {t} z={z:.2f} done={done} hard_exit={hard_exit} gate_on={gate_on}"
                )

        # ---------- 2) choose entries if capacity allows ----------
        entries = []
        for t, row in snap.iterrows():
            if t in held:
                continue

            z = float(row["bb_z"])
            if not np.isfinite(z):
                print(
                    f"[SidewaysSleeve] WARNING: Non-finite z for {t}: {z}, skipping entry"
                )
                continue

            gate_on = bool(self.state.in_sideways.get(t, False))
            if gate_on and (z <= -cfg.entry_z):
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
                entries = sorted(entries, key=lambda x: x[1], reverse=True)[:slots]
            else:
                entries = []
        # if unlimited, take all entries

        for t, _ in entries:
            self.state.positions[t] = True
            held.add(t)

        # ---------- 3) allocate among actually-held positions ----------
        if not held:
            return {}

        desired_scores = {}
        for t in held:
            z = float(snap["bb_z"].get(t, np.nan))
            strength = max(0.0, -z)

            invvol = 1.0
            if cfg.use_inverse_vol:
                v = float(snap["vol"].get(t, np.nan))
                if np.isfinite(v) and v > 0:
                    invvol = 1.0 / v

            desired_scores[t] = strength * invvol

        s = (
            pd.Series(desired_scores, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        if s.sum() <= 0:
            s[:] = 1.0

        w = (s / s.sum()).clip(upper=cfg.w_max_per_asset)
        w = w / w.sum()

        # keep state consistent with actual holdings
        for t in list(self.state.positions.keys()):
            self.state.positions[t] = t in w.index

        print(
            "held_all:",
            sorted([t for t, v in self.state.positions.items() if v]),
            "allocated:",
            list(w.index),
        )

        return {t: float(x) for t, x in w.items()}
