from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from datetime import datetime
import numpy as np
import pandas as pd

# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from signal_engine import SignalEngine


RegimeName = Literal["bull", "correction", "bear", "crisis", "sideways"]


@dataclass
class RegimeConfig:
    benchmark: str = "SPY"
    interval: str = "1d"
    vol_window: int = 20
    mom_window: int = 252
    mom_fast_window: int = 63
    fast_ma: int = 50
    slow_ma: int = 200
    dd_lookback: int = 252  # 1y trading days
    vol_norm_window: int = 252  # for z-scoring vol
    # volatility estimator mode
    vol_mode: str = "rolling"  # "rolling" | "ewm"
    ewm_vol_halflife: int = 20  # in trading days; if None, fall back to vol_window
    # rates/bonds stress veto
    bond_benchmark: str = "BND"  # or "IEF"
    bond_trend_window: int = 200
    bond_mom_window: int = 126


class RegimeEngine:
    def __init__(self, signals: SignalEngine, config: RegimeConfig | None = None):
        self.signals = signals
        self.config = config or RegimeConfig()

    def get_regime_frame(
        self,
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by date with:
          - features: trend_score, vol, vol_score, dd_1y, mom_252
          - regime scores: bull_score, correction_score, bear_score, crisis_score, sideways_score
          - primary_regime (string label)
        """
        cfg = self.config
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # --- 1) Pull benchmark price and core signals via SignalEngine/MarketDataStore ---

        # We can grab OHLCV directly from the MarketDataStore behind SignalEngine
        df_px = self.signals.mds.get_ohlcv(
            ticker=cfg.benchmark,
            start=start_dt,
            end=end_dt,
            interval=cfg.interval,
        )
        if df_px.empty:
            raise ValueError(
                f"No price data for {cfg.benchmark} in [{start_dt}, {end_dt}]"
            )

        close = df_px["Close"].astype(float)
        trend = self.signals.get_series(
            cfg.benchmark,
            "trend_score",
            start=start_dt,
            end=end_dt,
            interval=cfg.interval,
            fast_window=cfg.fast_ma,
            slow_window=cfg.slow_ma,
        )
        vol: pd.Series | None = None
        vol_mode = cfg.vol_mode
        if vol_mode == "ewm":
            vol = self.signals.get_series(
                cfg.benchmark,
                "ewm_vol",
                start=start_dt,
                end=end_dt,
                interval=cfg.interval,
                halflife=cfg.ewm_vol_halflife,
            )
        else:  # default to "rolling"
            vol = self.signals.get_series(
                cfg.benchmark,
                "vol",
                start=start_dt,
                end=end_dt,
                interval=cfg.interval,
                window=cfg.vol_window,
            )
        mom_252 = self.signals.get_series(
            cfg.benchmark,
            "ts_mom",
            start=start_dt,
            end=end_dt,
            interval=cfg.interval,
            window=cfg.mom_window,
        )
        mom_63 = self.signals.get_series(
            cfg.benchmark,
            "ts_mom",
            start=start_dt,
            end=end_dt,
            interval=cfg.interval,
            window=cfg.mom_fast_window,
        )

        # Bond/rates trend & momentum for stress veto
        bond_trend = self.signals.get_series(
            cfg.bond_benchmark,
            "trend_score",
            start=start_dt,
            end=end_dt,
            interval=cfg.interval,
            fast_window=cfg.fast_ma,
            slow_window=cfg.bond_trend_window,
        )
        bond_mom = self.signals.get_series(
            cfg.bond_benchmark,
            "ts_mom",
            start=start_dt,
            end=end_dt,
            interval=cfg.interval,
            window=cfg.bond_mom_window,
        )

        # --- 2) Derived features: drawdown & normalized vol ---

        rolling_max = close.rolling(cfg.dd_lookback, min_periods=1).max()
        dd_1y = close / rolling_max - 1.0

        vol_mean = vol.rolling(cfg.vol_norm_window, min_periods=20).mean()
        vol_std = vol.rolling(cfg.vol_norm_window, min_periods=20).std()
        vol_score = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # Align everything in a single DataFrame
        features = pd.DataFrame(
            {
                "close": close,
                "trend_score": trend,
                "vol": vol,
                "vol_score": vol_score,
                "dd_1y": dd_1y,
                "mom_63": mom_63,
                "mom_252": mom_252,
                "bond_trend_score": bond_trend,
                "bond_mom_126": bond_mom,
            }
        ).dropna(subset=["trend_score", "vol", "vol_score", "dd_1y", "mom_252"])

        # --- 3) Compute regime scores (soft membership) ---
        scores = self._compute_regime_scores(features)

        # --- 4) Normalize scores and choose primary regime ---
        score_cols = list(scores.columns)
        score_sum = scores[score_cols].sum(axis=1).replace(0, np.nan)
        scores_norm = scores.div(score_sum, axis=0)

        primary = scores_norm.idxmax(axis=1).rename("primary_regime")
        out = pd.concat([features, scores_norm, primary], axis=1)

        return out

    # -------------------------------------------------
    # Internal: fuzzy membership rules for each regime
    # -------------------------------------------------

    def _compute_regime_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        trend = df["trend_score"]
        vol_score = df["vol_score"]
        dd_1y = df["dd_1y"]
        mom = df["mom_252"]
        mom_fast = df["mom_63"]
        bond_trend = df["bond_trend_score"]
        bond_mom = df["bond_mom_126"]

        def ramp(x, x0, x1):
            return ((x - x0) / (x1 - x0)).clip(lower=0.0, upper=1.0)

        # Bond stress veto
        bond_break = ramp(-bond_mom, 0.00, 0.10) * ramp(-bond_trend, 0.10, 0.60)

        # -----------------
        # Bull
        # -----------------
        bull_trend = ramp(trend, 0.1, 0.7)  # need a clearer positive trend
        bull_vol = 1.0 - ramp(vol_score, 0.5, 2.0)  # penalize high vol
        bull_dd = 1.0 - ramp(-dd_1y, 0.10, 0.30)  # penalize 10-30% DD

        bull_mom_ok = 1.0 - ramp(-mom, 0.00, 0.10)  # penalize negative 12m mom
        bull_score = (
            (bull_trend * 0.45)
            + (bull_vol * 0.30)
            + (bull_dd * 0.15)
            + (bull_mom_ok * 0.10)
        )
        # Boost bull score when momentum is strong
        bull_score *= 1.0 + 0.25 * ramp(mom, 0.08, 0.15)

        # -----------------
        # Bear
        # -----------------
        bear_trend = ramp(-trend, 0.2, 0.7)
        bear_dd = ramp(-dd_1y, 0.20, 0.40)
        bear_vol = ramp(vol_score, 0.0, 1.5)
        bear_mom = ramp(-mom, 0.0, 0.20)

        bear_core = (bear_trend * 0.4) + (bear_dd * 0.4) + (bear_vol * 0.2)
        bear_score = bear_core * 0.6 + bear_mom * 0.4
        bear_score *= 1.0 + 0.30 * bond_break  # up to +30%

        # -----------------
        # Crisis
        # -----------------
        crisis_vol = ramp(vol_score, 2.0, 3.5)
        crisis_dd = ramp(-dd_1y, 0.30, 0.50)
        crisis_score = (crisis_vol * 0.7) + (crisis_dd * 0.3)

        # -----------------
        # Correction
        # -----------------
        # 1) Long-term uptrend: positive momentum / trend
        corr_uptrend = ramp(mom, 0.02, 0.12)
        corr_trend_ok = ramp(trend, 0.15, 0.55)
        # but not TOO strong (otherwise it should just be bull)
        corr_not_strong_bull = 1.0 - ramp(trend, 0.7, 1.0)

        # 2) Mild drawdown: 5-20% below 1y high, but fade out >25%
        corr_dd_mid = ramp(-dd_1y, 0.05, 0.20)
        corr_dd_not_deep = 1.0 - ramp(-dd_1y, 0.25, 0.40)
        corr_dd = corr_dd_mid * corr_dd_not_deep

        # 3) Vol: somewhat elevated, but explicitly *not* crisis-level
        corr_vol_up = ramp(vol_score, 0.0, 1.5)
        corr_vol_not_panic = 1.0 - ramp(vol_score, 2.0, 3.5)
        corr_vol = corr_vol_up * corr_vol_not_panic

        # Base correction score
        correction_score = corr_uptrend * corr_trend_ok * corr_not_strong_bull
        correction_score *= 0.5 * corr_dd + 0.5 * corr_vol

        # 4) Explicitly suppress correction when bear/crisis is strong
        #    (use the unnormalized bear/crisis scores we just computed)
        suppress_bear = 1.0 - ramp(bear_score, 0.3, 0.8)
        suppress_crisis = 1.0 - ramp(crisis_score, 0.2, 0.6)
        correction_score *= suppress_bear * suppress_crisis

        # 5) Prefer when short-term momentum is not strongly negative
        corr_fast_mom_ok = ramp(mom_fast, -0.02, 0.05)  # prefer not strongly negative
        correction_score *= corr_fast_mom_ok

        # 6) Suppress correction if bonds are also breaking down
        correction_score *= 1.0 - 0.50 * bond_break

        # -----------------
        # Sideways as residual
        # -----------------
        # non_side = bull_score + correction_score + bear_score + crisis_score
        # sideways_score = (1.0 - non_side.clip(0.0, 1.0)).clip(lower=0.0)
        # -----------------
        # Sideways (explicit, not residual)
        # -----------------
        side_trend_flat = 1.0 - ramp(trend.abs(), 0.10, 0.35)
        side_vol_low = 1.0 - ramp(vol_score, 0.25, 1.25)
        side_dd_shallow = 1.0 - ramp(-dd_1y, 0.03, 0.12)
        sideways_score = (
            side_trend_flat * 0.45 + side_vol_low * 0.35 + side_dd_shallow * 0.20
        ).clip(lower=0.0, upper=1.0)
        # Suppress sideways if bonds are breaking down
        sideways_score *= 1.0 - 0.30 * bond_break

        scores = pd.DataFrame(
            {
                "bull": bull_score,
                "correction": correction_score,
                "bear": bear_score,
                "crisis": crisis_score,
                "sideways": sideways_score,
            },
            index=df.index,
        )
        scores[scores < 0] = 0.0
        return scores
