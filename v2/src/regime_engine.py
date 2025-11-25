from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from datetime import datetime
import numpy as np
import pandas as pd

from .signal_engine import SignalEngine


RegimeName = Literal["bull", "correction", "bear", "crisis", "sideways"]


@dataclass
class RegimeConfig:
    benchmark: str = "SPY"
    interval: str = "1d"
    vol_window: int = 20
    mom_window: int = 252
    fast_ma: int = 50
    slow_ma: int = 200
    dd_lookback: int = 252  # 1y trading days
    vol_norm_window: int = 252  # for z-scoring vol


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

        vol = self.signals.get_series(
            cfg.benchmark,
            "vol",
            start=start_dt,
            end=end_dt,
            interval=cfg.interval,
            window=cfg.vol_window,
        )

        mom = self.signals.get_series(
            cfg.benchmark,
            "ts_mom",
            start=start_dt,
            end=end_dt,
            interval=cfg.interval,
            window=cfg.mom_window,
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
                "mom_252": mom,
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
        """
        Heuristic, interpretable scoring rules.

        Each regime score is in [0, +inf); we will normalize later.
        The shape is mostly piecewise-linear around intuitive thresholds.
        """

        trend = df["trend_score"]
        vol_score = df["vol_score"]
        dd_1y = df["dd_1y"]
        mom = df["mom_252"]

        # Helper to create smooth-ish ramps
        def ramp(x, x0, x1):
            """Linear ramp: 0 at x<=x0, 1 at x>=x1."""
            return ((x - x0) / (x1 - x0)).clip(lower=0.0, upper=1.0)

        # Bull: positive trend, moderate vol, mild drawdown
        bull_trend = ramp(trend, 0.0, 0.5)  # stronger for higher trend_score
        bull_vol = 1.0 - ramp(vol_score, 0.5, 2.0)  # penalize very high vol
        bull_dd = 1.0 - ramp(-dd_1y, 0.15, 0.30)  # penalize deep drawdown

        bull_score = (bull_trend * 0.5) + (bull_vol * 0.3) + (bull_dd * 0.2)

        # Correction: long-term uptrend but noticeable drawdown & higher vol
        corr_trend = ramp(trend, -0.2, 0.3)  # still not strongly negative
        corr_dd = ramp(-dd_1y, 0.05, 0.20)  # 5~20% off highs
        corr_vol = ramp(vol_score, 0.0, 1.5)  # vol elevated but not panic
        corr_mom = ramp(mom, -0.05, 0.10)  # prefer >= 0, fade if very negative

        correction_score = (corr_trend * 0.3) + (corr_dd * 0.4) + (corr_vol * 0.3)
        correction_score *= corr_mom

        # Bear: negative trend, deep drawdown, elevated vol
        bear_trend = ramp(-trend, 0.2, 0.7)
        bear_dd = ramp(-dd_1y, 0.20, 0.40)
        bear_vol = ramp(vol_score, 0.0, 1.5)
        bear_mom = ramp(-mom, 0.0, 0.20)

        bear_score = (bear_trend * 0.4) + (bear_dd * 0.4) + (bear_vol * 0.2)
        bear_score = bear_score * 0.6 + bear_mom * 0.4

        # Crisis: very high vol and large recent drops
        # crisis_vol = ramp(vol_score, 1.5, 3.0)
        # crisis_dd = ramp(-dd_1y, 0.25, 0.50)
        crisis_vol = ramp(vol_score, 2.0, 3.5)  # require really high vol
        crisis_dd = ramp(-dd_1y, 0.30, 0.50)  # deeper drawdowns

        crisis_score = (crisis_vol * 0.7) + (crisis_dd * 0.3)

        # Sideways: weak trend, modest dd, normal-ish vol.
        # Compute as an independent "how sideways are we" measure.
        # side_trend_flat = 1.0 - ramp(trend.abs(), 0.2, 0.7)
        # side_dd = 1.0 - ramp(-dd_1y, 0.10, 0.30)
        # side_vol = 1.0 - ramp(vol_score.abs(), 0.5, 2.0)

        # sideways_score = (side_trend_flat * 0.5) + (side_dd * 0.3) + (side_vol * 0.2)

        # Sideways - alternative: make it the residual regime
        non_side = bull_score + correction_score + bear_score + crisis_score
        sideways_score = (1.0 - non_side.clip(0, 1.0)).clip(lower=0.0)

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

        # Avoid negative scores due to numerical noise
        scores[scores < 0] = 0.0

        return scores
