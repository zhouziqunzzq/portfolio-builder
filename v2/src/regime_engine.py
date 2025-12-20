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


RegimeName = Literal["bull", "correction", "bear", "crisis", "stress"]


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
    # NEW: volatility estimator mode
    vol_mode: str = "rolling"  # "rolling" | "ewm"
    ewm_vol_halflife: int = 20  # in trading days; if None, fall back to vol_window


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

        # primary = scores_norm.idxmax(axis=1).rename("primary_regime")
        # Dropping "stress" as it's not a primary regime label and is used only for blending weights at top-level.
        primary = (
            scores_norm[["bull", "correction", "bear", "crisis"]]
            .idxmax(axis=1)
            .rename("primary_regime")
        )
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

        def ramp(x, x0, x1):
            return ((x - x0) / (x1 - x0)).clip(lower=0.0, upper=1.0)

        # -------------------------------------------------
        # Bull: positive trend + not-too-high vol + shallow DD
        # -------------------------------------------------
        bull_trend = ramp(trend, 0.10, 0.60)
        bull_vol = 1.0 - ramp(vol_score, 0.50, 2.00)
        bull_dd = 1.0 - ramp(-dd_1y, 0.08, 0.25)  # slightly tighter than 10-30%

        bull_score = (0.50 * bull_trend + 0.30 * bull_vol + 0.20 * bull_dd).clip(
            0.0, 1.0
        )

        # -------------------------------------------------
        # Bear: negative trend + deeper DD + elevated vol + negative 12m mom
        # -------------------------------------------------
        bear_trend = ramp(-trend, 0.10, 0.60)
        bear_dd = ramp(-dd_1y, 0.15, 0.35)
        bear_vol = ramp(vol_score, 0.00, 1.50)
        bear_mom = ramp(-mom, 0.00, 0.20)

        bear_score = (
            0.40 * bear_trend + 0.35 * bear_dd + 0.15 * bear_vol + 0.10 * bear_mom
        ).clip(0.0, 1.0)

        # -------------------------------------------------
        # Crisis: very high vol and/or very deep DD (rare by design)
        # -------------------------------------------------
        crisis_vol = ramp(vol_score, 2.00, 3.50)
        crisis_dd = ramp(-dd_1y, 0.30, 0.50)

        crisis_score = (0.70 * crisis_vol + 0.30 * crisis_dd).clip(0.0, 1.0)

        # -------------------------------------------------
        # Correction: still “uptrend-ish” but under stress
        #   - positive-ish 12m mom (or not strongly negative)
        #   - trend still >= small positive
        #   - moderate DD and/or elevated vol
        # -------------------------------------------------
        corr_trend_ok = ramp(trend, 0.20, 0.40)
        corr_mom_ok = ramp(
            mom, -0.02, 0.10
        )  # allow small negative, but prefers positive
        corr_dd_mid = ramp(-dd_1y, 0.05, 0.20)
        corr_vol_up = ramp(vol_score, 0.20, 1.50)  # avoid firing on ultra-low vol

        # avoid calling it "correction" when trend is *very* strong (i.e. it's just bull)
        corr_not_strong_bull = 1.0 - ramp(trend, 0.40, 0.55)

        correction_score = (corr_trend_ok * corr_mom_ok * corr_not_strong_bull) * (
            0.50 * corr_dd_mid + 0.50 * corr_vol_up
        )
        correction_score = correction_score.clip(0.0, 1.0)

        # -----------------
        # Stress
        # -----------------
        # "How much should we lean into ballast sleeves?"
        stress_vol = ramp(
            vol_score, 0.75, 2.25
        )  # starts showing up when vol is elevated
        # penalize deep bear drawdowns — stress is not about trend collapse
        stress_dd_cap = 1.0 - ramp(-dd_1y, 0.20, 0.40)
        stress_score = (stress_vol * stress_dd_cap).clip(0.0, 1.0)

        # optional: reduce stress score when bear is already high
        stress_score *= 1.0 - ramp(bear_score, 0.50, 0.80)

        # print(trend.describe())
        # print(trend.quantile([0.05, 0.25, 0.5, 0.75, 0.95]))
        # print(trend.round(2).value_counts().head(10))
        # print("bull_trend")
        # print(bull_trend.describe())
        # print("bear_trend")
        # print(bear_trend.describe())
        # print("corr_trend_ok")
        # print(corr_trend_ok.describe())
        # print("corr_not_strong_bull")
        # print(corr_not_strong_bull.describe())
        # print("side_trend_flat")
        # print(side_trend_flat.describe())
        # print("Bear and Stress correlations:")
        # print(
        #     pd.DataFrame(
        #         {"bear_score": bear_score, "stress_score": stress_score}
        #     ).corr()
        # )

        scores = pd.DataFrame(
            {
                "bull": bull_score,
                "correction": correction_score,
                "bear": bear_score,
                "crisis": crisis_score,
                "stress": stress_score,
            },
            index=df.index,
        )
        scores[scores < 0] = 0.0

        # Calculate primary regime using scores
        # scores_copy = scores.copy()
        # scores_copy["primary_regime"] = scores_copy.idxmax(axis=1)
        # print(
        #     scores_copy["primary_regime"]
        #     .groupby([scores_copy.index.year // 10 * 10])
        #     .value_counts(normalize=True)
        # )

        return scores
