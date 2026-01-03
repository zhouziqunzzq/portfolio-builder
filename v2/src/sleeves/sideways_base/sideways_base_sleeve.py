from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set
from datetime import datetime

import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[2]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from universe_manager import UniverseManager
from market_data_store import MarketDataStore
from signal_engine import SignalEngine
from sleeves.base import BaseSleeve
from sleeves.common.rebalance_helpers import should_rebalance
from context.rebalance import RebalanceContext

from .sideways_base_config import SidewaysBaseConfig
from states.base_state import BaseState


@dataclass
class SidewaysBaseState(BaseState):
    STATE_KEY = "sleeve.sideways_base"
    SCHEMA_VERSION = 1

    last_rebalance_ts: Optional[pd.Timestamp] = None
    last_weights: Optional[Dict[str, float]] = None

    def to_payload(self) -> Dict[str, Any]:
        last_rebalance_ts = (
            self.last_rebalance_ts.isoformat()
            if self.last_rebalance_ts is not None
            else None
        )
        last_weights = (
            {str(k): float(v) for k, v in self.last_weights.items()}
            if self.last_weights is not None
            else None
        )
        return {
            "last_rebalance_ts": last_rebalance_ts,
            "last_weights": last_weights,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SidewaysBaseState":
        raw_ts = payload.get("last_rebalance_ts")
        last_rebalance_ts = pd.to_datetime(raw_ts) if raw_ts else None

        raw_weights = payload.get("last_weights")
        last_weights = None
        if isinstance(raw_weights, Mapping):
            last_weights = {str(k): float(v) for k, v in raw_weights.items()}

        return cls(
            last_rebalance_ts=last_rebalance_ts,
            last_weights=last_weights,
        )

    @classmethod
    def empty(cls) -> "SidewaysBaseState":
        return cls()


class SidewaysBaseSleeve(BaseSleeve):
    """
    Sideways Base Sleeve (Option A) — long-only whipsaw insurance.

    ETFs only; score is:
      - low realized vol (dominant)
      - low max drawdown (dominant)
      - small positive drift (tie-breaker)

    Then allocate by asset class (equity/bond/gold/cash) with slow rebalancing.
    """

    def __init__(
        self,
        mds: MarketDataStore,
        signals: SignalEngine,
        config: Optional[SidewaysBaseConfig] = None,
    ):
        super().__init__(
            market_data_store=mds,
            universe_manager=None,
            signal_engine=signals,
            vectorized_signal_engine=None,
        )
        # Aliases for convenience
        self.mds = self.market_data_store
        self.signals = self.signal_engine

        self.config = config or SidewaysBaseConfig()
        self.state = SidewaysBaseState()

        # Logger (instance-level, mirror DefensiveSleeve pattern)
        self.log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self, as_of: Optional[datetime | str] = None) -> Set[str]:
        """
        Get the universe, i.e. all tickers tradable for the sleeve.
        If as_of is provided, get the universe as-of that date. Otherwise, return the
        universe of all time (including tickers no longer in the index).
        """
        return set(self._get_sideways_universe())

    def _get_sideways_universe(self) -> List[str]:
        # ETFs only by design (robust + low churn)
        return self.config.sideways_etfs

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
                    continue

                last_price = float(df["Close"].iloc[-1])
                if last_price < cfg.min_price:
                    continue

                if "Volume" not in df.columns:
                    continue

                adv = df["Volume"].rolling(cfg.min_adv_window).mean().iloc[-1]
                if pd.isna(adv) or float(adv) < cfg.min_adv:
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
        start_for_signals: datetime | str,
        regime: str = "sideways",
        rebalance_ctx: Optional[RebalanceContext] = None,
    ) -> Dict[str, float]:
        as_of = pd.to_datetime(as_of)
        _ = pd.to_datetime(start_for_signals)  # kept for interface consistency
        cfg = self.config

        # ---------- Rebalance timing ----------
        reb_ts = rebalance_ctx.rebalance_ts if rebalance_ctx is not None else as_of
        if not self.should_rebalance(reb_ts):
            self.log.info(
                "Skipping rebalance at %s; last rebalance at %s",
                reb_ts.date(),
                (
                    self.state.last_rebalance_ts.date()
                    if self.state.last_rebalance_ts is not None
                    else "never"
                ),
            )
            return self.state.last_weights

        self.log.info(
            "Rebalancing at %s using data as of %s; last rebalance at %s",
            reb_ts.date(),
            as_of.date(),
            (
                self.state.last_rebalance_ts.date()
                if self.state.last_rebalance_ts is not None
                else "never"
            ),
        )

        universe = self._get_sideways_universe()
        universe = self._apply_liquidity_filters(universe, as_of)
        if not universe:
            return {}

        sigs = self._compute_signals_snapshot(universe, as_of)
        if sigs.empty:
            return {}

        scored = self._compute_scores_only(sigs)
        weights = self.allocate_by_asset_class(scored, regime=regime)

        self.state.last_rebalance_ts = pd.Timestamp(reb_ts)
        self.state.last_weights = weights
        return weights

    def should_rebalance(
        self,
        now: datetime | str,
    ) -> bool:
        if self.state.last_weights is None:
            # Never rebalanced before -> must rebalance
            return True

        now = pd.to_datetime(now)
        cfg = self.config
        return should_rebalance(self.state.last_rebalance_ts, now, cfg.rebalance_freq)

    def get_last_rebalance_datetime(self) -> Optional[datetime]:
        return (
            self.state.last_rebalance_ts.to_pydatetime()
            if self.state.last_rebalance_ts is not None
            else None
        )

    # ------------------------------------------------------------------
    # Signal computation (MVP uses OHLCV directly)
    # ------------------------------------------------------------------

    def _compute_signals_snapshot(
        self, tickers: List[str], end: pd.Timestamp
    ) -> pd.DataFrame:
        cfg = self.config
        buffer = pd.Timedelta(days=cfg.signals_extra_buffer_days)

        rows = []
        for t in tickers:
            try:
                # Need enough history for dd_window + slope_window + vol_window
                need = max(cfg.dd_window, cfg.slope_window, cfg.vol_window)
                start = (
                    end - pd.Timedelta(days=int(need * 2.2)) - buffer
                )  # ~business-day cushion

                df = self.mds.get_ohlcv(
                    ticker=t,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                )
                if df is None or df.empty or "Close" not in df.columns:
                    continue

                close = df["Close"].dropna()
                if len(close) < need + 5:
                    continue

                rets = close.pct_change().dropna()

                vol = self._realized_vol(
                    rets, window=cfg.vol_window, ann_factor=cfg.ann_factor
                )
                mdd = self._max_drawdown(close, window=cfg.dd_window)
                slope = self._trend_slope(close, window=cfg.slope_window)

                rows.append(
                    {
                        "ticker": t,
                        "vol": vol,
                        "mdd": mdd,
                        "slope": slope,
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(rows).set_index("ticker")
        return df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["vol", "mdd", "slope"]
        )

    @staticmethod
    def _realized_vol(returns: pd.Series, window: int, ann_factor: int) -> float:
        v = returns.rolling(window).std().iloc[-1]
        return float(v) * np.sqrt(ann_factor) if pd.notna(v) else np.nan

    @staticmethod
    def _max_drawdown(close: pd.Series, window: int) -> float:
        s = close.iloc[-window:].copy()
        peak = s.cummax()
        dd = (s / peak) - 1.0
        mdd = dd.min()
        return float(mdd) if pd.notna(mdd) else np.nan  # negative number

    @staticmethod
    def _trend_slope(close: pd.Series, window: int) -> float:
        """
        Simple linear slope on log-prices over `window` days.
        Returns daily slope in log space (small number).
        """
        s = close.iloc[-window:].copy()
        y = np.log(s.values.astype(float))
        x = np.arange(len(y), dtype=float)
        # robust enough for MVP: ordinary least squares via polyfit
        b = np.polyfit(x, y, 1)[0]
        return float(b)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_scores_only(self, sigs: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = sigs.copy()

        def r01(s: pd.Series, ascending: bool) -> pd.Series:
            return s.rank(method="average", ascending=ascending, pct=True)

        # lower vol is better
        df["r_vol"] = r01(df["vol"], ascending=True)

        # drawdown: mdd is negative; "less negative" is better -> sort descending
        df["r_mdd"] = r01(df["mdd"], ascending=False)

        # slope: higher is better (but only tie-breaker)
        df["r_slope"] = r01(df["slope"], ascending=False)

        df["score"] = (
            cfg.w_low_vol * df["r_vol"]
            + cfg.w_low_dd * df["r_mdd"]
            + cfg.w_pos_slope * df["r_slope"]
        )

        return df.sort_values("score", ascending=False)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_by_asset_class(
        self, scored: pd.DataFrame, regime: str
    ) -> Dict[str, float]:
        cfg = self.config

        def assign_class(t: str) -> str:
            return cfg.asset_class_for_etf.get(t, "bond")

        df = scored.copy()
        df["asset_class"] = [assign_class(t) for t in df.index]

        # Select global top-k first (MVP)
        df = df.sort_values("score", ascending=False).head(min(cfg.top_k, len(df)))

        class_alloc = cfg.asset_class_allocations
        final: Dict[str, float] = {}

        # Allocate within each class proportional to score / vol (soft risk-adjust)
        for asset_class, class_w in class_alloc.items():
            if class_w <= 0:
                continue
            sub = df[df["asset_class"] == asset_class]
            if sub.empty:
                continue

            raw = sub["score"] / sub["vol"]
            raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if raw.sum() <= 0:
                w = pd.Series(1.0, index=sub.index)
            else:
                w = raw / raw.sum()

            for t, w_i in w.items():
                final[t] = final.get(t, 0.0) + class_w * float(w_i)

        # normalize
        total = sum(final.values())
        if total <= 0:
            return {}

        final = {t: w / total for t, w in final.items()}

        # cap per name and renormalize (simple iterative clip once)
        final = self._cap_and_renorm(final, cap=cfg.max_weight_per_name)

        # optional: drop tiny weights
        if cfg.min_weight_cutoff > 0:
            final = {t: w for t, w in final.items() if w >= cfg.min_weight_cutoff}
            total = sum(final.values())
            if total > 0:
                final = {t: w / total for t, w in final.items()}

        return final

    @staticmethod
    def _cap_and_renorm(w: Dict[str, float], cap: float) -> Dict[str, float]:
        if not w:
            return w
        w2 = dict(w)
        # clip once, then renorm leftover
        over = {t: x for t, x in w2.items() if x > cap}
        if not over:
            return w2

        clipped_sum = 0.0
        for t in over:
            clipped_sum += cap
            w2[t] = cap

        under = {t: x for t, x in w2.items() if x < cap}
        under_sum = sum(under.values())
        if under_sum <= 0:
            # everything capped -> just renorm
            s = sum(w2.values())
            return {t: x / s for t, x in w2.items()} if s > 0 else {}

        # scale unders to fill remaining mass
        remaining = 1.0 - clipped_sum
        scale = remaining / under_sum
        for t in under:
            w2[t] = w2[t] * scale

        # final renorm for numerical safety
        s = sum(w2.values())
        return {t: x / s for t, x in w2.items()} if s > 0 else {}

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------
    def precompute(
        self,
        start: datetime | str,
        end: datetime | str,
        sample_dates: Optional[List[datetime | str]] = None,
        warmup_buffer: Optional[int] = None,  # in days
    ) -> pd.DataFrame:
        # Not implemented for this sleeve
        self.log.debug("Precompute not implemented; returning empty DataFrame")
        return pd.DataFrame()
