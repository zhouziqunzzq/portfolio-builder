from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Mapping, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd

from src.regime_engine import RegimeEngine


@dataclass
class MultiSleeveConfig:
    """
    Configuration for the multi-sleeve allocator.

    Later you can wire this to v2/config/strategy.yml; for now we keep it
    hard-coded and simple.
    """

    # How far back sleeves should look for signals by default
    signal_lookback_days: int = 365

    # How far back to feed into RegimeEngine.get_regime_frame
    regime_lookback_days: int = 252

    # Names of regime-score columns in RegimeEngine output
    regime_score_columns: Tuple[str, ...] = (
        "bull",
        "correction",
        "bear",
        "crisis",
        "sideways",
    )

    # Mapping: regime_name -> {sleeve_name -> weight}
    # These are top-level weights per regime. The allocator will
    # blend them using regime scores, then normalize across sleeves.
    sleeve_regime_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            # With only the defensive sleeve wired initially, every regime
            # just maps to defensive: 1.0. You can extend this later.
            "bull": {
                "defensive": 1.0,
            },
            "correction": {
                "defensive": 1.0,
            },
            "bear": {
                "defensive": 1.0,
            },
            "crisis": {
                "defensive": 1.0,
            },
            "sideways": {
                "defensive": 1.0,
            },
        }
    )

    def get_primary_sleeve_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """
        Fallback: return normalized sleeve weights for a *single* regime.

        Used when we don't have regime scores or when blending fails.
        """
        if regime not in self.sleeve_regime_weights:
            # fallback order: sideways -> bull -> equal-weight-over-all-configured
            for fb in ("sideways", "bull"):
                if fb in self.sleeve_regime_weights:
                    regime = fb
                    break

        mapping = self.sleeve_regime_weights.get(regime, {})
        if not mapping:
            # equal-weight across all sleeves mentioned anywhere
            all_sleeves: Dict[str, float] = {}
            for rm, m in self.sleeve_regime_weights.items():
                for s in m.keys():
                    all_sleeves[s] = 1.0
            mapping = all_sleeves

        total = float(sum(mapping.values()))
        if total <= 0:
            return {}
        return {name: w / total for name, w in mapping.items()}


class MultiSleeveAllocator:
    """
    Top-level allocator that:

      - uses RegimeEngine.get_regime_frame() to get both:
            * primary_regime label
            * regime_score distribution across regimes

      - computes *effective* sleeve weights as a regime-score-weighted blend
        of per-regime sleeve allocations.

      - asks each sleeve for internal target weights
      - combines into a global {ticker -> weight} portfolio.
    """

    def __init__(
        self,
        regime_engine: RegimeEngine,
        sleeves: Mapping[str, Any],
        config: Optional[MultiSleeveConfig] = None,
    ):
        """
        Parameters
        ----------
        regime_engine:
            Instance of RegimeEngine (v2/src/regime_engine.py)

        sleeves:
            Dict mapping sleeve_name -> sleeve instance.
            Each sleeve must implement:
                generate_target_weights_for_date(as_of, start_for_signals, regime)

            The `regime` argument passed to each sleeve will be the *primary*
            regime label (for any local regime-dependent logic inside sleeves).

        config:
            MultiSleeveConfig instance. If None, default config is used.
        """
        self.regime_engine = regime_engine
        self.sleeves: Dict[str, Any] = dict(sleeves)
        self.config = config or MultiSleeveConfig()

        # Optional state
        self.last_as_of: Optional[pd.Timestamp] = None
        self.last_primary_regime: Optional[str] = None
        self.last_regime_scores: Optional[Dict[str, float]] = None
        self.last_sleeve_weights: Optional[Dict[str, float]] = None
        self.last_portfolio: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_global_target_weights(
        self,
        as_of: datetime | str,
        start_for_signals: Optional[datetime | str] = None,
    ) -> Dict[str, float]:
        """
        Main entry point for V2 multi-sleeve allocation.

        Steps:
            1) Query RegimeEngine over a lookback window.
            2) Extract primary_regime and regime_score distribution.
            3) Blend per-regime sleeve weights using regime scores.
            4) Ask each sleeve for internal weights and combine.
        """
        as_of_ts = pd.to_datetime(as_of)

        # Figure out how much history sleeves need for signals
        if start_for_signals is None:
            start_for_signals = as_of_ts - timedelta(
                days=self.config.signal_lookback_days
            )
        start_for_signals_ts = pd.to_datetime(start_for_signals)

        # 1) Get regime context (primary label + score distribution)
        primary_regime, regime_scores = self._get_regime_context(as_of_ts)

        # 2) Compute effective sleeve weights via regime-score blending
        sleeve_alloc = self._compute_effective_sleeve_weights(
            primary_regime, regime_scores
        )
        if not sleeve_alloc:
            return {}

        # 3) Query sleeves and combine
        combined: Dict[str, float] = {}

        for name, sleeve in self.sleeves.items():
            alloc = sleeve_alloc.get(name, 0.0)
            if alloc <= 0:
                continue

            local_weights = sleeve.generate_target_weights_for_date(
                as_of=as_of_ts,
                start_for_signals=start_for_signals_ts,
                regime=primary_regime,
            )
            if not local_weights:
                continue

            for ticker, w in local_weights.items():
                combined[ticker] = combined.get(ticker, 0.0) + alloc * w

        # 4) Normalize global weights for safety
        total = float(sum(combined.values()))
        if total <= 0:
            out: Dict[str, float] = {}
        else:
            out = {t: w / total for t, w in combined.items()}

        # store state
        self.last_as_of = as_of_ts
        self.last_primary_regime = primary_regime
        self.last_regime_scores = regime_scores
        self.last_sleeve_weights = sleeve_alloc
        self.last_portfolio = out

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_regime_context(self, as_of: pd.Timestamp) -> Tuple[str, Dict[str, float]]:
        """
        Call RegimeEngine.get_regime_frame() and return:
            - primary_regime label (string)
            - regime_scores: {regime_name -> score in [0,1], sum ~ 1}
        """
        lookback_days = self.config.regime_lookback_days
        start = as_of - timedelta(days=lookback_days)

        df = self.regime_engine.get_regime_frame(start=start, end=as_of)
        if df is None or df.empty:
            # Fallback: sideways, no scores
            return "sideways", {}

        last = df.iloc[-1]

        # Primary regime label
        primary = str(last.get("primary_regime", "sideways"))

        # Regime scores (soft assignment)
        scores_raw: Dict[str, float] = {}
        for col in self.config.regime_score_columns:
            if col in last.index:
                val = last[col]
                try:
                    f = float(val)
                except Exception:
                    continue
                if pd.isna(f):
                    continue
                scores_raw[col] = max(f, 0.0)

        # Map from "bull_score" -> "bull" if needed
        regime_scores: Dict[str, float] = {}
        for key, val in scores_raw.items():
            name = key.replace("_score", "")
            regime_scores[name] = regime_scores.get(name, 0.0) + val

        total = float(sum(regime_scores.values()))
        if total <= 0:
            # If no usable scores, just return primary with full weight
            return primary, {primary: 1.0}

        regime_scores = {k: v / total for k, v in regime_scores.items()}
        return primary, regime_scores

    def _compute_effective_sleeve_weights(
        self,
        primary_regime: str,
        regime_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Blend per-regime sleeve weights using the regime_scores distribution.

        effective_sleeve_weight[sleeve] =
            sum_over_regimes( regime_scores[r] * sleeve_regime_weights[r][sleeve] )

        If blending fails (e.g., no scores), fall back to single-regime weights.
        """
        # If no regime scores, use primary-regime weights directly
        if not regime_scores:
            return self.config.get_primary_sleeve_weights_for_regime(primary_regime)

        # Soft-blended sleeve allocations
        accum: Dict[str, float] = {}

        for regime_name, r_weight in regime_scores.items():
            if r_weight <= 0:
                continue
            per_regime = self.config.sleeve_regime_weights.get(regime_name, {})
            if not per_regime:
                continue
            for sleeve_name, w in per_regime.items():
                accum[sleeve_name] = accum.get(sleeve_name, 0.0) + r_weight * w

        if not accum:
            # Fallback if no regime in sleeve_regime_weights matched
            return self.config.get_primary_sleeve_weights_for_regime(primary_regime)

        total = float(sum(accum.values()))
        if total <= 0:
            return {}

        return {sleeve: w / total for sleeve, w in accum.items()}
