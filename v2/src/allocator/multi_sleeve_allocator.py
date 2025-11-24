from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Mapping, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd

from src.regime_engine import RegimeEngine
from .multi_sleeve_config import MultiSleeveConfig


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

        # 4) Normalize global weights for safety / optional cash preservation
        total = float(sum(combined.values()))
        if total <= 0:
            out: Dict[str, float] = {}
        else:
            # If config allows preserving cash and total <= 1.0, do NOT scale up.
            # This leaves leftover (1-total) as implicit cash.
            if self.config.preserve_cash_if_under_target and total <= 1.0:
                out = {t: float(w) for t, w in combined.items() if w > 0}
            else:
                # If total > 1.0 or cash preservation disabled, scale to 1.
                out = {t: w / total for t, w in combined.items()}

        # store state
        self.last_as_of = as_of_ts
        self.last_primary_regime = primary_regime
        self.last_regime_scores = regime_scores
        self.last_sleeve_weights = sleeve_alloc
        self.last_portfolio = out

        return out

    # ------------------------------------------------------------------
    # Vectorized Precompute for All Sleeves
    # ------------------------------------------------------------------
    def precompute(
        self,
        start: datetime | str,
        end: datetime | str,
        rebalance_dates: Optional[list[datetime | str]] = None,
        warmup_buffer: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """Precompute sleeve-level weight matrices over [start, end].

        For each managed sleeve that exposes a `precompute` method with the
        signature matching TrendSleeve / DefensiveSleeve expectations, we invoke
        it and store the resulting Date x Ticker weight matrix.

        Parameters
        ----------
        start, end : datetime | str
            Target inclusive date range for final (post-warmup) weights.
        rebalance_dates : list[datetime | str], optional
            If provided, sleeves will sample their internal matrix to only
            those dates.
        warmup_buffer : int
            Extra days added to max signal window during sleeve warmup.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping sleeve_name -> precomputed weight matrix.
        """
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        if end_ts < start_ts:
            raise ValueError("end must be >= start for precompute")

        print(f"[MultiSleeveAllocator] Starting precompute for sleeves over [{start_ts.date()}, {end_ts.date()}]")
        if rebalance_dates is not None:
            print(f"[MultiSleeveAllocator] Rebalance dates supplied: {len(rebalance_dates)}")

        results: Dict[str, pd.DataFrame] = {}

        for name, sleeve in self.sleeves.items():
            if not hasattr(sleeve, "precompute"):
                print(f"[MultiSleeveAllocator] Sleeve '{name}' has no precompute(); skipping.")
                continue
            print(f"[MultiSleeveAllocator] Precomputing sleeve '{name}' ...")
            try:
                wmat = sleeve.precompute(
                    start=start_ts,
                    end=end_ts,
                    rebalance_dates=rebalance_dates,
                    warmup_buffer=warmup_buffer,
                )
                if wmat is None or wmat.empty:
                    print(f"[MultiSleeveAllocator] Sleeve '{name}' returned empty matrix.")
                else:
                    print(
                        f"[MultiSleeveAllocator] Sleeve '{name}' weights shape: {wmat.shape} (dates={wmat.shape[0]}, tickers={wmat.shape[1]})"
                    )
                results[name] = wmat.copy() if wmat is not None else pd.DataFrame()
            except Exception as e:
                print(f"[MultiSleeveAllocator] Sleeve '{name}' precompute failed: {e}")
                results[name] = pd.DataFrame()

        print("[MultiSleeveAllocator] Precompute finished.")
        return results

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
