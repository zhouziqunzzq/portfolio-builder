from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Mapping, Any, Set, Tuple
from datetime import datetime, timedelta
import math

import pandas as pd
import logging

import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from regime_engine import RegimeEngine
from signal_engine import SignalEngine
from friction_control.friction_controller import FrictionController
from context.rebalance import RebalanceContext
from context.friction_control import FrictionControlContext
from sleeves.base import BaseSleeve
from sleeves.common.rebalance_helpers import should_rebalance
from .multi_sleeve_config import MultiSleeveConfig
from states.base_state import BaseState


@dataclass
class MultiSleeveAllocatorState(BaseState):
    STATE_KEY = "allocator.multi_sleeve"
    SCHEMA_VERSION = 1

    # Regime engine states
    last_regime_sample_ts: Optional[pd.Timestamp] = None
    last_regime_context: Optional[Tuple[str, Dict[str, float]]] = None

    # Trend filter states
    last_trend_sample_ts: Optional[pd.Timestamp] = None
    last_trend_status: Optional[str] = None

    last_as_of: Optional[pd.Timestamp] = None
    last_rebalance_ts: Optional[pd.Timestamp] = None
    last_sleeve_weights: Optional[Dict[str, float]] = None
    last_portfolio: Optional[Dict[str, float]] = None

    def to_payload(self) -> Dict[str, Any]:
        def _ts(x: Optional[pd.Timestamp]) -> Optional[str]:
            return x.isoformat() if x is not None else None

        regime_ctx = None
        if self.last_regime_context is not None:
            primary, scores = self.last_regime_context
            regime_ctx = {
                "primary": str(primary),
                "scores": {str(k): float(v) for k, v in dict(scores).items()},
            }
        sleeve_w = (
            {str(k): float(v) for k, v in self.last_sleeve_weights.items()}
            if self.last_sleeve_weights is not None
            else None
        )
        portfolio_w = (
            {str(k): float(v) for k, v in self.last_portfolio.items()}
            if self.last_portfolio is not None
            else None
        )
        return {
            "last_regime_sample_ts": _ts(self.last_regime_sample_ts),
            "last_regime_context": regime_ctx,
            "last_trend_sample_ts": _ts(self.last_trend_sample_ts),
            "last_trend_status": self.last_trend_status,
            "last_as_of": _ts(self.last_as_of),
            "last_rebalance_ts": _ts(self.last_rebalance_ts),
            "last_sleeve_weights": sleeve_w,
            "last_portfolio": portfolio_w,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MultiSleeveAllocatorState":
        def _pdt(x: Any) -> Optional[pd.Timestamp]:
            return pd.to_datetime(x) if x else None

        last_regime_sample_ts = _pdt(payload.get("last_regime_sample_ts"))
        last_trend_sample_ts = _pdt(payload.get("last_trend_sample_ts"))
        last_as_of = _pdt(payload.get("last_as_of"))
        last_rebalance_ts = _pdt(payload.get("last_rebalance_ts"))

        # Regime context: preferred dict form, but accept legacy tuple/list
        last_regime_context = None
        raw_rc = payload.get("last_regime_context")
        if isinstance(raw_rc, Mapping):
            primary = raw_rc.get("primary")
            if primary is None:
                primary = raw_rc.get("primary_regime")
            scores_raw = raw_rc.get("scores")
            if scores_raw is None:
                scores_raw = raw_rc.get("regime_scores")
            if isinstance(primary, str) and isinstance(scores_raw, Mapping):
                last_regime_context = (
                    primary,
                    {str(k): float(v) for k, v in scores_raw.items()},
                )
        elif isinstance(raw_rc, (list, tuple)) and len(raw_rc) == 2:
            primary, scores_raw = raw_rc
            if isinstance(primary, str) and isinstance(scores_raw, Mapping):
                last_regime_context = (
                    primary,
                    {str(k): float(v) for k, v in scores_raw.items()},
                )

        last_trend_status = payload.get("last_trend_status")
        if last_trend_status is not None:
            last_trend_status = str(last_trend_status)

        raw_sleeve_w = payload.get("last_sleeve_weights")
        last_sleeve_weights = None
        if isinstance(raw_sleeve_w, Mapping):
            last_sleeve_weights = {str(k): float(v) for k, v in raw_sleeve_w.items()}

        raw_portfolio = payload.get("last_portfolio")
        last_portfolio = None
        if isinstance(raw_portfolio, Mapping):
            last_portfolio = {str(k): float(v) for k, v in raw_portfolio.items()}

        return cls(
            last_regime_sample_ts=last_regime_sample_ts,
            last_regime_context=last_regime_context,
            last_trend_sample_ts=last_trend_sample_ts,
            last_trend_status=last_trend_status,
            last_as_of=last_as_of,
            last_rebalance_ts=last_rebalance_ts,
            last_sleeve_weights=last_sleeve_weights,
            last_portfolio=last_portfolio,
        )

    @classmethod
    def empty(cls) -> "MultiSleeveAllocatorState":
        return cls()


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
        sleeves: Mapping[str, BaseSleeve],
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
        self.regime_engine: RegimeEngine = regime_engine
        self.signals: SignalEngine = (
            regime_engine.signals
        )  # "steal" SignalEngine from RegimeEngine
        self.sleeves: Dict[str, BaseSleeve] = dict(sleeves)
        self.config = config or MultiSleeveConfig()
        self.enabled_sleeves = set()
        for regime_weights in self.config.sleeve_regime_weights.values():
            for sleeve_name in regime_weights.keys():
                self.enabled_sleeves.add(sleeve_name)

        # Check if there are any enabled sleeves not in the provided sleeves (except "cash")
        for sleeve_name in self.enabled_sleeves:
            if sleeve_name != "cash" and sleeve_name not in self.sleeves:
                raise ValueError(
                    f"Sleeve '{sleeve_name}' is enabled in config but not provided in sleeves."
                )

        # Logger
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info("Enabled sleeves: %s", self.enabled_sleeves)

        # Friction Control
        self.friction_controller: Optional[FrictionController] = None
        if self.config.enable_friction_control:
            self.friction_controller = FrictionController(
                keep_cash=True,
                config=self.config.friction_control_config,
            )

        # State
        self.state = MultiSleeveAllocatorState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_universe(self, as_of: Optional[datetime | str] = None) -> Set[str]:
        """
        Get the overall universe, i.e. all tickers tradable across all enabled sleeves.
        If as_of is provided, get the universe as-of that date. Otherwise, return the
        universe of all time (including tickers no longer in the index of individual sleeves).
        """
        tickers: Set[str] = set()
        for name in self.enabled_sleeves:
            if name == "cash":
                continue
            sleeve = self.sleeves.get(name)
            if sleeve is None:
                continue
            sleeve_univ = sleeve.get_universe(as_of)
            tickers.update(sleeve_univ)
        return tickers

    def generate_global_target_weights(
        self,
        as_of: datetime | str,
        start_for_signals: Optional[datetime | str] = None,
        rebalance_ctx: Optional[RebalanceContext] = None,
    ) -> tuple[Dict[str, float], Dict[str, Any]]:
        """Main entry point for V2 multi-sleeve allocation.

        Returns a tuple `(weights, context)` where `context` is a dict with
        keys `primary_regime`, `regime_scores`, and `sleeve_weights` that were
        used to form the returned global `weights`.

        Steps:
            1) Query RegimeEngine over a lookback window.
            2) Extract primary_regime and regime_score distribution.
              2.5) Apply temperature shaping to regime scores.
            3) Blend per-regime sleeve weights using regime scores.
            4) Scale sleeve weights via trend filter if enabled.
            5) Ask each sleeve for internal weights and combine.
            6) Normalize global weights (with optional cash preservation).
            7) Apply friction control if enabled.
        """
        as_of_ts = pd.to_datetime(as_of)
        rebalance_ts = pd.to_datetime(
            rebalance_ctx.rebalance_ts if rebalance_ctx else as_of_ts
        )

        # Figure out how much history sleeves need for signals
        if start_for_signals is None:
            start_for_signals = as_of_ts - timedelta(
                days=self.config.signal_lookback_days
            )
        start_for_signals_ts = pd.to_datetime(start_for_signals)

        # 1) Get regime context (primary label + score distribution)
        primary_regime, regime_scores = self._get_regime_context(as_of_ts, rebalance_ts)
        self.log.info(
            "As of %s: primary_regime=%s, regime_scores=%s",
            as_of_ts.date(),
            primary_regime,
            regime_scores,
        )
        # Apply temperature shaping to regime scores
        regime_scores = self._shape_regime_scores_temperature(
            regime_scores,
            tau=self.config.regime_temperature_tau,
        )
        self.log.debug("Shaped regime_scores=%s", regime_scores)

        # 2) Compute effective sleeve weights via regime-score blending
        sleeve_alloc = self._compute_effective_sleeve_weights(
            primary_regime, regime_scores
        )
        # Adjust sleeve weights via trend filter if enabled
        if self.config.trend_filter_enabled:
            trend_status = self._compute_trend_filter_status(as_of_ts, rebalance_ts)
            sleeve_alloc = self._apply_trend_filter(sleeve_alloc, trend_status)
        # Build context to return to caller
        context: Dict[str, Any] = {
            "primary_regime": primary_regime,
            "regime_scores": regime_scores,
            "sleeve_weights": sleeve_alloc,
            "trend_status": trend_status if self.config.trend_filter_enabled else None,
        }
        if not sleeve_alloc:
            return {}, context

        # 3) Query sleeves and combine
        combined: Dict[str, float] = {}
        # Enrich rebalance context with regime info
        if rebalance_ctx is not None:
            rebalance_ctx.primary_regime = primary_regime
            rebalance_ctx.regime_scores = regime_scores

        for name, sleeve in self.sleeves.items():
            self.log.debug("Processing sleeve '%s' ...", name)
            alloc = sleeve_alloc.get(name, 0.0)
            if alloc <= 0:
                self.log.debug("Skipping sleeve '%s' with zero allocation.", name)
                continue

            local_weights = sleeve.generate_target_weights_for_date(
                as_of=as_of_ts,
                start_for_signals=start_for_signals_ts,
                regime=primary_regime,  # deprecated; use regime in rebalance_ctx
                rebalance_ctx=rebalance_ctx,
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

        # 5) Apply friction control if enabled and if the weights have meaningfully changed
        if self.friction_controller is not None:
            last = self.state.last_portfolio or {}
            if _weights_close(last, out):
                out_adj = out
            else:
                fc_ctx = FrictionControlContext(
                    aum=rebalance_ctx.aum if rebalance_ctx else 0.0
                )
                out_adj = self.friction_controller.apply(
                    w_prev=last,
                    w_new=out,
                    ctx=fc_ctx,
                )
            # Debug: log % changes due to friction control
            # changes = {}
            # for t in set(out.keys()).union(set(out_adj.keys())):
            #     w_prev = (
            #         self.state.last_portfolio.get(t, 0.0)
            #         if self.state.last_portfolio
            #         else 0.0
            #     )
            #     w_new = out.get(t, 0.0)
            #     w_eff = out_adj.get(t, 0.0)
            #     if w_new != w_eff:
            #         changes[t] = (w_prev, w_new, w_eff)
            # if changes:
            #     self.log.debug("Friction control adjustments:")
            #     for t, (w_prev, w_new, w_eff) in changes.items():
            #         self.log.debug(
            #             "%s: prev=%0.6f, new=%0.6f, eff=%0.6f (delta=%+0.6f)",
            #             t,
            #             w_prev,
            #             w_new,
            #             w_eff,
            #             w_eff - w_new,
            #         )
            out = out_adj

        # store state
        self.state.last_rebalance_ts = rebalance_ts
        self.state.last_as_of = as_of_ts
        self.state.last_sleeve_weights = sleeve_alloc
        self.state.last_portfolio = out
        return out, context

    def should_rebalance(self, now: datetime | str) -> bool:
        """Check if rebalance is needed at `now` timestamp.
        Returns True if any enabled sleeve requires rebalance, or if
        the regime engine or trend filter wants state update (which may also
        trigger a rebalance).
        """
        # If any enabled sleeve requires rebalance, we rebalance
        for name in self.enabled_sleeves:
            if name == "cash":
                continue
            sleeve = self.sleeves.get(name)
            if sleeve is None:
                self.log.warning(
                    "Sleeve '%s' is enabled but not found in sleeves; skipping rebalance check.",
                    name,
                )
                continue
            if sleeve.should_rebalance(now):
                self.log.debug("Sleeve '%s' requires rebalance at %s.", name, now)
                return True

        # Check if regime engine wants rebalance
        if self.state.last_regime_context is None:
            self.log.debug("No prior regime context; rebalance required.")
            return True
        if should_rebalance(
            self.state.last_regime_sample_ts,
            pd.to_datetime(now),
            self.config.regime_sample_freq,
        ):
            self.log.debug("Regime engine requires rebalance at %s.", now)
            return True

        # Check if trend filter wants rebalance
        if self.config.trend_filter_enabled:
            if self.state.last_trend_status is None:
                self.log.debug("No prior trend status; rebalance required.")
                return True
            if should_rebalance(
                self.state.last_trend_sample_ts,
                pd.to_datetime(now),
                self.config.trend_sample_freq,
            ):
                self.log.debug("Trend filter requires rebalance at %s.", now)
                return True

        return False

    def get_last_rebalance_datetime(self) -> Optional[datetime]:
        """Get the datetime of the last rebalance executed by the allocator.

        Returns:
            A datetime object representing the last rebalance time, or None if no rebalances have occurred.
        """
        candidates: List[Optional[datetime]] = [
            # Allocator-level last rebalance
            self.state.last_rebalance_ts,
            # Regime engine last sample
            self.state.last_regime_sample_ts,
            # Trend filter last sample
            self.state.last_trend_sample_ts,
        ]
        # Sleeve-level last rebalances
        for name in self.enabled_sleeves:
            if name == "cash":
                continue
            sleeve = self.sleeves.get(name)
            if sleeve is None:
                continue
            sleeve_last = sleeve.get_last_rebalance_datetime()
            candidates.append(sleeve_last)
        # Return the most recent datetime among candidates
        valid_candidates = [dt for dt in candidates if dt is not None]
        if not valid_candidates:
            return None
        return max(valid_candidates)

    # ------------------------------------------------------------------
    # Vectorized Precompute for All Sleeves
    # ------------------------------------------------------------------
    def precompute(
        self,
        start: datetime | str,
        end: datetime | str,
        sample_dates: Optional[list[datetime | str]] = None,
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
        sample_dates : list[datetime | str], optional
            If provided, sleeves may choose to sample their internal matrix to only
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

        if sample_dates is not None:
            # Adjust start date and end date to cover all sample dates
            min_sample_date = min(pd.to_datetime(d) for d in sample_dates)
            max_sample_date = max(pd.to_datetime(d) for d in sample_dates)
            start_ts = min(start_ts, min_sample_date)
            end_ts = max(end_ts, max_sample_date)
            self.log.info("Sample dates supplied: %d", len(sample_dates))
            self.log.info(
                "Adjusted precompute date range to [%s, %s]",
                start_ts.date(),
                end_ts.date(),
            )

        self.log.info(
            "Starting precompute for sleeves over [%s, %s]",
            start_ts.date(),
            end_ts.date(),
        )

        results: Dict[str, pd.DataFrame] = {}
        for name, sleeve in self.sleeves.items():
            if name not in self.enabled_sleeves:
                self.log.info("Sleeve '%s' is not enabled; skipping precompute.", name)
                continue
            self.log.info("Precomputing sleeve '%s' ...", name)
            try:
                wmat = sleeve.precompute(
                    start=start_ts,
                    end=end_ts,
                    sample_dates=sample_dates,
                    warmup_buffer=warmup_buffer,
                )
                if wmat is None or wmat.empty:
                    self.log.info("Sleeve '%s' returned empty matrix.", name)
                else:
                    self.log.info(
                        "Sleeve '%s' weights shape: %s (dates=%d, tickers=%d)",
                        name,
                        wmat.shape,
                        wmat.shape[0],
                        wmat.shape[1],
                    )
                results[name] = wmat.copy() if wmat is not None else pd.DataFrame()
            except Exception as e:
                # Log the exception and continue
                self.log.exception("Sleeve '%s' precompute failed: %s", name, e)
                results[name] = pd.DataFrame()

        self.log.info("Precompute finished.")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_regime_context(
        self, as_of: pd.Timestamp, rebalance_ts: pd.Timestamp
    ) -> Tuple[str, Dict[str, float]]:
        """
        Call RegimeEngine.get_regime_frame() and return:
            - primary_regime label (string)
            - regime_scores: {regime_name -> score in [0,1], sum ~ 1}
        """
        # Check if we should reuse last regime context
        if self.state.last_regime_context is not None and not should_rebalance(
            self.state.last_regime_sample_ts,
            rebalance_ts,
            self.config.regime_sample_freq,
        ):
            return self.state.last_regime_context
        # Otherwise, proceed to compute new regime context

        lookback_days = self.config.regime_lookback_days
        start = as_of - timedelta(days=lookback_days)

        df = self.regime_engine.get_regime_frame(start=start, end=as_of)
        if df is None or df.empty:
            # Fallback: sideways, no scores
            return "sideways", {}

        # Use the last available date <= as_of; Not necessarily as_of itself because
        # of weekends/holidays
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

        # Make sure to update regime states
        self.state.last_regime_sample_ts = rebalance_ts
        self.state.last_regime_context = (primary, regime_scores)
        return primary, regime_scores

    @staticmethod
    def _shape_regime_scores_temperature(
        regime_scores: Dict[str, float],
        tau: float = 0.6,
        method: str = "power",
        eps: float = 1e-12,
        risk_on_regimes: List[str] = ["bull", "correction"],
    ) -> Dict[str, float]:
        """
        Temperature-shape a regime score distribution into a more/less decisive distribution.
        Temperature shaping controls "decisiveness":
            tau < 1  => sharper / more decisive (top regime gets more weight)
            tau = 1  => unchanged
            tau > 1  => softer / more blended
        Asymmetric version:
        - Apply temperature shaping ONLY to risk-on regimes (bull, correction)
        - Leave defensive regimes (bear, crisis, sideways) unchanged

        Inputs:
        regime_scores: Dict[regime, score]. Can be any nonnegative scale; will be cleaned + normalized.
        tau: temperature parameter. Recommended search: {0.35, 0.6}. Start at 0.6.
        method:
            - "power": p_i <- p_i^(1/tau)  (fast, robust, no exp overflow)
            - "softmax": p_i <- exp(p_i/tau) (more sensitive to score scaling; usually unnecessary)
        eps: stability epsilon.

        Returns:
        A new Dict[regime, prob] that sums to 1.0 (or {} if input unusable).
        """
        if not regime_scores:
            return {}

        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")

        # 1) Clean + clip negatives + coerce to float
        cleaned: Dict[str, float] = {}
        for k, v in regime_scores.items():
            try:
                x = float(v)
            except Exception:
                continue
            if x <= 0:
                continue
            cleaned[k] = x

        if not cleaned:
            return {}

        # 2) Normalize to probabilities
        s = sum(cleaned.values())
        if s <= eps:
            return {}
        p = {k: v / s for k, v in cleaned.items()}

        # 3) Temperature shaping
        if abs(tau - 1.0) < 1e-9:
            return p

        if method == "power":
            inv_tau = 1.0 / tau
            shaped = {k: max(eps, pv) ** inv_tau for k, pv in p.items()}
        elif method == "softmax":
            # Softmax on probabilities is usually redundant, but included for completeness.
            # This version is numerically safe via max-subtraction.
            import math

            vals = list(p.values())
            m = max(vals)
            shaped = {k: math.exp((pv - m) / tau) for k, pv in p.items()}
        else:
            raise ValueError(f"Unknown method={method!r}; use 'power' or 'softmax'.")

        s2 = sum(shaped.values())
        if s2 <= eps:
            return {}
        return {k: v / s2 for k, v in shaped.items()}

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

    def _compute_trend_filter_status(
        self,
        as_of: pd.Timestamp,
        rebalance_ts: pd.Timestamp,
    ) -> str:
        """
        Compute trend filter status ("risk-on" or "risk-off") based on
        configured benchmark and window.

        Returns "risk-on" or "risk-off".
        """
        # Reuse cached trend status if sampling frequency doesn't require recompute
        if self.state.last_trend_status is not None and not should_rebalance(
            self.state.last_trend_sample_ts, rebalance_ts, self.config.trend_sample_freq
        ):
            return self.state.last_trend_status

        benchmark = self.config.trend_benchmark
        window = self.config.trend_window

        smas = self.signals.get_series(
            ticker=benchmark,
            signal="sma",
            start=as_of - timedelta(days=window * 7),  # extra buffer
            end=as_of,
            window=window,
        )
        if smas is None or smas.empty:
            return "risk-on"
        prices = self.signals.get_series(
            ticker=benchmark,
            signal="last_price",
            start=as_of - timedelta(days=window * 7),  # extra buffer
            end=as_of,
        )
        if prices is None or prices.empty:
            return "risk-on"

        # Use the last available price and SMA as of `as_of` if not too far back
        # Otherwise use None
        last_trading_date = min(prices.index.max(), smas.index.max())
        if (as_of - last_trading_date).days > 7:
            price_as_of, sma_as_of = None, None
            self.log.info("Trend filter: no recent data as of %s", as_of.date())
        else:
            price_as_of = prices.iloc[-1]
            sma_as_of = smas.iloc[-1]
            self.log.debug(
                "Trend filter for %s (use %s): price=%s, sma=%s",
                as_of.date(),
                last_trading_date.date(),
                price_as_of,
                sma_as_of,
            )

        if price_as_of is None or sma_as_of is None:
            trend_status = "risk-on"
        elif price_as_of > sma_as_of:
            trend_status = "risk-on"
        else:
            trend_status = "risk-off"

        # update cached trend state
        self.state.last_trend_sample_ts = rebalance_ts
        self.state.last_trend_status = trend_status
        return trend_status

    def _apply_trend_filter(
        self,
        sleeve_weights: Dict[str, float],
        trend_status: str,
    ) -> Dict[str, float]:
        """
        Apply trend filter to sleeve weights.

        If trend_status is "risk-off", scale down certain sleeves
        according to config.

        Returns adjusted sleeve weights.
        """
        adjusted: Dict[str, float] = {}
        for sleeve, weight in sleeve_weights.items():
            if trend_status == "risk-off":
                off_scale = self.config.sleeve_risk_off_equity_frac.get(sleeve, 1.0)
                adjusted[sleeve] = weight * off_scale
            else:
                on_scale = self.config.sleeve_risk_on_equity_frac.get(sleeve, 1.0)
                adjusted[sleeve] = weight * on_scale
        return adjusted


def _weights_close(
    a: Dict[str, float],
    b: Dict[str, float],
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-12,
) -> bool:
    a_up = {str(k).upper(): float(v) for k, v in (a or {}).items()}
    b_up = {str(k).upper(): float(v) for k, v in (b or {}).items()}
    for k in set(a_up.keys()).union(b_up.keys()):
        av = a_up.get(k, 0.0)
        bv = b_up.get(k, 0.0)
        if not math.isclose(av, bv, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True
