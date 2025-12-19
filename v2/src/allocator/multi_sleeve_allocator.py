from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Mapping, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd

from src.regime_engine import RegimeEngine
from src.signal_engine import SignalEngine
from src.context.rebalance import RebalanceContext
from src.sleeves.common.rebalance_helpers import should_rebalance
from .multi_sleeve_config import MultiSleeveConfig


@dataclass
class MultiSleeveAllocatorState:
    # Regime engine states
    last_regime_sample_ts: Optional[pd.Timestamp] = None
    last_regime_context: Optional[Tuple[str, Dict[str, float]]] = None
    # Effective (post-hysteresis) primary regime
    last_primary: Optional[str] = None
    # Raw primary regime tracking (pre-hysteresis)
    last_raw_primary: Optional[str] = None
    raw_primary_streak: int = 0

    # Trend filter states
    last_trend_sample_ts: Optional[pd.Timestamp] = None
    last_trend_status: Optional[str] = None

    last_as_of: Optional[pd.Timestamp] = None
    last_rebalance_ts: Optional[pd.Timestamp] = None
    last_sleeve_weights: Optional[Dict[str, float]] = None
    last_portfolio: Optional[Dict[str, float]] = None


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
        self.regime_engine: RegimeEngine = regime_engine
        self.signals: SignalEngine = (
            regime_engine.signals
        )  # "steal" SignalEngine from RegimeEngine
        self.sleeves: Dict[str, Any] = dict(sleeves)
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

        print(f"[MultiSleeveAllocator] Enabled sleeves: {self.enabled_sleeves}")

        # State
        self.state = MultiSleeveAllocatorState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            3) Blend per-regime sleeve weights using regime scores.
            4) Ask each sleeve for internal weights and combine.
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
        # print(f"[MultiSleeveAllocator] As of {as_of_ts.date()}: primary_regime={primary_regime}, regime_scores={regime_scores}")

        # 2) Compute effective sleeve weights via regime-score blending
        sleeve_alloc = self._compute_effective_sleeve_weights(
            primary_regime, regime_scores
        )
        # Apply optional regime-based modifiers (e.g., downweight sideways_base
        # when the 'sideways' score is weak)
        sleeve_alloc = self._apply_sleeve_modifiers(sleeve_alloc, regime_scores)
        # Apply hard floors on the effective sleeve weights
        sleeve_alloc = self._apply_sleeve_floors(sleeve_alloc, primary_regime)
        # Adjust sleeve weights via trend filter if enabled
        if self.config.trend_filter_enabled:
            trend_status = self._compute_trend_filter_status(as_of_ts, rebalance_ts)
            sleeve_alloc = self._apply_trend_filter(
                sleeve_alloc, trend_status, primary_regime
            )
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

        for name, sleeve in self.sleeves.items():
            # print(f"[MultiSleeveAllocator] Processing sleeve '{name}' ...")
            alloc = sleeve_alloc.get(name, 0.0)
            if alloc <= 0:
                # print(f"[MultiSleeveAllocator]  Skipping sleeve '{name}' with zero allocation.")
                continue

            local_weights = sleeve.generate_target_weights_for_date(
                as_of=as_of_ts,
                start_for_signals=start_for_signals_ts,
                regime=primary_regime,
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

        # store state
        self.state.last_as_of = as_of_ts
        self.state.last_sleeve_weights = sleeve_alloc
        self.state.last_portfolio = out
        return out, context

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
            print(f"[MultiSleeveAllocator] Sample dates supplied: {len(sample_dates)}")
            print(
                f"[MultiSleeveAllocator] Adjusted precompute date range to [{start_ts.date()}, {end_ts.date()}]"
            )

        print(
            f"[MultiSleeveAllocator] Starting precompute for sleeves over [{start_ts.date()}, {end_ts.date()}]"
        )

        results: Dict[str, pd.DataFrame] = {}

        for name, sleeve in self.sleeves.items():
            if name not in self.enabled_sleeves:
                print(
                    f"[MultiSleeveAllocator] Sleeve '{name}' is not enabled; skipping precompute."
                )
                continue
            if not hasattr(sleeve, "precompute"):
                print(
                    f"[MultiSleeveAllocator] Sleeve '{name}' has no precompute(); skipping."
                )
                continue
            print(f"[MultiSleeveAllocator] Precomputing sleeve '{name}' ...")
            try:
                wmat = sleeve.precompute(
                    start=start_ts,
                    end=end_ts,
                    sample_dates=sample_dates,
                    warmup_buffer=warmup_buffer,
                )
                if wmat is None or wmat.empty:
                    print(
                        f"[MultiSleeveAllocator] Sleeve '{name}' returned empty matrix."
                    )
                else:
                    print(
                        f"[MultiSleeveAllocator] Sleeve '{name}' weights shape: {wmat.shape} (dates={wmat.shape[0]}, tickers={wmat.shape[1]})"
                    )
                results[name] = wmat.copy() if wmat is not None else pd.DataFrame()
            except Exception as e:
                # Print the exception and stacktrace
                import traceback

                traceback.print_exc()
                print(f"[MultiSleeveAllocator] Sleeve '{name}' precompute failed: {e}")
                results[name] = pd.DataFrame()

        print("[MultiSleeveAllocator] Precompute finished.")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_regime_hysteresis(self, raw_primary: str) -> Tuple[str, bool]:
        """
        Apply enter-fast / exit-slow hysteresis to raw primary regime.

        Returns:
            (effective_primary, switched)
        where `switched` indicates whether we accepted switching to raw_primary.

        Logic:
          - Maintain a streak counter for consecutive occurrences of raw_primary.
          - Define regime "defensiveness" order; moving to a MORE defensive regime
            uses the "enter" threshold for the candidate regime.
          - Moving to a LESS defensive regime uses the "exit" threshold for the
            CURRENT effective regime (i.e., require more confirmation to exit).
        """
        cfg = self.config

        # Regime order: increasing = more defensive / risk-off
        # (sideways vs correction can be debated; choose an order and be consistent)
        regime_order = {
            "bull": 0,
            "sideways": 1,
            "correction": 2,
            "bear": 3,
            "crisis": 4,
        }

        # Safety fallback if unknown labels appear
        if raw_primary not in regime_order:
            return raw_primary, True

        prev_effective = self.state.last_primary or raw_primary

        # Update raw streak
        if self.state.last_raw_primary == raw_primary:
            self.state.raw_primary_streak += 1
        else:
            self.state.last_raw_primary = raw_primary
            self.state.raw_primary_streak = 1

        # If we have no previous effective regime, accept immediately
        if self.state.last_primary is None:
            self.state.last_primary = raw_primary
            return raw_primary, True

        # If no change, nothing to do
        if raw_primary == prev_effective:
            return prev_effective, False

        moving_more_defensive = regime_order[raw_primary] > regime_order[prev_effective]

        enter_map = cfg.regime_hysteresis.get("enter", {})
        exit_map = cfg.regime_hysteresis.get("exit", {})

        if moving_more_defensive:
            # Enter risk-off faster
            needed = int(enter_map.get(raw_primary, 1))
        else:
            # Exit risk-off slower: require confirms to leave prev_effective
            needed = int(exit_map.get(prev_effective, 2))

        # Decision: accept switch only if raw streak reaches required confirms
        if self.state.raw_primary_streak >= needed:
            self.state.last_primary = raw_primary
            return raw_primary, True

        # Otherwise, hold previous effective regime
        return prev_effective, False

    def _get_regime_context(
        self, as_of: pd.Timestamp, rebalance_ts: pd.Timestamp
    ) -> Tuple[str, Dict[str, float]]:
        """
        Returns:
        - effective_primary: hysteresis-stabilized discrete regime label
        - regime_scores: fresh soft distribution (NOT hysteresis'd), sum ~ 1
        """

        # If we are not due to resample regimes, reuse last returned context
        if self.state.last_regime_context is not None and not should_rebalance(
            self.state.last_regime_sample_ts,
            rebalance_ts,
            self.config.regime_sample_freq,
        ):
            return self.state.last_regime_context

        # ---- compute fresh scores ----
        lookback_days = self.config.regime_lookback_days
        start = as_of - timedelta(days=lookback_days)

        df = self.regime_engine.get_regime_frame(start=start, end=as_of)
        if df is None or df.empty:
            # fallback: keep last primary if exists, else sideways
            primary = self.state.last_primary or "sideways"
            scores = {primary: 1.0}
            self.state.last_regime_sample_ts = rebalance_ts
            self.state.last_regime_context = (primary, scores)
            return primary, scores

        last = df.iloc[-1]

        # Extract scores (already normalized in your regime_engine output)
        scores_raw: Dict[str, float] = {}
        for col in self.config.regime_score_columns:
            if col in last.index:
                v = float(last[col])
                if not pd.isna(v):
                    scores_raw[col] = max(v, 0.0)

        total = float(sum(scores_raw.values()))
        if total <= 0:
            candidate = "sideways"
            scores = {candidate: 1.0}
        else:
            scores = {k: v / total for k, v in scores_raw.items()}
            candidate = max(scores, key=scores.get)

        # ---- hysteresis on PRIMARY ONLY ----
        last_primary = self.state.last_primary
        if last_primary is None:
            effective_primary = candidate
            self.state.primary_streak = 1
        else:
            if candidate == last_primary:
                # staying put
                effective_primary = last_primary
                self.state.primary_streak = 0  # reset streak for switching
            else:
                # switching attempt: count consecutive "votes" for candidate
                self.state.primary_streak += 1

                enter_n = self.config.regime_hysteresis.get("enter", {}).get(
                    candidate, 1
                )
                exit_n = self.config.regime_hysteresis.get("exit", {}).get(
                    last_primary, 1
                )

                # conservative rule: require BOTH:
                # - candidate has enough consecutive votes (enter_n)
                # - and we've persisted long enough to exit the old one (exit_n)
                # With monthly sampling, primary_streak approximates consecutive months.
                if self.state.primary_streak >= max(enter_n, exit_n):
                    effective_primary = candidate
                    self.state.primary_streak = 0
                else:
                    effective_primary = last_primary

        # update state
        self.state.last_primary = effective_primary
        self.state.last_regime_sample_ts = rebalance_ts
        # NOTE: store effective_primary but ALWAYS store fresh scores
        self.state.last_regime_context = (effective_primary, scores)

        return effective_primary, scores

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
            print(
                f"[MultiSleeveAllocator] Trend filter: no recent data as of {as_of.date()}"
            )
        else:
            price_as_of = prices.iloc[-1]
            sma_as_of = smas.iloc[-1]
            # print(
            #     f"[MultiSleeveAllocator] Trend filter for {as_of.date()} (use {last_trading_date.date()}): price={price_as_of}, sma={sma_as_of}"
            # )

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
        primary_regime: str,
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
                # Risk off: apply regime-dependent scaling
                off_scale = self.config.sleeve_risk_off_equity_frac.get(sleeve, {}).get(
                    primary_regime, 1.0
                )
                adjusted[sleeve] = weight * off_scale
            else:
                # Risk on doesn't differentiate by regime
                on_scale = self.config.sleeve_risk_on_equity_frac.get(sleeve, 1.0)
                adjusted[sleeve] = weight * on_scale
        return adjusted

    def _apply_sleeve_floors(
        self, sleeve_alloc: Dict[str, float], primary_regime: str
    ) -> Dict[str, float]:
        """
        Enforce hard floors from `self.config.sleeve_regime_weights_floor` for
        the provided `primary_regime`. Any sleeve with a configured floor that
        is higher than its current allocation will be raised to the floor.

        After applying floors, allocations are normalized to sum to 1. If the
        resulting total is zero or negative, an empty dict is returned.
        """
        if not sleeve_alloc:
            return {}

        floor_map = self.config.sleeve_regime_weights_floor.get(primary_regime, {})
        if not floor_map:
            return sleeve_alloc

        alloc = dict(sleeve_alloc)  # shallow copy

        # Ensure sleeves present in floor_map exist in alloc (with 0 if needed)
        for s, floor_val in floor_map.items():
            if floor_val <= 0:
                continue
            curr = float(alloc.get(s, 0.0))
            if curr < floor_val:
                print(
                    f"[MultiSleeveAllocator] Applying floor {floor_val} to sleeve '{s}' (current={curr}) for regime '{primary_regime}'"
                )
                alloc[s] = floor_val

        total = float(sum(alloc.values()))
        if total <= 0:
            return {}

        # Normalize to sum to 1
        return {s: (w / total) for s, w in alloc.items()}

    def _apply_sleeve_modifiers(
        self, sleeve_alloc: Dict[str, float], regime_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply `sleeve_regime_modifiers` from config.

        Current semantics: for each modifier-regime `r` in
        `config.sleeve_regime_modifiers`, if the observed `regime_scores.get(r,0)`
        is less than 0.5, multiply the listed sleeves by the provided
        multiplier. Returns a new allocation dict (shallow-copied).
        """
        if not sleeve_alloc:
            return {}

        mods = self.config.sleeve_regime_modifiers or {}
        if not mods:
            return sleeve_alloc

        alloc = dict(sleeve_alloc)

        for mod_regime, mod_map in mods.items():
            score = float(regime_scores.get(mod_regime, 0.0))
            # Apply modifier only when the regime's score is weak
            if score >= self.config.modifier_regime_score_threshold:
                continue
            for sleeve_name, m in (mod_map or {}).items():
                if sleeve_name in alloc:
                    print(
                        f"[MultiSleeveAllocator] Applying modifier {m} to sleeve '{sleeve_name}' due to weak regime '{mod_regime}' (score={score:.3f})"
                    )
                    alloc[sleeve_name] = float(alloc.get(sleeve_name, 0.0)) * m

        # If nothing changed, return as-is. Otherwise return modified alloc.
        return alloc
