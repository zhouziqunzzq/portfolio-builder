from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[2]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from universe_manager import UniverseManager
from market_data_store import MarketDataStore
from signal_engine import SignalEngine
from vec_signal_engine import VectorizedSignalEngine
from context.rebalance import RebalanceContext
from .fast_alpha_config import FastAlphaConfig


@dataclass
class FastAlphaState:
    # Current holdings
    holdings: Set[str] = field(default_factory=set)
    # Track entry date for min-hold constraint
    entry_date: Dict[str, pd.Timestamp] = field(default_factory=dict)
    # Last weights (for turnover cap blending)
    last_weights: Dict[str, float] = field(default_factory=dict)
    last_as_of: Optional[pd.Timestamp] = None
    last_removed: Dict[str, pd.Timestamp] = field(default_factory=dict)


class FastAlphaSleeve:
    """
    Fast Overnight Alpha Sleeve (Spread-momentum) with deployability guardrails.

    Notes:
      - This sleeve is *not* sector-weighted. It is top-K selection across the universe.
      - Execution timing (Open vs Mid) is handled by your backtest executor.
      - This sleeve produces target weights "as-of" a date.
    """

    def __init__(
        self,
        universe: UniverseManager,
        mds: MarketDataStore,
        signals: SignalEngine,
        vec_engine: Optional[VectorizedSignalEngine] = None,
        config: Optional[FastAlphaConfig] = None,
    ) -> None:
        self.um = universe
        self.mds = mds
        self.signals = signals
        self.vec_engine = vec_engine
        self.config = config or FastAlphaConfig()
        self.state = FastAlphaState()

        # Cached precomputed weights (Date x Ticker)
        self.precomputed_weights_mat: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_target_weights_for_date(
        self,
        as_of: datetime | str,
        start_for_signals: datetime | str,
        regime: str = "bull",  # deprecated; use rebalance_ctx instead
        rebalance_ctx: RebalanceContext | None = None,
    ) -> Dict[str, float]:
        as_of = pd.to_datetime(as_of).normalize()
        start_for_signals = pd.to_datetime(start_for_signals).normalize()
        cfg = self.config
        # regime_key = (regime or "").lower()

        # Regime gating
        if (
            cfg.use_regime_gating
            and rebalance_ctx is not None
            and rebalance_ctx.regime_scores
        ):
            for gated_off_regime in cfg.gated_off_regimes:
                score = rebalance_ctx.regime_scores.get(gated_off_regime, 0.0)
                if score >= cfg.gated_off_threshold:
                    print(
                        f"[FastAlphaSleeve] Regime gating active on {as_of.date()}: "
                        f"gated_off_regime={gated_off_regime} score={score:.3f} "
                        f">= threshold={cfg.gated_off_threshold:.3f} => returning zero weights"
                    )
                    return {}

        # Precomputed path
        if (
            self.precomputed_weights_mat is not None
            and not self.precomputed_weights_mat.empty
        ):
            if as_of in self.precomputed_weights_mat.index:
                row = self.precomputed_weights_mat.loc[as_of]
                # print(f"[FastAlphaSleeve] precomputed weights on {as_of.date()}: {row}")
                w = {t: float(v) for t, v in row.items() if v > 0}
                s = sum(w.values())
                return {t: v / s for t, v in w.items()} if s > 0 else {}

        # Non-vectorized single-date path
        universe = self._get_universe(as_of)
        if not universe:
            return {}

        scores = self._compute_spread_scores_snapshot(
            universe, start_for_signals, as_of
        )
        if scores.empty:
            return {}

        target = self._select_and_weight(as_of, scores)

        # Save last_weights for turnover control continuity
        self.state.last_as_of = as_of
        self.state.last_weights = target.copy()
        return target

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self, as_of: Optional[datetime | str] = None) -> Set[str]:
        """
        Get the universe, i.e. all tickers tradable for the sleeve.
        If as_of is provided, get the universe as-of that date. Otherwise, return the
        universe of all time (including tickers no longer in the index).
        """
        tickers = set(self._get_universe(as_of))
        # And don't forget about the spread-mom benchmark(s)
        if self.config.spread_benchmark:
            tickers.add(self.config.spread_benchmark)
        return tickers

    def _get_universe(self, as_of: pd.Timestamp) -> List[str]:
        """
        Time-aware membership universe similar to TrendSleeve:
          - Use membership mask as-of date
          - Filter out tickers without valid sectors (optional; mostly to remove Unknown)
        """
        smap = self.um.sector_map or {}
        active_tickers: Optional[set[str]] = None

        mask = self.um.membership_mask(
            start=as_of.strftime("%Y-%m-%d"),
            end=as_of.strftime("%Y-%m-%d"),
        )
        if not mask.empty:
            row = mask.iloc[0]
            active_tickers = set(row.index[row.astype(bool)])

        out: List[str] = []
        for t, sec in smap.items():
            if not isinstance(sec, str) or sec.strip() in ("", "Unknown"):
                continue
            if active_tickers is not None and t not in active_tickers:
                continue
            out.append(t)

        return sorted(set(out))

    # ------------------------------------------------------------------
    # Non-vectorized signals
    # ------------------------------------------------------------------
    def _compute_spread_scores_snapshot(
        self,
        tickers: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        """
        Compute spread-mom composite score per ticker for a single date.
        Applies optional liquidity filters and optional vol penalty.

        Returns:
          Series indexed by ticker with 'rank_score' values.
        """
        cfg = self.config
        buffer = pd.Timedelta(days=cfg.signals_extra_buffer_days or 30)

        rows = []
        for t in tickers:
            try:
                # Liquidity filters (optional)
                if cfg.use_liquidity_filters:
                    adv_start = end - pd.Timedelta(days=cfg.adv_window) - buffer
                    mv_start = (
                        end - pd.Timedelta(days=cfg.median_volume_window) - buffer
                    )
                    px_start = end - buffer

                    adv = self.signals.get_series(
                        t, "adv", start=adv_start, end=end, window=cfg.adv_window
                    )
                    mv = self.signals.get_series(
                        t,
                        "median_volume",
                        start=mv_start,
                        end=end,
                        window=cfg.median_volume_window,
                    )
                    px = self.signals.get_series(
                        t, "last_price", start=px_start, end=end
                    )

                    adv_val = adv.iloc[-1] if not adv.empty else np.nan
                    mv_val = mv.iloc[-1] if not mv.empty else np.nan
                    px_val = px.iloc[-1] if not px.empty else np.nan

                    if np.isnan(px_val) or px_val < cfg.min_price:
                        continue
                    if np.isnan(adv_val) or adv_val < cfg.min_adv:
                        continue
                    if np.isnan(mv_val) or mv_val < cfg.min_median_volume:
                        continue

                # Spread-mom composite
                bench = cfg.spread_benchmark or "SPY"
                score = 0.0
                ok = True
                for w, wt in zip(cfg.spread_mom_windows, cfg.spread_mom_window_weights):
                    smom_start = end - pd.Timedelta(days=w) - buffer
                    smom = self.signals.get_series(
                        t,
                        "spread_mom",
                        start=smom_start,
                        end=end,
                        window=w,
                        benchmark=bench,
                    )
                    if smom.empty:
                        ok = False
                        break
                    score += float(wt) * float(smom.iloc[-1])

                if not ok or not np.isfinite(score):
                    continue

                # Optional vol penalty
                if cfg.use_vol_penalty and cfg.vol_penalty and cfg.vol_penalty > 0:
                    vol_start = end - pd.Timedelta(days=cfg.vol_window) - buffer
                    vol = self.signals.get_series(
                        t, "vol", start=vol_start, end=end, window=cfg.vol_window
                    )
                    vol_val = float(vol.iloc[-1]) if not vol.empty else np.nan
                    if np.isfinite(vol_val):
                        score -= float(cfg.vol_penalty) * vol_val

                rows.append((t, score))
            except Exception:
                continue

        if not rows:
            return pd.Series(dtype=float)

        s = pd.Series({t: sc for t, sc in rows}, dtype=float)
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return s

    # ------------------------------------------------------------------
    # Selection + turnover-capped weighting
    # ------------------------------------------------------------------
    def _select_and_weight(
        self, as_of: pd.Timestamp, scores: pd.Series
    ) -> Dict[str, float]:
        """
        Stateful membership selection with:
        - enter/exit rank buffers
        - min-hold constraint
        - max replacements per day
        - optional dominance swap (rank gap + score gap)
        - turnover cap via partial weight adjustment

        Robustness:
        - NEVER uses inf ranks in downstream logic/logging
        - handles held tickers missing from today's scores (treated as very bad rank/score)
        """
        cfg = self.config

        # ----------------------------
        # Prep / ranking
        # ----------------------------
        scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
        if scores.empty:
            self.state.holdings = set()
            return {}

        # Sort candidates by score descending (best first)
        sorted_scores = scores.sort_values(ascending=False)
        # print(f"[FastAlphaSleeve] computed scores on {as_of.date()}: {sorted_scores}")

        # Rank: 1 = best (finite ranks only for tickers in `scores`)
        ranks = scores.rank(ascending=False, method="first")

        # Finite fallbacks for missing tickers (e.g., filtered out by liq mask)
        worst_rank_default = float(len(sorted_scores) + 1)  # finite "worse than worst"
        worst_score_default = float(sorted_scores.iloc[-1]) - 1e9  # finite "very bad"

        def _rank_of(t: str) -> float:
            r = float(ranks.get(t, worst_rank_default))
            # extra safety: if r is nan/inf for any reason, use worst_rank_default
            return r if np.isfinite(r) else worst_rank_default

        def _score_of(t: str) -> float:
            s = float(scores.get(t, worst_score_default))
            return s if np.isfinite(s) else worst_score_default

        def _fmt_rank(r: float) -> str:
            return str(int(r)) if np.isfinite(r) else "NA"

        def _fmt_score(s: float) -> str:
            return f"{s:.6f}" if np.isfinite(s) else "NA"

        # print(f"[FastAlphaSleeve] scores on {as_of.date()}: {sorted_scores}")

        # holdings may be a set[str] OR a dict[str,float] (you do both in your code)
        if isinstance(self.state.holdings, dict):
            holdings = set(self.state.holdings.keys())
        else:
            holdings = set(self.state.holdings)

        print(f"[FastAlphaSleeve] initial holdings on {as_of.date()}: {holdings}")

        # ----------------------------
        # Eligible exits: held & rank >= exit_k & min_hold_days satisfied
        # (tickers missing from today's scores are treated as very bad rank => eligible)
        # ----------------------------
        exits: List[str] = []
        for t in list(holdings):
            r = _rank_of(t)
            if r < float(cfg.exit_k):
                continue

            entry_dt = self.state.entry_date.get(t)
            if entry_dt is None:
                exits.append(t)
                continue

            held_days = (as_of - entry_dt).days
            if held_days >= int(cfg.min_hold_days):
                exits.append(t)

        # Worst first (largest rank)
        exits.sort(key=lambda x: _rank_of(x), reverse=True)
        print(f"[FastAlphaSleeve] exits on {as_of.date()}: {exits}")

        # ----------------------------
        # Eligible entries: not held & from top slice of ranked list
        # ----------------------------
        topN = int(max(cfg.enter_k, cfg.target_k))
        candidates = [t for t in sorted_scores.index[:topN] if t not in holdings]
        print(f"[FastAlphaSleeve] candidates on {as_of.date()}: {candidates}")

        # ----------------------------
        # Apply regular swaps (exit/enter buffers + churn cap)
        # ----------------------------
        max_rep = int(cfg.max_replacements_per_day)
        n_swap = min(max_rep, len(exits), len(candidates))

        for i in range(n_swap):
            out_t = exits[i]
            in_t = candidates[i]

            if out_t in holdings:
                holdings.remove(out_t)
                self.state.entry_date.pop(out_t, None)
                self.state.last_removed[out_t] = as_of

            holdings.add(in_t)
            self.state.entry_date[in_t] = as_of

        # Top up holdings if below target_k
        if len(holdings) < int(cfg.target_k):
            need = int(cfg.target_k) - len(holdings)
            print(f"[FastAlphaSleeve] top-up needed on {as_of.date()}: {need}")
            extra = [t for t in candidates[n_swap:] if t not in holdings]
            for t in extra[:need]:
                holdings.add(t)
                self.state.entry_date[t] = as_of

        # Trim if above target_k (keep best ranks)
        if len(holdings) > int(cfg.target_k):
            held_list = list(holdings)
            held_list.sort(key=lambda x: _rank_of(x))  # best first
            holdings = set(held_list[: int(cfg.target_k)])
            print(f"[FastAlphaSleeve] trimmed on {as_of.date()}: {holdings}")
            for t in list(self.state.entry_date.keys()):
                if t not in holdings:
                    self.state.entry_date.pop(t, None)

        # ----------------------------
        # Dominance swap (only when NO exits fired, i.e., "dominance gap" situation)
        # ----------------------------
        do_dom = (
            hasattr(cfg, "dominance_gap")
            and int(getattr(cfg, "dominance_gap", 0)) > 0
            and hasattr(cfg, "dominance_gap_score")
            and float(getattr(cfg, "dominance_gap_score", 0.0)) > 0.0
            and len(holdings) == int(cfg.target_k)
            and len(exits) == 0
            and len(candidates) > 0
        )

        if do_dom:
            # pick best outsider among candidates that is sufficiently good by rank
            max_out_rank = int(getattr(cfg, "max_out_rank", cfg.enter_k))
            best_out = None
            best_out_rank = None
            best_out_score = None

            # OPTIONAL cooldown: block re-entry for recently removed names
            cooldown_days = int(getattr(cfg, "dominance_cooldown_days", 0))
            last_removed = getattr(self.state, "last_removed", None)
            can_use_cooldown = isinstance(last_removed, dict) and cooldown_days > 0

            for t in candidates:
                r = _rank_of(t)
                if r > float(max_out_rank):
                    continue

                if can_use_cooldown:
                    last_dt = last_removed.get(t)
                    if last_dt is not None and (as_of - last_dt).days < cooldown_days:
                        continue

                best_out = t
                best_out_rank = r
                best_out_score = _score_of(t)
                break  # candidates already in best-first order

            if best_out is not None:
                # choose worst held by rank (largest rank), but respect min-hold
                worst_held = None
                worst_rank = -1.0
                worst_score = None

                for t in list(holdings):
                    # min-hold check (dominance respects min_hold unless you add a "panic" override)
                    entry_dt = self.state.entry_date.get(t)
                    if entry_dt is not None:
                        held_days = (as_of - entry_dt).days
                        if held_days < int(cfg.min_hold_days):
                            continue

                    r = _rank_of(t)
                    if r > worst_rank:
                        worst_rank = r
                        worst_held = t
                        worst_score = _score_of(t)

                if worst_held is not None:
                    rank_gap = float(worst_rank) - float(best_out_rank)
                    score_gap = float(best_out_score) - float(worst_score)
                    swap = rank_gap >= float(
                        getattr(cfg, "dominance_gap", 0)
                    ) and score_gap >= float(getattr(cfg, "dominance_gap_score", 0.0))

                    print(
                        f"[FastAlphaSleeve] dominance_check on {as_of.date()}: "
                        f"best_out={best_out}(r={_fmt_rank(best_out_rank)}, s={_fmt_score(best_out_score)}) "
                        f"worst_held={worst_held}(r={_fmt_rank(worst_rank)}, s={_fmt_score(worst_score)}) "
                        f"rank_gap={rank_gap:.0f} score_gap={score_gap:.6f} swap={swap} "
                        f"(score_gap={score_gap:.6f} >= {float(getattr(cfg, 'dominance_gap_score', 0.0)):.6f})"
                    )

                    if swap:
                        # perform the dominance swap
                        holdings.remove(worst_held)
                        self.state.entry_date.pop(worst_held, None)
                        self.state.last_removed[worst_held] = as_of

                        holdings.add(best_out)
                        self.state.entry_date[best_out] = as_of
                        print(
                            f"[FastAlphaSleeve] dominance_swap on {as_of.date()}: {worst_held} -> {best_out}"
                        )

        # ----------------------------
        # Target weights (equal weight across holdings)
        # ----------------------------
        if not holdings:
            self.state.holdings = set()
            return {}

        target = {t: 1.0 / float(len(holdings)) for t in sorted(holdings)}

        # Turnover cap: blend from prev -> target
        prev = self.state.last_weights or {}
        capped = self._apply_turnover_cap(
            prev, target, cap=float(cfg.max_daily_turnover)
        )

        # enforce sparsity
        eps = float(getattr(cfg, "min_weight", 1e-4))
        capped = {t: w for t, w in capped.items() if w >= eps}

        # cap to K again (paranoid)
        if len(capped) > int(cfg.target_k):
            capped = dict(
                sorted(capped.items(), key=lambda kv: kv[1], reverse=True)[
                    : int(cfg.target_k)
                ]
            )

        # renorm
        s = float(sum(capped.values()))
        capped = {t: w / s for t, w in capped.items()} if s > 0 else {}

        # Update holdings state based on capped weights
        for t in capped:
            self.state.entry_date.setdefault(t, as_of)

        self.state.holdings = capped
        print(f"[FastAlphaSleeve] final holdings on {as_of.date()}: {capped}")
        return capped

    @staticmethod
    def _turnover(prev: Dict[str, float], new: Dict[str, float]) -> float:
        keys = set(prev.keys()) | set(new.keys())
        diff = 0.0
        for k in keys:
            diff += abs(float(new.get(k, 0.0)) - float(prev.get(k, 0.0)))
        return 0.5 * diff

    def _apply_turnover_cap(
        self,
        prev: Dict[str, float],
        target: Dict[str, float],
        cap: float,
    ) -> Dict[str, float]:
        cfg = self.config

        def _normalize(d: Dict[str, float]) -> Dict[str, float]:
            s = float(sum(d.values()))
            return {k: float(v) / s for k, v in d.items()} if s > 0 else {}

        if cap is None or cap <= 0:
            # No cap => return clean sparse target
            out = {k: float(v) for k, v in target.items() if v > 0}
            return _normalize(out)

        tv = self._turnover(prev, target)
        if tv <= cap + 1e-12:
            out = {k: float(v) for k, v in target.items() if v > 0}
            return _normalize(out)

        lam = max(min(cap / tv, 1.0), 0.0)

        # Blend on the union, but we will sparsify hard afterwards
        keys = set(prev.keys()) | set(target.keys())
        w = {
            k: (1.0 - lam) * float(prev.get(k, 0.0)) + lam * float(target.get(k, 0.0))
            for k in keys
        }

        # --------------------------
        # NEW: prune dust
        # --------------------------
        eps = float(getattr(cfg, "min_weight", 1e-4))
        w = {k: v for k, v in w.items() if v >= eps}

        if not w:
            # fallback: at least keep target
            out = {k: float(v) for k, v in target.items() if v > 0}
            return _normalize(out)

        # --------------------------
        # NEW: hard cap number of names
        # --------------------------
        hard_max = int(getattr(cfg, "hard_max_names", cfg.target_k))
        if hard_max > 0 and len(w) > hard_max:
            # keep top weights only
            top_items = sorted(w.items(), key=lambda kv: kv[1], reverse=True)[:hard_max]
            w = dict(top_items)

        # After blending + pruning + normalize...
        # enforce membership = target keys
        target_keys = set(target.keys())
        w = {k: v for k, v in w.items() if k in target_keys}
        # renorm
        return _normalize(w)

    # ------------------------------------------------------------------
    # Vectorized precompute (signals vectorized, selection stateful)
    # ------------------------------------------------------------------
    def precompute(
        self,
        start: datetime | str,
        end: datetime | str,
        sample_dates: Optional[List[datetime | str]] = None,
        warmup_buffer: Optional[int] = None,  # days; optional override
    ) -> pd.DataFrame:
        """
        Vectorized-ish precompute:
          1) Load close price matrix (warmup_start..end)
          2) Compute spread-mom matrices (Date x Ticker)
          3) Build a daily score series per date
          4) Run stateful selection day-by-day to produce weights matrix
          5) Slice to [start, end] and optionally sample rebalance_dates

        Note: selection is intentionally stateful, so we iterate dates.
        """
        if self.vec_engine is None:
            raise ValueError("vec_engine is not set; cannot precompute.")

        cfg = self.config
        start_ts = pd.to_datetime(start).normalize()
        end_ts = pd.to_datetime(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        # Warmup: need max spread window + liq windows + min-hold buffer (small)
        window_candidates = list(cfg.spread_mom_windows or [])
        if cfg.use_liquidity_filters:
            window_candidates += [cfg.adv_window, cfg.median_volume_window]
        if cfg.use_vol_penalty and cfg.vol_window:
            window_candidates += [cfg.vol_window]
        max_window = max(window_candidates) if window_candidates else 0
        warmup_days = int(np.ceil(max_window * (365 / 252))) + (
            warmup_buffer or 0
        )  # extra buffer
        warmup_start = start_ts - pd.Timedelta(days=warmup_days)
        print(f"[FastAlphaSleeve] precompute warmup_days: {warmup_days}")
        print(
            f"[FastAlphaSleeve] precompute warmup_start: {warmup_start.date()}, start: {start_ts.date()}, end: {end_ts.date()}"
        )

        # 1) price matrix (Close)
        price_mat = self.um.get_price_matrix(
            price_loader=self.mds,
            start=warmup_start,
            end=end_ts,
            tickers=self.um.tickers,
            field="Close",
            interval="1d",
            auto_adjust=True,
            auto_apply_membership_mask=True,
            local_only=getattr(self.mds, "local_only", False),
        )
        price_mat = price_mat.dropna(axis=1, how="all")
        if price_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        price_mat.columns = [c.upper() for c in price_mat.columns]
        tickers = price_mat.columns.tolist()
        VSE = self.vec_engine

        # 2) spread-mom matrices
        bench = cfg.spread_benchmark or "SPY"
        spread_dict = VSE.get_spread_momentum(
            price_mat, cfg.spread_mom_windows, benchmark=bench
        )

        # composite score matrix
        score_mat = None
        for w, wt in zip(cfg.spread_mom_windows, cfg.spread_mom_window_weights):
            m = spread_dict[w].reindex_like(price_mat)
            part = float(wt) * m
            score_mat = (
                part if score_mat is None else score_mat.add(part, fill_value=0.0)
            )

        # Optional vol penalty (keep off for MVP unless you need it)
        if cfg.use_vol_penalty and cfg.vol_penalty and cfg.vol_penalty > 0:
            vol_mat = VSE.get_volatility(price_mat, window=int(cfg.vol_window))
            score_mat = score_mat.sub(float(cfg.vol_penalty) * vol_mat, fill_value=0.0)

        # 3) liquidity mask (optional)
        liq_mask = None
        if cfg.use_liquidity_filters:
            # volume matrix
            try:
                volume_mat = VSE.get_field_matrix(
                    tickers=tickers,
                    start=price_mat.index.min(),
                    end=price_mat.index.max(),
                    field="Volume",
                    interval="1d",
                    local_only=getattr(self.mds, "local_only", False),
                    auto_adjust=True,
                    membership_aware=False,
                    treat_unknown_as_always_member=True,
                )
                # print(f"[FastAlphaSleeve] volume_mat: {volume_mat}")
            except Exception:
                print(
                    "[FastAlphaSleeve] WARNING: volume matrix load failed; using empty volume matrix"
                )
                volume_mat = pd.DataFrame(index=price_mat.index, columns=tickers)

            volume_mat = volume_mat.reindex_like(price_mat)
            dollar_vol = price_mat * volume_mat
            # Debug: compute price_mat, volume_mat nan ratios
            # price_nan_ratio = price_mat.isna().sum().sum() / (
            #     price_mat.shape[0] * price_mat.shape[1]
            # )
            # volume_nan_ratio = volume_mat.isna().sum().sum() / (
            #     volume_mat.shape[0] * volume_mat.shape[1]
            # )
            # print(f"[FastAlphaSleeve] price_mat nan ratio: {price_nan_ratio:.4f}")
            # print(f"[FastAlphaSleeve] volume_mat nan ratio: {volume_nan_ratio:.4f}")

            adv_mat = dollar_vol.rolling(
                cfg.adv_window, min_periods=max(5, cfg.adv_window // 2)
            ).mean()
            medvol_mat = volume_mat.rolling(
                cfg.median_volume_window,
                min_periods=max(5, cfg.median_volume_window // 2),
            ).median()

            liq_mask = pd.DataFrame(True, index=price_mat.index, columns=tickers)
            liq_mask &= price_mat >= float(cfg.min_price)
            # print(f"[FastAlphaSleeve] liq_mask after price filter:\n{liq_mask}")
            liq_mask &= adv_mat >= float(cfg.min_adv)
            # print(f"[FastAlphaSleeve] adv_mat:\n{adv_mat}")
            # Debug: compute adv_mat nan ratio
            # nan_ratio = adv_mat.isna().sum().sum() / (
            #     adv_mat.shape[0] * adv_mat.shape[1]
            # )
            # print(f"[FastAlphaSleeve] adv_mat nan ratio: {nan_ratio:.4f}")
            # print(f"[FastAlphaSleeve] liq_mask after adv filter:\n{liq_mask}")
            liq_mask &= medvol_mat >= float(cfg.min_median_volume)
            # print(f"[FastAlphaSleeve] liq_mask after median volume filter:\n{liq_mask}")
            liq_mask = liq_mask.fillna(False)
            # print(f"[FastAlphaSleeve] final liq_mask:\n{liq_mask}")

        # 4) iterate dates with stateful selection
        dates = score_mat.index
        out = pd.DataFrame(0.0, index=dates, columns=tickers)

        # Reset state for deterministic precompute
        self.state = FastAlphaState()

        for dt in dates:
            s = score_mat.loc[dt].replace([np.inf, -np.inf], np.nan)

            if liq_mask is not None:
                elig = liq_mask.loc[dt]
                s = s[elig.astype(bool)]

            s = s.dropna()
            # print(f"[FastAlphaSleeve] scores on {dt.date()}: {s}")
            if s.empty:
                # no scores => keep previous weights (or zero)
                prev = self.state.last_weights or {}
                row = {t: float(w) for t, w in prev.items() if w > 0}
                if row:
                    # normalize defensively
                    sm = sum(row.values())
                    row = {t: w / sm for t, w in row.items()} if sm > 0 else {}
                out.loc[dt, list(row.keys())] = list(row.values())
                continue

            w = self._select_and_weight(dt.normalize(), s)
            out.loc[dt, list(w.keys())] = list(w.values())
            self.state.last_as_of = dt.normalize()
            self.state.last_weights = w.copy()

        # 5) slice to [start, end] and asfreq fill
        out = out.loc[(out.index >= warmup_start) & (out.index <= end_ts)]
        out = out.asfreq("D", method="ffill").fillna(0.0)

        # optional sampling
        if sample_dates:
            target_dates = pd.to_datetime(sample_dates).normalize()
            out = out[out.index.isin(target_dates)]

        self.precomputed_weights_mat = out.copy()
        return self.precomputed_weights_mat

    def get_precomputed_weights(self) -> Optional[pd.DataFrame]:
        return self.precomputed_weights_mat
