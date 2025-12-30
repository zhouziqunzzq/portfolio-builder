from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set
from datetime import datetime

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import logging

# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
_ROOT_SRC = Path(__file__).resolve().parents[2]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from universe_manager import UniverseManager
from market_data_store import MarketDataStore
from signal_engine import SignalEngine
from vec_signal_engine import VectorizedSignalEngine
from sleeves.base import BaseSleeve
from sleeves.common.rebalance_helpers import (
    should_rebalance,
    get_closest_date_on_or_before,
)
from context.rebalance import RebalanceContext
from .defensive_config import DefensiveConfig
from states.base_state import BaseState


@dataclass
class DefensiveState(BaseState):
    STATE_KEY = "sleeve.defensive"
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
    def from_payload(cls, payload: Mapping[str, Any]) -> "DefensiveState":
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
    def empty(cls) -> "DefensiveState":
        return cls()


class DefensiveSleeve(BaseSleeve):
    """
    Defensive Sleeve (Multi-Asset)

    - Defensive equities (S&P 500 stocks in defensive sectors)
    - Defensive equity ETFs (XLP, XLV, XLU)
    - Bonds (BND, IEF, TLT, SHY, LQD)
    - Gold (GLD)
    """

    def __init__(
        self,
        mds: MarketDataStore,
        universe: UniverseManager,
        signals: SignalEngine,
        config: Optional[DefensiveConfig] = None,
    ):
        super().__init__(
            market_data_store=mds,
            universe_manager=universe,
            signal_engine=signals,
            vectorized_signal_engine=VectorizedSignalEngine(universe, mds),
        )
        # Aliases for convenience
        self.um = self.universe_manager
        self.mds = self.market_data_store
        self.signals = self.signal_engine
        self.vec_engine = self.vectorized_signal_engine or VectorizedSignalEngine(
            universe, mds
        )

        self.config = config or DefensiveConfig()
        self.state = DefensiveState()
        # Logger
        self.log = logging.getLogger(self.__class__.__name__)

        # Cached precompute results
        self._cached_scores_mat: pd.DataFrame = pd.DataFrame()
        self._cached_signal_mats: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_universe(self, as_of: Optional[datetime | str] = None) -> Set[str]:
        """
        Get the universe, i.e. all tickers tradable for the sleeve.
        If as_of is provided, get the universe as-of that date. Otherwise, return the
        universe of all time (including tickers no longer in the index).
        """
        return set(self._get_defensive_universe(as_of))

    def _get_defensive_universe(
        self,
        as_of: Optional[datetime | str] = None,
    ) -> List[str]:
        """
        Build the defensive universe:

        - Start from S&P 500 membership as-of `as_of` (if provided)
        - Filter to defensive sectors (Staples, Health Care, Utilities)
        - Add defensive ETFs (equity/bond/gold) from config

        This helps avoid weird micro-cap / non-index tickers accidentally
        creeping into the defensive sleeve.
        """
        smap = self.um.sector_map or {}

        active_tickers: Optional[set[str]] = None
        if as_of is not None:
            as_of_dt = pd.to_datetime(as_of).normalize()
            # membership_mask returns [date x ticker] bool DataFrame
            mask = self.um.membership_mask(
                start=as_of_dt.strftime("%Y-%m-%d"),
                end=as_of_dt.strftime("%Y-%m-%d"),
            )
            if not mask.empty:
                row = mask.iloc[0]
                self.log.debug(
                    "Active S&P 500 members on %s: %d",
                    as_of_dt.date(),
                    int(row.sum()),
                )
                active_tickers = set(row.index[row.astype(bool)])

        core: List[str] = []
        for t, sec in smap.items():
            if sec not in self.config.defensive_sectors:
                continue
            if active_tickers is not None and t not in active_tickers:
                # Exclude tickers that are not active index members on as_of date
                continue
            core.append(t)

        # All configured multi-asset defensive ETFs (not part of S&P membership)
        extra = self.config.defensive_etfs

        return sorted(set(core + extra))

    def _apply_liquidity_filters(
        self, tickers: List[str], end: pd.Timestamp
    ) -> List[str]:
        """
        Filter tickers by:
            - min price
            - min average daily volume
        """
        cfg = self.config
        keep = []

        for t in tickers:
            try:
                # Get last ~2 month of OHLCV
                df = self.mds.get_ohlcv(
                    ticker=t,
                    start=end - pd.Timedelta(days=60),
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                )

                if df is None or df.empty:
                    continue

                # --- price filter ---
                last_price = df["Close"].iloc[-1]
                if last_price < cfg.min_price:
                    continue

                # --- ADV filter ---
                if "Volume" in df.columns:
                    adv = df["Volume"].rolling(cfg.min_adv_window).mean().iloc[-1]
                    if adv < cfg.min_adv:
                        continue
                else:
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
        regime: str = "bull",
        rebalance_ctx: Optional[RebalanceContext] = None,
    ) -> Dict[str, float]:
        as_of = pd.to_datetime(as_of)
        start_for_signals = pd.to_datetime(start_for_signals)
        cfg = self.config

        # ---------- Rebalance timing check ----------
        # Note: We need this because the global scheduler may call this function
        # more frequently than the sleeve's intended rebalance frequency.
        # If it's not time to rebalance yet, we return the last weights.
        if not self.should_rebalance(
            rebalance_ctx.rebalance_ts if rebalance_ctx is not None else as_of
        ):
            self.log.info(
                "Skipping rebalance at %s; last rebalance at %s",
                (
                    rebalance_ctx.rebalance_ts.date()
                    if rebalance_ctx is not None
                    else as_of.date()
                ),
                (
                    self.state.last_rebalance_ts.date()
                    if self.state.last_rebalance_ts is not None
                    else "never"
                ),
            )
            return self.state.last_weights

        self.log.info(
            "Rebalancing at %s using data as of %s; last rebalance at %s",
            (
                rebalance_ctx.rebalance_ts.date()
                if rebalance_ctx is not None
                else as_of.date()
            ),
            as_of.date(),
            (
                self.state.last_rebalance_ts.date()
                if self.state.last_rebalance_ts is not None
                else "never"
            ),
        )

        # Try to use cached precompute outputs (score + vol) first; otherwise
        # attempt to assemble signals from cached raw signal mats; finally
        # fall back to non-vectorized per-ticker path.
        date_key = as_of.normalize()

        scored_df: Optional[pd.DataFrame] = None

        # helper: fetch row by closest available date on-or-before d
        def _fetch_row_using_closest(
            df: Optional[pd.DataFrame], d: pd.Timestamp
        ) -> Optional[pd.Series]:
            if df is None or df.empty:
                return None
            idx = get_closest_date_on_or_before(d, df.index)
            if idx is None or pd.isna(idx):
                return None
            row = df.loc[idx]
            if row.isna().all():
                return None
            return row

        # 1) Use cached composite score + cached vol if available
        if (
            getattr(self, "_cached_scores_mat", None) is not None
            and not self._cached_scores_mat.empty
            and date_key >= self._cached_scores_mat.index.min()
        ):
            score_row = _fetch_row_using_closest(self._cached_scores_mat, date_key)
            vol_row = _fetch_row_using_closest(
                (
                    self._cached_signal_mats.get("vol")
                    if getattr(self, "_cached_signal_mats", None) is not None
                    else None
                ),
                date_key,
            )
            if score_row is not None and vol_row is not None:
                tmp_scored = pd.DataFrame(
                    {"score": score_row.astype(float), "vol": vol_row.astype(float)}
                )
                tmp_scored = tmp_scored.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=["score", "vol"], how="any"
                )
                if not tmp_scored.empty:
                    # `allocate_by_asset_class` expects a DataFrame with at least 'score' and 'vol'
                    scored_df = tmp_scored
                    self.log.info(
                        "Using cached precomputed scores for %s", date_key.date()
                    )

        # 2) If no scored_df from cache, try to assemble raw signals from cached signal mats
        if scored_df is None:
            self.log.warning(
                "no cached scores found, attempting to assemble from raw signal mats"
            )
            sigs_df: Optional[pd.DataFrame] = None
            if getattr(self, "_cached_signal_mats", None) is not None:
                mom_fast_s = _fetch_row_using_closest(
                    self._cached_signal_mats.get("mom_fast"), date_key
                )
                mom_slow_s = _fetch_row_using_closest(
                    self._cached_signal_mats.get("mom_slow"), date_key
                )
                vol_s = _fetch_row_using_closest(
                    self._cached_signal_mats.get("vol"), date_key
                )
                beta_s = _fetch_row_using_closest(
                    self._cached_signal_mats.get("beta"), date_key
                )
                if (
                    mom_fast_s is not None
                    and mom_slow_s is not None
                    and vol_s is not None
                ):
                    sigs_df = pd.DataFrame(
                        {
                            "mom_fast": mom_fast_s.astype(float),
                            "mom_slow": mom_slow_s.astype(float),
                            "vol": vol_s.astype(float),
                            "beta": (
                                beta_s.astype(float) if beta_s is not None else np.nan
                            ),
                        }
                    )
                    sigs_df = sigs_df.replace([np.inf, -np.inf], np.nan).dropna(
                        subset=["mom_fast", "mom_slow", "vol"], how="any"
                    )
                    if not sigs_df.empty:
                        # compute scored from raw signals
                        scored_df = self._compute_scores_only(sigs_df)

        # 3) Final fallback: compute signals one-by-one (non-vec path)
        if scored_df is None or scored_df.empty:
            self.log.warning(
                "no cached signals or scores found, falling back to per-ticker signal computation"
            )
            universe = self._get_defensive_universe(as_of=as_of)
            universe = self._apply_liquidity_filters(universe, as_of)
            if not universe:
                return {}
            sigs = self._compute_signals_snapshot(universe, start_for_signals, as_of)
            if sigs.empty:
                return {}
            scored_df = self._compute_scores_only(sigs)

        # Single allocation call (unified)
        weights = self.allocate_by_asset_class(scored_df, regime)
        self.state.last_rebalance_ts = (
            rebalance_ctx.rebalance_ts if rebalance_ctx is not None else as_of
        )
        self.state.last_weights = weights
        return weights

    def should_rebalance(self, now: datetime | str) -> bool:
        if self.state.last_weights is None:
            # Never rebalanced before; must rebalance now
            return True

        now = pd.to_datetime(now)
        cfg = self.config
        return should_rebalance(self.state.last_rebalance_ts, now, cfg.rebalance_freq)

    # ------------------------------------------------------------------
    # Multi-Asset Allocation (Option 1)
    # ------------------------------------------------------------------

    def allocate_by_asset_class(
        self, scored: pd.DataFrame, regime: str
    ) -> Dict[str, float]:
        """
        Allocate by asset class according to config.
        Steps:
        1) Assign asset class: equity / bond / gold
        2) Per-class: select top_k * class_alloc fraction
        3) Weight by score / inverse-vol
        4) Multiply by class allocation
        5) Normalize final vector
        """
        cfg = self.config

        # 1) assign asset class
        def assign_class(t):
            if t in cfg.asset_class_for_etf:
                return cfg.asset_class_for_etf[t]
            return "equity"  # defensive stock

        scored = scored.copy()
        scored["asset_class"] = [assign_class(t) for t in scored.index]

        # 2) get regime-level class allocations
        class_alloc = cfg.asset_class_allocations_by_regime.get(regime)
        if class_alloc is None:
            raise ValueError(
                f"Unknown regime '{regime}' in asset_class_allocations_by_regime"
            )

        final: Dict[str, float] = {}

        for asset_class in ["equity", "bond", "gold"]:
            sub = scored[scored["asset_class"] == asset_class]
            if sub.empty:
                continue

            # ---- per-class top-k selection ----
            k_class = int(round(cfg.top_k * class_alloc.get(asset_class, 0.0)))
            k_class = max(1, min(len(sub), k_class))  # ensure at least one if exists

            sub = sub.sort_values("score", ascending=False).head(k_class)

            # ---- score / inverse-vol ----
            raw = sub["score"] / sub["vol"]
            raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)

            if raw.sum() <= 0:
                w = pd.Series(1.0, index=sub.index)
            else:
                w = raw / raw.sum()

            # ---- scale by class allocation ----
            class_w = class_alloc.get(asset_class, 0.0)
            for ticker, w_i in w.items():
                final[ticker] = class_w * w_i

        # ---- normalize entire result ----
        total = sum(final.values())
        if total <= 0:
            return {}

        final = {t: w / total for t, w in final.items()}
        return final

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_signals_snapshot(
        self,
        tickers: List[str],
        start: pd.Timestamp,  # not used; individual signal start dates are inferred from window and buffer
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Compute all required signals for the given tickers as-of `end` date.
        Returns a DataFrame with one row per ticker and columns for each
        signal.

        Note that if the as-of end date does not have data (e.g. weekend/holiday),
        we use the most recent available data before that date.
        """

        cfg = self.config
        buffer = pd.Timedelta(days=cfg.signals_extra_buffer_days)
        rows = []

        for t in tickers:
            try:
                mom_fast_start = end - pd.Timedelta(days=cfg.mom_fast_window) - buffer
                mom_fast = self.signals.get_series(
                    t,
                    "ts_mom",
                    start=mom_fast_start,
                    end=end,
                    window=cfg.mom_fast_window,
                )
                mom_slow_start = end - pd.Timedelta(days=cfg.mom_slow_window) - buffer
                mom_slow = self.signals.get_series(
                    t,
                    "ts_mom",
                    start=mom_slow_start,
                    end=end,
                    window=cfg.mom_slow_window,
                )
                vol_start = end - pd.Timedelta(days=cfg.vol_window) - buffer
                vol = self.signals.get_series(
                    t, "vol", start=vol_start, end=end, window=cfg.vol_window
                )
                beta_start = end - pd.Timedelta(days=cfg.beta_window) - buffer
                beta = self.signals.get_series(
                    t,
                    "beta",
                    start=beta_start,
                    end=end,
                    window=cfg.beta_window,
                    benchmark="SPY",
                )

                rows.append(
                    {
                        "ticker": t,
                        "mom_fast": mom_fast.iloc[-1] if not mom_fast.empty else np.nan,
                        "mom_slow": mom_slow.iloc[-1] if not mom_slow.empty else np.nan,
                        "vol": vol.iloc[-1] if not vol.empty else np.nan,
                        "beta": (
                            beta.iloc[-1]
                            if (beta is not None and not beta.empty)
                            else np.nan
                        ),
                    }
                )
            except Exception:
                continue

        df = pd.DataFrame(rows).set_index("ticker")
        return df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["mom_fast", "mom_slow", "vol"]
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_scores_only(self, sigs: pd.DataFrame) -> pd.DataFrame:
        """
        Produce composite score but DO NOT select top_k globally.
        """
        cfg = self.config
        df = sigs.copy()

        def r01(s, ascending):
            return s.rank(method="average", ascending=ascending, pct=True)

        df["r_mom_fast"] = r01(df["mom_fast"], ascending=False)
        df["r_mom_slow"] = r01(df["mom_slow"], ascending=False)
        df["r_vol"] = r01(df["vol"], ascending=True)
        df["r_beta"] = (
            r01(df["beta"], ascending=True) if df["beta"].notna().any() else 0.5
        )

        df["score"] = (
            cfg.w_mom_fast * df["r_mom_fast"]
            + cfg.w_mom_slow * df["r_mom_slow"]
            + cfg.w_low_vol * df["r_vol"]
            + cfg.w_low_beta * df["r_beta"]
        )

        return df.sort_values("score", ascending=False)

    # ------------------------------------------------------------------
    # Precompute helpers
    # ------------------------------------------------------------------
    def _compute_warmup_start(
        self, cfg: DefensiveConfig, start_ts: pd.Timestamp, warmup_buffer: Optional[int]
    ) -> pd.Timestamp:
        mom_fast_w = int(getattr(cfg, "mom_fast_window", 50))
        mom_slow_w = int(getattr(cfg, "mom_slow_window", 200))
        vol_w = int(getattr(cfg, "vol_window", 20))
        beta_w = int(getattr(cfg, "beta_window", 63))

        max_window = max(mom_fast_w, mom_slow_w, vol_w, beta_w)
        warmup_days = int(np.ceil(max_window * (365 / 252))) + (warmup_buffer or 0)
        return start_ts - pd.Timedelta(days=warmup_days)

    def _load_full_price_matrix(
        self, warmup_start: pd.Timestamp, end_ts: pd.Timestamp, tickers: List[str]
    ):
        price_mat = self.um.get_price_matrix(
            price_loader=self.mds,
            start=warmup_start,
            end=end_ts,
            tickers=tickers,
            field="Close",
            interval="1d",
            auto_adjust=True,
            auto_apply_membership_mask=False,
            local_only=getattr(self.mds, "local_only", False),
        )
        price_mat = price_mat.dropna(axis=1, how="all")
        if price_mat.empty:
            return pd.DataFrame()
        price_mat.columns = [c.upper() for c in price_mat.columns]
        return price_mat

    def _compute_vectorized_signal_matrices(
        self,
        price_mat: pd.DataFrame,
        cfg: DefensiveConfig,
        warmup_start: pd.Timestamp,
        end_ts: pd.Timestamp,
    ):
        VSE = self.vec_engine

        mom_fast_w = int(getattr(cfg, "mom_fast_window", 50))
        mom_slow_w = int(getattr(cfg, "mom_slow_window", 200))
        vol_w = int(getattr(cfg, "vol_window", 20))
        beta_w = int(getattr(cfg, "beta_window", 63))

        mom_dict = VSE.get_momentum(price_mat, [mom_fast_w, mom_slow_w])
        mom_fast_mat = mom_dict.get(
            mom_fast_w, pd.DataFrame(index=price_mat.index, columns=price_mat.columns)
        )
        mom_slow_mat = mom_dict.get(
            mom_slow_w, pd.DataFrame(index=price_mat.index, columns=price_mat.columns)
        )

        vol_mat = VSE.get_volatility(
            price_mat, window=vol_w, annualize=True, interval="1d"
        )

        bench = getattr(cfg, "beta_benchmark", "SPY")
        beta_mat = VSE.get_beta(
            price_mat, window=beta_w, benchmark=bench, price_col="Close", interval="1d"
        )

        return mom_fast_mat, mom_slow_mat, vol_mat, beta_mat

    def _build_scores_from_signal_matrices(
        self,
        price_mat: pd.DataFrame,
        mom_fast_mat: pd.DataFrame,
        mom_slow_mat: pd.DataFrame,
        vol_mat: pd.DataFrame,
        beta_mat: pd.DataFrame,
        warmup_start: pd.Timestamp,
        end_ts: pd.Timestamp,
        cfg: DefensiveConfig,
    ):
        # Liquidity filters
        min_price = float(getattr(cfg, "min_price", 0.0))
        min_adv = float(getattr(cfg, "min_adv", 0.0))
        adv_window = int(getattr(cfg, "min_adv_window", 20))

        tickers = price_mat.columns.tolist()
        VSE = self.vec_engine
        vol_field_mat = VSE.get_field_matrix(
            tickers,
            start=warmup_start,
            end=end_ts,
            field="Volume",
            interval="1d",
            local_only=getattr(self.mds, "local_only", False),
            auto_adjust=True,
            membership_aware=False,
            treat_unknown_as_always_member=True,
        )

        adv_mat = vol_field_mat.rolling(
            window=adv_window, min_periods=int(np.floor(adv_window / 2))
        ).mean()

        price_aligned = price_mat.reindex(
            index=vol_field_mat.index, columns=vol_field_mat.columns
        )
        price_mask = price_aligned >= min_price
        adv_mask = adv_mat >= min_adv
        liq_mask = price_mask & adv_mask

        mom_fast_mat = mom_fast_mat.where(liq_mask)
        mom_slow_mat = mom_slow_mat.where(liq_mask)
        vol_mat = vol_mat.where(liq_mask)
        beta_mat = beta_mat.where(liq_mask)

        # --- Apply membership mask to raw signals BEFORE ranking/scores ---
        try:
            mem_mask = self.um.membership_mask(start=warmup_start, end=end_ts)
        except Exception:
            self.log.warning("failed to load membership mask, skipping")
            mem_mask = None

        if mem_mask is not None and not mem_mask.empty:
            mem_mask.index = pd.to_datetime(mem_mask.index)
            mem_mask = mem_mask.reindex(index=price_mat.index)
            mem_mask = mem_mask.reindex(columns=price_mat.columns)
            # Treat unknown / non-index tickers (NaN cols) as always member
            mem_mask = mem_mask.fillna(True)
            mom_fast_mat = mom_fast_mat.where(mem_mask)
            mom_slow_mat = mom_slow_mat.where(mem_mask)
            vol_mat = vol_mat.where(mem_mask)
            beta_mat = beta_mat.where(mem_mask)

        def rank_rowwise(df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
            return df.rank(axis=1, method="average", pct=True, ascending=ascending)

        r_mom_fast = rank_rowwise(mom_fast_mat, ascending=False)
        r_mom_slow = rank_rowwise(mom_slow_mat, ascending=False)
        r_vol = rank_rowwise(vol_mat, ascending=True)
        r_beta = rank_rowwise(beta_mat, ascending=True)

        w_mom_fast = float(getattr(cfg, "w_mom_fast", 0.5))
        w_mom_slow = float(getattr(cfg, "w_mom_slow", 0.0))
        w_low_vol = float(getattr(cfg, "w_low_vol", 0.5))
        w_low_beta = float(getattr(cfg, "w_low_beta", 0.0))

        score_mat = (
            w_mom_fast * r_mom_fast
            + w_mom_slow * r_mom_slow
            + w_low_vol * r_vol
            + w_low_beta * r_beta
        )

        # Keep warmup..end range
        score_mat = score_mat.loc[
            (score_mat.index >= warmup_start) & (score_mat.index <= end_ts)
        ]

        return (
            score_mat,
            mom_fast_mat.loc[score_mat.index],
            mom_slow_mat.loc[score_mat.index],
            vol_mat.loc[score_mat.index],
            beta_mat.loc[score_mat.index],
        )

    def _cache_precompute_results(
        self,
        score_mat: pd.DataFrame,
        mom_fast_mat: pd.DataFrame,
        mom_slow_mat: pd.DataFrame,
        vol_mat: pd.DataFrame,
        beta_mat: pd.DataFrame,
    ):
        # Cache results (membership already applied to raw signals)
        self._cached_scores_mat = score_mat
        self._cached_signal_mats = {
            "mom_fast": mom_fast_mat,
            "mom_slow": mom_slow_mat,
            "vol": vol_mat,
            "beta": beta_mat,
        }

        return score_mat

    # ------------------------------------------------------------------
    # Precompute (vectorized end-to-end)
    # ------------------------------------------------------------------
    def precompute(
        self,
        start: datetime | str,
        end: datetime | str,
        sample_dates: Optional[List[datetime | str]] = None,
        warmup_buffer: Optional[int] = None,  # in days
    ) -> pd.DataFrame:
        """
        Vectorized precompute for DefensiveSleeve.

        Computes signal matrices (mom_fast, mom_slow, vol, beta) and a composite
        score matrix for all tickers in the defensive universe over the specified
        date range [start, end]. Caches the results for later use during rebalancing.
        """
        cfg = self.config
        start_ts = pd.to_datetime(start).normalize()
        end_ts = pd.to_datetime(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        warmup_start = self._compute_warmup_start(cfg, start_ts, warmup_buffer)

        # Build defensive universe
        # Note: as_of=None to get full universe, because we apply membership mask later
        tickers = self._get_defensive_universe(as_of=None)
        if not tickers:
            self._cached_scores_mat = pd.DataFrame()
            self._cached_signal_mats = {}
            return self._cached_scores_mat

        price_mat = self._load_full_price_matrix(warmup_start, end_ts, tickers)
        if price_mat.empty:
            self.log.warning("empty price matrix in precompute, skipping")
            self._cached_scores_mat = pd.DataFrame()
            self._cached_signal_mats = {}
            return self._cached_scores_mat

        mom_fast_mat, mom_slow_mat, vol_mat, beta_mat = (
            self._compute_vectorized_signal_matrices(
                price_mat, cfg, warmup_start, end_ts
            )
        )

        score_mat, mom_fast_cut, mom_slow_cut, vol_cut, beta_cut = (
            self._build_scores_from_signal_matrices(
                price_mat,
                mom_fast_mat,
                mom_slow_mat,
                vol_mat,
                beta_mat,
                warmup_start,
                end_ts,
                cfg,
            )
        )

        score_mat = self._cache_precompute_results(
            score_mat, mom_fast_cut, mom_slow_cut, vol_cut, beta_cut
        )

        return score_mat
