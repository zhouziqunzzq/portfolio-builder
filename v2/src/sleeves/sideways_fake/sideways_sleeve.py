from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Make v2/src importable by adding it to sys.path.
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[2]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from universe_manager import UniverseManager
from market_data_store import MarketDataStore
from signal_engine import SignalEngine
from vec_signal_engine import VectorizedSignalEngine
from utils.stats import (
    zscore_matrix_column_wise,
    windorized_zscore_matrix_column_wise,
    sector_mean_matrix,
)
from .sideways_config import SidewaysConfig


# ----------------------------------------------------------------------
# State (very light; mainly here for API symmetry)
# ----------------------------------------------------------------------


@dataclass
class SidewaysState:
    last_as_of: Optional[pd.Timestamp] = None
    last_sector_weights: Optional[pd.Series] = None


# ----------------------------------------------------------------------
# SidewaysSleeve
# ----------------------------------------------------------------------


class SidewaysSleeve:
    """
    Sideways / Mean-Reversion Sleeve (stock-based, sector-aware).

    MVP design:
      - Uses the same vectorized pipeline as TrendSleeve.
      - BUT the composite stock_score is *sign-flipped* (anti-momentum):
            stock_score_sideways = - stock_score_trend
        so that high scores correspond to *expected mean-reversion* winners.

      - Only implements the precomputed, fully vectorized path:
            * precompute(...)
            * generate_target_weights_for_date(...) reading from cache

      - Regime gating:
            - Uses SidewaysConfig.gated_off_regimes if use_regime_gating=True.
            - In practice you will mostly control ON/OFF at the allocator level
              by giving this sleeve nonzero weights only in sideways regimes.
    """

    def __init__(
        self,
        universe: UniverseManager,
        mds: MarketDataStore,
        signals: SignalEngine,  # Not used directly; only for API compatibility
        vec_engine: Optional[VectorizedSignalEngine] = None,
        config: Optional[SidewaysConfig] = None,
    ) -> None:
        self.um = universe
        self.mds = mds
        self.signals = signals  # Not used directly; only for API compatibility
        self.vec_engine = vec_engine
        self.config: SidewaysConfig = config or SidewaysConfig()
        self.state = SidewaysState()
        self.precomputed_weights_mat: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_target_weights_for_date(
        self,
        as_of: datetime | str,
        start_for_signals: datetime | str,  # unused for MVP; kept for API compatibility
        regime: str = "sideways",
    ) -> Dict[str, float]:
        """
        For MVP, we *only* support the precomputed vectorized path.

        Workflow:
          - MultiSleeveAllocator.precompute(...) calls self.precompute(...)
          - Later, on each rebalance date, allocator calls this method.
          - We simply look up the cached weights for `as_of` date.

        Regime gating:
          - If config.use_regime_gating is True and the regime is in
            config.gated_off_regimes, we return an empty dict.
          - Otherwise we use the precomputed weights if present.
        """
        as_of = pd.to_datetime(as_of).normalize()
        regime_key = (regime or "").lower()
        cfg = self.config

        # ---------- Regime-based gating (optional) ----------
        if getattr(cfg, "use_regime_gating", False):
            gated_off = {r.lower() for r in cfg.gated_off_regimes}
            if regime_key in gated_off:
                print(
                    f"[SidewaysSleeve] Regime '{regime_key}' is gated off; returning empty weights."
                )
                return {}

        # ---------- Cached vectorized weights ----------
        if (
            self.precomputed_weights_mat is not None
            and not self.precomputed_weights_mat.empty
        ):
            if as_of in self.precomputed_weights_mat.index:
                print(f"[SidewaysSleeve] Using precomputed weights for {as_of.date()}")
                row = self.precomputed_weights_mat.loc[as_of]
                weights = {t: float(w) for t, w in row.items() if w > 0}
                total = sum(weights.values())
                if total > 0:
                    return {t: w / total for t, w in weights.items()}
                return {}

        # If no precomputed weights (or date missing), return empty.
        return {}

    # ------------------------------------------------------------------
    # Vectorized stock scoring (anti-momentum)
    # ------------------------------------------------------------------

    def _compute_stock_scores_vectorized(
        self,
        price_mat: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fully vectorized stock score computation over ALL dates.

        Produces a Date x Ticker matrix with:
            - stock_score_sideways(t, ticker)

        This is derived from the TrendSleeve stock_score, but sign-flipped:
            stock_score_sideways = - (mom_part + vol_penalty * vol_score)

        No liquidity filtering here (that's handled downstream).
        """
        cfg = self.config
        VSE = self.vec_engine
        if VSE is None:
            raise ValueError("vec_engine is not set; cannot compute stock scores.")

        # --------------------------------------------------------
        # 1) Vectorized raw signals
        # --------------------------------------------------------
        mom_dict = VSE.get_momentum(price_mat, cfg.mom_windows)
        vol_mat = VSE.get_volatility(price_mat, window=cfg.vol_window)

        feature_mats: Dict[str, pd.DataFrame] = {}

        # Momentum matrices
        for w in cfg.mom_windows:
            feature_mats[f"mom_{w}"] = mom_dict[w]

        # Volatility matrix
        feature_mats["vol"] = vol_mat

        # --------------------------------------------------------
        # 2) Z scoring per date (cross-sectional)
        # --------------------------------------------------------
        z_mats: Dict[str, pd.DataFrame] = {}

        for w in cfg.mom_windows:
            mat = feature_mats[f"mom_{w}"]
            if cfg.use_zscore_winsorization:
                z = windorized_zscore_matrix_column_wise(mat, clip=cfg.zscore_clip)
            else:
                z = zscore_matrix_column_wise(mat)
            z_mats[f"z_mom_{w}"] = z

        # Vol Z-score (column-wise)
        if cfg.use_zscore_winsorization:
            z_vol = windorized_zscore_matrix_column_wise(
                feature_mats["vol"],
                clip=cfg.zscore_clip,
            )
        else:
            z_vol = zscore_matrix_column_wise(feature_mats["vol"])

        # For stock selection we want *lower* vol to be better:
        z_mats["vol_score"] = -z_vol

        # --------------------------------------------------------
        # 3) Composite score (before flip)
        # --------------------------------------------------------
        mom_part: Optional[pd.DataFrame] = None
        for w, wt in zip(cfg.mom_windows, cfg.mom_weights):
            if mom_part is None:
                mom_part = wt * z_mats[f"z_mom_{w}"]
            else:
                mom_part += wt * z_mats[f"z_mom_{w}"]

        if mom_part is None:
            # Should not happen if mom_windows is non-empty
            return pd.DataFrame(index=price_mat.index, columns=price_mat.columns)

        stock_score_mat_trend_like = mom_part.add(
            cfg.vol_penalty * z_mats["vol_score"],
            fill_value=0.0,
        )

        # --------------------------------------------------------
        # 4) Sideways stock score = negative of trend-like score
        # --------------------------------------------------------
        stock_score_mat = -stock_score_mat_trend_like
        stock_score_mat.name = "stock_score"

        return stock_score_mat

    # ------------------------------------------------------------------
    # Sector scoring & smoothing (vectorized)
    # ------------------------------------------------------------------

    def _compute_sector_scores_vectorized(
        self,
        stock_score_mat: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Vectorized sector scores over time.

        Returns Date x Sector matrix of mean stock_score per sector.
        """
        smap = sector_map or self.um.sector_map or {}
        if stock_score_mat is None or stock_score_mat.empty:
            return pd.DataFrame(index=getattr(stock_score_mat, "index", None))

        return sector_mean_matrix(
            stock_score_mat=stock_score_mat,
            sector_map=smap,
        )

    def _softmax(self, scores: pd.Series) -> pd.Series:
        cfg = self.config
        x = cfg.sector_softmax_alpha * scores.astype(float)
        x = x.fillna(0.0)
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _apply_caps(self, w: pd.Series) -> pd.Series:
        cfg = self.config
        w2 = w.clip(lower=cfg.sector_w_min, upper=cfg.sector_w_max)
        total = w2.sum()
        if total <= 0:
            return pd.Series(1.0 / len(w), index=w.index)
        return w2 / total

    def _smooth_weights(
        self,
        prev_weights: Optional[pd.Series],
        new_weights: pd.Series,
    ) -> pd.Series:
        """
        Exponential smoothing:
            w_smoothed = (1 - beta) * prev + beta * new
        """
        if prev_weights is None:
            return new_weights

        cfg = self.config
        beta = cfg.sector_smoothing_beta

        prev_weights = prev_weights.reindex(new_weights.index).fillna(0.0)
        w_smoothed = (1 - beta) * prev_weights + beta * new_weights

        total = w_smoothed.sum()
        if total <= 0:
            return pd.Series(1.0 / len(w_smoothed), index=w_smoothed.index, dtype=float)
        return w_smoothed / total

    def _top_k(self, w: pd.Series) -> pd.Series:
        cfg = self.config
        k = cfg.sector_top_k
        if k is None or k >= len(w):
            total = w.sum()
            if total > 0:
                return w / total
            return pd.Series(1.0 / len(w), index=w.index)

        keep = w.nlargest(k).index
        w2 = w.copy()
        w2.loc[~w2.index.isin(keep)] = 0.0
        total = w2.sum()
        if total <= 0:
            w2.loc[keep] = 1.0 / len(keep)
            return w2
        return w2 / total

    def _compute_sector_weights_vectorized(
        self,
        sector_scores_mat: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Vectorized sector weights: Date x Sector matrix, with
        per-day softmax/caps/top-k and forward smoothing.
        """
        if sector_scores_mat.empty:
            return pd.DataFrame()

        dates = sector_scores_mat.index
        sectors = list(sector_scores_mat.columns)
        weights_list: List[pd.Series] = []
        prev_smoothed: Optional[pd.Series] = None

        for dt in dates:
            scores_t = sector_scores_mat.loc[dt]
            scores_t_nonan = scores_t.dropna()

            if scores_t_nonan.empty:
                # carry forward or uniform
                if prev_smoothed is not None:
                    w_final = prev_smoothed.reindex(sectors).fillna(0.0)
                    w_smoothed = w_final
                else:
                    w_final = pd.Series(1.0 / len(sectors), index=sectors)
                    w_smoothed = w_final
            else:
                w_raw = self._softmax(scores_t)
                w_capped = self._apply_caps(w_raw)
                w_smoothed = self._smooth_weights(prev_smoothed, w_capped)
                w_selected = self._top_k(w_smoothed)
                w_final = w_selected.reindex(sectors).fillna(0.0)

            weights_list.append(w_final)
            prev_smoothed = w_smoothed

        sector_weights_mat = pd.DataFrame(weights_list, index=dates, columns=sectors)
        return sector_weights_mat

    # ------------------------------------------------------------------
    # Vectorized stock allocation (with liquidity mask)
    # ------------------------------------------------------------------

    def _allocate_to_stocks_vectorized(
        self,
        price_mat: pd.DataFrame,
        stock_score_mat: pd.DataFrame,
        sector_weights_mat: pd.DataFrame,
        vol_mat: Optional[pd.DataFrame] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Vectorized stock allocation:

            Inputs:
                price_mat          : Date x Ticker (Close)
                stock_score_mat    : Date x Ticker (sideways stock_score)
                sector_weights_mat : Date x Sector
                vol_mat            : Date x Ticker (realized vol); if None, compute.

            Output:
                Date x Ticker portfolio weights.
        """
        cfg = self.config
        smap = sector_map or self.um.sector_map or {}
        VSE = self.vec_engine
        if VSE is None:
            raise ValueError("vec_engine is not set; cannot allocate to stocks.")

        dates = stock_score_mat.index
        tickers = stock_score_mat.columns.tolist()

        out = pd.DataFrame(0.0, index=dates, columns=tickers)

        # 1) Liquidity mask
        liq_mask: Optional[pd.DataFrame] = None
        if getattr(cfg, "use_liquidity_filters", False):
            adv_window = getattr(cfg, "adv_window", 20)
            median_volume_window = getattr(cfg, "median_volume_window", 20)
            min_price = getattr(cfg, "min_price", None)
            min_adv = getattr(cfg, "min_adv20", None)
            min_medvol = getattr(cfg, "min_median_volume20", None)

            try:
                volume_mat = VSE.get_field_matrix(
                    tickers=tickers,
                    start=price_mat.index.min(),
                    end=price_mat.index.max(),
                    field="Volume",
                    interval="1d",
                    local_only=getattr(self.mds, "local_only", False),
                    auto_adjust=False,
                    membership_aware=False,
                    treat_unknown_as_always_member=True,
                )
            except Exception:
                volume_mat = pd.DataFrame(index=price_mat.index, columns=tickers)

            volume_mat = volume_mat.reindex_like(price_mat)

            dollar_vol = price_mat * volume_mat
            adv_mat = dollar_vol.rolling(
                adv_window,
                min_periods=max(5, adv_window // 2),
            ).mean()
            medvol_mat = volume_mat.rolling(
                median_volume_window,
                min_periods=max(5, median_volume_window // 2),
            ).median()

            liq_mask = pd.DataFrame(True, index=dates, columns=tickers)
            if min_price is not None:
                liq_mask &= price_mat >= float(min_price)
            if min_adv is not None:
                liq_mask &= adv_mat >= float(min_adv)
            if min_medvol is not None:
                liq_mask &= medvol_mat >= float(min_medvol)

            liq_mask = liq_mask.fillna(False)

        # 2) sector -> tickers map
        sector_to_tickers: Dict[str, List[str]] = {}
        for t in tickers:
            sec = smap.get(t)
            if isinstance(sec, str) and sec.strip() not in ("", "Unknown"):
                sector_to_tickers.setdefault(sec, []).append(t)

        # 3) Intra-day allocation
        if cfg.weighting_mode == "inverse-vol" and vol_mat is None:
            vol_mat = VSE.get_volatility(
                price_mat,
                window=cfg.vol_window,
            )

        for dt in dates:
            scores_t = stock_score_mat.loc[dt]
            if cfg.weighting_mode == "inverse-vol":
                vols_t = vol_mat.loc[dt]
            sector_w_t = sector_weights_mat.loc[dt]

            eligible_tickers = None
            if liq_mask is not None:
                eligible_tickers = liq_mask.loc[dt]

            for sector, w_sec in sector_w_t.items():
                if w_sec <= 0 or pd.isna(w_sec):
                    continue

                candidates = sector_to_tickers.get(sector)
                if not candidates:
                    continue

                if eligible_tickers is not None:
                    candidates = [
                        t for t in candidates if bool(eligible_tickers.get(t, False))
                    ]
                if not candidates:
                    continue

                # Top-k by *sideways* stock_score within sector
                s = scores_t[candidates].dropna()
                if s.empty:
                    continue
                top = s.nlargest(cfg.top_k_per_sector).index.tolist()

                if cfg.weighting_mode == "inverse-vol":
                    v = vols_t[top].astype(float)
                    v = v.replace({0.0: np.nan})
                    inv = 1.0 / v
                    inv = inv.replace({np.inf: np.nan}).dropna()
                    if inv.empty:
                        intra = pd.Series(1.0 / len(top), index=top)
                    else:
                        intra = inv / inv.sum()
                else:
                    intra = pd.Series(1.0 / len(top), index=top)

                out.loc[dt, intra.index] += w_sec * intra

        # 4) Normalize per-day
        row_sums = out.sum(axis=1).replace(0.0, np.nan)
        out = out.div(row_sums, axis=0).fillna(0.0)

        return out

    # ------------------------------------------------------------------
    # Precompute (vectorized end-to-end)
    # ------------------------------------------------------------------

    def precompute(
        self,
        start: datetime | str,
        end: datetime | str,
        rebalance_dates: Optional[List[datetime | str]] = None,
        warmup_buffer: int = 30,
    ) -> pd.DataFrame:
        """
        Vectorized precomputation of final stock weights over [start, end].

        Steps:
          1) Build price matrix over [warmup_start, end]
          2) Vectorized sideways stock scores (Date x Ticker)
          3) Vectorized sector scores (Date x Sector)
          4) Vectorized sector weights (Date x Sector)
          5) Vectorized stock allocation (Date x Ticker)
          6) Slice to [start, end]; optionally sample by rebalance_dates

        Result stored in self.precomputed_weights_mat.
        """
        if self.vec_engine is None:
            raise ValueError("vec_engine is not set; cannot precompute.")

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        cfg = self.config

        # Warmup length from signal-related windows
        window_candidates: List[int] = []
        window_candidates.extend(getattr(cfg, "mom_windows", []) or [])
        for attr_name in [
            "vol_window",
            "adv_window",
            "median_volume_window",
            "liquidity_window",
        ]:
            w_val = getattr(cfg, attr_name, None)
            if isinstance(w_val, int) and w_val > 0:
                window_candidates.append(w_val)
        max_window = max(window_candidates) if window_candidates else 0
        warmup_days = max_window + int(warmup_buffer)
        warmup_start = start_ts - pd.Timedelta(days=warmup_days)

        # 1) Price matrix (membership-aware)
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
        sector_map = self.um.sector_map
        if sector_map is not None:
            sector_map = {t.upper(): s for t, s in sector_map.items()}
            sector_map = {t: s for t, s in sector_map.items() if t in price_mat.columns}

        # 2) Vectorized sideways stock scores
        stock_score_mat = self._compute_stock_scores_vectorized(price_mat)
        if stock_score_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # 3) Sector scores
        sector_scores_mat = self._compute_sector_scores_vectorized(
            stock_score_mat, sector_map=sector_map
        )
        if sector_scores_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # 4) Sector weights
        sector_weights_mat = self._compute_sector_weights_vectorized(sector_scores_mat)
        if sector_weights_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # 5) Stock allocations
        alloc_mat = self._allocate_to_stocks_vectorized(
            price_mat=price_mat,
            stock_score_mat=stock_score_mat,
            sector_weights_mat=sector_weights_mat,
            sector_map=sector_map,
        )
        if alloc_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # 6) Slice to [start, end] and align to calendar
        alloc_mat = alloc_mat.loc[
            (alloc_mat.index >= start_ts) & (alloc_mat.index <= end_ts)
        ]
        alloc_mat = alloc_mat.asfreq("D", method="ffill").fillna(0.0)

        if rebalance_dates:
            target_dates = pd.to_datetime(rebalance_dates).normalize()
            alloc_mat = alloc_mat[alloc_mat.index.isin(target_dates)]

        self.precomputed_weights_mat = alloc_mat.copy()
        return self.precomputed_weights_mat

    # Convenience accessor
    def get_precomputed_weights(self) -> Optional[pd.DataFrame]:
        return self.precomputed_weights_mat
