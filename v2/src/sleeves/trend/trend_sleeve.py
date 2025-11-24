from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from src.vec_signal_engine import VectorizedSignalEngine
from src.utils.stats import winsorized_zscore, sector_mean_snapshot, sector_mean_matrix
from .trend_config import TrendConfig


@dataclass
class TrendState:
    last_as_of: Optional[pd.Timestamp] = None
    last_sector_weights: Optional[pd.Series] = None


class TrendSleeve:
    """
    Trend / Momentum Sleeve (stock-based, sector-aware).

    Key features:
      - Uses S&P 500 membership as-of date
      - Sector-aware scoring & smoothing
      - Momentum + volatility composite stock scoring
      - Liquidity filters (ADV / median volume / price)
      - Top-k per sector selection
      - Pure-stock sleeve (no ETFs for now)
    """

    def __init__(
        self,
        universe: UniverseManager,
        mds: MarketDataStore,
        signals: SignalEngine,
        vec_engine: Optional[VectorizedSignalEngine] = None,
        config: Optional[TrendConfig] = None,
    ) -> None:
        self.um = universe
        self.mds = mds
        self.signals = signals
        self.vec_engine = vec_engine
        self.config = config or TrendConfig()
        self.state = TrendState()
        self.precomputed_weights_mat: Optional[pd.DataFrame] = None

        self._approx_rebalance_days = self._infer_approx_rebalance_days(
            self.config.rebalance_freq
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_target_weights_for_date(
        self,
        as_of: datetime | str,
        start_for_signals: datetime | str,
        regime: str = "bull",
    ) -> Dict[str, float]:

        as_of = pd.to_datetime(as_of)
        start_for_signals = pd.to_datetime(start_for_signals)

        cfg = self.config
        regime_key = (regime or "").lower()

        # ---------- Regime-based gating ----------
        if cfg.use_regime_gating:
            gated_off = {r.lower() for r in cfg.gated_off_regimes}
            if regime_key in gated_off:
                # Sleeve is turned completely OFF in these regimes.
                # We *don't* update smoothing state here; next active call
                # will treat it as a fresh start or a long gap.
                return {}

        # ---------- Cached vectorized weights (if precomputed) ----------
        # If precompute() was called, we can directly return weights for this date
        # without re-running the single-date pipeline. We still performed regime gating above.
        if self.precomputed_weights_mat is not None and not self.precomputed_weights_mat.empty:
            date_key = as_of.normalize()
            if date_key in self.precomputed_weights_mat.index:
                row = self.precomputed_weights_mat.loc[date_key]
                weights = {t: float(w) for t, w in row.items() if w > 0}
                total = sum(weights.values())
                if total > 0:
                    return {t: w / total for t, w in weights.items()}
                return {}

        # ---------- Normal trend sleeve logic below ----------

        # 1) TIME-AWARE UNIVERSE
        universe = self._get_trend_universe(as_of)
        if not universe:
            return {}

        # 2) Compute signals (with liquidity filters)
        sigs = self._compute_signals_snapshot(universe, start_for_signals, as_of)
        if sigs.empty:
            return {}

        # 3) Stock scores
        scored = self._compute_stock_scores(sigs)
        if scored.empty:
            return {}

        # 4) Sector scores & smoothed sector weights
        sector_scores = self._compute_sector_scores(scored)
        if sector_scores.empty:
            return {}

        sector_weights = self._compute_smoothed_sector_weights(as_of, sector_scores)
        if sector_weights.isna().all() or sector_weights.sum() <= 0:
            return {}

        # 5) Allocate sector -> stocks
        stock_weights = self._allocate_to_stocks(scored, sector_weights)
        if not stock_weights:
            return {}

        total = sum(stock_weights.values())
        if total <= 0:
            return {}

        # Normalize
        stock_weights = {t: w / total for t, w in stock_weights.items()}
        return stock_weights

    # ------------------------------------------------------------------
    # TIME-AWARE UNIVERSE
    # ------------------------------------------------------------------

    def _get_trend_universe(
        self,
        as_of: Optional[datetime | str] = None,
    ) -> List[str]:
        """
        Build a time-aware S&P 500 universe:
          - Use S&P membership as-of the given date
          - Filter out tickers with missing / invalid sectors
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
                active_tickers = set(row.index[row.astype(bool)])

        core: List[str] = []
        for t, sec in smap.items():
            # must have a valid sector
            if not isinstance(sec, str) or sec.strip() in ("", "Unknown"):
                continue

            if active_tickers is not None and t not in active_tickers:
                continue

            core.append(t)

        return sorted(set(core))

    # ------------------------------------------------------------------
    # Signal computation (with liquidity filters)
    # ------------------------------------------------------------------

    def _compute_signals_snapshot(
        self,
        tickers: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        For each ticker, compute:
          - multi-horizon momentum (mom_{w})
          - realized volatility (vol)
          - liquidity metrics: adv, median_volume, last_price

        Apply hard liquidity filters:
          - last_price >= cfg.min_price
          - adv >= cfg.min_adv
          - median_volume >= cfg.min_median_volume

        Only tickers passing all filters are kept.
        """
        cfg = self.config
        rows = []

        # Pull thresholds & windows (with graceful defaults if missing)
        adv_window = getattr(cfg, "adv_window", 20)
        mv_window = getattr(cfg, "median_volume_window", 20)
        min_adv = getattr(cfg, "min_adv", None)
        min_median_vol = getattr(cfg, "min_median_volume", None)
        min_price = getattr(cfg, "min_price", None)

        for t in tickers:
            try:
                row = {"ticker": t}

                # --- Liquidity metrics ---
                adv = self.signals.get_series(
                    t,
                    "adv",
                    start=start,
                    end=end,
                    window=adv_window,
                )
                mv = self.signals.get_series(
                    t,
                    "median_volume",
                    start=start,
                    end=end,
                    window=mv_window,
                )
                px = self.signals.get_series(
                    t,
                    "last_price",
                    start=start,
                    end=end,
                )

                adv_val = adv.iloc[-1] if not adv.empty else np.nan
                mv_val = mv.iloc[-1] if not mv.empty else np.nan
                px_val = px.iloc[-1] if not px.empty else np.nan

                # Hard liquidity filters (if thresholds provided)
                if min_price is not None and (np.isnan(px_val) or px_val < min_price):
                    continue
                if min_adv is not None and (np.isnan(adv_val) or adv_val < min_adv):
                    continue
                if min_median_vol is not None and (
                    np.isnan(mv_val) or mv_val < min_median_vol
                ):
                    continue

                row["adv"] = adv_val
                row["median_volume"] = mv_val
                row["last_price"] = px_val

                # --- Multi-horizon momentum ---
                for w in cfg.mom_windows:
                    mom = self.signals.get_series(
                        t,
                        "ts_mom",
                        start=start,
                        end=end,
                        window=w,
                    )
                    row[f"mom_{w}"] = mom.iloc[-1] if not mom.empty else np.nan

                # --- Realized volatility ---
                vol = self.signals.get_series(
                    t,
                    "vol",
                    start=start,
                    end=end,
                    window=cfg.vol_window,
                )
                row["vol"] = vol.iloc[-1] if not vol.empty else np.nan

                rows.append(row)

            except Exception:
                # For robustness: skip this ticker if anything fails
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")

        mom_cols = [f"mom_{w}" for w in cfg.mom_windows]
        keep_cols = mom_cols + ["vol"]

        # Drop rows missing key signal inputs
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols, how="any")
        return df

    # ------------------------------------------------------------------
    # Stock scoring (with winsorized z-scores)
    # ------------------------------------------------------------------

    def _zscore_series(self, s: pd.Series) -> pd.Series:
        """
        Standard z-score with optional winsorization based on cfg.zscore_clip.
        """
        cfg = self.config
        s = s.astype(float)
        mean = s.mean()
        std = s.std()
        if std is None or std == 0 or np.isnan(std):
            z = pd.Series(0.0, index=s.index)
        else:
            z = (s - mean) / std

        zclip = getattr(cfg, "zscore_clip", None)
        if zclip is not None and zclip > 0:
            z = z.clip(lower=-zclip, upper=zclip)

        return z

    def _compute_stock_scores(self, sigs: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = sigs.copy()

        # z-score momentum components
        for w in cfg.mom_windows:
            col = f"mom_{w}"
            df[f"z_mom_{w}"] = self._zscore_series(df[col])

        # weighted momentum score
        momentum_score = pd.Series(0.0, index=df.index)
        for w, wt in zip(cfg.mom_windows, cfg.mom_weights):
            momentum_score += wt * df[f"z_mom_{w}"]

        df["momentum_score"] = momentum_score

        # volatility score (lower vol -> higher score)
        vol_z = self._zscore_series(df["vol"])
        df["vol_score"] = -vol_z

        # stock_score = momentum_score + vol_penalty * vol_score
        df["stock_score"] = df["momentum_score"] + cfg.vol_penalty * df["vol_score"]

        return df

    # ------------------------------------------------------------------
    # Sector scoring & smoothing
    # ------------------------------------------------------------------

    def _compute_sector_scores(self, scored: pd.DataFrame) -> pd.Series:
        """
        Single-date sector scores from a stock_score column.

        scored: DataFrame indexed by ticker, must contain 'stock_score'.
        """
        smap = self.um.sector_map or {}
        if "stock_score" not in scored.columns or scored.empty:
            return pd.Series(dtype=float)

        return sector_mean_snapshot(
            stock_scores=scored["stock_score"],
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

    def _compute_raw_sector_weights(self, sector_scores: pd.Series) -> pd.Series:
        w = self._softmax(sector_scores)
        w = self._apply_caps(w)
        w = self._top_k(w)
        return w

    def _compute_smoothed_sector_weights(
        self,
        as_of: pd.Timestamp,
        sector_scores: pd.Series,
    ) -> pd.Series:
        cfg = self.config
        raw = self._compute_raw_sector_weights(sector_scores)

        if self.state.last_as_of is None or self.state.last_sector_weights is None:
            smoothed = raw
        else:
            gap = (as_of - self.state.last_as_of).days

            if gap <= 0:
                smoothed = raw
            elif gap <= 2 * self._approx_rebalance_days:
                prev = self.state.last_sector_weights.reindex(raw.index).fillna(0.0)
                smoothed = (
                    1 - cfg.sector_smoothing_beta
                ) * prev + cfg.sector_smoothing_beta * raw
                total = smoothed.sum()
                if total > 0:
                    smoothed = smoothed / total
                else:
                    smoothed = raw
            else:
                smoothed = raw

        self.state.last_as_of = as_of
        self.state.last_sector_weights = smoothed.copy()
        return smoothed

    # ------------------------------------------------------------------
    # Allocate sector -> stocks
    # ------------------------------------------------------------------

    def _tickers_in_sector(self, sector: str, candidates: List[str]) -> List[str]:
        smap = self.um.sector_map or {}
        return [t for t in candidates if t in smap and smap[t] == sector]

    def _select_top_k_for_sector(self, sector: str, scored: pd.DataFrame) -> List[str]:
        cfg = self.config
        tickers = self._tickers_in_sector(sector, scored.index.tolist())
        if not tickers:
            return []

        s = scored.loc[tickers, "stock_score"].dropna()
        if s.empty:
            return []
        top = s.sort_values(ascending=False).head(cfg.top_k_per_sector)
        return top.index.tolist()

    def _intra_sector_weights(
        self, tickers: List[str], scored: pd.DataFrame
    ) -> pd.Series:
        cfg = self.config
        if not tickers:
            return pd.Series(dtype=float)

        if cfg.weighting_mode == "inverse-vol":
            vol = scored.loc[tickers, "vol"].astype(float)
            vol = vol.replace({0.0: np.nan})
            if vol.isna().all():
                return pd.Series(1.0 / len(tickers), index=tickers)
            inv = 1.0 / vol
            inv = inv.replace({np.inf: np.nan}).dropna()
            if inv.empty:
                return pd.Series(1.0 / len(tickers), index=tickers)
            w = (inv / inv.sum()).reindex(tickers).fillna(0.0)
            return w

        # equal-weight
        return pd.Series(1.0 / len(tickers), index=tickers)

    def _allocate_to_stocks(
        self,
        scored: pd.DataFrame,
        sector_weights: pd.Series,
    ) -> Dict[str, float]:

        final: Dict[str, float] = {}

        for sector, w_sec in sector_weights.items():
            if w_sec <= 0 or pd.isna(w_sec):
                continue

            tickers = self._select_top_k_for_sector(sector, scored)
            if not tickers:
                continue

            intra = self._intra_sector_weights(tickers, scored)

            for t, w in intra.items():
                final[t] = final.get(t, 0.0) + w_sec * float(w)

        return final

    # --------------------------------------------------
    # Vectorized Implementations
    # --------------------------------------------------
    def _compute_stock_scores_vectorized(
        self,
        price_mat: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fully vectorized stock score computation over ALL dates.

        Produces a Date x Ticker matrix with:
            - mom_{w}(t, ticker)
            - vol(t, ticker)
            - z_mom_{w}(t, ticker)
            - momentum_score(t, ticker)
            - vol_score(t, ticker)
            - stock_score(t, ticker)

        No liquidity filtering here.
        """

        cfg = self.config
        VSE = self.vec_engine

        # --------------------------------------------------------
        # 1) Vectorized raw signals
        # --------------------------------------------------------
        mom_dict = VSE.get_momentum(price_mat, cfg.mom_windows)
        vol_mat = VSE.get_volatility(price_mat, window=cfg.vol_window)

        # Shape: Date x Ticker
        idx = price_mat.index

        # Build a dict: feature_name -> matrix
        feature_mats = {}

        # Momentum matrices
        for w in cfg.mom_windows:
            feature_mats[f"mom_{w}"] = mom_dict[w]

        # Volatility matrix
        feature_mats["vol"] = vol_mat

        # --------------------------------------------------------
        # 2) Z scoring (winsorized) *per date*, cross-sectionally
        # --------------------------------------------------------
        z_mats = {}

        for w in cfg.mom_windows:
            mat = feature_mats[f"mom_{w}"]
            # apply rowwise z-scoring
            z = mat.apply(winsorized_zscore, axis=1, clip=cfg.zscore_clip)
            z_mats[f"z_mom_{w}"] = z

        # Vol Z-score (rowwise)
        z_vol = feature_mats["vol"].apply(
            winsorized_zscore,
            axis=1,
            clip=cfg.zscore_clip,
        )
        z_mats["vol_score"] = -z_vol  # invert: lower vol -> higher score

        # --------------------------------------------------------
        # 3) Composite scores
        # --------------------------------------------------------
        # Weighted momentum score
        mom_part = None
        for w, wt in zip(cfg.mom_windows, cfg.mom_weights):
            if mom_part is None:
                mom_part = wt * z_mats[f"z_mom_{w}"]
            else:
                mom_part += wt * z_mats[f"z_mom_{w}"]

        stock_score_mat = mom_part + cfg.vol_penalty * z_mats["vol_score"]

        # --------------------------------------------------------
        # 4) Construct final output: keep only stock_score for sector scoring
        # --------------------------------------------------------
        stock_score_mat.name = "stock_score"

        return stock_score_mat

    def _compute_sector_scores_vectorized(
        self,
        stock_score_mat: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Vectorized sector scores over time.

        Parameters
        ----------
        stock_score_mat : DataFrame
            Date x Ticker matrix of stock scores (may contain NaNs where
            a ticker is out-of-universe / illiquid on a given date).

        Returns
        -------
        DataFrame
            Date x Sector matrix of mean stock_score per sector.
        """
        smap = self.um.sector_map or {}
        if stock_score_mat is None or stock_score_mat.empty:
            return pd.DataFrame(index=getattr(stock_score_mat, "index", None))

        return sector_mean_matrix(
            stock_score_mat=stock_score_mat,
            sector_map=smap,
        )

    def _compute_sector_weights_vectorized(
        self,
        sector_scores_mat: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Vectorized sector weights: Date x Sector matrix, with
        per-day softmax/caps/top-k and forward beta-smoothing.

        This is a *pure* function with respect to sleeve state:
        it does not mutate or read self.state. Smoothing happens
        along the time axis using sector_smoothing_beta.
        """
        if sector_scores_mat.empty:
            return pd.DataFrame()

        cfg = self.config
        beta = getattr(cfg, "sector_smoothing_beta", 0.0) or 0.0

        dates = sector_scores_mat.index
        sectors = list(sector_scores_mat.columns)

        weights_list = []
        prev_smoothed: Optional[pd.Series] = None

        for dt in dates:
            scores_t = sector_scores_mat.loc[dt]

            # Drop sectors with all-NaN scores on this date
            scores_t = scores_t.dropna()
            if scores_t.empty:
                # Fallback: if absolutely no scores, either carry forward
                # or use uniform weights over all sectors.
                if prev_smoothed is not None:
                    raw = prev_smoothed.copy()
                else:
                    raw = pd.Series(1.0 / len(sectors), index=sectors)
            else:
                # Compute raw weights from the non-NaN sectors
                raw = self._compute_raw_sector_weights(scores_t)
                # Expand to all sectors, missing ones default to 0
                raw = raw.reindex(sectors).fillna(0.0)

            # Smoothing
            if prev_smoothed is None or beta <= 0.0:
                smoothed = raw
            else:
                smoothed = (1.0 - beta) * prev_smoothed + beta * raw
                total = smoothed.sum()
                if total > 0:
                    smoothed = smoothed / total
                else:
                    smoothed = raw

            weights_list.append(smoothed)
            prev_smoothed = smoothed

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
    ) -> pd.DataFrame:
        """
        Vectorized stock allocation:
            Inputs:
                price_mat          : Date x Ticker (Close prices)
                stock_score_mat    : Date x Ticker (stock_score per day)
                sector_weights_mat : Date x Sector (sector weights per day)
                vol_mat            : Date x Ticker (realized vol per day); if None, will be computed internally

            Output:
                Date x Ticker portfolio weights.

        Mirrors the non-vectorized `_allocate_to_stocks`, but:
          - applies a time-varying liquidity mask using ADV / median volume / price
          - runs across all dates in a single pass
        """
        cfg = self.config
        smap = self.um.sector_map or {}
        VSE = self.vec_engine  # VectorizedSignalEngine

        dates = stock_score_mat.index
        tickers = stock_score_mat.columns.tolist()

        # Result holder
        out = pd.DataFrame(0.0, index=dates, columns=tickers)

        # ============================================================
        # 1) Build liquidity mask (Date x Ticker of bool)
        # ============================================================
        liq_mask: Optional[pd.DataFrame] = None

        if getattr(cfg, "use_liquidity_filters", False):
            # liquidity_window = getattr(cfg, "liquidity_window", 20)
            adv_window = getattr(cfg, "adv_window", 20)
            median_volume_window = getattr(cfg, "median_volume_window", 20)
            min_price = getattr(cfg, "min_price", None)
            min_adv = getattr(cfg, "min_adv20", None)
            min_medvol = getattr(cfg, "min_median_volume20", None)

            # Fetch volume matrix via VSE
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

            # Align to price_mat
            volume_mat = volume_mat.reindex_like(price_mat)

            # Dollar volume & rolling stats
            dollar_vol = price_mat * volume_mat
            adv_mat = dollar_vol.rolling(
                adv_window, min_periods=max(5, cfg.adv_window // 2)
            ).mean()
            medvol_mat = volume_mat.rolling(
                median_volume_window, min_periods=max(5, cfg.median_volume_window // 2)
            ).median()

            # Start with all True, then AND constraints in
            liq_mask = pd.DataFrame(True, index=dates, columns=tickers)

            if min_price is not None:
                liq_mask &= price_mat >= float(min_price)
            if min_adv is not None:
                liq_mask &= adv_mat >= float(min_adv)
            if min_medvol is not None:
                liq_mask &= medvol_mat >= float(min_medvol)

            liq_mask = liq_mask.fillna(False)

        # ============================================================
        # 2) Precompute sector -> ticker mapping
        # ============================================================
        sector_to_tickers: Dict[str, List[str]] = {}
        for t in tickers:
            sec = smap.get(t)
            if isinstance(sec, str) and sec.strip() not in ("", "Unknown"):
                sector_to_tickers.setdefault(sec, []).append(t)

        # ============================================================
        # 3) Per-day sector -> stock allocation
        # ============================================================
        if vol_mat is None:
            vol_mat = VSE.get_volatility(
                price_mat,
                window=cfg.vol_window,
            )
        for dt in dates:
            scores_t = stock_score_mat.loc[dt]
            vols_t = vol_mat.loc[dt]
            sector_w_t = sector_weights_mat.loc[dt]

            # Per-day liquidity eligibility
            eligible_tickers = None
            if liq_mask is not None:
                eligible_tickers = liq_mask.loc[dt]

            for sector, w_sec in sector_w_t.items():
                if w_sec <= 0 or pd.isna(w_sec):
                    continue

                candidates = sector_to_tickers.get(sector)
                if not candidates:
                    continue

                # Apply liquidity mask if present
                if eligible_tickers is not None:
                    candidates = [
                        t for t in candidates if bool(eligible_tickers.get(t, False))
                    ]
                if not candidates:
                    continue

                # Top-k by stock_score within sector
                s = scores_t[candidates].dropna()
                if s.empty:
                    continue
                top = s.nlargest(cfg.top_k_per_sector).index.tolist()

                # Intra-sector weights
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
                    # equal-weight
                    intra = pd.Series(1.0 / len(top), index=top)

                # Aggregate into final matrix
                out.loc[dt, intra.index] += w_sec * intra

        # ============================================================
        # 4) Normalize per-day (ensure weights sum to 1)
        # ============================================================
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
        Vectorized precomputation of final stock weights over [start, end], using a
        warmup period for signals. The warmup start date is:

            warmup_start = start - (max(signal_windows) + warmup_buffer) days

        Steps:
          1) Build price matrix over [warmup_start, end]
          2) Vectorized stock scores (Date x Ticker stock_score matrix)
          3) Vectorized sector scores (Date x Sector)
          4) Vectorized sector weights (smoothing applied across time)
          5) Vectorized stock allocation (Date x Ticker weights)
          6) Slice to [start, end]; optionally sample by provided rebalance dates

        Caches result in `self.precomputed_weights_mat` and returns it.
        """
        if self.vec_engine is None:
            raise ValueError("vec_engine is not set; cannot precompute.")

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        if end_ts < start_ts:
            raise ValueError("end must be >= start")

        cfg = self.config

        # Determine warmup length from signal-related windows
        window_candidates: List[int] = []
        window_candidates.extend(getattr(cfg, "mom_windows", []) or [])
        for attr_name in ["vol_window", "adv_window", "median_volume_window", "liquidity_window"]:
            w_val = getattr(cfg, attr_name, None)
            if isinstance(w_val, int) and w_val > 0:
                window_candidates.append(w_val)
        max_window = max(window_candidates) if window_candidates else 0
        warmup_days = max_window + int(warmup_buffer)
        warmup_start = start_ts - pd.Timedelta(days=warmup_days)

        # --------------------------------------------------
        # 1) Price matrix
        # --------------------------------------------------
        universe_all = self._get_trend_universe(end_ts)  # broad universe as of end
        if not universe_all:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        price_mat = self.vec_engine.get_price_matrix(
            tickers=universe_all,
            start=warmup_start,
            end=end_ts,
            local_only=getattr(self.mds, "local_only", False),
            membership_aware=True,
            treat_unknown_as_always_member=True,
        )

        # Drop tickers with no data at all
        price_mat = price_mat.dropna(axis=1, how="all")
        if price_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # --------------------------------------------------
        # 2) Vectorized stock scores (Date x Ticker)
        # --------------------------------------------------
        stock_score_mat = self._compute_stock_scores_vectorized(price_mat)
        if stock_score_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # --------------------------------------------------
        # 3) Vectorized sector scores (Date x Sector)
        # --------------------------------------------------
        sector_scores_mat = self._compute_sector_scores_vectorized(stock_score_mat)
        if sector_scores_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # --------------------------------------------------
        # 4) Vectorized sector weights (Date x Sector)
        # --------------------------------------------------
        sector_weights_mat = self._compute_sector_weights_vectorized(sector_scores_mat)
        if sector_weights_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # --------------------------------------------------
        # 5) Vectorized stock allocations (Date x Ticker)
        # --------------------------------------------------
        alloc_mat = self._allocate_to_stocks_vectorized(
            price_mat=price_mat,
            stock_score_mat=stock_score_mat,
            sector_weights_mat=sector_weights_mat,
        )
        if alloc_mat.empty:
            self.precomputed_weights_mat = pd.DataFrame()
            return self.precomputed_weights_mat

        # --------------------------------------------------
        # 6) Slice to [start, end] (drop warmup portion)
        # --------------------------------------------------
        alloc_mat = alloc_mat.loc[(alloc_mat.index >= start_ts) & (alloc_mat.index <= end_ts)]

        # Optional sampling by rebalance schedule
        if rebalance_dates:
            target_dates = pd.to_datetime(rebalance_dates).normalize()
            alloc_mat = alloc_mat[alloc_mat.index.isin(target_dates)]

        self.precomputed_weights_mat = alloc_mat.copy()
        return self.precomputed_weights_mat

    # Convenience accessor
    def get_precomputed_weights(self) -> Optional[pd.DataFrame]:
        return self.precomputed_weights_mat

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_approx_rebalance_days(freq: str) -> int:
        if not freq:
            return 21
        f = freq.upper().strip()

        if f.endswith("D") and f[:-1].isdigit():
            return max(int(f[:-1]), 1)
        if f.startswith("W"):
            return 7
        if f.startswith("M"):
            return 30
        if f.startswith("Q"):
            return 90
        if f.startswith("A") or f.startswith("Y"):
            return 365
        return 30
