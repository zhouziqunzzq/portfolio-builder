from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd


# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
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
from utils.stats import (
    zscore,
    winsorized_zscore,
    zscore_matrix_column_wise,
    windorized_zscore_matrix_column_wise,
    sector_mean_snapshot,
    sector_mean_matrix,
)
from sleeves.common.rebalance_helpers import (
    should_rebalance,
    infer_approx_rebalance_days,
    get_closest_date_on_or_before,
)
from .trend_config import TrendConfig


@dataclass
class TrendState:
    # Timestamp of last rebalance
    last_rebalance_ts: Optional[pd.Timestamp] = None
    # Latent smoothed sector weights (before top-k)
    last_sector_weights: Optional[pd.Series] = None
    # Stock weights from last rebalance
    last_stock_weights: Optional[Dict[str, float]] = None


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

        self._approx_rebalance_days = infer_approx_rebalance_days(
            self.config.rebalance_freq
        )

        # Precomputed caches for score matrices (lightweight caching)
        self._cached_stock_scores_mat: Optional[pd.DataFrame] = None  # Date x Ticker
        self._cached_sector_scores_mat: Optional[pd.DataFrame] = None  # Date x Sector

        # Cached components for sector scoring (used by vectorized path)
        self._base_stock_score_mat: Optional[pd.DataFrame] = (
            None  # CS-mom + spread-mom + vol
        )
        self._ts_stock_score_mat: Optional[pd.DataFrame] = None  # TS-mom component
        # Cached raw feature matrices (without z-scoring)
        self._cached_feature_mats: Optional[Dict[str, pd.DataFrame]] = None

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
        """
        Generate target stock weights for the given as-of date.
        Args:
            as_of (datetime | str): As-of date for weight generation. Signals should NEVER use data
                from after this date (i.e., no lookahead).
            start_for_signals (datetime | str): Start date for signal computation.
            regime (str): Current market regime for gating.
            rebalance_ctx (Optional[RebalanceContext]): Rebalance context. Note that the context
                may contain a rebalance timestamp that is different from `as_of` (usually later),
                which should be ONLY used for rebalance timing checks and NOT for signal computation.
        Returns:
            Dict[str, float]: Target stock weights (ticker -> weight).
        """
        as_of = pd.to_datetime(as_of)
        start_for_signals = pd.to_datetime(start_for_signals)

        cfg = self.config
        regime_key = (regime or "").lower()

        # # ---------- Rebalance timing check ----------
        # # Note: We need this because the global scheduler may call this function
        # # more frequently than the sleeve's intended rebalance frequency.
        # # If it's not time to rebalance yet, we return the last weights.
        # if self.state.last_stock_weights is not None and not should_rebalance(
        #     self.state.last_rebalance_ts,
        #     rebalance_ctx.rebalance_ts if rebalance_ctx is not None else as_of,
        #     cfg.rebalance_freq,
        # ):
        #     print(
        #         f"[TrendSleeve] Skipping rebalance at {rebalance_ctx.rebalance_ts.date()}; last rebalance at {self.state.last_rebalance_ts.date() if self.state.last_rebalance_ts is not None else 'never'}"
        #     )
        #     return self.state.last_stock_weights
        # # Otherwise, proceed to compute new weights
        # print(
        #     f"[TrendSleeve] Rebalancing at {rebalance_ctx.rebalance_ts.date()} using data as of {as_of.date()}; last rebalance at {self.state.last_rebalance_ts.date() if self.state.last_rebalance_ts is not None else 'never'}"
        # )

        # ---------- Regime-based gating ----------
        if cfg.use_regime_gating:
            gated_off = {r.lower() for r in cfg.gated_off_regimes}
            if regime_key in gated_off:
                # Sleeve is turned completely OFF in these regimes.
                # We *don't* update smoothing state here; next active call
                # will treat it as a fresh start or a long gap.
                print(
                    f"[TrendSleeve] Regime {regime_key} is gated off; skipping weights generation."
                )
                return {}

        # ---------- Rebalance timing check ----------
        # Note: We need this because the global scheduler may call this function
        # more frequently than the sleeve's intended rebalance frequency.
        # If it's not time to rebalance yet, we return the last weights.
        if self.state.last_stock_weights is not None and not should_rebalance(
            self.state.last_rebalance_ts,
            rebalance_ctx.rebalance_ts if rebalance_ctx is not None else as_of,
            cfg.rebalance_freq,
        ):
            print(
                f"[TrendSleeve] Skipping rebalance at {rebalance_ctx.rebalance_ts.date()}; last rebalance at {self.state.last_rebalance_ts.date() if self.state.last_rebalance_ts is not None else 'never'}"
            )
            return self.state.last_stock_weights
        # Otherwise, proceed to compute new weights
        print(
            f"[TrendSleeve] Rebalancing at {rebalance_ctx.rebalance_ts.date()} using data as of {as_of.date()}; last rebalance at {self.state.last_rebalance_ts.date() if self.state.last_rebalance_ts is not None else 'never'}"
        )

        # ---------- Compute or extract stock scores and sector scores ----------
        date_key = as_of.normalize()

        # Try to use cached scores from precompute
        if (
            self._cached_stock_scores_mat is not None
            and not self._cached_stock_scores_mat.empty
            and date_key >= self._cached_stock_scores_mat.index.min()
            and date_key <= self._cached_stock_scores_mat.index.max()
        ):
            stock_as_of = get_closest_date_on_or_before(
                date_key, self._cached_stock_scores_mat.index
            )
            sector_as_of = get_closest_date_on_or_before(
                date_key, self._cached_sector_scores_mat.index
            )
            if stock_as_of != sector_as_of:
                print(
                    f"Warning: Stock and sector scores are out of sync at {date_key.date()} (stock: {stock_as_of.date()}, sector: {sector_as_of.date()})"
                )
            date_key = stock_as_of  # Use the closest available date
            print(f"Using cached scores for {date_key.date()}")

            # Extract from cache
            stock_scores_series = self._cached_stock_scores_mat.loc[date_key]
            sector_scores = self._cached_sector_scores_mat.loc[date_key].dropna()

            # Convert to DataFrame format
            stock_scores = pd.DataFrame({"stock_score": stock_scores_series})
            stock_scores = stock_scores[stock_scores["stock_score"].notna()]

            # Get vol data if needed for inverse-vol weighting
            if cfg.weighting_mode == "inverse-vol":
                if (
                    self._cached_feature_mats is not None
                    and "vol" in self._cached_feature_mats
                ):
                    vol_mat = self._cached_feature_mats["vol"]
                    vol_as_of = get_closest_date_on_or_before(date_key, vol_mat.index)
                    if vol_as_of != date_key:
                        print(
                            f"Warning: Vol data is out of sync at {date_key.date()} (vol: {vol_as_of.date()})"
                        )
                    # vol_mat_daily = vol_mat.asfreq("D", method="ffill")
                    # if date_key in vol_mat_daily.index:
                    vol_series = vol_mat.loc[vol_as_of]
                    stock_scores["vol"] = vol_series.reindex(stock_scores.index)
                else:
                    # Fallback: compute vol on-the-fly if cache not available
                    universe = stock_scores.index.tolist()
                    sigs = self._compute_signals_snapshot(
                        universe, start_for_signals, as_of
                    )
                    if not sigs.empty and "vol" in sigs.columns:
                        stock_scores["vol"] = sigs["vol"]
        else:
            # Compute from scratch
            universe = self._get_trend_universe(as_of)
            if not universe:
                return {}

            sigs = self._compute_signals_snapshot(universe, start_for_signals, as_of)
            if sigs.empty:
                return {}

            stock_scores = self._compute_stock_scores(sigs)
            sector_scores = self._compute_sector_scores(stock_scores)

        # ---------- Common path: sector weighting and stock allocation ----------
        if stock_scores.empty or sector_scores.empty:
            return {}

        # Compute sector scores for smoothing if needed
        smoothing_freq = getattr(cfg, "sector_smoothing_freq", "rebalance")
        intermediate_sector_scores = None
        if (
            smoothing_freq == "signal"
            and self.state.last_rebalance_ts is not None
            and self.state.last_sector_weights is not None
        ):
            # Check if we'll need interpolation
            gap = (as_of - self.state.last_rebalance_ts).days
            if gap > 0 and gap <= np.ceil(2 * self._approx_rebalance_days):
                # Compute scores for [last_as_of+1, as_of]
                start_date = self.state.last_rebalance_ts + pd.Timedelta(days=1)
                intermediate_sector_scores = self._compute_sector_scores_for_range(
                    start_date, as_of, start_for_signals
                )

        sector_weights = self._compute_smoothed_sector_weights(
            as_of, sector_scores, intermediate_sector_scores
        )
        if sector_weights.isna().all() or sector_weights.sum() <= 0:
            return {}

        stock_weights = self._allocate_to_stocks(stock_scores, sector_weights)
        if not stock_weights:
            return {}

        total = sum(stock_weights.values())
        if total <= 0:
            return {}

        # Normalize
        stock_weights = {t: w / total for t, w in stock_weights.items()}

        # Don't forget to update state for a successful rebalance
        self.state.last_rebalance_ts = (
            rebalance_ctx.rebalance_ts if rebalance_ctx is not None else as_of
        )
        self.state.last_stock_weights = stock_weights

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
        buffer = pd.Timedelta(days=cfg.signals_extra_buffer_days or 30)
        rows = []

        # Pull thresholds & windows (with graceful defaults if missing)
        adv_window = getattr(cfg, "adv_window", 20)
        mv_window = getattr(cfg, "median_volume_window", 20)
        min_adv = getattr(cfg, "min_adv", None)
        min_median_vol = getattr(cfg, "min_median_volume", None)
        min_price = getattr(cfg, "min_price", None)

        signal_as_of: Optional[pd.Timestamp] = None
        for t in tickers:
            try:
                row = {"ticker": t}
                sigs: List[pd.Series | None] = []

                # --- Liquidity metrics ---
                adv_start = end - pd.Timedelta(days=adv_window) - buffer
                adv = self.signals.get_series(
                    t,
                    "adv",
                    start=adv_start,
                    end=end,
                    interval=cfg.signals_interval,
                    window=adv_window,
                )
                mv_start = end - pd.Timedelta(days=mv_window) - buffer
                mv = self.signals.get_series(
                    t,
                    "median_volume",
                    start=mv_start,
                    end=end,
                    interval=cfg.signals_interval,
                    window=mv_window,
                )
                px_start = end - buffer
                px = self.signals.get_series(
                    t,
                    "last_price",
                    start=px_start,
                    end=end,
                    interval=cfg.signals_interval,
                )

                # Use the last available values
                adv_val = adv.iloc[-1] if not adv.empty else np.nan
                mv_val = mv.iloc[-1] if not mv.empty else np.nan
                px_val = px.iloc[-1] if not px.empty else np.nan
                sigs.extend([adv, mv, px])

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
                    mom_start = end - pd.Timedelta(days=w) - buffer
                    mom = self.signals.get_series(
                        t,
                        "ts_mom",
                        start=mom_start,
                        end=end,
                        interval=cfg.signals_interval,
                        window=w,
                    )
                    # Use the last available value
                    row[f"mom_{w}"] = mom.iloc[-1] if not mom.empty else np.nan
                    sigs.append(mom)

                # --- Multi-horizon time-series momentum ---
                if cfg.use_ts_mom:
                    # TODO: Implement ts_mom for non-vectorized path
                    raise NotImplementedError("ts_mom for non-vectorized path")

                # --- Multi-horizon spread momentum ---
                if cfg.use_spread_mom:
                    benchmark = cfg.spread_benchmark or "SPY"
                    for w in cfg.spread_mom_windows:
                        smom_start = end - pd.Timedelta(days=w) - buffer
                        smom = self.signals.get_series(
                            t,
                            "spread_mom",
                            start=smom_start,
                            end=end,
                            interval=cfg.signals_interval,
                            window=w,
                            benchmark=benchmark,
                        )
                        # Use the last available value
                        row[f"spread_mom_{w}"] = (
                            smom.iloc[-1] if not smom.empty else np.nan
                        )
                        sigs.append(smom)

                # --- Realized volatility ---
                vol_mode = getattr(cfg, "vol_mode", "rolling")
                ewm_halflife = getattr(cfg, "ewm_vol_halflife", None) or cfg.vol_window
                if vol_mode == "rolling":
                    vol_start = end - pd.Timedelta(days=cfg.vol_window) - buffer
                    vol = self.signals.get_series(
                        t,
                        "vol",
                        start=vol_start,
                        end=end,
                        interval=cfg.signals_interval,
                        window=cfg.vol_window,
                    )
                    # Use the last available value
                    vol_val = vol.iloc[-1] if not vol.empty else np.nan
                    sigs.append(vol)
                elif vol_mode == "ewm":
                    # NEW: EM-vol from last_price series
                    # Use a longer lookback window to make sure we have enough px data
                    vol_lookback_days = max(cfg.vol_window, ewm_halflife)
                    vol_start = end - pd.Timedelta(days=vol_lookback_days) - buffer
                    vol = self.signals.get_series(
                        t,
                        "ewm_vol",
                        start=vol_start,
                        end=end,
                        interval=cfg.signals_interval,
                        halflife=ewm_halflife,
                    )
                    vol_val = vol.iloc[-1] if not vol.empty else np.nan
                    sigs.append(vol)
                else:
                    raise ValueError(f"Unknown vol_mode: {vol_mode}")
                row["vol"] = vol_val

                # Record signal as-of date for debugging
                for s in sigs:
                    if signal_as_of is None:
                        signal_as_of = s.index.max()
                        continue
                    if s is not None and not s.empty:
                        s_as_of = s.index.max()
                        if signal_as_of is not None and signal_as_of != s_as_of:
                            print(
                                f"Warning: {t} signal data is out of sync at {end.date()} (adv: {s_as_of.date()})"
                            )
                        signal_as_of = max(signal_as_of, s.index.max())

                rows.append(row)

            except Exception as e:
                # For robustness: skip this ticker if anything fails
                # Print stacktrace for debugging
                print(
                    f"Warning: Failed to compute signals for {t} at {end.date()}: {e}"
                )
                traceback.print_exc()
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")

        mom_cols = [f"mom_{w}" for w in cfg.mom_windows]
        keep_cols = mom_cols + ["vol"]

        # Drop rows missing key signal inputs
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols, how="any")

        print(f"Using signal data as-of {signal_as_of.date()}")
        return df

    # ------------------------------------------------------------------
    # Stock scoring (with winsorized z-scores)
    # ------------------------------------------------------------------

    def _zscore_series(self, s: pd.Series) -> pd.Series:
        """
        Standard z-score with optional winsorization based on cfg.zscore_clip.
        """
        cfg = self.config
        zclip_enabled = getattr(cfg, "use_zscore_winsorization", False)
        zclip = getattr(cfg, "zscore_clip", None)
        if zclip_enabled and zclip is not None and zclip > 0:
            z = winsorized_zscore(s, clip=zclip)
        else:
            z = zscore(s)

        return z

    def _compute_stock_scores(self, sigs: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = sigs.copy()

        # z-score momentum components
        # Note: By z-scoring we turn the sampled ts-mom into a cross-sectional ranking, i.e. cs-mom
        for w in cfg.mom_windows:
            col = f"mom_{w}"
            df[f"z_mom_{w}"] = self._zscore_series(df[col])

        # weighted momentum scores for z_mom and spread_mom
        momentum_score = pd.Series(0.0, index=df.index)
        # CS-momentum
        for w, wt in zip(cfg.mom_windows, cfg.mom_weights):
            momentum_score += wt * df[f"z_mom_{w}"]
        # Spread-momentum
        if cfg.use_spread_mom:
            for w, wt in zip(cfg.spread_mom_windows, cfg.spread_mom_window_weights):
                col = f"spread_mom_{w}"
                # Raw addition; no z-scoring of spread_mom
                momentum_score += cfg.spread_mom_weight * wt * df[f"spread_mom_{w}"]
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

    def _compute_sector_scores_for_range(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        start_for_signals: pd.Timestamp,
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        Computes sector scores for the date range [start_date, end_date].

        If vec precompute was called, returns scores from cache.
        Otherwise, computes scores using non-vec logic.

        Returns dict mapping date -> sector_scores (does not modify caches).
        """
        result = {}
        date_range = self._get_trading_calendar(start_date, end_date, interval="1d")

        # If both caches exist, extract from cache (vec path)
        if (
            self._cached_sector_scores_mat is not None
            and not self._cached_sector_scores_mat.empty
        ):
            # Extract sector scores from cache for the date range
            for date in date_range:
                date_key = pd.Timestamp(date).normalize()
                if date_key in self._cached_sector_scores_mat.index:
                    sector_scores = self._cached_sector_scores_mat.loc[
                        date_key
                    ].dropna()
                    if not sector_scores.empty:
                        result[date_key] = sector_scores
            return result

        # Cache missing - compute using non-vec logic (without caching)
        for date in date_range:
            print(
                f"[TrendSleeve] Computing sector scores for {date.date()} for sector weights interpolation"
            )
            date_ts = pd.Timestamp(date).normalize()
            universe = self._get_trend_universe(date_ts)
            if not universe:
                continue

            sigs = self._compute_signals_snapshot(universe, start_for_signals, date_ts)
            if sigs.empty:
                continue

            scored = self._compute_stock_scores(sigs)
            if scored.empty or "stock_score" not in scored.columns:
                continue

            # Compute sector scores
            sector_scores = self._compute_sector_scores(scored)
            if not sector_scores.empty:
                result[date_ts] = sector_scores

        return result

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
        Exponential smoothing / hysteresis:
        w_smoothed = (1 - beta) * prev + beta * new

        If no prev_weights, just return new_weights.
        """
        if prev_weights is None:
            return new_weights

        cfg = self.config
        beta = cfg.sector_smoothing_beta
        # Align indices
        prev_weights = prev_weights.reindex(new_weights.index).fillna(0.0)

        w_smoothed = (1 - beta) * prev_weights + beta * new_weights
        # Renormalize to sum to 1
        total = w_smoothed.sum()
        if total <= 0:
            return pd.Series(
                1.0 / len(w_smoothed),
                index=w_smoothed.index,
                dtype=float,
            )
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

    def _compute_smoothed_sector_weights(
        self,
        as_of: pd.Timestamp,
        sector_scores: pd.Series,
        intermediate_sector_scores: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    ) -> pd.Series:
        """
        Compute smoothed sector weights with optional daily interpolation.

        For daily mode, uses intermediate_sector_scores if provided.

        Parameters
        ----------
        as_of : pd.Timestamp
            Target date for sector weights
        sector_scores : pd.Series
            Sector scores for the target date
        intermediate_sector_scores : dict, optional
            Dict mapping date -> sector_scores for daily interpolation
        """
        cfg = self.config

        # 1) Softmax -> base sector weights
        w_soft = self._softmax(sector_scores)

        # 2) Apply caps/floors
        w_capped = self._apply_caps(w_soft)

        # 3) Time-based smoothing vs previous (on full uncropped vector)
        if (
            self.state.last_rebalance_ts is None
            or self.state.last_sector_weights is None
        ):
            # First time: no smoothing, just use capped weights
            smoothed_pre_topk = w_capped
        else:
            gap = (as_of - self.state.last_rebalance_ts).days

            if gap <= 0:
                # Same day or out-of-order: don't smooth, just use new capped weights
                smoothed_pre_topk = w_capped
            elif gap > np.ceil(2 * self._approx_rebalance_days):
                # Large gap -> reset smoothing, treat as fresh start
                smoothed_pre_topk = w_capped
            else:
                # Normal smoothing case
                smoothing_freq = getattr(cfg, "sector_smoothing_freq", "rebalance")

                if (
                    smoothing_freq == "signal"
                    and intermediate_sector_scores is not None
                ):
                    # interpolation: recompute target weights for each signal date
                    smoothed = self.state.last_sector_weights
                    beta = cfg.sector_smoothing_beta

                    # Generate trading dates from last_as_of + 1 day to as_of
                    start_date = self.state.last_rebalance_ts + pd.Timedelta(days=1)
                    daily_dates = self._get_trading_calendar(
                        start_date,
                        as_of,
                        interval="1d",
                    )

                    # Use provided sector scores for each day
                    for date in daily_dates:
                        date_key = pd.Timestamp(date).normalize()
                        if date_key in intermediate_sector_scores:  # On signal dates
                            sector_scores_sig = intermediate_sector_scores[date_key]

                            if not sector_scores_sig.empty:
                                print(
                                    f"[TrendSleeve] Applying sector weights smoothing for {date_key.date()}"
                                )
                                # Apply softmax and caps for this day
                                w_soft_day = self._softmax(sector_scores_sig)
                                w_capped_day = self._apply_caps(w_soft_day)

                                # Apply smoothing
                                smoothed = smoothed.reindex(w_capped_day.index).fillna(
                                    0.0
                                )
                                smoothed = (1 - beta) * smoothed + beta * w_capped_day
                                total = smoothed.sum()
                                if total > 0:
                                    smoothed = smoothed / total
                        # If date not in scores dict, skip that day (keep prev smoothed)
                        # This makes sure smoothing is applied only on signal dates

                    smoothed_pre_topk = smoothed
                else:
                    # "rebalance": single-step smoothing (original behavior)
                    prev = self.state.last_sector_weights.reindex(
                        w_capped.index
                    ).fillna(0.0)
                    beta = cfg.sector_smoothing_beta
                    smoothed_pre_topk = (1 - beta) * prev + beta * w_capped

                    total = smoothed_pre_topk.sum()
                    if total > 0:
                        smoothed_pre_topk = smoothed_pre_topk / total
                    else:
                        smoothed_pre_topk = w_capped

        # 4) Apply top-k on the smoothed weights for actual allocation
        smoothed_out = self._top_k(smoothed_pre_topk)

        # 5) Update state with pre-top-k smoothed weights (latent preference)
        self.state.last_rebalance_ts = as_of
        self.state.last_sector_weights = smoothed_pre_topk.copy()

        return smoothed_out

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
        membership_mask: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Fully vectorized stock score computation over ALL dates.

        Produces a Date x Ticker matrix with:
            - stock_score(t, ticker)

        No liquidity filtering here.

        Parameters
        ----------
        membership_mask : pd.DataFrame, optional
            Boolean mask (Date x Ticker) indicating universe membership.
            If provided, raw signals will be masked before z-scoring.
        """

        cfg = self.config
        VSE = self.vec_engine

        # --------------------------------------------------------
        # 1) Vectorized raw signals
        # --------------------------------------------------------
        # CS-momentum
        cs_mom_dict = VSE.get_momentum(price_mat, cfg.mom_windows)
        # TS-momentum (raw, no cross-sectional z-score)
        if cfg.use_ts_mom:
            ts_mom_dict = VSE.get_ts_momentum(price_mat, cfg.ts_mom_windows)
        else:
            ts_mom_dict = {}
        # Spread-momentum (raw, no cross-sectional z-score)
        if cfg.use_spread_mom:
            benchmark = cfg.spread_benchmark or "SPY"
            spread_mom_dict = VSE.get_spread_momentum(
                price_mat,
                cfg.spread_mom_windows,
                benchmark=benchmark,
            )
        else:
            spread_mom_dict = {}

        # Realized volatility
        vol_mat = None
        # volatility matrix (EM-vol or rolling)
        vol_mode = getattr(cfg, "vol_mode", "rolling")
        ewm_halflife = getattr(cfg, "ewm_vol_halflife", None) or cfg.vol_window
        if vol_mode == "ewm":
            vol_mat = VSE.get_ewm_volatility(
                price_mat,
                halflife=ewm_halflife,
            )
        else:  # rolling
            vol_mat = VSE.get_volatility(
                price_mat,
                window=cfg.vol_window,
            )

        # Build a dict: feature_name -> matrix
        feature_mats = {}
        # Momentum matrices (CS-momentum)
        for w in cfg.mom_windows:
            feature_mats[f"mom_{w}"] = cs_mom_dict[w]
        # Momentum matrices (TS-momentum)
        if cfg.use_ts_mom:
            for w in cfg.ts_mom_windows:
                feature_mats[f"ts_mom_{w}"] = ts_mom_dict[w]
        # Spread-momentum matrices
        if cfg.use_spread_mom:
            for w in cfg.spread_mom_windows:
                feature_mats[f"spread_mom_{w}"] = spread_mom_dict[w]
        # Volatility matrix
        feature_mats["vol"] = vol_mat

        # Save cached feature matrices
        self._cached_feature_mats = feature_mats

        # --------------------------------------------------------
        # 1.5) Apply membership mask to raw signals (BEFORE z-scoring)
        # --------------------------------------------------------
        if membership_mask is not None and not membership_mask.empty:
            # Reindex without ffill - use exact membership status for each date
            membership_mask_aligned = membership_mask.reindex(
                index=feature_mats[f"mom_{cfg.mom_windows[0]}"].index,
                fill_value=False,  # No ffill - missing dates default to False
            )
            membership_mask_aligned = membership_mask_aligned.reindex(
                columns=feature_mats[f"mom_{cfg.mom_windows[0]}"].columns,
                fill_value=False,
            )

            # Apply mask to all raw feature matrices
            for key in feature_mats:
                feature_mats[key] = feature_mats[key].where(membership_mask_aligned)

        # --------------------------------------------------------
        # 1.6) Filter out stocks with NaN momentum/vol (match non-vec behavior)
        # --------------------------------------------------------
        # The non-vec path drops stocks with missing mom/vol before z-scoring
        # The vec path must do the same to ensure identical z-score universes PER DATE
        # For each date, a stock is valid only if it has non-NaN values in ALL mom windows AND vol

        # Compute a combined validity mask: valid if all required signals are non-NaN
        combined_valid_mask = feature_mats["vol"].notna()  # Start with vol
        for w in cfg.mom_windows:
            combined_valid_mask = combined_valid_mask & feature_mats[f"mom_{w}"].notna()

        # Apply the combined mask to ALL feature matrices
        for key in feature_mats:
            feature_mats[key] = feature_mats[key].where(combined_valid_mask)

        # --------------------------------------------------------
        # 2) Z scoring (winsorized if enabled) *per date*, cross-sectionally (column-wise)
        # --------------------------------------------------------
        z_mats = {}

        # CS-momentum Z-score (column-wise)
        for w in cfg.mom_windows:
            mat = feature_mats[f"mom_{w}"]

            # apply rowwise z-scoring if enabled
            if cfg.use_zscore_winsorization:
                z = windorized_zscore_matrix_column_wise(mat, clip=cfg.zscore_clip)
            else:
                z = zscore_matrix_column_wise(mat)
            z_mats[f"z_mom_{w}"] = z

        # TS-momentum Z-score (column-wise)
        if cfg.use_ts_mom:
            for w in cfg.ts_mom_windows:
                mat = feature_mats[f"ts_mom_{w}"]

                # apply rowwise z-scoring if enabled
                if cfg.use_zscore_winsorization:
                    z = windorized_zscore_matrix_column_wise(mat, clip=cfg.zscore_clip)
                else:
                    z = zscore_matrix_column_wise(mat)
                z_mats[f"ts_mom_{w}"] = z

        # Vol Z-score (column-wise)
        if cfg.use_zscore_winsorization:
            z_vol = windorized_zscore_matrix_column_wise(
                feature_mats["vol"],
                clip=cfg.zscore_clip,
            )
        else:
            z_vol = zscore_matrix_column_wise(feature_mats["vol"])
        z_mats["vol_score"] = -z_vol  # invert: lower vol -> higher score

        # --------------------------------------------------------
        # 3) Composite scores
        # --------------------------------------------------------
        # Weighted CS-momentum score
        mom_part = None
        for w, wt in zip(cfg.mom_windows, cfg.mom_weights):
            if mom_part is None:
                mom_part = wt * z_mats[f"z_mom_{w}"]
            else:
                mom_part += wt * z_mats[f"z_mom_{w}"]

        # stock_score_mat = mom_part + cfg.vol_penalty * z_mats["vol_score"]
        base_score = mom_part.add(cfg.vol_penalty * z_mats["vol_score"], fill_value=0.0)

        # CS-momentum component (after cs_weight)
        cs_stock_score_mat = base_score * cfg.cs_weight

        # Weighted TS-momentum score
        # ts_mom_windows / ts_mom_weights are independent of mom_windows.
        ts_stock_score_mat = None  # will hold pure TS component (after ts_weight)
        ts_stock_score_mat_raw = None  # will hold raw TS component before weighting
        if cfg.use_ts_mom and ts_mom_dict:
            ts_part = None
            for w, wt in zip(cfg.ts_mom_windows, cfg.ts_mom_weights):
                ts_mat_z = z_mats[f"ts_mom_{w}"]
                # ensure same index/columns as base_score
                ts_mat_z = ts_mat_z.reindex_like(base_score)

                if ts_part is None:
                    ts_part = wt * ts_mat_z
                else:
                    ts_part += wt * ts_mat_z
            ts_stock_score_mat_raw = ts_part
            # TS-mom stock component (after ts_weight)
            ts_stock_score_mat = cfg.ts_weight * ts_part

        # Spread-momentum contribution (raw, absolute)
        spread_stock_score_mat = (
            None  # will hold pure spread-mom component (after spread_mom_weight)
        )
        if cfg.use_spread_mom and spread_mom_dict:
            spread_part = None
            for w, wt in zip(cfg.spread_mom_windows, cfg.spread_mom_window_weights):
                smat = feature_mats[f"spread_mom_{w}"]
                # ensure same index/columns as base_score
                smat = smat.reindex_like(base_score)

                if spread_part is None:
                    spread_part = wt * smat
                else:
                    spread_part += wt * smat
            # Spread-mom stock component (after spread_mom_weight)
            spread_stock_score_mat = cfg.spread_mom_weight * spread_part

        # Combined stock score BEFORE gating
        stock_score_mat = cs_stock_score_mat.copy()
        if ts_stock_score_mat is not None:
            stock_score_mat = stock_score_mat.add(ts_stock_score_mat, fill_value=0.0)
        if spread_stock_score_mat is not None:
            stock_score_mat = stock_score_mat.add(
                spread_stock_score_mat, fill_value=0.0
            )

        # Apply TS-mom gating if enabled
        if cfg.use_ts_mom and cfg.use_ts_gate and ts_stock_score_mat_raw is not None:
            # TS-mom gating: zero out stock_score where TS-mom < threshold
            gate_threshold = cfg.ts_gate_threshold
            if gate_threshold is not None:
                row_min = stock_score_mat.min(axis=1)
                penalty = row_min - 0.5  # ensure some negative buffer
                bad_ts = ts_stock_score_mat_raw < gate_threshold
                # stock_score_mat = stock_score_mat.mask(bad_ts) # set to NaN
                stock_score_mat = stock_score_mat.where(
                    ~bad_ts,
                    penalty,
                    axis=0,
                )  # set to penalty instead of hard gating

        # --------------------------------------------------------
        # 4) Construct final output: keep only stock_score for sector scoring
        # --------------------------------------------------------
        # Apply membership mask AGAIN to final stock scores
        # This ensures stocks that are not in the universe on each date have NaN scores
        if membership_mask is not None and not membership_mask.empty:
            # Reindex membership mask to match stock_score_mat dates
            membership_mask_final = membership_mask.reindex(
                index=stock_score_mat.index, fill_value=False
            )
            # Ensure columns match
            membership_mask_final = membership_mask_final.reindex(
                columns=stock_score_mat.columns, fill_value=False
            )
            # Apply mask: set non-members to NaN
            stock_score_mat = stock_score_mat.where(membership_mask_final)
            # print(f"[TrendSleeve] Applied final membership mask to stock scores")

        # Cache components for sector scoring
        self._base_stock_score_mat = cs_stock_score_mat
        self._ts_stock_score_mat = ts_stock_score_mat
        stock_score_mat.name = "stock_score"
        return stock_score_mat

    def _compute_sector_scores_vectorized(
        self,
        stock_score_mat: pd.DataFrame,
        sector_map: Optional[Dict[str, str]] = None,
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
            Date x Sector matrix of blended CS + TS sector scores.
        """
        smap = sector_map or self.um.sector_map or {}
        if stock_score_mat is None or stock_score_mat.empty:
            return pd.DataFrame(index=getattr(stock_score_mat, "index", None))
        cfg = self.config

        # ------------------------------------------------------------------
        # 1) Get cached stock-level components
        # ------------------------------------------------------------------
        base_stock = getattr(self, "_base_stock_score_mat", None)
        ts_stock = getattr(self, "_ts_stock_score_mat", None)

        # Fallback if for some reason cache is missing, all-NaN, all-zero,
        # or mostly-zero (e.g. cache was zeroed out). Use a small epsilon
        # and a fractional threshold to detect "mostly zero" matrices.
        def _is_mostly_zero(
            df: pd.DataFrame, eps: float = 1e-12, frac_thresh: float = 0.01
        ) -> bool:
            if df is None:
                return True
            if df.empty:
                return True
            # all-NaN
            try:
                if df.isna().all().all():
                    return True
            except Exception:
                pass
            # count non-negligible entries
            try:
                total = df.size
                nonzero = (df.fillna(0).abs() > eps).sum().sum()
                if total == 0:
                    return True
                if nonzero == 0:
                    return True
                if (nonzero / float(total)) < float(frac_thresh):
                    return True
            except Exception:
                # If anything goes wrong, err on the side of recomputing
                return True
            return False

        if _is_mostly_zero(base_stock):
            base_stock = stock_score_mat

        # ------------------------------------------------------------------
        # 2) Sector base score: mean of FULL stock scores (CS-mom + spread-mom + vol) within each sector
        # ------------------------------------------------------------------
        # NOTE: We use stock_score_mat (the full composite score), not base_stock (base component only).
        # The base_stock and ts_stock caches are only used for blended base+TS sector scoring,
        # but standard sector aggregation should use the full composite stock score.

        sector_base = sector_mean_matrix(
            stock_score_mat=stock_score_mat,  # Full composite: CS-mom + spread-mom + vol
            sector_map=smap,
        )

        # ------------------------------------------------------------------
        # 3) Sector TS score: mean of TS stock scores within each sector
        # ------------------------------------------------------------------
        if cfg.use_ts_mom and ts_stock is not None:
            sector_ts = sector_mean_matrix(
                stock_score_mat=ts_stock,
                sector_map=smap,
            )
        else:
            # no TS contribution
            sector_ts = pd.DataFrame(
                0.0, index=sector_base.index, columns=sector_base.columns
            )

        # ------------------------------------------------------------------
        # 4) Blend sector base (CS + spread-mom) + TS
        # ------------------------------------------------------------------
        sector_scores_mat = (
            cfg.sector_cs_weight * sector_base + cfg.sector_ts_weight * sector_ts
        )

        return sector_scores_mat

    # ------------------------------------------------------------------
    # Vectorized stock allocation - DEPRECATED (Legacy code kept for reference)
    # Now we precompute only scores and reuse non-vec allocation logic
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
                price_mat          : Date x Ticker (Close prices)
                stock_score_mat    : Date x Ticker (stock_score per day)
                sector_weights_mat : Date x Sector (sector weights per day / rebalance date)
                vol_mat            : Date x Ticker (realized vol per day); if None, will be computed internally

            Output:
                Date x Ticker portfolio weights.

        Mirrors the non-vectorized `_allocate_to_stocks`, but:
          - applies a time-varying liquidity mask using ADV / median volume / price
          - runs across all dates in a single pass
        """
        cfg = self.config
        smap = sector_map or self.um.sector_map or {}
        VSE = self.vec_engine  # VectorizedSignalEngine

        # Align dates on sector_weights_mat
        dates = sector_weights_mat.index
        stock_score_mat = stock_score_mat.reindex(dates, method="ffill")
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
        if cfg.weighting_mode == "inverse-vol" and vol_mat is None:
            # First try to get cached vol matrix
            vol_mat = getattr(self, "_cached_feature_mats", {}).get("vol")
            if vol_mat is not None:
                # Forward-fill vol_mat to daily frequency to match stock_score_mat
                vol_mat = vol_mat.asfreq("D", method="ffill")
            if vol_mat is None:
                print(
                    "[TrendSleeve] Warning: volatility matrix cache missing; recomputing vol_mat."
                )
                # Fallback: compute on the fly
                vol_mode = getattr(cfg, "vol_mode", "rolling")
                ewm_halflife = getattr(cfg, "ewm_vol_halflife", None) or cfg.vol_window
                if vol_mode == "ewm":
                    vol_mat = VSE.get_ewm_volatility(
                        price_mat,
                        halflife=ewm_halflife,
                    )
                else:  # rolling
                    vol_mat = VSE.get_volatility(
                        price_mat,
                        window=cfg.vol_window,
                    )
        for dt in dates:
            scores_t = stock_score_mat.loc[dt]
            if cfg.weighting_mode == "inverse-vol":
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

                # Debug: Print stock scores for Financials on first date
                if dt == dates[0] and sector == "Financials":
                    print(
                        f"[TrendSleeve] Vec first date {dt.date()}, Financials stock scores:"
                    )
                    print(f"  {s.sort_values(ascending=False).head(5).to_dict()}")

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
        sample_dates: Optional[List[datetime | str]] = None,
        warmup_buffer: Optional[
            int
        ] = None,  # in days; unused for vectorized trend sleeve
    ) -> pd.DataFrame:
        """
        Vectorized precomputation of final stock weights over [start, end], using a
        warmup period for signals. The warmup start date is:
            warmup_start = start - max_signal_window - sector_smoothing_buffer
        where
            max_signal_window = max of all signal windows (mom, vol, adv, median_volume, etc)
            sector_smoothing_buffer = days needed for sector weight smoothing to stabilize.

        Steps:
          1) Build price matrix over [warmup_start, end]
          2) Vectorized stock scores (Date x Ticker stock_score matrix)
          3) Vectorized sector scores (Date x Sector)
          4) Vectorized sector weights (smoothing applied across time)
          5) Vectorized stock allocation (Date x Ticker weights)
          6) Slice to [start, end]; optionally sample by provided sample dates

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
        # Note: The warmup is extremely critical for correct sector weight smoothing.
        beta = cfg.sector_smoothing_beta
        # Calculate the extra buffer days needed for smoothing to stabilize
        if beta > 0 and beta < 1:
            sector_weight_smoothing_days = int(np.ceil(np.log(1e-3) / np.log(1 - beta)))
        else:
            sector_weight_smoothing_days = 0
        sector_weight_smoothing_days = int(
            sector_weight_smoothing_days * (365 / 252)
        )  # trading days to calendar days
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
        max_window_abs_days = int(
            self._infer_abs_days_from_window_size(max_window, cfg.signals_interval)
        )  # convert trading days to calendar days
        warmup_days = max_window_abs_days + sector_weight_smoothing_days
        warmup_start = start_ts - pd.Timedelta(days=warmup_days)

        # --------------------------------------------------
        # 1) Price matrix (WITHOUT membership masking to avoid NaN gaps)
        # --------------------------------------------------
        # Fetch clean price data without membership masking to ensure
        # signal calculations (especially .shift() operations) work correctly.
        # We'll apply membership masking AFTER signal calculation.
        print(
            f"[TrendSleeve] Precomputing price matrix from {warmup_start.date()} to {end_ts.date()}"
        )
        price_mat = self.um.get_price_matrix(
            price_loader=self.mds,
            start=warmup_start,
            end=end_ts,
            tickers=self.um.tickers,
            interval=cfg.signals_interval,
            auto_apply_membership_mask=False,  # Changed: Don't mask prices
            local_only=getattr(self.mds, "local_only", False),
        )

        # Drop tickers with no data at all
        price_mat = price_mat.dropna(axis=1, how="all")
        if price_mat.empty:
            self._cached_stock_scores_mat = pd.DataFrame()
            self._cached_sector_scores_mat = pd.DataFrame()
            return pd.DataFrame()

        # Normalize tickers to uppercase and align sector_map to columns
        price_mat.columns = [c.upper() for c in price_mat.columns]
        sector_map = self.um.sector_map
        if sector_map is not None:
            sector_map = {t.upper(): s for t, s in sector_map.items()}
            # Keep sector_map only for tickers present in prices
            sector_map = {t: s for t, s in sector_map.items() if t in price_mat.columns}

        # Get membership mask to apply after signal calculation
        membership_mask = self.um.membership_mask(
            start=warmup_start.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
        )
        # Ensure mask columns match price_mat columns
        if not membership_mask.empty:
            membership_mask = membership_mask.reindex(
                columns=price_mat.columns, fill_value=False
            )

        # --------------------------------------------------
        # 2) Vectorized stock scores (Date x Ticker)
        # --------------------------------------------------
        stock_score_mat = self._compute_stock_scores_vectorized(
            price_mat, membership_mask=membership_mask
        )
        if stock_score_mat.empty:
            self._cached_stock_scores_mat = pd.DataFrame()
            self._cached_sector_scores_mat = pd.DataFrame()
            return pd.DataFrame()

        # Membership mask already applied inside _compute_stock_scores_vectorized
        # before z-scoring to ensure cross-sectional rankings are correct

        # --------------------------------------------------
        # 3) Vectorized sector scores (Date x Sector)
        # --------------------------------------------------
        sector_scores_mat = self._compute_sector_scores_vectorized(
            stock_score_mat, sector_map=sector_map
        )
        if sector_scores_mat.empty:
            self._cached_stock_scores_mat = pd.DataFrame()
            self._cached_sector_scores_mat = pd.DataFrame()
            return pd.DataFrame()

        # --------------------------------------------------
        # 4) Cache the computed matrices for use in generate_target_weights_for_date
        # --------------------------------------------------
        # Forward-fill stock scores to daily frequency to handle non-trading days
        # (e.g., 2025-01-01 should use scores from 2024-12-31)
        # stock_score_mat_daily = stock_score_mat.asfreq("D", method="ffill")
        # sector_scores_mat_daily = sector_scores_mat.asfreq("D", method="ffill")

        # Slice to [warmup_start, end]
        # Keep warmup period to ensure the first rebalance date has proper signals
        stock_score_mat = stock_score_mat.loc[
            (stock_score_mat.index >= warmup_start) & (stock_score_mat.index <= end_ts)
        ]
        sector_scores_mat = sector_scores_mat.loc[
            (sector_scores_mat.index >= warmup_start)
            & (sector_scores_mat.index <= end_ts)
        ]

        # Cache the scores for use in generate_target_weights_for_date
        self._cached_stock_scores_mat = stock_score_mat.copy()
        self._cached_sector_scores_mat = sector_scores_mat.copy()

        return pd.DataFrame()

    # Convenience accessor (deprecated - returns empty DataFrame)
    def get_precomputed_weights(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_trading_calendar(
        self,
        start: datetime | str,
        end: datetime | str,
        reference_ticker: Optional[str] = "SPY",
        interval: str = "1d",
    ) -> pd.DatetimeIndex:
        if self.mds is None:
            raise ValueError("mds is not set; cannot get trading calendar.")

        # Fetch OHLCV data for the reference ticker to determine trading days
        ohlcv = self.mds.get_ohlcv(
            ticker=reference_ticker,
            start=start,
            end=end,
            interval=interval,
            local_only=getattr(self.mds, "local_only", False),
        )
        # Ensure the index is a DatetimeIndex, sorted, and within the specified range
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv.index = pd.to_datetime(ohlcv.index)
        ohlcv.index = ohlcv.index.sort_values()
        ohlcv.index = ohlcv.index[(ohlcv.index >= start) & (ohlcv.index <= end)]
        return ohlcv.index

    @staticmethod
    def _infer_abs_days_from_window_size(window_size: int, signal_interval: str) -> int:
        if window_size <= 0:
            return 0
        freq = signal_interval.lower().strip()
        if freq.endswith("d") and freq[:-1].isdigit():
            return (
                window_size * int(freq[:-1]) * (365 / 252)
            )  # trading days to calendar days
        if freq.endswith("wk"):
            return window_size * 7
        if freq.endswith("mo"):
            return window_size * 30

        raise ValueError(f"Unknown signal interval: {signal_interval}")
        # return window_size * 30
