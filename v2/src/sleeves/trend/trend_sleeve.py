from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
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
        config: Optional[TrendConfig] = None,
    ) -> None:
        self.um = universe
        self.mds = mds
        self.signals = signals
        self.config = config or TrendConfig()
        self.state = TrendState()

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

        # 5) Allocate sector â stocks
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

        # volatility score (lower vol â higher score)
        vol_z = self._zscore_series(df["vol"])
        df["vol_score"] = -vol_z

        # stock_score = momentum_score + vol_penalty * vol_score
        df["stock_score"] = df["momentum_score"] + cfg.vol_penalty * df["vol_score"]

        return df

    # ------------------------------------------------------------------
    # Sector scoring & smoothing
    # ------------------------------------------------------------------

    def _compute_sector_scores(self, scored: pd.DataFrame) -> pd.Series:
        smap = self.um.sector_map or {}
        # filter only valid stocks
        valid = {
            t: smap[t]
            for t in scored.index
            if t in smap
            and isinstance(smap[t], str)
            and smap[t].strip() not in ("", "Unknown")
        }
        if not valid:
            return pd.Series(dtype=float)

        sector_series = pd.Series(valid)
        sector_scores = (
            scored.loc[valid.keys(), "stock_score"].groupby(sector_series).mean()
        )
        sector_scores.name = "sector_score"
        return sector_scores

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
    # Allocate sector â stocks
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
