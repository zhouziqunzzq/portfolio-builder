from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from src.universe_manager import UniverseManager
from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from .defensive_config import DefensiveConfig


@dataclass
class DefensiveState:
    last_rebalance: Optional[pd.Timestamp] = None
    last_weights: Optional[Dict[str, float]] = None


class DefensiveSleeve:
    """
    Defensive Sleeve (Multi-Asset)

    - Defensive equities (S&P 500 stocks in defensive sectors)
    - Defensive equity ETFs (XLP, XLV, XLU)
    - Bonds (BND, IEF, TLT, SHY, LQD)
    - Gold (GLD)
    """

    def __init__(
        self,
        universe: UniverseManager,
        mds: MarketDataStore,
        signals: SignalEngine,
        config: Optional[DefensiveConfig] = None,
    ):
        self.um = universe
        self.mds = mds
        self.signals = signals
        self.config = config or DefensiveConfig()
        self.state = DefensiveState()

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def get_defensive_universe(
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
                # Get last ~1 month of OHLCV
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
    ) -> Dict[str, float]:

        as_of = pd.to_datetime(as_of)
        start_for_signals = pd.to_datetime(start_for_signals)

        # Use as_of when building universe so membership is date-aware
        universe = self.get_defensive_universe(as_of=as_of)
        # --- apply liquidity filter ---
        universe = self._apply_liquidity_filters(universe, as_of)
        if not universe:
            return {}

        # 1) compute raw signals
        sigs = self._compute_signals_snapshot(universe, start_for_signals, as_of)
        if sigs.empty:
            return {}

        # 2) compute composite scores (NO global top-k)
        scored = self._compute_scores_only(sigs)

        # 3) class-aware allocation with per-class top-k selection
        weights = self.allocate_by_asset_class(scored, regime)

        self.state.last_rebalance = as_of
        self.state.last_weights = weights
        return weights

    # ------------------------------------------------------------------
    # Multi-Asset Allocation (Option 1)
    # ------------------------------------------------------------------

    def allocate_by_asset_class(
        self, df: pd.DataFrame, regime: str
    ) -> Dict[str, float]:
        """
        Steps:
        1) Assign asset class: equity / bond / gold
        2) Per-class: select top_k * class_alloc fraction
        3) Weight by score Ã inverse-vol
        4) Multiply by class allocation
        5) Normalize final vector
        """
        cfg = self.config

        # 1) assign asset class
        def assign_class(t):
            if t in cfg.asset_class_for_etf:
                return cfg.asset_class_for_etf[t]
            return "equity"  # defensive stock

        df = df.copy()
        df["asset_class"] = [assign_class(t) for t in df.index]

        # 2) get regime-level class allocations
        class_alloc = cfg.asset_class_allocations_by_regime.get(regime)
        if class_alloc is None:
            raise ValueError(
                f"Unknown regime '{regime}' in asset_class_allocations_by_regime"
            )

        final: Dict[str, float] = {}

        for asset_class in ["equity", "bond", "gold"]:
            sub = df[df["asset_class"] == asset_class]
            if sub.empty:
                continue

            # ---- per-class top-k selection ----
            k_class = int(round(cfg.top_k * class_alloc.get(asset_class, 0.0)))
            k_class = max(1, min(len(sub), k_class))  # ensure at least one if exists

            sub = sub.sort_values("score", ascending=False).head(k_class)

            # ---- score Ã inverse-vol ----
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
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:

        cfg = self.config
        rows = []

        for t in tickers:
            try:
                mom_fast = self.signals.get_series(
                    t, "ts_mom", start=start, end=end, window=cfg.mom_fast_window
                )
                mom_slow = self.signals.get_series(
                    t, "ts_mom", start=start, end=end, window=cfg.mom_slow_window
                )
                vol = self.signals.get_series(
                    t, "vol", start=start, end=end, window=cfg.vol_window
                )

                beta = self.signals.get_series(
                    t,
                    "beta",
                    start=start,
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
