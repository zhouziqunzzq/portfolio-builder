from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SidewaysBaseConfig:
    # -------------------------------
    # Universe (ETFs only)
    # -------------------------------
    defensive_equity_etfs: List[str] = field(
        default_factory=lambda: [
            "XLP",  # staples
            "XLV",  # healthcare
            "XLU",  # utilities
            # optional: low-vol broad equity
            "USMV",
        ]
    )
    bond_etfs: List[str] = field(
        default_factory=lambda: [
            "SHY",  # 1-3y treasuries
            "IEF",  # 7-10y treasuries
            "BND",  # agg bonds
        ]
    )
    gold_etfs: List[str] = field(default_factory=lambda: ["GLD"])

    # Optional: treat cash proxy as a "bond-like" low risk parking asset
    cash_proxy_etfs: List[str] = field(default_factory=lambda: ["BIL"])

    @property
    def sideways_etfs(self) -> List[str]:
        return sorted(
            set(
                self.defensive_equity_etfs
                + self.bond_etfs
                + self.gold_etfs
                + self.cash_proxy_etfs
            )
        )

    asset_class_for_etf: Dict[str, str] = field(
        default_factory=lambda: {
            # equity
            "XLP": "equity",
            "XLV": "equity",
            "XLU": "equity",
            "USMV": "equity",
            # bonds / cash-proxy
            "SHY": "bond",
            "IEF": "bond",
            "BND": "bond",
            "BIL": "cash",
            # gold
            "GLD": "gold",
        }
    )

    # -------------------------------
    # Liquidity filters (simple)
    # -------------------------------
    min_price: float = 5.0
    min_adv: float = 500_000
    min_adv_window: int = 20

    # -------------------------------
    # Windows / metrics
    # -------------------------------
    signals_extra_buffer_days: int = 60
    vol_window: int = 20  # realized vol window
    dd_window: int = 63  # drawdown lookback
    slope_window: int = 126  # drift lookback (small weight)
    ann_factor: int = 252  # for annualizing daily vol

    # Scoring weights (Option A)
    # low vol + low drawdown dominate; slope is a tie-breaker
    w_low_vol: float = 0.50
    w_low_dd: float = 0.45
    w_pos_slope: float = 0.05

    # -------------------------------
    # Portfolio construction
    # -------------------------------
    top_k: int = 6
    max_weight_per_name: float = 0.40  # keep concentration bounded
    min_weight_cutoff: float = 0.00  # can raise later (e.g. 0.02)

    # Allocations *within this sleeve*
    # (Top-level allocator will multiply by sleeve weight.)
    asset_class_allocations: Dict[str, float] = field(
        default_factory=lambda: {
            "equity": 0.35,
            "bond": 0.45,
            "gold": 0.15,
            "cash": 0.05,
        }
    )

    # -------------------------------
    # Rebalancing
    # -------------------------------
    rebalance_freq: str = "M"  # keep slow to survive whipsaws
