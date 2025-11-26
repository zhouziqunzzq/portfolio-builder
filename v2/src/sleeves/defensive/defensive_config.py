from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DefensiveConfig:
    # -------------------------------
    # Universe / sectors
    # -------------------------------
    # Individual stocks: defensive sectors inside S&P 500
    defensive_sectors: List[str] = field(
        default_factory=lambda: [
            "Consumer Staples",
            "Health Care",
            "Utilities",
        ]
    )

    # ETFs grouped by asset class
    defensive_equity_etfs: List[str] = field(
        default_factory=lambda: [
            # defensive equity sector ETFs
            "XLP",
            "XLV",
            "XLU",
        ]
    )
    bond_etfs: List[str] = field(
        default_factory=lambda: [
            # aggregate + treasury + IG credit
            "BND",  # US aggregate bonds
            "IEF",  # 7~10Y Treasuries
            "TLT",  # 20Y+ Treasuries
            "SHY",  # 1~3Y Treasuries
            "LQD",  # IG corporates
        ]
    )
    gold_etfs: List[str] = field(
        default_factory=lambda: [
            # gold hedge (crisis / inflation hedge)
            "GLD",
        ]
    )

    # Convenience: all defensive ETFs in one list (read-only derived field)
    @property
    def defensive_etfs(self) -> List[str]:
        return sorted(set(self.defensive_equity_etfs + self.bond_etfs + self.gold_etfs))

    # Explicit asset-class mapping for ETFs
    # (stocks from defensive_sectors will be treated as "equity" in the sleeve logic)
    asset_class_for_etf: Dict[str, str] = field(
        default_factory=lambda: {
            # defensive equity
            "XLP": "equity",
            "XLV": "equity",
            "XLU": "equity",
            # gold
            "GLD": "gold",
            # bonds
            "BND": "bond",
            "IEF": "bond",
            "TLT": "bond",
            "SHY": "bond",
            "LQD": "bond",
        }
    )

    # Liquidity filters
    min_price: float = 5.0  # remove penny/low-liquidity names
    min_adv: float = 500_000  # min average daily volume
    min_adv_window: int = 20  # 1-month ADV

    # -------------------------------
    # Signals / windows
    # -------------------------------
    mom_fast_window: int = 63
    mom_slow_window: int = 252
    vol_window: int = 20
    beta_window: int = 63

    # Ranking weights for per-asset score
    # (used inside each asset class; then combined with class-level allocations)
    w_mom_fast: float = 0.3
    w_mom_slow: float = 0.3
    w_low_vol: float = 0.4
    w_low_beta: float = 0.2

    # -------------------------------
    # Portfolio construction
    # -------------------------------
    top_k: int = 10
    max_weight_per_name: float = 0.10
    # TODO: soft constraint; can enforce later
    max_weight_per_sector: float = 0.40

    # -------------------------------
    # Asset-class allocations by regime
    # -------------------------------
    # These are *within the Defensive Sleeve* (before the top-level allocator multiplies
    # by the sleeve's overall weight). They should sum to ~1 per regime.
    #
    # Keys: global regime labels from RegimeEngine
    # Values: per-asset-class allocations in {"equity", "bond", "gold"}.
    asset_class_allocations_by_regime: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            # Strong uptrend, normal vol
            "bull": {
                "equity": 0.40,
                "bond": 0.25,
                "gold": 0.35,
            },
            # Uptrend but pullback / higher vol
            "correction": {
                "equity": 0.10,
                "bond": 0.40,
                "gold": 0.50,
            },
            # Downtrend, elevated vol
            "bear": {
                "equity": 0.00,
                "bond": 0.45,
                "gold": 0.55,
            },
            # Panic / crisis regime
            "crisis": {
                "equity": 0.00,
                "bond": 0.25,
                "gold": 0.75,
            },
            # Choppy / sideways
            "sideways": {
                "equity": 0.05,
                "bond": 0.50,
                "gold": 0.45,
            },
        }
    )

    # -------------------------------
    # Rebalancing
    # -------------------------------
    rebalance_freq: str = "M"  # "M" for monthly (pandas offset), or use N days
