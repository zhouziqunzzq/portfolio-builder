from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FastAlphaConfig:
    """
    Fast Overnight Alpha Sleeve (Spread-momentum) with strong guardrails.

    Core mechanics:
      - Rank spread-mom signals daily (as-of T-1 close)
      - Execute at T open (handled by backtest executor)
      - Maintain a top-K portfolio with hysteresis and churn limits
    """

    # ----------------------------
    # Universe / signals
    # ----------------------------
    signals_extra_buffer_days: int = 30  # non-vec path

    use_liquidity_filters: bool = True
    adv_window: int = 20
    median_volume_window: int = 20
    min_adv: float = 5_000_000.0  # USD
    min_median_volume: int = 50_000  # shares
    min_price: float = 1.0  # dollars

    # Spread momentum definition
    spread_benchmark: str = "SPY"
    spread_mom_windows: List[int] = field(default_factory=lambda: [63])
    spread_mom_window_weights: List[float] = field(default_factory=lambda: [1.0])

    # Optional: if you want to down-weight volatile names in ranking.
    # Keep off for MVP unless you have a clear reason.
    use_vol_penalty: bool = False
    vol_window: int = 20
    vol_penalty: float = 0.0  # if use_vol_penalty=True, rank_score -= vol_penalty * vol

    # ----------------------------
    # Selection guardrails
    # ----------------------------
    target_k: int = 10  # target number of names to hold
    enter_k: int = 10  # can enter if rank <= enter_k
    exit_k: int = 15  # can exit if rank >= exit_k (hysteresis)

    dominance_gap: int = 7 # if worst_held_rank - best_out_rank >= dominance_gap, swap out worst_held
    dominance_gap_score: float = 0.08
    dominance_panic_score_gap: float = 0.20
    dominance_cooldown_days: int = 3
    max_out_rank: int = 5
    max_replacements_per_day: int = 2  # cap membership churn
    min_hold_days: int = 1  # cannot exit until held >= this many trading days

    # Daily turnover cap (sum(|w_new - w_old|)/2)
    max_daily_turnover: float = 0.20

    # Pruning / sparsity guard
    min_weight: float = 0.005     # anything below this is dust; drop it
    hard_max_names: int = 10     # enforce output sparsity (usually == target_k)

    # ----------------------------
    # Regime gating (optional)
    # ----------------------------
    use_regime_gating: bool = True
    gated_off_regimes: Tuple[str, ...] = ("crisis", "bear")
