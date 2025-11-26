from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class MultiSleeveConfig:
    """
    Configuration for the multi-sleeve allocator (V2).
    """

    # Lookback windows
    signal_lookback_days: int = 365
    regime_lookback_days: int = 252

    # Regime score columns supplied by RegimeEngine
    regime_score_columns: Tuple[str, ...] = (
        "bull",
        "correction",
        "bear",
        "crisis",
        "sideways",
    )

    # If True: allow implicit cash by NOT normalizing global sleeve-combined
    # equity weights up to 1.0 when their raw sum is <= 1.0. We still
    # normalize (scale down) if the sum exceeds 1.0 (to avoid leverage).
    preserve_cash_if_under_target: bool = True

    # Regime -> sleeve -> weight BEFORE normalization.
    #
    # These represent *ideal* allocations for each sleeve under each regime.
    # MultiSleeveAllocator will blend these using regime scores.

    # All trend for testing
    # sleeve_regime_weights: Dict[str, Dict[str, float]] = field(
    #     default_factory=lambda: {
    #         "bull": {
    #             "trend": 1.0,
    #             # "defensive": 0.2,
    #         },
    #         "correction": {
    #             "trend": 1.0,
    #             # "defensive": 0.5,
    #         },
    #         "bear": {
    #             "trend": 1.0,
    #             # "defensive": 1.0,
    #         },
    #         "crisis": {
    #             "trend": 1.0,
    #             # "defensive": 1.0,
    #         },
    #         "sideways": {
    #             "trend": 1.0,
    #             # "defensive": 0.5,
    #         },
    #     }
    # )

    sleeve_regime_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "bull": {
                "trend": 1.0,
                "defensive": 0.0,
                "cash": 0.0,
            },
            "correction": {
                "trend": 0.7,
                "defensive": 0.2,
                "cash": 0.1,
            },
            "bear": {
                "trend": 0.1,
                "defensive": 0.4,
                "cash": 0.5,
            },
            "crisis": {
                "trend": 0.0,
                "defensive": 0.3,
                "cash": 0.7,
            },
            "sideways": {
                "trend": 0.3,
                "defensive": 0.3,
                "cash": 0.4,
            },
        }
    )

    def get_primary_sleeve_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """
        Fallback: return normalized sleeve weights for a *single* regime.
        Used when blending is unavailable.
        """
        mapping = self.sleeve_regime_weights.get(regime)

        # fallback
        if not mapping:
            mapping = self.sleeve_regime_weights.get(
                "sideways"
            ) or self.sleeve_regime_weights.get("bull")

        total = float(sum(mapping.values()))
        if total <= 0:
            return {}
        return {k: v / total for k, v in mapping.items()}
