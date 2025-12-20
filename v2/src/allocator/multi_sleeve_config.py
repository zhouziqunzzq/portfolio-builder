from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

SINGLE_SLEEVE_TESTING = "trend"


@dataclass
class MultiSleeveConfig:
    """
    Configuration for the multi-sleeve allocator (V2).
    """

    # Lookback windows
    signal_lookback_days: int = 365
    regime_lookback_days: int = 252

    # Regime configs
    regime_score_columns: Tuple[str, ...] = (
        "bull",
        "correction",
        "bear",
        "crisis",
        "stress",
    )
    regime_sample_freq: str = "M"  # frequency for regime scoring

    # Trend filter on benchmark for risk-on/risk-off scaling
    trend_filter_enabled: bool = True
    trend_benchmark: str = "SPY"
    trend_window: int = 200  # in trading days
    trend_sample_freq: str = "M"  # frequency for trend filter
    sleeve_risk_on_equity_frac: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": 1.0,
            "fast_alpha": 1.0,
        }
    )
    sleeve_risk_off_equity_frac: Dict[str, float] = field(
        default_factory=lambda: {
            "trend": 0.7,
            "fast_alpha": 0.7,
        }
    )

    # If True: allow implicit cash by NOT normalizing global sleeve-combined
    # equity weights up to 1.0 when their raw sum is <= 1.0. We still
    # normalize (scale down) if the sum exceeds 1.0 (to avoid leverage).
    preserve_cash_if_under_target: bool = True

    # Regime -> sleeve -> weight BEFORE normalization.
    #
    # These represent *ideal* allocations for each sleeve under each regime.
    # MultiSleeveAllocator will blend these using regime scores.

    # Single sleeve for testing
    # sleeve_regime_weights: Dict[str, Dict[str, float]] = field(
    #     default_factory=lambda: {
    #         "bull": {
    #             SINGLE_SLEEVE_TESTING: 1.0,
    #         },
    #         "correction": {
    #             SINGLE_SLEEVE_TESTING: 1.0,
    #         },
    #         "bear": {
    #             SINGLE_SLEEVE_TESTING: 1.0,
    #         },
    #         "crisis": {
    #             SINGLE_SLEEVE_TESTING: 1.0,
    #         },
    #         "sideways": {
    #             SINGLE_SLEEVE_TESTING: 1.0,
    #         },
    #     }
    # )

    sleeve_regime_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "bull": {
                "trend": 0.88,
                "sideways_base": 0.06,
                "defensive": 0.06,
                "cash": 0.00,
            },
            "correction": {
                "trend": 0.55,
                "sideways_base": 0.15,
                "defensive": 0.25,
                "cash": 0.05,
            },
            "bear": {
                "trend": 0.08,
                "sideways_base": 0.02,
                "defensive": 0.60,
                "cash": 0.30,
            },
            "crisis": {
                "trend": 0.00,
                "sideways_base": 0.00,
                "defensive": 0.75,
                "cash": 0.25,
            },
            "stress": {
                "trend": 0.35,
                "sideways_base": 0.30,
                "defensive": 0.30,
                "cash": 0.05,
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
                "stress"
            ) or self.sleeve_regime_weights.get("bull")

        total = float(sum(mapping.values()))
        if total <= 0:
            return {}
        return {k: v / total for k, v in mapping.items()}
