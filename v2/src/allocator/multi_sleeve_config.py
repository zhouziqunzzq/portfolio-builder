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

    # Regime score columns supplied by RegimeEngine
    regime_score_columns: Tuple[str, ...] = (
        "bull",
        "correction",
        "bear",
        "crisis",
        "sideways",
    )

    # Trend filter on benchmark for risk-on/risk-off scaling
    trend_filter_enabled: bool = True
    trend_benchmark: str = "SPY"
    trend_window: int = 200  # in trading days
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
            # Strong uptrend, normal vol
            "bull": {
                "trend": 0.90,
                # "sideways": 0.04,  # tiny, exploratory
                "defensive": 0.06,  # small stabilizer
                "cash": 0.00,
            },
            # Uptrend but pullback / higher vol
            "correction": {
                "trend": 0.28,
                # "sideways": 0.12,  # modest contribution
                "defensive": 0.40,
                "cash": 0.20,
            },
            # Downtrend, elevated vol
            "bear": {
                "trend": 0.03,
                # "sideways": 0.02,  # tiny, avoids overexposure to laggards
                "defensive": 0.53,
                "cash": 0.42,
            },
            # Panic / crisis regime
            "crisis": {
                "trend": 0.00,
                # "sideways": 0.00,  # fully suppressed
                "defensive": 0.72,
                "cash": 0.28,
            },
            # Choppy / sideways
            "sideways": {
                "trend": 0.15,  # down from 0.25
                # "sideways": 0.25,  # main role here, but not dominant
                "defensive": 0.40,  # stabilizer
                "cash": 0.20,
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
