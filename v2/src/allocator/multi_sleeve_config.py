from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Mapping, Any, Tuple


@dataclass
class MultiSleeveConfig:
    """
    Configuration for the multi-sleeve allocator.

    Later you can wire this to v2/config/strategy.yml; for now we keep it
    hard-coded and simple.
    """

    # How far back sleeves should look for signals by default
    signal_lookback_days: int = 365

    # How far back to feed into RegimeEngine.get_regime_frame
    regime_lookback_days: int = 252

    # Names of regime-score columns in RegimeEngine output
    regime_score_columns: Tuple[str, ...] = (
        "bull",
        "correction",
        "bear",
        "crisis",
        "sideways",
    )

    # Mapping: regime_name -> {sleeve_name -> weight}
    # These are top-level weights per regime. The allocator will
    # blend them using regime scores, then normalize across sleeves.
    sleeve_regime_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            # With only the defensive sleeve wired initially, every regime
            # just maps to defensive: 1.0. You can extend this later.
            "bull": {
                "defensive": 1.0,
            },
            "correction": {
                "defensive": 1.0,
            },
            "bear": {
                "defensive": 1.0,
            },
            "crisis": {
                "defensive": 1.0,
            },
            "sideways": {
                "defensive": 1.0,
            },
        }
    )

    def get_primary_sleeve_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """
        Fallback: return normalized sleeve weights for a *single* regime.

        Used when we don't have regime scores or when blending fails.
        """
        if regime not in self.sleeve_regime_weights:
            # fallback order: sideways -> bull -> equal-weight-over-all-configured
            for fb in ("sideways", "bull"):
                if fb in self.sleeve_regime_weights:
                    regime = fb
                    break

        mapping = self.sleeve_regime_weights.get(regime, {})
        if not mapping:
            # equal-weight across all sleeves mentioned anywhere
            all_sleeves: Dict[str, float] = {}
            for rm, m in self.sleeve_regime_weights.items():
                for s in m.keys():
                    all_sleeves[s] = 1.0
            mapping = all_sleeves

        total = float(sum(mapping.values()))
        if total <= 0:
            return {}
        return {name: w / total for name, w in mapping.items()}
