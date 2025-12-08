"""
Quick sanity test for MultiSleeveAllocator

This test DOES NOT depend on real universe, real signals,
or real sleeves â it uses stubs/mocks so we exercise only the
allocator logic + regime-score blending.
"""

from datetime import datetime
import pandas as pd

from src.allocator.multi_sleeve_allocator import (
    MultiSleeveAllocator,
    MultiSleeveConfig,
)

# ------------------------------------------------------------
# 1) Stub RegimeEngine
# ------------------------------------------------------------


class StubRegimeEngine:
    """
    Always returns a DataFrame with:
        primary_regime = "bull"
        bull = 0.6
        bear = 0.4
    """

    def get_regime_frame(self, start, end):
        return pd.DataFrame(
            {
                "primary_regime": ["bull"],
                "bull": [0.6],
                "correction": [0.0],
                "bear": [0.4],
                "crisis": [0.0],
                "sideways": [0.0],
            },
            index=[pd.to_datetime(end)],
        )


# ------------------------------------------------------------
# 2) Stub Sleeve
# ------------------------------------------------------------


class StubSleeve:
    """
    Returns deterministic internal weights that sum to 1.
    """

    def __init__(self, name):
        self.name = name

    def generate_target_weights_for_date(self, as_of, start_for_signals, regime):
        # Different sleeves return different tickers for visibility
        if self.name == "defensive":
            return {"DEF1": 0.5, "DEF2": 0.5}
        elif self.name == "growth":
            return {"GRW1": 0.7, "GRW2": 0.3}
        elif self.name == "tactical":
            return {"TAC1": 1.0}
        else:
            return {}


# ------------------------------------------------------------
# 3) Multi-sleeve config for test
# ------------------------------------------------------------

test_config = MultiSleeveConfig(
    sleeve_regime_weights={
        # bull regime loves growth
        "bull": {
            "defensive": 0.2,
            "growth": 0.7,
            "tactical": 0.1,
        },
        # bear regime prefers defensive
        "bear": {
            "defensive": 0.7,
            "growth": 0.1,
            "tactical": 0.2,
        },
    },
    # keep defaults for everything else
)


# ------------------------------------------------------------
# 4) Instantiate allocator
# ------------------------------------------------------------

regime_engine = StubRegimeEngine()

sleeves = {
    "defensive": StubSleeve("defensive"),
    "growth": StubSleeve("growth"),
    "tactical": StubSleeve("tactical"),
}

allocator = MultiSleeveAllocator(
    regime_engine=regime_engine,
    sleeves=sleeves,
    config=test_config,
)


# ------------------------------------------------------------
# 5) Run the test
# ------------------------------------------------------------

if __name__ == "__main__":
    as_of = "2025-01-31"
    print("=== MultiSleeveAllocator Test ===")

    weights, ctx = allocator.generate_global_target_weights(as_of)

    for t, w in weights.items():
        print(f"{t:6} : {w:.4f}")

    print("Sum of weights =", sum(weights.values()))
