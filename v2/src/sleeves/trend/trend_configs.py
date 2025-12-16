# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[2]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from sleeves.trend.trend_config import TrendConfig

TREND_CONFIG_DAILY = TrendConfig(
    signals_interval="1d",
    # Keep the rest of the configuration default
)

TREND_CONFIG_WEEKLY = TrendConfig(
    signals_interval="1wk",

    # Weekly bars → convert “days” windows to “weeks”
    mom_windows=[13, 26, 52],
    mom_weights=[1.0, 1.0, 1.0],

    # If/when you enable TS-mom later, make it consistent too
    ts_mom_windows=[13, 26, 52],
    ts_mom_weights=[0.3, 1.0, 0.7],  # keep your default shape

    # Spread-momentum: shorter horizon tends to work best on weekly
    use_spread_mom=True,
    spread_mom_windows=[13],
    spread_mom_window_weights=[1.0],
    spread_mom_weight=0.4,  # keep default unless you’re retuning

    # Volatility scaling for weekly bars
    vol_mode="rolling",     # keep default; you can later test ewm here
    vol_window=13,          # ~3 months of weekly bars
    ewm_vol_halflife=10,    # ~10 weeks (roughly comparable “smoothing”)
    vol_penalty=0.5,        # keep default; weekly already reduces noise

    # Non-vectorized mode buffer: weekly windows need more history
    signals_extra_buffer_days=90,

    # Rebalance intent (even if global scheduler overrides)
    rebalance_freq="M",

    # Sector weights smoothing
    sector_smoothing_freq="rebalance_dates",

    # Everything else: keep defaults (sector weighting, top-k, gating, etc.)
)

if __name__ == "__main__":
    print("Daily Trend Configurations:")
    print(TREND_CONFIG_DAILY)

    print("Weekly Trend Configurations:")
    print(TREND_CONFIG_WEEKLY)
