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
    # CS-momentum
    # Weekly bars → convert “days” windows to “weeks”
    mom_windows=[13, 26, 52],
    mom_weights=[1.0, 1.0, 1.0],
    # TS-momentum
    # If/when you enable TS-mom later, make it consistent too
    use_ts_mom=False,
    ts_weight=0.0,
    ts_mom_windows=[13, 26, 52],
    ts_mom_weights=[0.3, 1.0, 0.7],  # keep your default shape
    use_ts_gate=False,
    ts_gate_threshold=0.0,
    # Spread-momentum: shorter horizon tends to work best on weekly
    use_spread_mom=True,
    spread_mom_windows=[13],
    spread_mom_window_weights=[1.0],
    spread_mom_weight=0.3,
    # Volatility scaling for weekly bars
    vol_mode="rolling",  # keep default; you can later test ewm here
    vol_window=13,  # ~3 months of weekly bars
    ewm_vol_halflife=10,  # ~10 weeks (roughly comparable “smoothing”)
    vol_penalty=0.3,  # keep default; weekly already reduces noise
    # Non-vectorized mode buffer: weekly windows need more history
    signals_extra_buffer_days=90,
    # Rebalance intent (even if global scheduler overrides)
    rebalance_freq="M",
    # Sector weighting
    sector_cs_weight=1.0,
    sector_ts_weight=0.0,
    # Sector weights smoothing
    sector_smoothing_beta=0.2,
    sector_smoothing_freq="signal",
    # Everything else: keep defaults (sector weighting, top-k, gating, etc.)
)

TREND_CONFIG_MONTHLY = TrendConfig(
    signals_interval="1mo",
    # --------------------------------------------------
    # CS momentum (monthly bars)
    # --------------------------------------------------
    # 3 / 6 / 12 months
    mom_windows=[3, 6, 12],
    mom_weights=[1.0, 1.0, 1.0],
    # --------------------------------------------------
    # TS momentum (disabled for diagnostic)
    # --------------------------------------------------
    use_ts_mom=False,
    ts_weight=0.0,
    use_ts_gate=False,
    # --------------------------------------------------
    # Spread momentum
    # --------------------------------------------------
    # Keep horizon comparable to weekly 13w ≈ 3mo
    use_spread_mom=True,
    spread_mom_windows=[3],
    spread_mom_window_weights=[1.0],
    spread_mom_weight=0.3,
    # --------------------------------------------------
    # Volatility penalty (monthly scale)
    # --------------------------------------------------
    vol_mode="rolling",
    vol_window=3,  # ~3 months
    vol_penalty=0.5,
    # --------------------------------------------------
    # Data buffer (monthly bars need fewer)
    # --------------------------------------------------
    signals_extra_buffer_days=120,
    # --------------------------------------------------
    # Rebalancing intent
    # --------------------------------------------------
    rebalance_freq="M",
    # --------------------------------------------------
    # Sector weighting
    # --------------------------------------------------
    sector_cs_weight=1.0,
    sector_ts_weight=0.0,
    # --------------------------------------------------
    # Sector smoothing (monthly)
    # --------------------------------------------------
    sector_smoothing_beta=0.2,
    sector_smoothing_freq="signal",  # monthly signal dates
    # --------------------------------------------------
    # Everything else: defaults
    # --------------------------------------------------
)


if __name__ == "__main__":
    print("Daily Trend Configurations:")
    print(TREND_CONFIG_DAILY)

    print("Weekly Trend Configurations:")
    print(TREND_CONFIG_WEEKLY)

    print("Monthly Trend Configurations:")
    print(TREND_CONFIG_MONTHLY)
