from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine
from src.regime_engine import RegimeEngine, RegimeConfig
from pathlib import Path
import os
import sys
import matplotlib

# Choose backend: prefer interactive when a display is available, otherwise use Agg
has_display = bool(os.environ.get("DISPLAY")) or sys.platform.startswith("win") or sys.platform == "darwin"
if not has_display:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    mds = MarketDataStore(data_root="data/prices", source="yfinance", local_only=False)
    signals = SignalEngine(mds)
    regime_engine = RegimeEngine(signals, RegimeConfig(benchmark="SPY"))

    # regimes = regime_engine.get_regime_frame("2015-01-01", "2025-01-01")
    regimes = regime_engine.get_regime_frame("2000-01-01", "2025-11-21")
    print(regimes.head())
    print(regimes.tail())
    print(regimes[["bull", "correction", "bear", "crisis", "sideways", "primary_regime"]].tail())

    # --- Quick plotting: stacked-area of regime scores (monthly aggregated) ---
    score_cols = [c for c in ["bull", "correction", "bear", "crisis", "sideways"] if c in regimes.columns]
    if score_cols:
        # Ensure datetime index
        regimes = regimes.sort_index()
        if not isinstance(regimes.index, (pd.DatetimeIndex,)):
            regimes.index = pd.to_datetime(regimes.index)

        # Aggregate monthly to reduce point density
        scores_monthly = regimes[score_cols].resample("M").mean()

        out_dir = Path("data/plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        scores_monthly.plot.area(ax=ax, stacked=True)
        ax.set_title("Regime Scores (monthly average)")
        ax.set_ylabel("Normalized score")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")
        fig.tight_layout()
        out_path = out_dir / "regime_scores.png"
        fig.savefig(out_path)
        print(f"Saved regime scores plot to {out_path}")
        # Show interactively if a display is available
        if has_display:
            try:
                plt.show()
            except Exception:
                # If show fails for any reason, still continue silently
                pass
    else:
        print("No regime score columns found to plot.")
