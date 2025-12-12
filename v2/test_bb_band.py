from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine, SignalStore, SignalKey
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Bollinger bandwidth + trend slope for a ticker")
    parser.add_argument("--ticker", "-t", default="SPY", help="Ticker to analyze (default: SPY)")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    mds = MarketDataStore(
        data_root="data/prices",
        source="yfinance",
        local_only=False,
    )
    signals = SignalEngine(mds)

    # Plot last ~20 years (conservative long window)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=20)

    BB_WINDOW = 20

    # Only keep Bollinger & related signals
    price_series = signals.get_series(ticker, "last_price", start, end, interval="1d")
    bb_bandwidth = signals.get_series(
        ticker, "bb_bandwidth", start, end, window=BB_WINDOW, k=2.0
    )
    trend_slope_log = signals.get_series(
        ticker, "trend_slope", start, end, window=BB_WINDOW, use_log_price=True
    )

    # Align series
    df = pd.concat(
        [
            price_series.rename("price"),
            bb_bandwidth.rename("bb_bandwidth"),
            trend_slope_log.rename("trend_slope"),
        ],
        axis=1,
        join="inner",
    )
    df = df.sort_index().dropna()

    if df.empty:
        print("Not enough data to plot Bollinger + trend signals.")
    else:
        # Sideways condition: use a rolling gate over the boolean condition
        bw_thresh = 0.040
        slope_thresh = 0.0010
        cond = (df["bb_bandwidth"] < bw_thresh) & (
            df["trend_slope"].abs() < slope_thresh
        )
        gate_score = cond.rolling(window=10, min_periods=10).mean()
        # Determine sideways state with hysteresis
        is_sideways = pd.Series(False, index=gate_score.index)
        in_state = False
        for i, v in enumerate(gate_score.values):
            if not np.isfinite(v):
                continue
            if (not in_state) and (v >= 0.8):
                in_state = True
            elif in_state and (v <= 0.4):
                in_state = False
            is_sideways.iloc[i] = in_state

        print("cond true rate:", cond.mean())
        print("is_sideways true rate:", is_sideways.mean())

        os.makedirs("plots", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]}
        )

        # Price on top axes
        ax1.plot(df.index, df["price"], label="Price", color="tab:blue")
        ax1.set_ylabel("Price")
        ax1.set_title(
            f"{ticker} Price, BB Bandwidth ({BB_WINDOW}) and Trend Slope (log) — last 20 years"
        )

        # Shade sideways periods on price plot
        mask = is_sideways
        # find contiguous True spans
        mask_shift = mask.astype(int).diff().fillna(0)
        starts = mask_shift[mask_shift == 1].index.tolist()
        ends = mask_shift[mask_shift == -1].index.tolist()
        # handle case mask starts True
        if mask.iloc[0]:
            starts.insert(0, mask.index[0])
        if mask.iloc[-1]:
            ends.append(mask.index[-1])

        for s, e in zip(starts, ends):
            ax1.axvspan(s, e, color="orange", alpha=0.2)

        ax1.legend(loc="upper left")

        # Bottom axes: bb_bandwidth and trend_slope
        ax2.plot(
            df.index,
            df["bb_bandwidth"],
            label=f"bb_bandwidth_{BB_WINDOW}",
            color="tab:green",
        )
        ax2.set_ylabel("BB Bandwidth")
        ax2b = ax2.twinx()
        ax2b.plot(
            df.index,
            df["trend_slope"],
            label=f"trend_slope_{BB_WINDOW}_log",
            color="tab:red",
            alpha=0.8,
        )
        ax2b.set_ylabel("Trend Slope (log)")

        # Legends
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")

        outpath = os.path.join("plots", f"{ticker}_bb_sideways.png")
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved sideways Bollinger plot to {outpath}")
