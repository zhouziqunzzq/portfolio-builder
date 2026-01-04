from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine, SignalStore, SignalKey
import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mds = MarketDataStore(
        data_root="data/prices",
        source="yfinance",
        local_only=False,
    )
    signals = SignalEngine(mds)

    start = "2020-01-01"
    end = "2025-01-01"

    spy_trend = signals.get_series(
        "SPY",
        "trend_score",
        start,
        end,
        fast_window=50,
        slow_window=200,
    )
    print(f"SPY Trend Score:\n{spy_trend}")

    spy_vol = signals.get_series("SPY", "vol", start, end, window=20)
    print(f"SPY Volatility:\n{spy_vol}")

    spy_ewm_vol = signals.get_series("SPY", "ewm_vol", start, end, halflife=20)
    print(f"SPY EWM Volatility:\n{spy_ewm_vol}")

    spy_mom = signals.get_series("SPY", "ts_mom", start, end, window=252)
    print(f"SPY Momentum:\n{spy_mom}")

    spy_ret = signals.get_series("SPY", "ret", start, end, window=252)
    print(f"SPY Return:\n{spy_ret}")
    spy_log_ret = signals.get_series("SPY", "log_ret", start, end, window=252)
    print(f"SPY Log Return:\n{spy_log_ret}")

    # A sleeve can later request something already computed:
    # another_view = signals.get_series("SPY", "vol", start, end, window=20)
    # -> served from cache, no re-computation
    # print(f"Another view of SPY Volatility:\n{another_view}")

    spy_sma_20 = signals.get_series("SPY", "sma", start, end, window=20)
    print(f"SPY 20-day SMA:\n{spy_sma_20}")
    # Re-request to test caching
    # spy_sma_20_again = signals.get_series("SPY", "sma", start, end, window=20)
    # print(f"SPY 20-day SMA (again, from cache):\n{spy_sma_20_again}")

    spy_last_price = signals.get_series("SPY", "last_price", start, end)
    print(f"SPY Last Price:\n{spy_last_price}")

    aapl_spy_beta = signals.get_series(
        "AAPL",
        "beta",
        start,
        end,
        benchmark="SPY",
        window=252,
    )
    print(f"AAPL Beta vs SPY:\n{aapl_spy_beta}")

    # Test spread momentum signal
    # Case 1: same ticker for ticker and benchmark - expect all zeros
    spy_spread_mom = signals.get_series(
        "SPY",
        "spread_mom",
        start,
        end,
        benchmark="SPY",
        window=252,
    )
    print(f"SPY Spread Momentum vs SPY:\n{spy_spread_mom}")
    # Case 2: different benchmark
    aapl_spy_spread_mom = signals.get_series(
        "AAPL",
        "spread_mom",
        start,
        end,
        benchmark="SPY",
        window=252,
    )
    print(f"AAPL Spread Momentum vs SPY:\n{aapl_spy_spread_mom}")

    # Test Bollinger signals
    BB_WINDOW = 20
    spy_bb_mid = signals.get_series(
        "SPY",
        "bb_mid",
        start,
        end,
        window=BB_WINDOW,
    )
    print(f"SPY Bollinger Mid:\n{spy_bb_mid}")
    spy_bb_std = signals.get_series(
        "SPY",
        "bb_std",
        start,
        end,
        window=BB_WINDOW,
    )
    print(f"SPY Bollinger Std:\n{spy_bb_std}")
    spy_bb_upper = signals.get_series(
        "SPY",
        "bb_upper",
        start,
        end,
        window=BB_WINDOW,
        k=2.0,
    )
    print(f"SPY Bollinger Upper:\n{spy_bb_upper}")
    spy_bb_lower = signals.get_series(
        "SPY",
        "bb_lower",
        start,
        end,
        window=BB_WINDOW,
        k=2.0,
    )
    print(f"SPY Bollinger Lower:\n{spy_bb_lower}")
    spy_bb_z = signals.get_series(
        "SPY",
        "bb_z",
        start,
        end,
        window=BB_WINDOW,
    )
    print(f"SPY Bollinger Z-Score:\n{spy_bb_z}")
    spy_bb_bandwidth = signals.get_series(
        "SPY",
        "bb_bandwidth",
        start,
        end,
        window=BB_WINDOW,
        k=2.0,
    )
    print(f"SPY Bollinger Bandwidth:\n{spy_bb_bandwidth}")
    spy_bb_percent_b = signals.get_series(
        "SPY",
        "bb_percent_b",
        start,
        end,
        window=BB_WINDOW,
        k=2.0,
    )
    print(f"SPY Bollinger %b:\n{spy_bb_percent_b}")

    # --- Small visualization: plot recent Bollinger Bands for SPY ---
    try:
        os.makedirs("plots", exist_ok=True)
        # Align price and bands on common index and take recent window
        df_plot = pd.concat(
            [
                spy_last_price.rename("price"),
                spy_bb_mid.rename("mid"),
                spy_bb_upper.rename("upper"),
                spy_bb_lower.rename("lower"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if not df_plot.empty:
            df_plot = df_plot.tail(200)
            plt.figure(figsize=(12, 6))
            plt.plot(df_plot.index, df_plot["price"], label="Price")
            plt.plot(df_plot.index, df_plot["mid"], label="BB Mid")
            plt.plot(df_plot.index, df_plot["upper"], label="BB Upper", linestyle="--")
            plt.plot(df_plot.index, df_plot["lower"], label="BB Lower", linestyle="--")
            plt.fill_between(
                df_plot.index,
                df_plot["lower"],
                df_plot["upper"],
                color="gray",
                alpha=0.2,
            )
            plt.title("SPY Bollinger Bands (last 200 points)")
            plt.legend()
            outpath = os.path.join("plots", "spy_bollinger.png")
            plt.savefig(outpath, bbox_inches="tight")
            # plt.show()
            plt.close()
            print(f"Saved Bollinger plot to {outpath}")
        else:
            print("Not enough aligned data to plot Bollinger bands.")
    except Exception as e:
        print(f"Failed to create Bollinger plot: {e}")

    # Trend slope signal
    spy_trend_slope = signals.get_series(
        "SPY",
        "trend_slope",
        start,
        end,
        window=BB_WINDOW,
        use_log_price=False,
    )
    print(f"SPY Trend Slope:\n{spy_trend_slope}")
    spy_trend_slope_log = signals.get_series(
        "SPY",
        "trend_slope",
        start,
        end,
        window=BB_WINDOW,
        use_log_price=True,
    )
    print(f"SPY Trend Slope (log price):\n{spy_trend_slope_log}")

    # Donchian Channel position signal
    spy_donchian_pos = signals.get_series(
        "SPY",
        "donchian_pos",
        start,
        end,
        window=BB_WINDOW,
    )
    print(f"SPY Donchian Channel Position 20:\n{spy_donchian_pos}")
