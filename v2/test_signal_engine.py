from src.market_data_store import MarketDataStore
from src.signal_engine import SignalEngine, SignalStore, SignalKey

if __name__ == "__main__":
    mds = MarketDataStore(
        data_root="data/prices",
        source="yfinance",
        local_only=False,
    )
    signals = SignalEngine(mds)

    start = "2020-01-01"
    end = "2025-01-01"

    # RegimeEngine might do:
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

    spy_mom = signals.get_series("SPY", "ts_mom", start, end, window=252)
    print(f"SPY Momentum:\n{spy_mom}")

    # A sleeve can later request something already computed:
    another_view = signals.get_series("SPY", "vol", start, end, window=20)
    # -> served from cache, no re-computation
    print(f"Another view of SPY Volatility:\n{another_view}")
