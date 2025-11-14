from datetime import datetime, timedelta
from market_data_store import MarketDataStore
from universe_manager import UniverseManager
from signal_engine import SignalEngine

# 1. Setup
store = MarketDataStore(data_root="./data")

universe = UniverseManager.from_csv(
    data_store=store,
    csv_path="data/universe/sp500_constituents.csv",
    ticker_col="Symbol",
    sector_col="GICS Sector",
    name="SP500_static",
)

end = datetime.today().date()
start = end - timedelta(days=365 * 5)

# (optional) warm up cache for all tickers
universe.ensure_ohlcv(start=start, end=end, interval="1d")

# 2. Build price matrix
prices = universe.get_price_matrix(start=start, end=end, field="Close", interval="1d")

# 3. Create SignalEngine
engine = SignalEngine(prices=prices, sector_map=universe.sector_map)

# 4. Compute signals
momentum_score = engine.compute_momentum_score(windows=(63, 126, 252))
vol_score = engine.compute_vol_score(window=20)
stock_score = engine.compute_stock_score(
    mom_windows=(63, 126, 252),
    mom_weights=None,      # equal weights
    vol_window=20,
    vol_weight=1.0,
)

print("Momentum Score:")
print(momentum_score.tail())
print("Volatility Score:")
print(vol_score.tail())
print("Stock Score:")
print(stock_score.tail())
