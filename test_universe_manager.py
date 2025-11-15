from datetime import datetime, timedelta
from market_data_store import MarketDataStore
from universe_manager import UniverseManager

# 1. Init data store
store = MarketDataStore(data_root="./data")

# 2. Load universe (e.g. S&P 500 list from CSV)
# universe = UniverseManager.from_csv(
#     data_store=store,
#     csv_path="./data/universe/sp500_constituents.csv",
#     ticker_col="Symbol",
#     sector_col="GICS Sector",    # or None if you don't have it yet
#     name="SP500_static",
# )
universe = UniverseManager.from_membership_csv(
    data_store=store,
    csv_path="data/universe/sp500_membership_history_enriched.csv",
    ticker_col="Symbol",
    sector_col="GICS Sector",
    date_added_col="DateAdded",
    date_removed_col="DateRemoved",
    name="SP500_hist",
)

# 3. Ensure all data is locally cached for your backtest window
end = datetime.today().date()
start = end - timedelta(days=365 * 10)
universe.ensure_ohlcv(start=start, end=end, interval="1d")

# 4. Build a daily close price matrix for signal computation
prices = universe.get_price_matrix(start=start, end=end, field="Close", interval="1d")
print(prices.head())
print(prices.shape)
