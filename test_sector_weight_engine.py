import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from market_data_store import MarketDataStore
from universe_manager import UniverseManager
from signal_engine import SignalEngine
from sector_weight_engine import SectorWeightEngine

# 1. Setup data + universe
store = MarketDataStore(data_root="./data")

universe = UniverseManager.from_csv(
    data_store=store,
    csv_path="data/universe/sp500_constituents.csv",
    ticker_col="Symbol",
    sector_col="GICS Sector",
    name="SP500_static",
)

end = datetime.today().date()
start = end - timedelta(days=365 * 10)

prices = universe.get_price_matrix(start=start, end=end, field="Close", interval="1d")

# 2. Signals
engine = SignalEngine(prices=prices, sector_map=universe.sector_map)
momentum_score = engine.compute_momentum_score(windows=(63, 126, 252))
vol_score = engine.compute_vol_score(window=20)
stock_score = engine.compute_stock_score(
    mom_windows=(63, 126, 252),
    mom_weights=None,
    vol_window=20,
    vol_weight=1.0,
)

sector_scores = engine.compute_sector_scores_from_stock_scores(
    stock_score, universe.sector_map)

# 3. Benchmark for trend filter (SPY)
spy = yf.download(
    "SPY", start=start, end=end, progress=False, group_by="column",
)
# Handle possible MultiIndex columns just in case
if isinstance(spy.columns, pd.MultiIndex):
    # Common layout: level 0 = Price, level 1 = Ticker
    # For single-ticker download we can just take the first level
    # or cross-section the specific ticker.
    try:
        # If indexed as (Price, Ticker) with ticker level:
        spy = spy.xs("SPY", axis=1, level=-1)
    except Exception:
        # Fallback: keep only the first level
        spy.columns = spy.columns.get_level_values(0)
spy = spy["Close"]
spy = spy.reindex(prices.index).ffill().bfill()

# 4. Sector weights over time
swe = SectorWeightEngine(
    sector_scores=sector_scores,
    benchmark_prices=spy,
    alpha=1.0,
    w_min=0.03,
    w_max=0.50,
    beta=0.3,
    trend_window=200,
    risk_on_equity_frac=1.0,
    risk_off_equity_frac=0.7,
)

sector_weights = swe.compute_weights()
# print("Momentum Score:")
# print(momentum_score.head())
# print(momentum_score.tail())
# print("Volatility Score:")
# print(vol_score.head())
# print(vol_score.tail())
# print("Stock Score:")
# print(stock_score.head())
# print(stock_score.tail())
# print("Sector Scores:")
# print(sector_scores.head())
# print(sector_scores.tail())
print("Sector Weights:")
print(sector_weights.head())
print(sector_weights.tail())
print("Cash weights (implicit):")
print(1 - sector_weights.sum(axis=1).tail())

# 5. Plot sector weights over time
plt.figure(figsize=(12, 6))

# Stacked area chart: each sector as a colored band over time
sector_weights.plot.area(ax=plt.gca(), linewidth=0)

plt.title("Sector Weights Over Time")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.legend(title="Sector", loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
plt.show()
