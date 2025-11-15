import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from backtest_helper import compute_benchmark
from market_data_store import MarketDataStore
from universe_manager import UniverseManager
from signal_engine import SignalEngine
from sector_weight_engine import SectorWeightEngine
from stock_allocator import StockAllocator
from portfolio_backtester import PortfolioBacktester

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
    w_max=0.30,
    beta=0.3,
    trend_window=200,
    risk_on_equity_frac=1.0,
    risk_off_equity_frac=0.7,
)
# swe = SectorWeightEngine(
#     sector_scores=sector_scores,
#     benchmark_prices=spy,
#     alpha=1.0,
#     w_min=0.00,
#     w_max=0.70,
#     beta=0.3,
#     trend_window=200,
#     risk_on_equity_frac=1.0,
#     risk_off_equity_frac=0.7,
# )

sector_weights_daily = swe.compute_weights()
print("Sector Weights Daily:")
print(sector_weights_daily.head())
print(sector_weights_daily.tail())
print("Cash weights (implicit):")
print(1 - sector_weights_daily.sum(axis=1).tail())

# Monthly sector weights: last value in each calendar month
sector_weights_monthly = sector_weights_daily.resample("M").last()
print("Sector Weights Monthly:")
print(sector_weights_monthly.head())
print(sector_weights_monthly.tail())


# === Stock Allocator ===

# 1. Compute stock-level signal
# stock_score already computed above

# 2. (Optional) compute stock vol for inverse-vol weighting
stock_vol = engine.compute_volatility(window=20)

# 3. Build allocator
allocator = StockAllocator(
    sector_weights=sector_weights_monthly,# Monthly Date × Sector
    stock_scores=stock_score,             # Date × Ticker
    sector_map=universe.sector_map,       # ticker -> sector
    stock_vol=stock_vol,                  # optional
    top_k=2,                              # or 3, etc.
    weighting_mode="equal",               # or "inverse_vol"
    preserve_cash=True,                   # keep cash as 1 - sum(weights)
)

stock_weights_monthly = allocator.compute_stock_weights()
print("Stock Weights Monthly:")
print(stock_weights_monthly.head())
print(stock_weights_monthly.tail())
print(stock_weights_monthly.index[:5])  # should match rebalance_dates

# === Backtest ===

# prices: Date × Ticker (Close)
# stock_weights: Date × Ticker (weights from StockAllocator)

bt = PortfolioBacktester(
    prices=prices,
    weights=stock_weights_monthly,
    trading_days_per_year=252,
    initial_value=100_000.0,
    cost_per_turnover=0.001,        # 10 bps per 100% turnover
)

result = bt.run()
print(result.tail())

# Auto-warmup only (start stats when weight_sum > 1%)
stats_auto = bt.stats(result, auto_warmup=True, warmup_days=0)
print("Auto-warmup window:", stats_auto["EffectiveStart"], "->", stats_auto["EffectiveEnd"])
for k, v in stats_auto.items():
    if k.startswith("Effective"):
        continue
    print(k, ":", v)

# Print debug info
print("Debug info:")
print("Initial equity:", result["equity"].iloc[0])
print("Final equity:", result["equity"].iloc[-1])
print("Returns stats:")
print(result[["portfolio_return"]].describe())
print("Avg daily turnover:", result["turnover"].mean())
print("Avg daily cost impact:", result["cost"].mean())

# Benchmark comparison
warmup_start = stats_auto["EffectiveStart"].strftime("%Y-%m-%d")
warmup_end = stats_auto["EffectiveEnd"].strftime("%Y-%m-%d")
spy_prices = store.get_ohlcv("SPY", "2014-01-01", "today")
bench = compute_benchmark(spy_prices, warmup_start, warmup_end)
print("Benchmark (SPY) stats:")
print("From", warmup_start, "to", warmup_end)
print("CAGR:", bench["CAGR"])
print("Sharpe:", bench["Sharpe"])
print("MaxDrawdown:", bench["MaxDrawdown"])
