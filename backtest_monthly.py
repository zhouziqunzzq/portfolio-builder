import numpy as np
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


def pretty_pct(x):
    if x is None or pd.isna(x):
        return "   n/a"
    return f"{x*100:6.2f}%"

def print_stats_table(strategy_stats, benchmark_stats, label_strategy="Strategy", label_bench="SPY"):
    rows = []
    rows.append([
        label_strategy,
        strategy_stats.get("CAGR"),
        strategy_stats.get("Volatility"),
        strategy_stats.get("Sharpe"),
        strategy_stats.get("MaxDrawdown"),
        strategy_stats.get("AvgDailyTurnover"),
    ])
    rows.append([
        label_bench,
        benchmark_stats.get("CAGR"),
        benchmark_stats.get("Volatility"),
        benchmark_stats.get("Sharpe"),
        benchmark_stats.get("MaxDrawdown"),
        None,
    ])

    df_stats = pd.DataFrame(
        rows,
        columns=["Model", "CAGR", "Volatility", "Sharpe", "Max Drawdown", "Avg Daily Turnover"],
    )

    def fmt(x, col):
        if col in ("CAGR", "Volatility", "Max Drawdown"):
            return pretty_pct(x)
        if col == "Avg Daily Turnover":
            return pretty_pct(x) if x is not None else "   n/a"
        if col == "Sharpe":
            return f"{x:6.2f}" if x is not None and not pd.isna(x) else "   n/a"
        return x

    # Build a nicely formatted string table
    headers = df_stats.columns.tolist()
    lines = []
    header_line = " | ".join(f"{h:>18}" for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for _, row in df_stats.iterrows():
        parts = []
        for col in headers:
            val = fmt(row[col], col)
            parts.append(f"{val:>18}")
        lines.append(" | ".join(parts))

    print("\n=== Performance Summary ===")
    print("\n".join(lines))


# === Main backtest script ===
# Define backtest window
FULL_HISTORY = True

start = None
end = datetime.today().date()
if FULL_HISTORY:
    start = datetime(2000, 1, 1).date()
else:
    start = end - timedelta(days=365 * 10)

# 1. Setup data + universe
store = MarketDataStore(data_root="./data")

universe = UniverseManager.from_csv(
    data_store=store,
    csv_path="data/universe/sp500_constituents.csv",
    ticker_col="Symbol",
    sector_col="GICS Sector",
    name="SP500_static",
)

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
bt = PortfolioBacktester(
    prices=prices,
    weights=stock_weights_monthly,
    trading_days_per_year=252,
    initial_value=100_000.0,
    cost_per_turnover=0.001,
)

result = bt.run()

# Auto-warmup only (start stats when weight_sum > 1%)
stats_auto = bt.stats(result, auto_warmup=True, warmup_days=0)

warmup_start = stats_auto["EffectiveStart"].strftime("%Y-%m-%d")
warmup_end = stats_auto["EffectiveEnd"].strftime("%Y-%m-%d")

print(f"\nAuto-warmup window: {warmup_start} -> {warmup_end}")

print("Debug info:")
print("Initial equity:", result["equity"].iloc[0])
print("Final equity:", result["equity"].iloc[-1])
print("Returns stats:")
print(result[["portfolio_return"]].describe())
print("Avg daily turnover:", result["turnover"].mean())
print("Avg daily cost impact:", result["cost"].mean())

# === Benchmark comparison ===
spy_prices = store.get_ohlcv("SPY", start, end)
bench = compute_benchmark(spy_prices, warmup_start, warmup_end)

# IMPORTANT: make sure compute_benchmark returns Volatility & Sharpe too
# (easy to add there if not already)

print_stats_table(
    strategy_stats={
        k: v for k, v in stats_auto.items()
        if not k.startswith("Effective")
    } | {"AvgDailyTurnover": result["turnover"].mean()},
    benchmark_stats=bench,
    label_strategy="Momentum V1.5",
    label_bench="SPY",
)

# === Plots ===
ENABLE_PLOTS = True

if ENABLE_PLOTS:
    # Align equity curves for the stats window
    strat_eq = result["equity"].loc[warmup_start:warmup_end]

    # bench should include an 'equity' series for SPY; if not, you can reconstruct it
    spy_eq = bench["equity"]  # assuming your compute_benchmark returns this

    combined = pd.DataFrame({
        "Strategy": strat_eq,
        "SPY": spy_eq * (strat_eq.iloc[0] / spy_eq.iloc[0]),  # normalize SPY to same starting equity
    }).dropna()

    # --- Equity curve (log scale) ---
    plt.figure(figsize=(10, 6))
    plt.plot(combined.index, combined["Strategy"], label="Momentum V1.5")
    plt.plot(combined.index, combined["SPY"], label="SPY (benchmark)", alpha=0.8)
    plt.yscale("log")
    plt.title("Equity Curve (Log Scale)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # --- Strategy drawdown ---
    eq = combined["Strategy"]
    running_max = eq.cummax()
    dd = eq / running_max - 1.0

    plt.figure(figsize=(10, 4))
    plt.plot(dd.index, dd, label="Drawdown")
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # === Annual returns: Strategy vs SPY ===
    # Use daily returns aligned to warmup window
    strat_rets = result["portfolio_return"].loc[warmup_start:warmup_end]
    spy_rets = bench["returns"].loc[warmup_start:warmup_end]

    strat_yearly = (1 + strat_rets).groupby(strat_rets.index.year).prod() - 1
    spy_yearly = (1 + spy_rets).groupby(spy_rets.index.year).prod() - 1

    years = strat_yearly.index.astype(int)
    x = range(len(years))

    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], strat_yearly.values, width=width, label="Momentum V1.5")
    plt.bar([i + width/2 for i in x], spy_yearly.values, width=width, label="SPY")

    plt.xticks(x, years, rotation=45)
    plt.axhline(0, linestyle="--", linewidth=0.5)
    plt.title("Calendar-Year Returns: Strategy vs SPY")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    # === Rolling 1-year Sharpe (252-day) ===
    window = 252

    def rolling_sharpe(returns, window):
        # simple rolling Sharpe using mean/std (no RF)
        roll_mean = returns.rolling(window).mean()
        roll_std = returns.rolling(window).std()
        return (roll_mean / roll_std) * np.sqrt(252)

    strat_roll_sharpe = rolling_sharpe(strat_rets, window)
    spy_roll_sharpe = rolling_sharpe(spy_rets, window)

    plt.figure(figsize=(10, 4))
    plt.plot(strat_roll_sharpe.index, strat_roll_sharpe, label="Momentum V1.5")
    plt.plot(spy_roll_sharpe.index, spy_roll_sharpe, label="SPY", linestyle="--")
    plt.axhline(0, linestyle="--", linewidth=0.5)
    plt.title("Rolling 1-Year Sharpe (252-day)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # === Turnover over time (monthly average) ===
    monthly_turnover = result["turnover"].resample("M").mean()

    plt.figure(figsize=(10, 4))
    plt.plot(monthly_turnover.index, monthly_turnover.values)
    plt.title("Average Monthly Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover (fraction of portfolio)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    # === Sector allocation over time (stacked area) ===
    # Restrict to stats window
    sector_alloc_window = sector_weights_daily.loc[warmup_start:warmup_end]

    # Optionally, downsample to monthly for readability
    sector_alloc_monthly = sector_alloc_window.resample("M").last()

    plt.figure(figsize=(10, 6))
    plt.stackplot(
        sector_alloc_monthly.index,
        [sector_alloc_monthly[col].values for col in sector_alloc_monthly.columns],
        labels=sector_alloc_monthly.columns,
    )
    plt.title("Sector Allocation Over Time (Monthly)")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()

    plt.show()
