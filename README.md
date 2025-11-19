## Portfolio Builder

Lightweight toolkit to build and test a sector-rotating equity portfolio using local OHLCV caching, a static S&P 500 universe, signal calculations, and a sector weighting engine.

### Components (production)

This repository ships a production-ready entrypoint under the `production/` directory. The
`production/src/` package contains self-contained, inlined copies of the core implementation
so production runners can import them without runtime path hacks.

- `production/src/market_data_store.py` — Local parquet cache for OHLCV via yfinance. Fetches missing ranges and keeps metadata.
- `production/src/universe_manager.py` — Loads a universe from CSV (membership histories are stored under `production/data/`) and can build price matrices.
- `production/src/signal_engine.py` — Computes momentum/volatility and aggregate stock/sector scores.
- `production/src/sector_weight_engine.py` — Converts sector scores into bounded, smoothed portfolio weights with a trend filter.
- `production/src/stock_allocator.py` — Allocates sector weights down to individual stock-level weights.

Note: Some earlier development helper scripts (for example, a standalone `get_sp500.py` scraper) have been removed or moved into the production runners. If you relied on those, use the production runner flags (see "Production CLI Tools") or provide your own universe CSV under `production/data/current_constituents.csv` or `production/data/sp500_membership.csv`.

### Data layout

- OHLCV cache (created on demand):
	- `data/ohlcv/1d/<TICKER>/data.parquet`
	- `data/ohlcv/1d/<TICKER>/meta.json`
- Universe CSV (input to `UniverseManager.from_csv`):
	- Recommended path: `data/universe/sp500_constituents.csv`

Note: Some files/scripts may refer to `data/universies/` (typo). Prefer `data/universe/` and update references accordingly.

## Quick start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install pandas yfinance requests lxml matplotlib pyarrow
```

Why these packages?

- `pandas` for data wrangling and `read_html`
- `yfinance` to download prices
- `requests` to fetch the Wikipedia page with headers
- `lxml` HTML parser used by `pandas.read_html`
- `matplotlib` for plotting (sector weights chart)
- `pyarrow` (or `fastparquet`) to write/read parquet cache files

### 2) Universe data

This repo no longer exposes a top-level `get_sp500.py` helper. Universe membership and related CSVs are stored under `production/data/` (for example `production/data/current_constituents.csv` and `production/data/sp500_membership*.csv`).

If you need to (re)generate a universe from Wikipedia or another source, either:

- Add your own CSV at `production/data/current_constituents.csv` (recommended), or
- Use the production runner to regenerate the universe if your configuration supports it:

```bash
python production/run_live.py --regenerate-universe
```

When creating a universe CSV manually, ensure tickers are normalized for yfinance (uppercase, replace `.` with `-`, e.g., `BRK.B` → `BRK-B`).

### 3) Warm the cache and run the pipeline

The production runners provide step-wise and full-pipeline execution. Common quick checks:

```bash
# Update prices and warm the parquet cache
python production/run_live.py --update-prices

# Compute signals
python production/run_live.py --compute-signals

# Compute sector weights
python production/run_live.py --compute-sector-weights

# Compute stock weights
python production/run_live.py --compute-stock-weights

# Or run the full rebalance pipeline
python production/run_live.py --rebalance
```

First runs that update prices will be slower while the local parquet cache is populated.

## Ticker normalization for yfinance

Wikipedia uses dot notation for share classes (e.g., `BRK.B`, `BF.B`), while yfinance expects dashes (`BRK-B`, `BF-B`). When preparing a universe CSV ensure tickers are normalized by:

- Uppercasing tickers
- Replacing `.` with `-` (e.g., `BRK.B` → `BRK-B`)

If you encounter additional symbol quirks, extend your normalization step or the universe generation logic used by your workflow.

## Troubleshooting

- HTTP 403 from Wikipedia: Use `requests.get(url, headers={...})` with a modern desktop browser User-Agent and an `Accept-Language` header, then pass `resp.text` into `pandas.read_html`.
- "Possibly delisted; no price data found": Ensure the ticker was normalized (dot → dash) and that it exists on Yahoo Finance for your date range.
- Parquet errors: Install `pyarrow` (or `fastparquet`).
- MultiIndex columns from yfinance: The code handles this, but if you customize downloads, flatten MultiIndex columns to price-only names.

## Notes

- This project favors a simple, file-based cache for repeatable backtests.
- Beware of API limits or short-term rate limiting from Yahoo Finance; stagger initial downloads if needed.

## Production CLI Tools

The `production/` directory contains higher-level runners that orchestrate the full pipeline, backtesting, exploration, and interactive rebalancing. Each script loads `strategy.yml` (unless you override with `--strategy`) to discover output paths and parameters.

### 1. Live Runner (`production/run_live.py`)

End-to-end or step-wise execution of the live (forward) pipeline. Core steps include universe regeneration, price/trend updates, signal computation, sector weight computation, stock allocation, and a rebalance pipeline combining those.

Key flags (subset):

```bash
# Regenerate universe (writes updated membership & dumps mask) then exit
python3 production/run_live.py --regenerate-universe

# Update prices & trend data only
python3 production/run_live.py --update-prices

# Compute signals (requires prices)
python3 production/run_live.py --compute-signals

# Compute sector weights (requires signals)
python3 production/run_live.py --compute-sector-weights

# Compute stock weights (requires sector weights)
python3 production/run_live.py --compute-stock-weights

# Full rebalance pipeline (signals → sector weights → stock weights)
python3 production/run_live.py --rebalance

# Start rebalance from a specific date (e.g., warm-start)
python3 production/run_live.py --rebalance --rebalance-start 2023-01-01

# Dump membership mask explicitly (also happens automatically after universe regeneration)
python3 production/run_live.py --dump-membership-mask
```

Artifacts are written under `output_root/` subfolders defined in `strategy.yml` (e.g., `prices/`, `signals/`, `weights/`, `masks/`, `plots/`, `logs/`).

### 2. Backtest Runner (`production/run_backtest.py`)

Runs the portfolio backtester over a historical window using previously computed monthly (or daily) stock weights and prices, then optionally generates performance & allocation plots.

Common usage:

```bash
# Basic backtest over a window
python3 production/run_backtest.py --backtest-start 2015-01-01 --backtest-end 2024-12-31

# Use only local cached data (skip any fetch logic inside helpers)
python3 production/run_backtest.py --local-only --backtest-start 2018-01-01

# Generate all plots and show them interactively
python3 production/run_backtest.py --backtest-start 2015-01-01 --plot-all --show

# Selectively plot equity + drawdown only (no interactive display)
python3 production/run_backtest.py --backtest-start 2015-01-01 --plot-equity --plot-drawdown
```

Plot flags available (all save images under `plots/`):
`--plot-equity`, `--plot-drawdown`, `--plot-annual`, `--plot-rolling`, `--plot-turnover`, `--plot-sectors`, or `--plot-all` (shortcut for all). Add `--show` to open interactive windows.

### 3. Interactive Rebalance Helper (`production/run_rebalance.py`)

Guides manual trade sizing using current (latest or specified date) stock & sector weights. It prints allocations, prompts for current positions + cash, then outputs target dollar holdings and a buy/sell execution plan.

```bash
# Use latest monthly weights (default frequency) interactively
python3 production/run_rebalance.py

# Use daily weights as of a specific date
python3 production/run_rebalance.py --frequency daily --as-of 2024-12-31

# Show a larger top slice of stock weights in the summary (e.g., top 40)
python3 production/run_rebalance.py --top 40

# Override strategy file (if you maintain multiple configs)
python3 production/run_rebalance.py --strategy production/config/strategy.yml
```

Workflow inside the helper:
1. Load latest (or specified) weights file set under `weights/`.
2. Print sector allocation (if available), top-N stock weights, and implied cash weight if weights sum < 1.
3. Prompt: enter lines `TICKER amount` until `done`.
4. Prompt: enter cash balance.
5. Compute current equity; size target positions; derive buys & sells and cash reconciliation; print trade plan.

### 4. Portfolio Analyzer (`production/explore_portfolio.py`)

Explores saved historical allocations (sector & stock) and produces snapshots or plots.

Examples:

```bash
# List allocation dates available
python3 production/explore_portfolio.py --list-dates

# Print summary for the latest date
python3 production/explore_portfolio.py --snapshot

# Print summary for a specific date
python3 production/explore_portfolio.py --snapshot --as-of 2024-06-30

# Plot stacked sector allocation (saved to plots/)
python3 production/explore_portfolio.py --plot-stacked

# Plot allocation heatmap (dates vs sectors)
python3 production/explore_portfolio.py --plot-heatmap

# Plot sector "bump" chart (rank evolution)
python3 production/explore_portfolio.py --plot-bump

# Plot time series for selected tickers (comma-separated)
python3 production/explore_portfolio.py --plot-tickers AAPL,MSFT,NVDA
```

Add `--show` (if supported in your environment) to display plots interactively; otherwise they are saved under `plots/`.

### General Tips

- If a runner reports missing weights, first execute the upstream steps in `run_live.py` (signals → sector weights → stock weights).
- All scripts honor the paths & parameters in `strategy.yml`; keep that file versioned alongside any changes in the pipeline.
- For reproducibility, avoid mixing daily and monthly weight frequencies mid-analysis unless explicitly intended.
- Implied cash = `1 - sum(stock_weights)`; maintain consistent interpretation across live, backtest, and rebalance.
