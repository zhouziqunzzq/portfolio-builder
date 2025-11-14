## Portfolio Builder

Lightweight toolkit to build and test a sector-rotating equity portfolio using local OHLCV caching, a static S&P 500 universe, signal calculations, and a sector weighting engine.

### Components

- `market_data_store.py` — Local parquet cache for OHLCV via yfinance. Fetches missing ranges and keeps metadata.
- `universe_manager.py` — Loads a universe from CSV and ensures data coverage; can build price matrices.
- `signal_engine.py` — Computes momentum/volatility and aggregate stock/sector scores.
- `sector_weight_engine.py` — Converts sector scores into bounded, smoothed portfolio weights with a trend filter.
- `get_sp500.py` — Scrapes the S&P 500 constituents from Wikipedia, normalizes tickers for yfinance, and writes a CSV.

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

### 2) Fetch S&P 500 universe

```bash
python get_sp500.py
```

This script:

- Requests the Wikipedia S&P 500 list with a browser-like User-Agent to avoid HTTP 403.
- Parses the constituents table using `pandas.read_html`.
- Normalizes tickers for yfinance by replacing dots with dashes (e.g., `BRK.B` → `BRK-B`, `BF.B` → `BF-B`).
- Writes the CSV. Recommended output path: `data/universe/sp500_constituents.csv`.

If you see a 403 Forbidden during scraping, ensure `get_sp500.py` sets headers with a realistic `User-Agent` and `Accept-Language`.

### 3) Warm the cache and inspect a price matrix

```bash
python test_universe_manager.py
```

This loads the universe CSV, ensures the local OHLCV cache for a backtest window, and prints a sample price matrix. First run will be slower as it downloads data and writes parquet files.

### 4) Signals and sector weights

```bash
python test_signal_engine.py
python test_sector_weight_engine.py
```

The sector weight script prints weights and (optionally) plots a stacked area chart of sector allocations over time.

## Ticker normalization for yfinance

Wikipedia uses dot notation for share classes (e.g., `BRK.B`, `BF.B`), while yfinance expects dashes (`BRK-B`, `BF-B`). The `get_sp500.py` script normalizes via:

- Uppercase
- Replace `.` with `-`

If you encounter additional symbol quirks, extend the normalization function in `get_sp500.py`.

## Troubleshooting

- HTTP 403 from Wikipedia: Use `requests.get(url, headers={...})` with a modern desktop browser User-Agent and an `Accept-Language` header, then pass `resp.text` into `pandas.read_html`.
- "Possibly delisted; no price data found": Ensure the ticker was normalized (dot → dash) and that it exists on Yahoo Finance for your date range.
- Parquet errors: Install `pyarrow` (or `fastparquet`).
- MultiIndex columns from yfinance: The code handles this, but if you customize downloads, flatten MultiIndex columns to price-only names.

## Notes

- This project favors a simple, file-based cache for repeatable backtests.
- Beware of API limits or short-term rate limiting from Yahoo Finance; stagger initial downloads if needed.

