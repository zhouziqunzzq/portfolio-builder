# Copilot / AI Agent Instructions — portfolio-builder

Purpose: give an AI coding agent the minimal, actionable knowledge to be productive
in this repository (where there are two active code lines: `v1/` and `v2/`).

- **Big picture**
  - This repo implements a sector-rotating portfolio pipeline with two largely
    separate codelines: `v1/` (production, imperative runners) and `v2/` (refactored,
    vectorized, sleeve-based allocator). Do not mix imports across versions.
  - Data flow: Universe CSV → MarketDataStore (parquet cache) → SignalEngine(s)
    → Sleeves (trend/defensive/sideways) → MultiSleeveAllocator → PortfolioBacktester.
  - Caching: price cache lives under `data/prices` (v2) or `data/ohlcv/1d` (v1).

- **Key entrypoints & common commands** (examples you can run locally)
  - Warm cache / live pipeline (v1): `python v1/run_live.py --update-prices` then
    `--compute-signals` / `--compute-sector-weights` / `--rebalance` as needed.
  - Backtest (v1): `python v1/run_backtest.py --backtest-start 2015-01-01 --backtest-end 2024-12-31`
  - Backtest (v2, vectorized): `python v2/src/backtest_runner.py --start 2015-01-01 --end 2024-12-31 --sample-frequency monthly`
  - Tests (v2): run `pytest v2/tests -q` from repo root (use virtualenv with dependencies).

- **Project-specific conventions & patterns**
  - Two parallel APIs: `v1/src/*` are self-contained modules; `v2/src/*` expects
    `v2/src` on `sys.path` (runners insert the path). See `v2/src/backtest_runner.py`.
  - Symbol normalization: universe CSV tickers must be uppercased and dots converted
    to dashes (`BRK.B` → `BRK-B`) for `yfinance` compatibility. See README examples.
  - Backtests treat weights as timestamped on execution day `t`; signals use data up
    to `t-1` (lookahead avoidance). v2 has explicit `signal_delay_days` and
    `sample_frequency` knobs in `v2/src/backtest_runner.py`.
  - Precompute: v2 supports sleeve precompute to speed vectorized backtests; caller
    can skip with `--skip-precompute`.

- **Important files to inspect when changing behavior**
  - High level runner + docs: [README.md](README.md)
  - v1 production runners: [v1/run_live.py](v1/run_live.py), [v1/run_backtest.py](v1/run_backtest.py)
  - v2 orchestrator: [v2/src/backtest_runner.py](v2/src/backtest_runner.py)
  - Market/cache abstractions: `v1/src/market_data_store.py` and `v2/src/market_data_store.py`
  - Allocator & sleeves (v2): [v2/src/allocator/multi_sleeve_allocator.py](v2/src/allocator/multi_sleeve_allocator.py) and `v2/src/sleeves`
  - Backtester: [v2/src/portfolio_backtester.py](v2/src/portfolio_backtester.py)

- **Testing & debug tips**
  - Unit tests in `v2/tests/` instantiate `UniverseManager`, `MarketDataStore`, and
    `SignalEngine`. Use the `--local-only` or create minimal synthetic price matrices
    to avoid external network calls in CI. See `v2/tests/*` for fixtures.
  - To debug data mismatches, compare v1 vs v2 with `compare.py` which instantiates
    both versions' `MarketDataStore` and `UniverseManager`.

- **Integration points / external dependencies**
  - Data fetch: `yfinance` (or local parquet); ensure `pyarrow` or `fastparquet` installed.
  - Universe CSVs live under `v1/data/` or `data/` (`v2` expects `data/sp500_membership.csv`).
  - Config: `v1/config/strategy.yml` and `v2/config/strategy.yml` control output roots.

- **What AI agents should not change without confirmation**
  - Do not refactor across `v1`/`v2` boundaries (mixing imports). Changes that alter
    the structure of the price cache or the `strategy.yml` layout require human review.

- **Quick examples (copyable)**
  - Install environment:

    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install pandas yfinance pyarrow matplotlib pytest

  - Run a quick v2 backtest (vectorized):

    python v2/src/backtest_runner.py --start 2018-01-01 --end 2020-12-31 --sample-frequency monthly --local-only

If anything is missing or you want more detail for a particular area (e.g. sleeve
implementation patterns, friction-control rules, or config fields in `strategy.yml`),
tell me which part and I'll expand the instructions.
