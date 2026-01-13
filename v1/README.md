v1 — Imperative runners
========================

What is v1
----------
- Production-style, imperative runners and helpers for backtests and live workflows.
- Stable, procedural codepath used for runner-style execution (vs the vectorized v2).

Strategy
--------
- Momentum-based cross-sectional strategy: stocks are ranked by momentum/relative-strength signals and aggregated into sector-level views. The pipeline applies sector weighting and a trend filter to determine sleeve/sector allocations, then allocates sector weights down to individual stocks. The approach focuses on cross-sectional momentum with explicit cash preservation and friction-control steps in allocation.

Architecture (high level)
-------------------------
- Data sources: universe CSVs under `v1/data/` or repo root.
- Market data: `v1/src/market_data_store.py` reads/writes a parquet price cache (price cache lives under `data/ohlcv/1d` in this codebase).
- Pipeline: universe → market data store → `SignalEngine` → sector/stock weight engines → `PortfolioBacktester` / live runner.
- Key modules: `v1/src/backtest_runner.py`, `v1/src/live_runner.py`, `v1/src/signal_engine.py`, `v1/src/sector_weight_engine.py`, `v1/src/portfolio_backtester.py`, `v1/src/universe_manager.py`, `v1/src/stock_allocator.py`.

Quick prerequisites
-------------------
- Create and activate a virtualenv and install dependencies (from the repo `requirements.txt`).

Backtest (example)
-------------------
Run the v1 backtest runner with date range flags. Example:

```bash
python v1/run_backtest.py --backtest-start 2015-01-01 --backtest-end 2024-12-31
```

Common backtest flags
- `--backtest-start` and `--backtest-end`: date range for the backtest
- check `v1/run_backtest.py` for additional options

Live runner (example)
----------------------
The live runner supports cache updating and stepwise actions. Typical workflow:

1. Update price cache:

```bash
python v1/run_live.py --update-prices
```

2. Compute signals / sector weights / rebalance as needed:

```bash
python v1/run_live.py --compute-signals
python v1/run_live.py --compute-sector-weights
python v1/run_live.py --rebalance
```

Rebalance helper (`run_rebalance.py`)
------------------------------------
`v1/run_rebalance.py` is a small interactive helper (wraps `v1/src/rebalance_helper.py`) that:

- Loads the latest computed stock/sector weights from the configured `output_root_path/weights` directory.
- Prints sector and stock allocation summaries for a chosen frequency (`monthly` or `daily`) and an "as-of" date (defaults to latest).
- Prompts the operator to type their current positions (ticker and dollar amount per line) and current cash.
- Computes dollar targets from the weights, reports buys/sells (dollar amounts), projected end cash, and any cash gap relative to the target cash allocation.

Usage example:

```bash
python v1/run_rebalance.py --frequency monthly --as-of latest --top 20
```

Important flags
- `--strategy`: path to `strategy.yml` (defaults to the repo `v1/config/strategy.yml`).
- `--frequency`: `monthly` or `daily` weights to use.
- `--as-of`: date (YYYY-MM-DD) or `latest` to pick the most recent weight date <= the given date.
- `--top`: how many top stocks to display in the summary.

Notes
- This helper only prints a suggested execution plan (dollar amounts). It does not place orders; use your broker tooling to execute.
- If no weights are present, compute weights first via the weight-generation step in the pipeline.

Configuration
-------------
- Strategy and runner configuration live under `v1/config/strategy.yml` and `v1/config/sectors.yml`.
- The runners read those files to determine universe, rebalancing rules, and output roots.

Notes & tips
------------
- v1 is intentionally imperative and simple to operate in production runners; for faster, vectorized research and testing use `v2/`.
- To debug data mismatches, compare v1 vs v2 outputs using the repo's `compare.py` helper.

See also
--------
- Root README for links to both `v1` and `v2` documentation.

