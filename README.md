## Portfolio Builder — Overview

Portfolio Builder is a lightweight toolkit to build, backtest, and operate a sector-rotating, momentum-driven equity portfolio using local price caches, modular signal engines, and sleeve-based allocation.

Two parallel codelines live in this repository. Pick the one that fits your workflow:

- `v1/` — Production-style, imperative runners and interactive helpers (backtests, live runners, rebalance helper).
- `v2/` — Refactored, vectorized research + live-ready auto-trader (sleeve-based allocator, IML/AT/EML, OpenTelemetry observability).

Quick start

- Create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r v2/requirements.txt  # or root requirements.txt for v1
```

- Run a simple v1 backtest:

```bash
python3 v1/run_backtest.py --backtest-start 2015-01-01 --backtest-end 2024-12-31
```

- Run a v2 vectorized backtest:

```bash
python v2/run_backtest.py --start 2018-01-01 --end 2020-12-31 --sample-frequency monthly --local-only
```

- Start the v2 observability stack locally:

```bash
docker compose -f v2/docker-compose.obs.yml up -d
```

Where to read more

- See `v1/README.md` for full v1 usage and CLI flags.
- See `v2/README.md` for v2 architecture, precompute behavior, live-run deployment, and observability.

Data & notes

- Price cache and output paths are configured per-version in their respective configs (`v1/config/strategy.yml`, `v2/config/app.yml`).
- Ticker normalization: uppercase tickers and replace `.` with `-` for yfinance compatibility (e.g., `BRK.B` → `BRK-B`).
