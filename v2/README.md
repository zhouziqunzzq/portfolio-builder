v2 — Vectorized, sleeve-based allocator
======================================

What is v2
----------
- A refactored, vectorized implementation focused on fast backtests and research.
- Uses a sleeve-based allocator (multiple sleeves such as `trend` and `defensive`) and a `MultiSleeveAllocator` to compose global targets.

Strategy
--------
- Regime-aware, vectorized momentum strategy: v2 implements a cross-sectional momentum approach split into modular "sleeves" (e.g. `trend`, `defensive`, `sideways_base`). A `RegimeEngine` produces regime labels and scores; the `MultiSleeveAllocator` blends per-regime sleeve allocations and asks each sleeve for vectorized date×ticker weight matrices (via `precompute`) or on-the-fly signals. The pipeline supports cash preservation, friction control, and optional signal delay to simulate real-world latency.

Architecture (high level)
-------------------------
- Config: `v2/config/app.yml` (and related YAMLs) control runtimes, universes, and outputs.
- Runtime manager: `v2/src/runtime_manager.py` wires singletons (UniverseManager, MarketDataStore, SignalEngine, RegimeEngine, sleeves, allocator).
- Event-driven runtime: v2 runs as an event-driven application composed of three logical layers:
	- IML (Information Market Link): sources market clock / bar events and publishes them to the event bus (`v2/src/iml/`).
	- AT (Auto Trader): subscribes to events and generates rebalance/cleanup plan requests (`v2/src/at/`).
	- EML (Execution Market Link): broker-facing execution service that translates plan requests into orders and publishes account/position snapshots (`v2/src/eml/`).
- Market data: `v2/src/market_data_store.py` with parquet price caches under `v2/data/prices` (local cache + optional live data dirs).
- Sleeves: modular sleeve implementations under `v2/src/sleeves/` (e.g. `trend`, `defensive`, `sideways_base`).
- Allocator: `v2/src/allocator/multi_sleeve_allocator.py` composes sleeve outputs into global target weights.
- Backtester: `v2/src/backtest_runner.py` (CLI) and `v2/src/portfolio_backtester.py` perform vectorized backtests and plotting.

Quick prerequisites
-------------------
- Create and activate a virtualenv and install v2 dependencies (see `v2/requirements.txt`).

Backtest (examples)
--------------------
Run the v2 backtest runner (wrapper at `v2/run_backtest.py`) with required start/end dates. Example:

```bash
python v2/run_backtest.py --start 2018-01-01 --end 2020-12-31 --sample-frequency monthly --local-only
```

Common backtest knobs
- `--sample-frequency`: how often the allocator runs (monthly, weekly, bi-weekly, semi-monthly, daily, ...).
- `--backtest-mode`: `vectorized` (fast) or `iterative` (day-by-day).
- `--skip-precompute`: skip sleeves precompute step for faster startup if already cached.
- `--local-only`: use only local caches (no network fetches).
- `--signal-delay-days`: simulate delayed signals by shifting the as-of date.
- `--force-live-mode`: force sleeves to run in live mode (danger: introduces lookahead bias).
- Cost and risk-free knobs: `--initial-equity`, `--cost-per-turnover`, `--bid-ask-bps-per-side`, `--rf-annual`.
- Plot controls: `--plot-all`, `--plot-equity`, `--plot-drawdown`, `--show`, etc.

Sleeve precompute
-----------------
- `precompute` invokes the allocator/sleeve *vectorized* codepath to compute date × ticker weight matrices for each sleeve over a lookback window. The runner calls `MultiSleeveAllocator.precompute(start, end)` which asks each sleeve to return a fully vectorized weight matrix (Date × Ticker). These precomputed matrices are then used when generating global target weights so the vectorized signal path is exercised instead of computing signals on-the-fly via non-vectorized methods.
- `--skip-precompute` skips this precompute step. Skipping is useful when precomputed outputs already exist or when you deliberately want sleeves to compute signals on-the-fly (non-vectorized). Omitting `--skip-precompute` ensures the vectorized path is used for consistency and performance in large backtests.

Testing
-------
- Unit tests live under `v2/tests/`. Run with:

```bash
pytest v2/tests -q
```

Notes & tips
------------
- Use the same `AppConfig` (YAML) across backtest and live runs to ensure consistent behavior.
-- For fast experiments, run in `vectorized` mode with `--local-only`. Keep precompute enabled (do not pass `--skip-precompute`) to use the vectorized signal path; use `--skip-precompute` only when cached precompute outputs already exist or you intentionally want on-the-fly computation.
- To compare outputs across versions, use the repo `compare.py` helper.

See also
--------
- Root README for high-level links and next steps.

Live-ready components (IML / AT / EML)
------------------------------------
- Information Market Link (IML): sources market time and bar events, polling or streaming market data and publishing `MarketClockEvent` / `NewBarsEvent` to the event bus. Implementations live under `v2/src/iml/` (e.g. `alpaca_polling_iml.py`).
- Auto Trader (AT): event-driven trader that subscribes to rebalance requests and position-cleanup intents; AT implementations live under `v2/src/at/` and emit execution plan requests to the EML.
- Execution Market Link (EML): broker-facing execution component that polls the broker, publishes account snapshots, translates rebalance/cleanup plan requests into broker orders, and records execution state. See `v2/src/eml/portfolio_eml.py` for an implementation instrumented with OpenTelemetry metrics.

Trading adapters
----------------
- Broker adapters live under `v2/src/trading_api/` and include adapters for Alpaca and Public.com. The EML selects the adapter via configuration.

Observability & deployment
--------------------------
- v2 is instrumented with OpenTelemetry. The EML and other services expose metrics that can be scraped by an OpenTelemetry collector / Prometheus.
- The repository includes a local observability stack: `docker-compose.obs.yml` to run an `otelcol` collector, `prometheus`, `grafana`, and `alertmanager` for local debugging and alerting.

Quick start: observability stack

```bash
# From repo root
docker compose -f v2/docker-compose.obs.yml up -d
# Visit Grafana: http://127.0.0.1:3000, Prometheus: http://127.0.0.1:9090
```

Live runner / deployment
------------------------
- The live application runner is `v2/run_app.py` (entrypoint to `v2/src/app_runner.py`). It constructs an `App` using `AppConfig` (YAML) and `RuntimeManager` singletons and runs all services (IML, AT, EML, state managers, event bus).
- Example local live run using the Alpaca config:

```bash
python v2/run_app.py --config v2/config/app_live_alpaca.yml
# or use the helper script
./v2/run_app_live_alpaca.sh
```

Docker-compose deployment examples
---------------------------------
- The repository includes example compose files for live setups (e.g. `v2/docker-compose.live_alpaca.yml`, `v2/docker-compose.live_publicdotcom.yml`) which wire the app with sidecars and observability when deploying to a host.

Testing & CI
-----------
- Unit tests are under `v2/tests/`. Run them locally with `pytest v2/tests -q`.
- For CI-friendly runs, use `--local-only` or provide minimal synthetic data via test fixtures to avoid external network calls.

Notes & tips
-----------
- Use `v2/src/runtime_manager.py` to construct the same runtime singletons used by the backtest and live runners; this ensures parity between research and production runs.
- Metrics: EML exposes detailed metrics (order fills, pending rebalances, account gauges) via OpenTelemetry; configure your collector to export to Prometheus and Grafana.
- Safety: EML includes startup/shutdown safety hooks (e.g., cancel open orders) and retry semantics — review `v2/src/eml/portfolio_eml.py` before enabling live trading.

