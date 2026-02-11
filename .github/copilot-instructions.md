# Copilot / AI Agent Instructions — portfolio-builder

Purpose: help an AI agent be productive in this repo with two parallel code lines.

- Big picture: sector-rotating, momentum-driven portfolio pipeline with two separate codelines: `v1/` (imperative runners) and `v2/` (vectorized, sleeve-based allocator). Do not mix imports across versions.
- Core data flow: Universe CSV → MarketDataStore (parquet cache) → SignalEngine(s) → Sleeves → MultiSleeveAllocator → PortfolioBacktester. v1 cache under `data/ohlcv/1d`, v2 cache under `v2/data/prices`.
- Event-driven runtime (v2): IML (market data) → AT (auto trader) → EML (execution), wired by `v2/src/runtime_manager.py` and launched via `v2/run_app.py`.

Key entrypoints and workflows
- Activate the virtualenv before running any Python or tests: `source .venv/bin/activate` from repo root.
- The repo expects the root `.venv` to be active; runners and tests assume its deps are installed.
- v1 live steps: `python v1/run_live.py --update-prices` then `--compute-signals` / `--compute-sector-weights` / `--rebalance`.
- v1 backtest: `python v1/run_backtest.py --backtest-start 2015-01-01 --backtest-end 2024-12-31`.
- v2 backtest (vectorized): `python v2/run_backtest.py --start 2018-01-01 --end 2020-12-31 --sample-frequency monthly --local-only`.
- v2 tests: `pytest v2/tests -q` (use `--local-only` or fixtures to avoid network).

Project-specific conventions
- v2 runners insert `v2/src` on `sys.path` (see `v2/src/backtest_runner.py`); v1 modules are self-contained.
- Tickers must be uppercased and `.` replaced with `-` for `yfinance` (e.g. `BRK.B` → `BRK-B`).
- v2 backtests use `signal_delay_days` and `sample_frequency` to avoid lookahead; precompute can be skipped via `--skip-precompute`.

Common library (`src/algotrading/lib`)
- Shared primitives used by v2 (and future v3) live workflows: eventing, runtime services, state persistence, types, and market-data utilities.
- Tests for the shared lib live under root `tests/` (not `v2/tests`).
- Eventing: `EventBus` fan-out with per-subscriber queues in [src/algotrading/lib/eventing/event_bus.py](src/algotrading/lib/eventing/event_bus.py); events are dataclasses inheriting `BaseEvent` with fixed `Topic` in `*.events` files (see [src/algotrading/lib/eventing/md_events.py](src/algotrading/lib/eventing/md_events.py) and [src/algotrading/lib/eventing/v2_events.py](src/algotrading/lib/eventing/v2_events.py)).
- Services: derive from `BaseService` ([src/algotrading/lib/runtime/base_service.py](src/algotrading/lib/runtime/base_service.py)) to get the standard run loop, STOP handling, and threadpool helpers; override `subscription_topics`, `_run_loop()`, and `_handle_event()`.
- State: persistent runtime state must implement `BaseState` ([src/algotrading/lib/state/base_state.py](src/algotrading/lib/state/base_state.py)); use `FileStateStore` for JSON persistence ([src/algotrading/lib/state/file_store.py](src/algotrading/lib/state/file_store.py)) and `BaseStateManager` for load/save/reset semantics ([src/algotrading/lib/state/manager.py](src/algotrading/lib/state/manager.py)).
- Types/utilities: `InstrumentRef`, `Timeframe`, and `OHLCVBar` in [src/algotrading/lib/types](src/algotrading/lib/types) are the canonical market-data types; time bucketing helpers live in [src/algotrading/lib/market_data/bucketing.py](src/algotrading/lib/market_data/bucketing.py).

Where to look when changing behavior
- Runners/docs: [README.md](README.md), [v1/README.md](v1/README.md), [v2/README.md](v2/README.md).
- v1 pipeline: [v1/src/market_data_store.py](v1/src/market_data_store.py), [v1/src/signal_engine.py](v1/src/signal_engine.py), [v1/src/portfolio_backtester.py](v1/src/portfolio_backtester.py).
- v2 pipeline: [v2/src/backtest_runner.py](v2/src/backtest_runner.py), [v2/src/allocator/multi_sleeve_allocator.py](v2/src/allocator/multi_sleeve_allocator.py), [v2/src/portfolio_backtester.py](v2/src/portfolio_backtester.py).
- Shared libs for event-driven work: [src/algotrading/lib/eventing](src/algotrading/lib/eventing) and [src/algotrading/lib/alpha](src/algotrading/lib/alpha).

Integration points and external deps
- Market data fetch uses `yfinance` with parquet caches (`pyarrow`/`fastparquet`).
- Config files: `v1/config/strategy.yml`, `v2/configs/app.yml` control universes and outputs.
- Observability: `docker compose -f v2/docker-compose.obs.yml up -d` starts Grafana/Prometheus/otelcol.

Do not change without confirmation
- Cross-imports between `v1/` and `v2/`.
- Cache layout or YAML schema under `v1/config` or `v2/configs`.
