from __future__ import annotations

import asyncio
from datetime import datetime
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from decimal import Decimal

from opentelemetry import metrics
from opentelemetry.metrics import Observation


from pathlib import Path
import sys

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.event_bus import EventBus
from events.events import (
    AccountSnapshotEvent,
    BrokerAccount,
    BrokerPosition,
    PositionCleanupIntent,
    PositionCleanupPlanRequestEvent,
    PositionCleanupPlanConfirmationEvent,
    RebalancePlanRequestEvent,
    RebalancePlanConfirmationEvent,
)
from runtime_manager import RuntimeManager

from .base_eml import BaseEML
from .config import EMLConfig
from .state import EMLState
from utils.decimals import to_decimal

from models.trading import (
    InstrumentRef,
    OrderFilter,
    OrderIntent,
    OrderSide,
    OrderStatus,
)
from trading_api.base import BaseTradingAPI
from trading_api.alpaca import AlpacaTradingAPI
from trading_api.exceptions import OrderNotFoundYet


class EMLShutdownRequested(Exception):
    """Raised internally to abort blocking execution during shutdown."""

    pass


class PortfolioEMLService(BaseEML):
    """Portfolio Execution Market Link (EML).

    Current scope:
    - Periodically fetch account + (optionally) positions from the configured broker
    - Publish an `AccountSnapshotEvent` to the event bus
    - Translate `RebalancePlanRequestEvent` into broker orders
    - Translate `PositionCleanupPlanRequestEvent` into broker orders
    - Publish order/fill updates to the bus
    """

    def __init__(
        self,
        bus: "EventBus",
        rm: Optional["RuntimeManager"] = None,
        *,
        config: Optional[EMLConfig] = None,
        trading_api: Optional[BaseTradingAPI] = None,
        name: str = "PortfolioEML",
    ):
        super().__init__(bus=bus, name=name)

        if config is None:
            self.log.warning("No EMLConfig provided; using default configuration")
            config = EMLConfig()
        config.validate()
        self.config = config

        if self.config.polling_interval_secs <= 0:
            raise ValueError("polling_interval_secs must be > 0")
        self._poll_interval_seconds = float(self.config.polling_interval_secs)

        self.rm = rm
        if self.rm is not None:
            # Register self to RuntimeManager for lifecycle management.
            self.rm.set("eml", self)
            self.rm.set("portfolio_eml", self)

        if trading_api is not None:
            self._trading_api = trading_api
        else:
            broker = str(self.config.broker or "alpaca").strip().lower()
            if broker != "alpaca":
                raise ValueError(
                    f"Unsupported EMLConfig.broker={self.config.broker!r}; only 'alpaca' is supported"
                )
            # Broker adapter resolves env vars/defaults.
            self._trading_api = AlpacaTradingAPI(name="AlpacaTradingAPI")

        self.log.info(
            "Initialized EML (broker=%s)",
            str(self.config.broker).strip().lower(),
        )

        # State
        # This will be managed by StateManager externally.
        self.state = EMLState()

        # Latest snapshots for metrics
        self._last_account: Optional[BrokerAccount] = None
        self._last_positions: List[BrokerPosition] = []

        self._init_metrics_instruments()

    def _init_metrics_instruments(self) -> None:
        meter = metrics.get_meter("portfolio_builder.v2.eml")

        self._heartbeat_counter = meter.create_counter(
            name="eml.loop_heartbeat",
            description=(
                "Heartbeat counter incremented once per EML background loop iteration"
            ),
        )

        self._loop_duration_hist = meter.create_histogram(
            name="eml.loop_duration_seconds",
            description="Wall-clock duration of each EML background loop iteration",
            unit="s",
        )

        self._rebalance_executions_counter = meter.create_counter(
            name="eml.rebalance_executions_total",
            description="Counts rebalance execution outcomes",
        )

        self._rebalance_execution_errors_counter = meter.create_counter(
            name="eml.rebalance_execution_errors",
            description="Counts errors encountered while executing rebalance plans",
        )

        self._executed_rebalance_count_counter = meter.create_counter(
            name="eml.executed_rebalance_count",
            description="Counts rebalance plans marked executed in EML state",
        )

        self._order_fills_counter = meter.create_counter(
            name="eml.order_fills_total",
            description="Count of broker orders filled during EML execution",
        )

        self._order_fill_latency_hist = meter.create_histogram(
            name="eml.order_fill_latency_seconds",
            description="Latency waiting for broker orders to fill",
            unit="s",
        )

        def _obs_account_value(field: str) -> list[Observation]:
            acct = self._last_account
            if acct is None:
                return [Observation(0.0, {"known": False})]
            v = getattr(acct, field, None)
            if v is None:
                return [Observation(0.0, {"known": False})]
            try:
                return [Observation(float(v), {"known": True})]
            except Exception:
                return [Observation(0.0, {"known": False})]

        meter.create_observable_gauge(
            name="eml.account_equity",
            description="Latest broker account equity",
            callbacks=[lambda _: _obs_account_value("equity")],
        )
        meter.create_observable_gauge(
            name="eml.account_portfolio_value",
            description="Latest broker account portfolio value",
            callbacks=[lambda _: _obs_account_value("portfolio_value")],
        )
        meter.create_observable_gauge(
            name="eml.account_cash",
            description="Latest broker account cash balance",
            callbacks=[lambda _: _obs_account_value("cash")],
        )

        def _obs_positions(field: str) -> list[Observation]:
            out: list[Observation] = []
            for p in list(self._last_positions or []):
                sym = str(getattr(p, "symbol", "") or "").strip().upper()
                if not sym:
                    continue
                v = getattr(p, field, None)
                if v is None:
                    continue
                try:
                    out.append(Observation(float(v), {"symbol": sym}))
                except Exception:
                    continue
            return out

        meter.create_observable_gauge(
            name="eml.position_qty",
            description="Latest broker position quantity per symbol",
            callbacks=[lambda _: _obs_positions("qty")],
        )
        meter.create_observable_gauge(
            name="eml.position_market_value",
            description="Latest broker position market value per symbol",
            callbacks=[lambda _: _obs_positions("market_value")],
        )
        meter.create_observable_gauge(
            name="eml.position_unrealized_pnl",
            description="Latest broker position unrealized P&L per symbol",
            callbacks=[lambda _: _obs_positions("unrealized_pnl")],
        )

        def _obs_pending_rebalances(_: object) -> list[Observation]:
            pending = getattr(self.state, "pending_rebalance_requests", None)
            if isinstance(pending, Mapping):
                return [Observation(float(len(pending)))]
            return [Observation(0.0)]

        meter.create_observable_gauge(
            name="eml.pending_rebalance_plans",
            description="Number of pending rebalance plans recorded in EML state",
            callbacks=[_obs_pending_rebalances],
        )

        def _obs_failed_rebalances(_: object) -> list[Observation]:
            failed = getattr(self.state, "failed_rebalance_requests", None)
            if isinstance(failed, list):
                return [Observation(float(len(failed)))]
            return [Observation(0.0)]

        meter.create_observable_gauge(
            name="eml.failed_rebalance_requests",
            description="Number of failed rebalance requests recorded in EML state",
            callbacks=[_obs_failed_rebalances],
        )

        def _obs_pending_execution_retries(_: object) -> list[Observation]:
            pending = getattr(self.state, "pending_rebalance_requests", None)
            if not isinstance(pending, Mapping) or not pending:
                return [Observation(0.0)]

            max_failures = 0
            for payload in dict(pending).values():
                if not isinstance(payload, Mapping):
                    continue
                v = payload.get("execution_failures", 0)
                try:
                    max_failures = max(max_failures, int(v))
                except Exception:
                    continue
            return [Observation(float(max_failures))]

        meter.create_observable_gauge(
            name="eml.pending_execution_retries",
            description="Maximum execution retry count among pending rebalance plans",
            callbacks=[_obs_pending_execution_retries],
        )

        # --- Position cleanup metrics ---
        def _obs_pending_position_cleanup(_: object) -> list[Observation]:
            pending = getattr(self.state, "pending_position_cleanup_requests", None)
            if isinstance(pending, Mapping):
                return [Observation(float(len(pending)))]
            return [Observation(0.0)]

        meter.create_observable_gauge(
            name="eml.pending_position_cleanup_plans",
            description="Number of pending position cleanup plans recorded in EML state",
            callbacks=[_obs_pending_position_cleanup],
        )

        def _obs_failed_position_cleanup(_: object) -> list[Observation]:
            failed = getattr(self.state, "failed_position_cleanup_requests", None)
            if isinstance(failed, list):
                return [Observation(float(len(failed)))]
            return [Observation(0.0)]

        meter.create_observable_gauge(
            name="eml.failed_position_cleanup_requests",
            description="Number of failed position cleanup requests recorded in EML state",
            callbacks=[_obs_failed_position_cleanup],
        )

        self._position_cleanup_skips_counter = meter.create_counter(
            name="eml.position_cleanup_skips_total",
            description=(
                "Counts position cleanup symbol-level skips (label: reason). "
                "Reasons include: no_position, already_flat, short_detected."
            ),
            unit="1",
        )

        self._position_cleanup_safety_refusals_counter = meter.create_counter(
            name="eml.position_cleanup_safety_refusals_total",
            description=(
                "Counts times position cleanup execution was refused due to safety checks (label: reason)"
            ),
            unit="1",
        )

        self._orders_submitted_counter = meter.create_counter(
            name="eml.orders_submitted",
            description="Count of broker orders submitted by EML",
        )

    async def _on_startup(self) -> None:
        await super()._on_startup()

        if not getattr(self.config, "cancel_open_orders_on_startup", True):
            self.log.info("Startup open-order cancel disabled by config")
            return

        # Best-effort safety cleanup. This is intentionally not fatal.
        try:
            await self._run_in_thread(self._cancel_all_open_orders)
        except Exception:
            self.log.exception("Failed to cancel open orders on startup (continuing)")

    async def _on_shutdown_requested(self) -> None:
        # Best-effort safety cleanup on shutdown. This is intentionally not fatal.
        if getattr(self.config, "cancel_open_orders_on_shutdown", False):
            try:
                await self._run_in_thread(self._cancel_all_open_orders)
            except Exception:
                self.log.exception(
                    "Failed to cancel open orders on shutdown (continuing)"
                )

        await super()._on_shutdown_requested()

    async def _run_loop(self) -> None:
        while self._running:
            iteration_start = time.monotonic()
            iteration_success = True
            iteration_cancelled = False
            try:
                failed_rebal = self.state.failed_rebalance_requests
                if isinstance(failed_rebal, list) and failed_rebal:
                    self.log.warning(
                        "EML has %d failed rebalance request(s); manual intervention may be required",
                        len(failed_rebal),
                    )

                failed_pc = self.state.failed_position_cleanup_requests
                if isinstance(failed_pc, list) and failed_pc:
                    self.log.warning(
                        "EML has %d failed position cleanup request(s); manual intervention may be required",
                        len(failed_pc),
                    )

                # Fetch account + positions
                account = await self._run_in_thread(self._get_account)
                positions: List[BrokerPosition] = []
                if self.config.include_positions:
                    positions = await self._run_in_thread(self._list_positions)

                # Update latest snapshots for metrics.
                self._last_account = account
                self._last_positions = list(positions or [])

                event = AccountSnapshotEvent(
                    ts=time.time(),
                    source=self.name,
                    account=account,
                    positions=positions,
                )
                self.log.debug(
                    "Fetched account snapshot: account=%s positions=%s",
                    account,
                    positions,
                )
                await self.emit_account_event(event)

                # Execute any pending rebalance plans (blocking per plan, off the event loop thread)
                await self._run_in_thread(self._execute_pending_rebalance_plans)

                # Execute any pending position cleanup plans (blocking per plan, off the event loop thread)
                await self._run_in_thread(self._execute_pending_position_cleanup_plans)

                # GC execution history (best-effort; keep state from growing unbounded)
                self._gc_execution_history()

                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                iteration_cancelled = True
                raise
            except Exception:
                iteration_success = False
                # Keep running; transient API/network issues are expected.
                self.log.exception("Error in PortfolioEMLService main loop")
                await asyncio.sleep(min(30.0, max(1.0, self._poll_interval_seconds)))
            finally:
                # Count one heartbeat per finished iteration, labeled by success.
                if self._running and not iteration_cancelled:
                    self._heartbeat_counter.add(1, {"success": iteration_success})
                    self._loop_duration_hist.record(
                        max(0.0, float(time.monotonic() - iteration_start)),
                        {"success": iteration_success},
                    )

    def _cancel_all_open_orders(self) -> None:
        """Cancel all currently-open orders at the broker (best-effort)."""
        openish = {
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        }

        # Broker-agnostic: ask adapter for open-ish orders, then cancel each.
        try:
            orders = self._trading_api.list_orders(
                OrderFilter(statuses=frozenset(openish))
            )
        except Exception:
            self.log.exception("Failed listing orders for cancel-all")
            return

        open_ids = [o.broker_order_id for o in orders if o.broker_order_id]

        if not open_ids:
            self.log.info("No open orders found to cancel")
            return

        self.log.info("Canceling %d open order(s)...", len(open_ids))
        for oid in open_ids:
            try:
                self._trading_api.cancel_order(oid)
            except Exception:
                self.log.exception("Failed canceling open order: order_id=%s", oid)

    def _list_open_orders_best_effort(self, *, limit: int = 100) -> List[Any]:
        """Best-effort retrieval of open orders."""
        # Deprecated: EML now uses trading_api.list_orders().
        try:
            orders = self._trading_api.list_orders(
                OrderFilter(
                    statuses=frozenset(
                        {
                            OrderStatus.NEW,
                            OrderStatus.ACCEPTED,
                            OrderStatus.OPEN,
                            OrderStatus.PARTIALLY_FILLED,
                        }
                    )
                )
            )
        except Exception:
            return []
        return orders[: int(limit)]

    async def execute_rebalance_plan(self, event: RebalancePlanRequestEvent) -> None:
        """Execute a rebalance plan request.
        This function simply records the pending rebalance request in state, and
        emits a RebalancePlanConfirmationEvent. Actual execution of the rebalance
        plan (placing orders) is handled asynchronously in the main loop. See `_run_loop()`
        and `_execute_pending_rebalance_plans()`.

        Args:
            event: RebalancePlanRequestEvent
        Returns:
            None
        """
        # Track pending rebalance requests in persisted state, and send back confirmation.
        now_ts = time.time()
        try:
            if self.state.has_pending_rebalance_request(event.rebalance_id):
                self.log.info(
                    "RebalancePlanRequestEvent already pending; ignoring duplicate: rebalance_id=%s",
                    event.rebalance_id,
                )
            else:
                self.state.remember_pending_rebalance_request(event)
            # We still send back confirmation even if duplicate.
            confirmation_event = RebalancePlanConfirmationEvent(
                ts=now_ts,
                rebalance_id=event.rebalance_id,
                confirmed_ts=now_ts,
                source=self.name,
            )
            await self.emit(confirmation_event)
            self.log.info(
                "Published RebalancePlanConfirmationEvent: rebalance_id=%s",
                event.rebalance_id,
            )
        except Exception:
            self.log.exception(
                "Failed to store pending rebalance request in state: rebalance_id=%s",
                getattr(event, "rebalance_id", None),
            )

        # Note: Actual execution of the rebalance plan (placing orders) is handled asynchronously
        # in the main loop, to avoid blocking the event handler.

    def _execute_pending_rebalance_plans(self) -> None:
        """Execute any pending rebalance requests recorded in `self.state`.

        This is a synchronous, potentially-blocking method intended to run off the
        asyncio event loop thread (e.g., via `BaseService._run_in_thread`).

        Behavior:
        - Reads pending requests from `self.state.pending_rebalance_requests`.
        - Market gating: executes only if `self._market_clock` is present and
            indicates `is_market_open is True`. If the clock is missing or the market
            is closed/unknown, the method logs and returns without side effects.
        - Deterministic ordering: processes requests oldest-first (by `request_ts`).
        - For each request:
            - Rehydrates a `RebalancePlanRequestEvent` from the stored payload.
            - Executes it via `_execute_rebalance_plan`.
            - On success, marks it executed via `self.state.mark_rebalance_executed`.

        Error handling:
        - `EMLShutdownRequested`: exits early and leaves remaining plans pending.
        - Any other exception: logs and continues to the next plan; the failed plan
            remains pending and is not marked executed.
        """
        pending = self.state.pending_rebalance_requests
        if not pending:
            self.log.debug("No pending rebalance plans to execute")
            return

        # Only execute when we are sure the market is open.
        clock = getattr(self, "_market_clock", None)
        if clock is None:
            self.log.info(
                "Skipping pending rebalance execution: market clock unknown (pending=%d)",
                len(pending),
            )
            try:
                self._rebalance_executions_counter.add(
                    len(pending),
                    {"result": "skipped_clock_unknown", "service": self.name},
                )
            except Exception:
                pass
            return

        is_open = getattr(clock, "is_market_open", None)
        if is_open is not True:
            self.log.debug(
                "Skipping pending rebalance execution: market not open (is_market_open=%s now=%s next_open=%s pending=%d)",
                is_open,
                getattr(clock, "now", None),
                getattr(clock, "next_market_open", None),
                len(pending),
            )
            try:
                self._rebalance_executions_counter.add(
                    len(pending),
                    {"result": "skipped_market_closed", "service": self.name},
                )
            except Exception:
                pass
            return

        # Process oldest-first for determinism.
        items: List[Tuple[str, Dict[str, Any]]] = []
        for rebalance_id, payload in dict(pending).items():
            if not isinstance(payload, dict):
                continue
            items.append((str(rebalance_id), dict(payload)))

        def _key(item: Tuple[str, Dict[str, Any]]) -> float:
            v = item[1].get("request_ts")
            try:
                return float(v)
            except Exception:
                return 0.0

        items.sort(key=_key)

        for rebalance_id, payload in items:
            # Another thread / loop iteration might have handled it already.
            if not self.state.has_pending_rebalance_request(rebalance_id):
                continue

            try:
                event = self._rebalance_request_from_state(payload)
                self._execute_rebalance_plan(event)
                self.state.mark_rebalance_executed(rebalance_id=rebalance_id)
                try:
                    self._rebalance_executions_counter.add(
                        1, {"result": "success", "service": self.name}
                    )
                    self._executed_rebalance_count_counter.add(
                        1, {"service": self.name}
                    )
                except Exception:
                    pass
                self.log.info(
                    "Rebalance executed successfully: rebalance_id=%s",
                    rebalance_id,
                )
            except EMLShutdownRequested:
                # Quiet exit on shutdown; leave pending state intact.
                self.log.info(
                    "Shutdown requested; aborting pending rebalance execution: rebalance_id=%s",
                    rebalance_id,
                )
                return
            except Exception:
                # Best-effort: keep processing other plans; do not mark as executed.
                self.log.exception(
                    "Failed executing pending rebalance plan: rebalance_id=%s",
                    rebalance_id,
                )
                try:
                    self._rebalance_executions_counter.add(
                        1, {"result": "error", "service": self.name}
                    )
                    self._rebalance_execution_errors_counter.add(
                        1, {"service": self.name}
                    )
                except Exception:
                    pass

                # Retry accounting + cap -> move to failed list and clear pending
                try:
                    failures = self.state.increment_pending_rebalance_execution_failure(
                        rebalance_id
                    )
                    max_retries = self.config.max_pending_rebalance_execution_retries
                    if failures >= max_retries:
                        self.state.mark_rebalance_failed(
                            rebalance_id=rebalance_id,
                            error="max retries exceeded",
                        )
                        self.log.warning(
                            "Pending rebalance marked failed after %d failed attempt(s): rebalance_id=%s",
                            failures,
                            rebalance_id,
                        )
                except Exception:
                    self.log.exception(
                        "Failed updating retry/failed state for pending rebalance: rebalance_id=%s",
                        rebalance_id,
                    )

    def _execute_rebalance_plan(self, event: RebalancePlanRequestEvent) -> None:
        """Synchronously execute a single rebalance plan by placing broker orders.

        This method is intentionally synchronous and may block while waiting for
        orders to fill. Call it from a worker thread (e.g., via `_run_in_thread`) and
        not directly on the asyncio event loop.

        High-level flow:
        1) Validate minimal event shape (`rebalance_id`, `weights`).
        2) Normalize/clean target weights (drop near-zero weights, normalize symbols).
        3) Fetch current account/positions and compute desired notional deltas.
        4) Build market orders subject to:
            - min order size (`self.config.min_order_size_notional`)
            - float-noise thresholds
            - deterministic symbol sorting
        5) Sanity check all symbols are tradable.
        6) Best-effort cancel all currently-open broker orders (pre-flight safety).
        7) Execute sells first, then buys, waiting for each order to fill.

        Shutdown/error semantics:
        - If shutdown is requested during blocking execution, helper methods raise
            `EMLShutdownRequested`, which is re-raised to allow upstream callers to
            abort without mutating pending state.
        - On any other exception, the method attempts a best-effort cancel of open
            orders (cleanup) and then re-raises.
        """
        # NOTE:
        # In this repo, v2 code often runs with `v2/src` injected onto `sys.path`.
        # Depending on how code is invoked (tests vs runners), the same dataclass
        # may be imported under different module names (e.g. `events.events` vs
        # `v2.src.events.events`), which makes `isinstance()` brittle.
        # For execution we only require a minimal event shape.
        rebalance_id = getattr(event, "rebalance_id", None)
        weights = getattr(event, "weights", None)
        if rebalance_id is None or weights is None:
            raise TypeError(
                "event must have attributes 'rebalance_id' and 'weights' (RebalancePlanRequestEvent-like)"
            )

        try:
            self.log.info(
                "Executing rebalance plan: rebalance_id=%s weights=%s",
                rebalance_id,
                weights,
            )

            # 1) Normalize target weights
            target_weights = self._normalize_target_weights(weights)
            self.log.debug(
                "Normalized target weights: rebalance_id=%s target_weights=%s",
                rebalance_id,
                target_weights,
            )

            # 2) Fetch current account + positions
            account = self._get_account()
            positions = self._list_positions() if self.config.include_positions else []

            equity = self._get_effective_equity(account)
            if equity <= 0:
                raise RuntimeError(f"Invalid account equity for execution: {equity}")
            pos_by_symbol = self._positions_by_symbol(positions)

            self.log.debug(
                "Account equity=%.2f positions=%s",
                equity,
                pos_by_symbol,
            )

            # 3) Compute desired notional deltas
            deltas = self._compute_target_deltas(
                target_weights=target_weights,
                equity=equity,
                positions_by_symbol=pos_by_symbol,
            )
            self.log.debug(
                "Computed target deltas: rebalance_id=%s deltas=%s",
                rebalance_id,
                deltas,
            )

            # 4) Build sell/buy orders subject to min-order size and float-noise thresholds
            sells, buys, dropped_by_min_size = self._build_market_orders(
                deltas=deltas,
                positions_by_symbol=pos_by_symbol,
                min_order_size_notional=float(self.config.min_order_size_notional),
            )

            if dropped_by_min_size:
                min_abs = float(self.config.min_order_size_notional)
                preview_n = 12
                preview = ", ".join(
                    [
                        (
                            f"{d.get('symbol')}:{d.get('side')}:{d.get('reason')}:"
                            f"{float(d.get('delta_value', 0.0) or 0.0):.2f}"
                        )
                        for d in dropped_by_min_size[:preview_n]
                    ]
                )
                extra = "" if len(dropped_by_min_size) <= preview_n else " ..."
                self.log.warning(
                    "Dropped %d delta(s) due to min_order_size_notional=%.2f (no orders created for these): rebalance_id=%s dropped=%s%s",
                    len(dropped_by_min_size),
                    min_abs,
                    rebalance_id,
                    preview,
                    extra,
                )

            if not sells and not buys:
                self.log.info(
                    "No executable orders after filtering; treating as executed: rebalance_id=%s",
                    rebalance_id,
                )
                return

            self.log.debug(
                "Built market orders: rebalance_id=%s sells=%s buys=%s",
                rebalance_id,
                sells,
                buys,
            )

            # 5) Sanity check tradability for all tickers in final plan
            self._assert_symbols_tradable([o["symbol"] for o in (sells + buys)])
            self.log.debug(
                "All symbols in rebalance plan are tradable: rebalance_id=%s",
                rebalance_id,
            )

            # 5b) Safety: cancel any outstanding orders before placing new ones.
            try:
                self._cancel_all_open_orders()
            except Exception:
                self.log.exception(
                    "Failed to cancel open orders before rebalance execution (continuing): rebalance_id=%s",
                    rebalance_id,
                )

            # 6) Execute sells first (cash generation), then buys; block until filled
            self._execute_orders_blocking(sells)
            self._execute_orders_blocking(buys)
        except EMLShutdownRequested:
            raise
        except Exception:
            # Best-effort cleanup: if execution fails mid-plan, try canceling any
            # potentially-open orders before returning control.
            try:
                self._cancel_all_open_orders()
            except Exception:
                self.log.exception(
                    "Failed to cancel open orders after rebalance execution error (continuing): rebalance_id=%s",
                    rebalance_id,
                )
            raise

    async def execute_position_cleanup_plan(
        self, event: PositionCleanupPlanRequestEvent
    ) -> None:
        """Record a position cleanup plan request and confirm receipt.

        This method does not place orders. It stores the request in persisted state and
        emits a `PositionCleanupPlanConfirmationEvent`. Actual broker execution happens
        asynchronously in the EML background loop via `_execute_pending_position_cleanup_plans()`.
        """

        now_ts = time.time()
        request_id = getattr(event, "request_id", None)
        if not request_id:
            self.log.warning(
                "PositionCleanupPlanRequestEvent missing request_id; ignoring: event=%s",
                event,
            )
            return

        try:
            if self.state.has_pending_position_cleanup_request(request_id):
                self.log.info(
                    "PositionCleanupPlanRequestEvent already pending; ignoring duplicate: request_id=%s",
                    request_id,
                )
            else:
                self.state.remember_pending_position_cleanup_request(event)

            confirmation_event = PositionCleanupPlanConfirmationEvent(
                ts=now_ts,
                request_id=str(request_id),
                confirmed_ts=now_ts,
                source=self.name,
            )
            await self.emit(confirmation_event)
            self.log.info(
                "Published PositionCleanupPlanConfirmationEvent: request_id=%s",
                request_id,
            )
        except Exception:
            self.log.exception(
                "Failed to store pending position cleanup request in state: request_id=%s",
                request_id,
            )

    # ------------------------------------------------------------------
    # Pending position cleanup execution
    # ------------------------------------------------------------------

    @staticmethod
    def _eastern_date_from_ts(ts: float):
        tz = ZoneInfo("America/New_York")
        return datetime.fromtimestamp(float(ts), tz=tz).date()

    def _has_executed_rebalance_today(self, *, now_ts: Optional[float] = None) -> bool:
        """Return True if EML state shows any executed rebalance on today's Eastern date."""

        now = float(now_ts if now_ts is not None else time.time())
        today = self._eastern_date_from_ts(now)
        hist = getattr(self.state, "executed_rebalance_history", None) or []
        for item in hist:
            if not isinstance(item, dict):
                continue
            ets = item.get("executed_ts")
            try:
                ets_f = float(ets)
            except Exception:
                continue
            try:
                if self._eastern_date_from_ts(ets_f) == today:
                    return True
            except Exception:
                continue
        return False

    def _position_cleanup_request_from_state(
        self, payload: Mapping[str, Any]
    ) -> PositionCleanupPlanRequestEvent:
        request_id = payload.get("request_id")
        if not request_id:
            raise ValueError(
                "Invalid pending position cleanup payload: missing request_id"
            )
        request_id = str(request_id)

        ts = payload.get("request_ts")
        try:
            ts_f = float(ts)
        except Exception:
            ts_f = time.time()

        intents_payload = payload.get("intents")
        if intents_payload is None:
            intents_payload = {}
        if not isinstance(intents_payload, Mapping):
            intents_payload = {}

        intents: Dict[str, PositionCleanupIntent] = {}
        for sym, info in dict(intents_payload).items():
            if not isinstance(info, Mapping):
                info = {}
            ticker = str(info.get("ticker") or sym)
            reason = str(info.get("reason") or "")
            intents[str(sym)] = PositionCleanupIntent(
                ticker=ticker,
                reason=reason,
                observed_qty=to_decimal(info.get("observed_qty")),
                qty_threshold=to_decimal(info.get("qty_threshold")),
                observed_market_value=to_decimal(info.get("observed_market_value")),
                market_value_threshold=to_decimal(info.get("market_value_threshold")),
            )

        return PositionCleanupPlanRequestEvent(
            ts=ts_f,
            request_id=request_id,
            intents=intents,
            source=str(payload.get("source") or ""),
            correlation_id=str(payload.get("correlation_id") or ""),
        )

    def _execute_pending_position_cleanup_plans(self) -> None:
        """Execute any pending position cleanup requests recorded in `self.state`.

        Safety rule
        -----------
        If ANY rebalance was executed today (Eastern), all pending cleanup plans are
        marked as "cancelled" (moved from pending to executed cleanup history).
        """

        pending = self.state.pending_position_cleanup_requests
        if not pending:
            self.log.debug("No pending position cleanup plans to execute")
            return

        # Only execute when we are sure the market is open.
        clock = self._market_clock
        if clock is None:
            self.log.info(
                "Skipping pending position cleanup execution: market clock unknown (pending=%d)",
                len(pending),
            )
            return
        is_open = clock.is_market_open
        if is_open is not True:
            self.log.debug(
                "Skipping pending position cleanup execution: market not open (is_market_open=%s now=%s next_open=%s pending=%d)",
                is_open,
                clock.now,
                clock.next_market_open,
                len(pending),
            )
            return

        cancel_all = self._has_executed_rebalance_today()
        if cancel_all:
            self.log.info(
                "Cancelling %d pending position cleanup plan(s): rebalance already executed today",
                len(pending),
            )

        # Process oldest-first for determinism.
        items: List[Tuple[str, Dict[str, Any]]] = []
        for request_id, payload in dict(pending).items():
            if not isinstance(payload, dict):
                continue
            items.append((str(request_id), dict(payload)))

        def _key(item: Tuple[str, Dict[str, Any]]) -> float:
            v = item[1].get("request_ts")
            try:
                return float(v)
            except Exception:
                return 0.0

        items.sort(key=_key)

        for request_id, payload in items:
            if not self.state.has_pending_position_cleanup_request(request_id):
                continue

            try:
                if cancel_all:
                    self.state.mark_position_cleanup_executed(
                        request_id=request_id,
                        status="cancelled",
                        note="rebalance executed today",
                    )
                    continue

                event = self._position_cleanup_request_from_state(payload)
                self._execute_position_cleanup_plan(event)
                self.state.mark_position_cleanup_executed(request_id=request_id)
                self.log.info(
                    "Position cleanup executed successfully: request_id=%s",
                    request_id,
                )
            except EMLShutdownRequested:
                self.log.info(
                    "Shutdown requested; aborting pending position cleanup execution: request_id=%s",
                    request_id,
                )
                return
            except Exception:
                self.log.exception(
                    "Failed executing pending position cleanup plan: request_id=%s",
                    request_id,
                )

                try:
                    failures = (
                        self.state.increment_pending_position_cleanup_execution_failure(
                            request_id
                        )
                    )
                    max_retries = (
                        self.config.max_pending_position_cleanup_execution_retries
                    )
                    if failures >= max_retries:
                        self.state.mark_position_cleanup_failed(
                            request_id=request_id,
                            error="max retries exceeded",
                        )
                        self.log.warning(
                            "Pending position cleanup marked failed after %d failed attempt(s): request_id=%s",
                            failures,
                            request_id,
                        )
                except Exception:
                    self.log.exception(
                        "Failed updating retry/failed state for pending position cleanup: request_id=%s",
                        request_id,
                    )

    def _execute_position_cleanup_plan(
        self, event: PositionCleanupPlanRequestEvent
    ) -> None:
        """Synchronously execute a single position cleanup plan.

        Translates cleanup intents into market orders that attempt to close the entire
        position for each intended symbol.
        """

        request_id = event.request_id
        intents = event.intents
        if request_id is None or intents is None:
            raise TypeError(
                "event must have attributes 'request_id' and 'intents' (PositionCleanupPlanRequestEvent-like)"
            )
        if not self.config.include_positions:
            try:
                self._position_cleanup_safety_refusals_counter.add(
                    1,
                    {
                        "service": self.name,
                        "reason": "include_positions_disabled",
                    },
                )
            except Exception:
                pass
            raise RuntimeError(
                "Cannot execute position cleanup plan when EMLConfig.include_positions is False"
            )

        self.log.info(
            "Executing position cleanup plan: request_id=%s intents=%s",
            request_id,
            list((intents or {}).keys()) if isinstance(intents, dict) else None,
        )
        # Fetch current positions; we rely on latest broker state.
        positions = self._list_positions()
        pos_by_symbol = self._positions_by_symbol(positions)
        symbols: List[str] = []
        if isinstance(intents, dict):
            symbols = [self._normalize_symbol(s) for s in intents.keys()]
        symbols = [s for s in symbols if s]

        sells: List[Dict[str, Any]] = []
        max_abs_qty = self.config.position_cleanup_max_abs_qty
        if max_abs_qty is not None:
            max_abs_qty = float(max_abs_qty)
        for sym in symbols:
            p = pos_by_symbol.get(sym)
            if p is None:
                try:
                    self._position_cleanup_skips_counter.add(
                        1,
                        {
                            "service": self.name,
                            "reason": "no_position",
                        },
                    )
                except Exception:
                    pass
                continue
            qty = float(p.qty or 0.0)
            if abs(qty) <= 0.0:
                try:
                    self._position_cleanup_skips_counter.add(
                        1,
                        {
                            "service": self.name,
                            "reason": "already_flat",
                        },
                    )
                except Exception:
                    pass
                self.log.warning(
                    "No position to clean up for symbol; skipping: request_id=%s symbol=%s qty=%s",
                    request_id,
                    sym,
                    qty,
                )
                continue
            if qty > 0:
                if max_abs_qty is not None and abs(qty) > max_abs_qty:
                    try:
                        self._position_cleanup_safety_refusals_counter.add(
                            1,
                            {
                                "service": self.name,
                                "reason": "max_abs_qty_exceeded",
                            },
                        )
                    except Exception:
                        pass
                    raise RuntimeError(
                        "Refusing to execute position cleanup sell exceeding qty safety threshold: "
                        f"request_id={request_id} symbol={sym} qty={qty} max_abs_qty={max_abs_qty}"
                    )
                # _execute_orders_blocking expects sells to use qty_fallback (notional optional).
                sells.append(
                    {
                        "symbol": sym,
                        "side": "sell",
                        "notional": None,
                        "qty_fallback": abs(qty),
                    }
                )
            else:
                # No short-position cleanup for now; warn and skip.
                try:
                    self._position_cleanup_skips_counter.add(
                        1,
                        {
                            "service": self.name,
                            "reason": "short_detected",
                        },
                    )
                except Exception:
                    pass
                self.log.warning(
                    "Residual short position detected during cleanup; skipping buy-to-cover: request_id=%s symbol=%s qty=%s",
                    request_id,
                    sym,
                    qty,
                )

        if not sells:
            self.log.info(
                "No executable cleanup orders (positions already flat); treating as executed: request_id=%s",
                request_id,
            )
            return

        # Sanity check tradability for all tickers in final plan.
        self._assert_symbols_tradable([o["symbol"] for o in sells])

        # Safety: cancel any outstanding orders before placing new ones.
        try:
            self._cancel_all_open_orders()
        except Exception:
            self.log.exception(
                "Failed to cancel open orders before position cleanup execution (continuing): request_id=%s",
                request_id,
            )

        # Execute sells only.
        self._execute_orders_blocking(sells)

    # ----------------------------
    # Helpers (testable)
    # ----------------------------

    @staticmethod
    def _normalize_symbol(symbol: Any) -> str:
        return str(symbol).strip().upper()

    @classmethod
    def _normalize_target_weights(
        cls,
        weights: Mapping[str, float],
        *,
        weight_epsilon: float = 1e-10,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not weights:
            return out
        for k, v in dict(weights).items():
            sym = cls._normalize_symbol(k)
            try:
                w = float(v)
            except Exception:
                continue
            if abs(w) <= weight_epsilon:
                continue
            out[sym] = w
        return out

    @staticmethod
    def _get_effective_equity(account: BrokerAccount) -> float:
        # Prefer adjusted equity (cash buffer), then equity, then portfolio_value.
        for v in (account.adj_equity, account.equity, account.portfolio_value):
            try:
                if v is None:
                    continue
                vf = float(v)
                if vf > 0:
                    return vf
            except Exception:
                continue
        return 0.0

    @classmethod
    def _positions_by_symbol(
        cls, positions: Iterable[BrokerPosition]
    ) -> Dict[str, BrokerPosition]:
        out: Dict[str, BrokerPosition] = {}
        for p in positions or []:
            sym = cls._normalize_symbol(getattr(p, "symbol", ""))
            if not sym:
                continue
            out[sym] = p
        return out

    @staticmethod
    def _compute_target_deltas(
        *,
        target_weights: Mapping[str, float],
        equity: float,
        positions_by_symbol: Mapping[str, BrokerPosition],
    ) -> Dict[str, Dict[str, float]]:
        # Returns per-symbol: current_value, target_value, delta_value.
        symbols = set(positions_by_symbol.keys()) | set(target_weights.keys())
        out: Dict[str, Dict[str, float]] = {}

        for sym in symbols:
            w = float(target_weights.get(sym, 0.0) or 0.0)
            target_value = equity * w

            p = positions_by_symbol.get(sym)
            mv = 0.0
            if p is not None:
                try:
                    mv = float(p.market_value or 0.0)
                except Exception:
                    mv = 0.0
            delta = target_value - mv
            out[sym] = {
                "current_value": mv,
                "target_value": target_value,
                "delta_value": delta,
            }

        return out

    @classmethod
    def _build_market_orders(
        cls,
        *,
        deltas: Mapping[str, Mapping[str, float]],
        positions_by_symbol: Mapping[str, BrokerPosition],
        min_order_size_notional: float,
        dollar_epsilon: float = 1e-6,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build deterministic sell/buy market orders from per-symbol notional deltas.

        Returns a 3-tuple: (sells, buys, dropped_by_min_size). The third element contains
        delta entries that were dropped specifically due to the `min_order_size_notional`
        threshold (including sell deltas that become too small after capping to current
        position market value).
        """
        sells: List[Dict[str, Any]] = []
        buys: List[Dict[str, Any]] = []
        dropped_by_min_size: List[Dict[str, Any]] = []

        min_abs = max(0.0, float(min_order_size_notional))

        for sym, info in dict(deltas).items():
            try:
                dv = float(info.get("delta_value", 0.0) or 0.0)
            except Exception:
                continue
            if abs(dv) <= dollar_epsilon:
                continue

            if dv < 0:
                # SELL: prefer notional if broker supports it, with qty fallback.
                if abs(dv) < min_abs:
                    dropped_by_min_size.append(
                        {
                            "symbol": sym,
                            "side": "sell",
                            "delta_value": dv,
                            "min_order_size_notional": min_abs,
                            "reason": "below_min_order_size_notional",
                        }
                    )
                    continue
                p = positions_by_symbol.get(sym)
                if p is None:
                    continue

                # Cap notional sells to current position market value (best-effort).
                desired_notional = float(abs(dv))
                try:
                    mv = float(p.market_value or 0.0)
                except Exception:
                    mv = 0.0
                if mv > 0:
                    desired_notional = min(desired_notional, float(mv))
                if desired_notional < min_abs:
                    dropped_by_min_size.append(
                        {
                            "symbol": sym,
                            "side": "sell",
                            "delta_value": dv,
                            "desired_notional": desired_notional,
                            "min_order_size_notional": min_abs,
                            "reason": "below_min_order_size_notional_after_cap",
                        }
                    )
                    continue

                qty_fallback = cls._estimate_qty_for_notional_sell(
                    p, notional=float(desired_notional)
                )
                if qty_fallback is None or qty_fallback <= 0:
                    continue

                sells.append(
                    {
                        "symbol": sym,
                        "side": "sell",
                        # Try a notional sell first if supported.
                        "notional": desired_notional,
                        # Fallback if broker/API rejects notional sells.
                        "qty_fallback": qty_fallback,
                    }
                )
            else:
                # BUY: use notional market orders for simplicity
                notional = float(dv)
                if notional < min_abs:
                    dropped_by_min_size.append(
                        {
                            "symbol": sym,
                            "side": "buy",
                            "delta_value": dv,
                            "min_order_size_notional": min_abs,
                            "reason": "below_min_order_size_notional",
                        }
                    )
                    continue
                buys.append(
                    {
                        "symbol": sym,
                        "side": "buy",
                        "qty": None,
                        "notional": notional,
                    }
                )

        # Deterministic order
        sells.sort(key=lambda x: x["symbol"])
        buys.sort(key=lambda x: x["symbol"])
        dropped_by_min_size.sort(key=lambda x: str(x.get("symbol") or ""))
        return sells, buys, dropped_by_min_size

    @staticmethod
    def _estimate_qty_for_notional_sell(
        position: BrokerPosition, *, notional: float
    ) -> Optional[float]:
        # Approximate qty to sell from position market_value/qty.
        # If we can't compute a reasonable unit price, fall back to selling full qty.
        try:
            qty = float(position.qty or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            return None

        try:
            mv = float(position.market_value or 0.0)
        except Exception:
            mv = 0.0

        if mv > 0:
            px = mv / qty
            if px > 0:
                est_qty = notional / px
                # Never sell more than we hold.
                return min(qty, max(0.0, est_qty))

        return qty

    def _assert_symbols_tradable(self, symbols: Iterable[str]) -> None:
        unique = []
        seen = set()
        for s in symbols:
            sym = self._normalize_symbol(s)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            unique.append(sym)

        not_tradable: List[str] = []
        for sym in unique:
            try:
                inst = self._trading_api.get_instrument(InstrumentRef(symbol=sym))
            except Exception:
                self.log.debug(
                    "Failed to fetch instrument info for symbol=%s; assumes not tradable",
                    sym,
                )
                not_tradable.append(sym)
                continue
            self.log.debug("Fetched instrument info for symbol=%s: %s", sym, inst)

            if inst.tradable is False:
                not_tradable.append(sym)

        if not_tradable:
            raise RuntimeError(
                f"Non-tradable symbols in rebalance plan: {sorted(not_tradable)}"
            )

    def _execute_orders_blocking(self, orders: List[Dict[str, Any]]) -> None:
        if not orders:
            return

        for order in orders:
            if self._shutdown_requested():
                raise EMLShutdownRequested("shutdown requested")

            symbol = self._normalize_symbol(order.get("symbol"))
            side = str(order.get("side")).strip().lower()
            if side not in {"buy", "sell"}:
                raise ValueError(f"Invalid order side: {side}")

            if side == "sell":
                order_id = self._submit_sell_market_order_prefer_notional(
                    symbol=symbol,
                    notional=order.get("notional"),
                    qty_fallback=order.get("qty_fallback"),
                )
            else:
                order_id = self._submit_market_order(
                    symbol=symbol,
                    side=side,
                    qty=order.get("qty"),
                    notional=order.get("notional"),
                )
            self.log.info(
                "Submitted market order: symbol=%s side=%s order_id=%s",
                symbol,
                side,
                order_id,
            )

            self.log.info(
                "Waiting for order fill: symbol=%s side=%s order_id=%s",
                symbol,
                side,
                order_id,
            )
            self._wait_for_order_fill(
                order_id,
                timeout_seconds=float(self.config.wait_for_order_fill_timeout_secs),
            )
            self.log.info(
                "Order filled: symbol=%s side=%s order_id=%s",
                symbol,
                side,
                order_id,
            )

    def _submit_sell_market_order_prefer_notional(
        self,
        *,
        symbol: str,
        notional: Any = None,
        qty_fallback: Any = None,
    ) -> str:
        """Submit a SELL market order.

        Prefers notional sells when available, but will fall back to qty if the
        broker/API rejects notional sells.
        """

        if notional is not None:
            try:
                return self._submit_market_order(
                    symbol=symbol,
                    side="sell",
                    qty=None,
                    notional=notional,
                )
            except Exception:
                self.log.warning(
                    "Notional sell rejected; falling back to qty sell: symbol=%s notional=%s",
                    symbol,
                    notional,
                    exc_info=True,
                )

        if qty_fallback is None:
            raise RuntimeError(
                f"Cannot submit sell order for {symbol}: no qty fallback available"
            )

        return self._submit_market_order(
            symbol=symbol,
            side="sell",
            qty=qty_fallback,
            notional=None,
        )

    def _submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: Any = None,
        notional: Any = None,
    ) -> str:
        if qty is not None and notional is not None:
            raise ValueError(
                "Market order must specify either qty or notional, not both"
            )

        if notional is not None:
            try:
                notional_f = float(notional)
            except Exception:
                raise ValueError(f"Invalid notional: {notional}")
            if notional_f <= 0:
                raise ValueError(f"Invalid notional: {notional}")

        self.log.debug(
            "Submitting market order: symbol=%s side=%s qty=%s notional=%s",
            symbol,
            side,
            qty,
            notional,
        )

        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        intent = OrderIntent(
            client_order_id=str(uuid.uuid4()),
            instrument=InstrumentRef(symbol=symbol),
            side=side_enum,
            qty=to_decimal(qty),
            notional=to_decimal(notional),
        )
        placed = self._trading_api.submit_order(intent)
        try:
            self._orders_submitted_counter.add(
                1,
                {
                    "symbol": self._normalize_symbol(symbol),
                    "side": str(side).strip().lower(),
                },
            )
        except Exception:
            pass
        return placed.broker_order_id

    def _wait_for_order_fill(
        self,
        order_id: str,
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        start = float(now_fn())

        while True:
            if self._shutdown_requested():
                raise EMLShutdownRequested("shutdown requested")

            if float(now_fn()) - start > float(timeout_seconds):
                raise TimeoutError(
                    f"Timed out waiting for order fill: order_id={order_id}"
                )

            try:
                o = self._trading_api.get_order(order_id)
            except OrderNotFoundYet:
                # Some brokers (e.g. Public.com) are eventually consistent: immediately after
                # order placement, the order may not yet be queryable.
                self.log.info("Order not found yet; will retry: order_id=%s", order_id)
                sleep_fn(float(poll_interval_seconds))
                continue
            status = o.status

            self.log.debug(
                "Polled order status: order_id=%s status=%s",
                order_id,
                status,
            )

            if status == OrderStatus.FILLED:
                try:
                    latency = max(0.0, float(now_fn()) - start)
                    self._order_fills_counter.add(1, {"service": self.name})
                    self._order_fill_latency_hist.record(
                        float(latency), {"service": self.name}
                    )
                except Exception:
                    pass
                return
            if status in {
                OrderStatus.CANCELED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            }:
                raise RuntimeError(
                    f"Order did not fill (status={status}): order_id={order_id}"
                )

            sleep_fn(float(poll_interval_seconds))

    @staticmethod
    def _rebalance_request_from_state(
        payload: Mapping[str, Any],
    ) -> RebalancePlanRequestEvent:
        rebalance_id = str(payload.get("rebalance_id") or "")
        if not rebalance_id:
            raise ValueError("Invalid pending rebalance payload: missing rebalance_id")

        ts = payload.get("request_ts")
        try:
            ts_f = float(ts)
        except Exception:
            ts_f = time.time()

        weights = payload.get("weights")
        if weights is None:
            weights = {}
        if not isinstance(weights, Mapping):
            weights = {}

        return RebalancePlanRequestEvent(
            ts=ts_f,
            rebalance_id=rebalance_id,
            weights=dict(weights),
            source=str(payload.get("source") or ""),
            correlation_id=str(payload.get("correlation_id") or ""),
        )

    def _gc_execution_history(self, *, now_ts: Optional[float] = None) -> None:
        """Discard executed execution history entries older than max_execution_history_days."""

        max_days_int = self.config.max_execution_history_days
        if max_days_int <= 0:
            self.log.debug(
                "EMLConfig: max_execution_history_days <= 0; skipping execution history GC."
            )
            return

        now = float(now_ts if now_ts is not None else time.time())
        cutoff = now - (max_days_int * 86400.0)

        def _gc_list(
            hist: List[Dict[str, Any]], *, id_field: str, label: str
        ) -> List[Dict[str, Any]]:
            before = len(hist)
            kept: List[Dict[str, Any]] = []
            for item in hist:
                if not isinstance(item, dict):
                    continue
                ts = item.get("executed_ts")
                try:
                    ts_f = float(ts)
                except Exception:
                    ts_f = 0.0
                if ts_f >= cutoff:
                    kept.append(item)
                else:
                    self.log.debug(
                        "GC'ing %s history entry: %s=%s executed_ts=%s",
                        label,
                        id_field,
                        item.get(id_field),
                        item.get("executed_ts"),
                    )

            if len(kept) != before:
                kept.sort(key=lambda x: float(x.get("executed_ts", 0.0) or 0.0))
                self.log.info(
                    "GC'd %s history: removed=%d kept=%d cutoff=%s",
                    label,
                    before - len(kept),
                    len(kept),
                    datetime.fromtimestamp(cutoff).isoformat(),
                )
            return kept

        self.state.executed_rebalance_history = _gc_list(
            self.state.executed_rebalance_history,
            id_field="rebalance_id",
            label="executed rebalance",
        )
        self.state.executed_position_cleanup_history = _gc_list(
            self.state.executed_position_cleanup_history,
            id_field="request_id",
            label="executed position cleanup",
        )

    def _get_account(self) -> BrokerAccount:
        acct = self._trading_api.get_account()
        return BrokerAccount(
            id=acct.id,
            status=acct.status,
            cash=acct.cash,
            buying_power=acct.buying_power,
            portfolio_value=acct.portfolio_value,
            equity=acct.equity,
            last_equity=acct.last_equity,
            adj_equity=self._get_equity_adj(acct.equity),
        )

    def _list_positions(self) -> List[BrokerPosition]:
        return list(self._trading_api.list_positions())

    def _get_equity_adj(self, equity_abs: Optional[Decimal]) -> Optional[Decimal]:
        """Compute adjusted equity after applying cash buffer settings.

        Returns None if equity_abs is None.
        """
        if equity_abs is None:
            return None

        try:
            equity_d = Decimal(str(equity_abs))
        except Exception:
            return None

        if self.config.cash_buffer_pct is not None:
            try:
                pct = Decimal(str(self.config.cash_buffer_pct))
            except Exception:
                pct = Decimal("0")
            buffer_amt = equity_d * pct
            return max(Decimal("0"), equity_d - buffer_amt)

        if self.config.cash_buffer_abs is not None:
            try:
                buffer_amt = Decimal(str(self.config.cash_buffer_abs))
            except Exception:
                buffer_amt = Decimal("0")
            return max(Decimal("0"), equity_d - buffer_amt)

        return equity_d
