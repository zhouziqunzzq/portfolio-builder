import asyncio
import time
import uuid
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from typing import Any, Dict, List, Mapping, Optional, Set
from decimal import Decimal
from opentelemetry import metrics
from opentelemetry.metrics import Observation
from .base_at import BaseATService
from .config import ATConfig
from runtime_manager import RuntimeManager
from events.topic import Topic
from events.events import (
    AccountSnapshotEvent,
    BaseEvent,
    V2MarketClockEvent,
    V2RebalancePlanRequestEvent,
    V2RebalancePlanConfirmationEvent,
    V2BarsCheckedEvent,
    V2PositionCleanupPlanRequestEvent,
    V2PositionCleanupPlanConfirmationEvent,
    V2PositionCleanupIntent,
)
from events.event_bus import EventBus
from allocator.multi_sleeve_allocator import MultiSleeveAllocator
from market_data_store import MarketDataStore
from utils.decimals import to_decimal
from utils.tz import to_canonical_eastern_naive
from states.base_state import BaseState
from context.rebalance import RebalanceContext
from models import PositionSnapshot


class MultiSleeveATState(BaseState):
    STATE_KEY = "at.multi_sleeve"
    SCHEMA_VERSION = 2

    # Pending rebalance info
    pending_rebalance_ts: Optional[datetime] = None
    pending_rebalance_id: Optional[str] = None
    pending_rebalance_weights: Optional[Dict[str, float]] = None

    # Last confirmed rebalance info
    last_rebalance_ts: Optional[datetime] = None
    last_rebalance_id: Optional[str] = None
    last_rebalance_weights: Optional[Dict[str, float]] = None

    # Market data freshness info
    last_market_data_ts: Optional[datetime] = None

    # Position cleanup info
    last_position_cleanup_ts: Optional[datetime] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "pending_rebalance_ts": (
                self.pending_rebalance_ts.isoformat()
                if self.pending_rebalance_ts
                else None
            ),
            "pending_rebalance_id": self.pending_rebalance_id,
            "pending_rebalance_weights": self.pending_rebalance_weights,
            "last_rebalance_ts": (
                self.last_rebalance_ts.isoformat() if self.last_rebalance_ts else None
            ),
            "last_rebalance_id": self.last_rebalance_id,
            "last_rebalance_weights": self.last_rebalance_weights,
            "last_market_data_ts": (
                self.last_market_data_ts.isoformat()
                if self.last_market_data_ts
                else None
            ),
            "last_position_cleanup_ts": (
                self.last_position_cleanup_ts.isoformat()
                if self.last_position_cleanup_ts
                else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MultiSleeveATState":
        state = cls()

        pending_ts = payload.get("pending_rebalance_ts")
        state.pending_rebalance_ts = (
            datetime.fromisoformat(pending_ts) if pending_ts else None
        )
        state.pending_rebalance_id = payload.get("pending_rebalance_id")
        state.pending_rebalance_weights = payload.get("pending_rebalance_weights")

        last_ts = payload.get("last_rebalance_ts")
        state.last_rebalance_ts = datetime.fromisoformat(last_ts) if last_ts else None
        state.last_rebalance_id = payload.get("last_rebalance_id")
        state.last_rebalance_weights = payload.get("last_rebalance_weights")
        market_data_ts = payload.get("last_market_data_ts")
        state.last_market_data_ts = (
            datetime.fromisoformat(market_data_ts) if market_data_ts else None
        )

        position_cleanup_ts = payload.get("last_position_cleanup_ts")
        state.last_position_cleanup_ts = (
            datetime.fromisoformat(position_cleanup_ts) if position_cleanup_ts else None
        )
        return state

    @classmethod
    def empty(cls) -> "MultiSleeveATState":
        return cls()


class MultiSleeveATService(BaseATService):
    """Multi-sleeve AutoTrader service.

    For now it only:
    - stores the latest market clock event
    - logs debug output for all incoming events
    """

    def __init__(
        self,
        bus: "EventBus",
        rm: "RuntimeManager",
        *,
        config: ATConfig,
        name: str = "MultiSleeveAT",
    ):
        super().__init__(bus=bus, name=name)
        if config is None:
            self.log.warning("No ATConfig provided, using default configuration values")
            config = ATConfig()
        # Validate config
        config.validate()
        self.config = config
        self.rm = rm

        if config.polling_interval_secs <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        self._poll_interval_seconds = float(config.polling_interval_secs)

        # Register self to RuntimeManager for lifecycle management.
        self.rm.set("at", self)
        self.rm.set("multi_sleeve_at", self)  # alias

        # State
        # This will be managed by StateManager externally.
        self.state: MultiSleeveATState = MultiSleeveATState()

        # Internal caches
        self._market_clock: Optional[V2MarketClockEvent] = None
        self._account_snapshot: Optional[AccountSnapshotEvent] = None

        # Metrics
        self._should_rebalance_last: Optional[bool] = None
        self._gate_account_snapshot_present: Optional[bool] = None
        self._gate_market_clock_present: Optional[bool] = None
        self._gate_market_data_fresh: Optional[bool] = None
        self._gate_allocator_wants_rebalance: Optional[bool] = None
        self._gate_market_open_now: Optional[bool] = None
        self._gate_market_open_today: Optional[bool] = None
        self._init_metrics_instruments()

    @property
    def subscription_topics(self) -> Set[Topic]:
        topics = {
            Topic.V2_BAR,  # MultiSleeveAT needs bar updates to track market data freshness
            Topic.V2_REBALANCE_PLAN,  # MultiSleeveAT handles rebalance plan confirmations
            Topic.V2_POSITION_CLEANUP_PLAN,  # MultiSleeveAT handles position cleanup confirmations
        }

        return super().subscription_topics.union(topics)

    def _init_metrics_instruments(self) -> None:
        meter = metrics.get_meter("portfolio_builder.v2.at")

        self._heartbeat_counter = meter.create_counter(
            name="at.loop_heartbeat",
            description=(
                "Heartbeat counter incremented once per AT background loop iteration"
            ),
        )

        self._loop_duration_hist = meter.create_histogram(
            name="at.loop_duration_seconds",
            description="Wall-clock duration of each AT background loop iteration",
            unit="s",
        )

        self._rebalance_plan_generation_duration_hist = meter.create_histogram(
            name="at.rebalance_plan_generation_duration_seconds",
            description="Wall-clock duration of AT rebalance plan generation",
            unit="s",
        )

        self._rebalance_plan_generation_errors_counter = meter.create_counter(
            name="at.rebalance_plan_generation_errors",
            description="Counts errors encountered while generating rebalance plans",
        )

        self._target_weight_count_hist = meter.create_histogram(
            name="at.target_weight_count",
            description="Number of symbols in generated target weight maps",
            unit="1",
        )

        def _obs_bool(v: Optional[bool]) -> list[Observation]:
            if v is None:
                return [Observation(0, {"known": False})]
            return [Observation(1 if v else 0, {"known": True})]

        def observe_time_since_last_rebalance(_: object) -> list[Observation]:
            last = getattr(self.state, "last_rebalance_ts", None)
            if last is None:
                return [Observation(0.0, {"known": False})]
            try:
                # `last` is tz-naive; treat as local wall time consistently.
                age = float(datetime.now().timestamp() - last.timestamp())
                return [Observation(max(0.0, age), {"known": True})]
            except Exception:
                return [Observation(0.0, {"known": False})]

        def observe_last_market_data_age(_: object) -> list[Observation]:
            last = getattr(self.state, "last_market_data_ts", None)
            if last is None:
                return [Observation(0.0, {"known": False})]
            try:
                age = float(datetime.now().timestamp() - last.timestamp())
                return [Observation(max(0.0, age), {"known": True})]
            except Exception:
                return [Observation(0.0, {"known": False})]

        def observe_has_pending_rebalance(_: object) -> list[Observation]:
            has_pending = (
                self.state.pending_rebalance_id is not None
                and self.state.pending_rebalance_weights is not None
            )
            return [Observation(1 if has_pending else 0)]

        meter.create_observable_gauge(
            name="at.has_pending_rebalance",
            description="1 if AT has a pending rebalance, else 0",
            callbacks=[observe_has_pending_rebalance],
        )

        def observe_should_rebalance(_: object) -> list[Observation]:
            v = self._should_rebalance_last
            if v is None:
                return [Observation(0, {"known": False})]
            return [Observation(1 if v else 0, {"known": True})]

        meter.create_observable_gauge(
            name="at.should_rebalance",
            description="Latest should_rebalance() evaluation (1=true, 0=false)",
            callbacks=[observe_should_rebalance],
        )

        meter.create_observable_gauge(
            name="at.time_since_last_rebalance_seconds",
            description="Seconds since the last confirmed rebalance (AT state)",
            callbacks=[observe_time_since_last_rebalance],
        )

        meter.create_observable_gauge(
            name="at.last_market_data_age_seconds",
            description="Seconds since AT last received BarsCheckedEvent (AT state)",
            callbacks=[observe_last_market_data_age],
        )

        meter.create_observable_gauge(
            name="at.gate_account_snapshot_present",
            description="Gating: 1 if AT has a recent AccountSnapshotEvent, else 0",
            callbacks=[lambda _: _obs_bool(self._gate_account_snapshot_present)],
        )
        meter.create_observable_gauge(
            name="at.gate_market_clock_present",
            description="Gating: 1 if AT has a MarketClockEvent, else 0",
            callbacks=[lambda _: _obs_bool(self._gate_market_clock_present)],
        )
        meter.create_observable_gauge(
            name="at.gate_market_data_fresh",
            description="Gating: 1 if market data is fresh today (BarsCheckedEvent), else 0",
            callbacks=[lambda _: _obs_bool(self._gate_market_data_fresh)],
        )
        meter.create_observable_gauge(
            name="at.gate_allocator_wants_rebalance",
            description="Gating: 1 if allocator indicates a rebalance is needed, else 0",
            callbacks=[lambda _: _obs_bool(self._gate_allocator_wants_rebalance)],
        )
        meter.create_observable_gauge(
            name="at.gate_market_open_now",
            description="Gating: 1 if market is open now (per MarketClockEvent), else 0",
            callbacks=[lambda _: _obs_bool(self._gate_market_open_now)],
        )
        meter.create_observable_gauge(
            name="at.gate_market_open_today",
            description="Gating: 1 if market will be open later today (per MarketClockEvent), else 0",
            callbacks=[lambda _: _obs_bool(self._gate_market_open_today)],
        )

        self._rebalance_plan_generated_counter = meter.create_counter(
            name="at.rebalance_plan_generated",
            description="Counts successful generation of new rebalance plans",
        )

        # --- Position cleanup metrics (planned/intent-level) ---
        self._position_cleanup_triggered_counter = meter.create_counter(
            name="at.position_cleanup_triggered",
            description=(
                "Counts times AT emitted a PositionCleanupPlanRequestEvent (cleanup triggered)"
            ),
        )
        self._position_cleanup_positions_cleaned_counter = meter.create_counter(
            name="at.position_cleanup_positions_cleaned",
            description=(
                "Counts positions (intents) included in emitted PositionCleanupPlanRequestEvent"
            ),
            unit="1",
        )
        self._position_cleanup_reasons_counter = meter.create_counter(
            name="at.position_cleanup_reason",
            description=(
                "Counts cleanup intents by reason (label: reason) for emitted PositionCleanupPlanRequestEvent"
            ),
        )

    async def _run_loop(self) -> None:
        """Background loop."""

        while self._running:
            iteration_start = time.monotonic()
            iteration_success = True
            iteration_cancelled = False
            try:
                now_native = datetime.now().astimezone()
                # Convert to tz-naive US/Eastern wall time
                now = to_canonical_eastern_naive(pd.Timestamp(now_native))

                # Re-submit pending rebalance if needed
                has_pending_rebalance = False
                if self.state.pending_rebalance_id is not None:
                    if self.state.pending_rebalance_weights is None:
                        self.log.error(
                            "Inconsistent state: pending_rebalance_id is set but weights are None"
                        )
                    else:
                        has_pending_rebalance = True
                    self.log.debug(
                        "Re-submitting pending rebalance: ts=%s id=%s weights=%s",
                        self.state.pending_rebalance_ts,
                        self.state.pending_rebalance_id,
                        self.state.pending_rebalance_weights,
                    )
                    event = V2RebalancePlanRequestEvent(
                        ts=now.timestamp(),
                        rebalance_id=self.state.pending_rebalance_id,
                        weights=self.state.pending_rebalance_weights,
                        source=self.name,
                    )
                    await self.emit_rebalance_plan_request(event)
                    self.log.info(
                        "Re-emitted RebalancePlanRequestEvent: rebalance_id=%s target_weights=%s",
                        event.rebalance_id,
                        event.weights,
                    )

                if has_pending_rebalance:
                    # Skip generating a new rebalance if one is already pending
                    self.log.debug("Skipping rebalance check due to pending rebalance")
                    await asyncio.sleep(self._poll_interval_seconds)
                    continue

                # If a rebalance is due and no pending rebalance exists, generate and emit a RebalancePlanRequestEvent
                should_rebalance = self._check_should_rebalance(now=now)
                self._should_rebalance_last = bool(should_rebalance)
                self.log.debug("Rebalance check: should_rebalance=%s", should_rebalance)
                if should_rebalance:
                    gen_start = time.monotonic()
                    try:
                        event = self._generate_rebalance_plan_request(now=now)
                    except Exception:
                        self._rebalance_plan_generation_errors_counter.add(
                            1, {"service": self.name}
                        )
                        self._rebalance_plan_generation_duration_hist.record(
                            max(0.0, float(time.monotonic() - gen_start)),
                            {"success": False, "service": self.name},
                        )
                        raise
                    else:
                        self._rebalance_plan_generation_duration_hist.record(
                            max(0.0, float(time.monotonic() - gen_start)),
                            {"success": True, "service": self.name},
                        )
                        try:
                            n_weights = len(getattr(event, "weights", {}) or {})
                            self._target_weight_count_hist.record(
                                float(n_weights), {"service": self.name}
                            )
                        except Exception:
                            # Never block live operation on metrics.
                            pass
                    # Update state with pending rebalance info first
                    self.state.pending_rebalance_ts = now.to_pydatetime()
                    self.state.pending_rebalance_id = event.rebalance_id
                    self.state.pending_rebalance_weights = event.weights
                    self.log.debug(
                        "Updated state with pending rebalance: ts=%s id=%s weights=%s",
                        self.state.pending_rebalance_ts,
                        self.state.pending_rebalance_id,
                        self.state.pending_rebalance_weights,
                    )
                    # Then emit the event
                    await self.emit_rebalance_plan_request(event)
                    self._rebalance_plan_generated_counter.add(
                        1,
                        {
                            "service": self.name,
                        },
                    )
                    self.log.info(
                        "Emitted RebalancePlanRequestEvent: rebalance_id=%s target_weights=%s",
                        event.rebalance_id,
                        event.weights,
                    )

                # Run position cleanup if needed
                should_cleanup = self._check_should_cleanup_positions(now=now)
                self.log.debug(
                    "Position cleanup check: should_cleanup=%s", should_cleanup
                )
                if should_cleanup:
                    cleanup_event = self._generate_position_cleanup_plan_request(
                        now=now
                    )
                    if cleanup_event is None:
                        self.log.info(
                            "Position cleanup gated on, but no residual positions detected; skipping cleanup emit"
                        )
                    else:
                        await self.emit_position_cleanup_plan_request(cleanup_event)

                        try:
                            intents = getattr(cleanup_event, "intents", None) or {}
                            intents_count = int(len(intents))
                            self._position_cleanup_triggered_counter.add(
                                1, {"service": self.name}
                            )
                            if intents_count > 0:
                                self._position_cleanup_positions_cleaned_counter.add(
                                    intents_count, {"service": self.name}
                                )
                                for intent in intents.values():
                                    reason = (
                                        getattr(intent, "reason", None) or "unknown"
                                    )
                                    self._position_cleanup_reasons_counter.add(
                                        1, {"service": self.name, "reason": str(reason)}
                                    )
                        except Exception:
                            # Never block live operation on metrics.
                            pass

                        self.log.info(
                            "Emitted PositionCleanupPlanRequestEvent: intents=%s",
                            list((cleanup_event.intents or {}).keys()),
                        )
                        # Update state with last cleanup timestamp
                        self.state.last_position_cleanup_ts = now.to_pydatetime()

                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                iteration_cancelled = True
                raise
            except Exception:
                iteration_success = False
                self.log.exception("Error in MultiSleeveATService main loop")
                await asyncio.sleep(self._poll_interval_seconds)
            finally:
                # Count one heartbeat per finished iteration, labeled by success.
                if self._running and not iteration_cancelled:
                    self._heartbeat_counter.add(1, {"success": iteration_success})
                    self._loop_duration_hist.record(
                        max(0.0, float(time.monotonic() - iteration_start)),
                        {"success": iteration_success},
                    )

    async def _handle_event(self, event: BaseEvent) -> None:
        self.log.debug(
            "AT event received: topic=%s type=%s source=%s ts=%s",
            getattr(event, "topic", None),
            type(event).__name__,
            getattr(event, "source", ""),
            getattr(event, "ts", None),
        )

        # Handle MarketClockEvent
        if isinstance(event, V2MarketClockEvent):
            self._market_clock = event
            self.log.debug(
                "Stored market clock: now=%s is_open=%s next_market_open=%s next_market_close=%s",
                event.now,
                event.is_market_open,
                event.next_market_open,
                event.next_market_close,
            )
            return

        # Handle BarsCheckedEvent
        if isinstance(event, V2BarsCheckedEvent):
            # Update last market data timestamp in state
            self.state.last_market_data_ts = datetime.fromtimestamp(event.ts)
            self.log.debug(
                "Updated last market data timestamp in state: %s",
                self.state.last_market_data_ts,
            )
            return

        # Handle AccountSnapshotEvent
        if isinstance(event, AccountSnapshotEvent):
            self._account_snapshot = event
            self.log.debug(
                "Stored account snapshot: equity=%s adj_equity=%s cash=%s buying_power=%s positions=%d",
                getattr(event.account, "equity", None),
                getattr(event.account, "adj_equity", None),
                getattr(event.account, "cash", None),
                getattr(event.account, "buying_power", None),
                len(getattr(event, "positions", []) or []),
            )
            return

        # Handle RebalancePlanConfirmationEvent
        if isinstance(event, V2RebalancePlanConfirmationEvent):
            if (
                self.state.pending_rebalance_id is not None
                and event.rebalance_id == self.state.pending_rebalance_id
            ):
                # Move pending rebalance to last confirmed rebalance
                self.state.last_rebalance_ts = datetime.fromtimestamp(event.ts)
                self.state.last_rebalance_id = event.rebalance_id
                self.state.last_rebalance_weights = self.state.pending_rebalance_weights
                self.state.pending_rebalance_ts = None
                self.state.pending_rebalance_id = None
                self.state.pending_rebalance_weights = None
                self.log.info(
                    "Rebalance confirmed: id=%s ts=%s. Moved pending rebalance to last confirmed rebalance.",
                    event.rebalance_id,
                    datetime.fromtimestamp(event.ts),
                )
            else:
                self.log.warning(
                    "Received RebalancePlanConfirmationEvent for unknown rebalance_id=%s",
                    event.rebalance_id,
                )
            return

        self.log.debug(
            "Ignoring event: topic=%s type=%s source=%s ts=%s",
            getattr(event, "topic", None),
            type(event).__name__,
            getattr(event, "source", ""),
            getattr(event, "ts", None),
        )

    def _check_should_rebalance(self, now: Optional[datetime] = None) -> bool:
        """Check if a rebalance should be triggered.
        A rebalance should be triggered if:
        - Market data is fresh for today (at least one BarsCheckedEvent received today), AND
        - The MultiSleeveAllocator indicates a rebalance is needed, AND
        - AT has received a valid AccountSnapshotEvent with adj_equity > 0, AND
        - The market is currently open, or will be open later today.

        Args:
            now: Current time as tz-naive US/Eastern. If None, uses current system time.
        Returns:
            True if a rebalance should be triggered, False otherwise.
        """
        if now is None:
            now_native = datetime.now().astimezone()
            now = to_canonical_eastern_naive(pd.Timestamp(now_native))

        # Reset gating state for this evaluation.
        self._gate_account_snapshot_present = None
        self._gate_market_clock_present = None
        self._gate_market_data_fresh = None
        self._gate_allocator_wants_rebalance = None
        self._gate_market_open_now = None
        self._gate_market_open_today = None

        # Ensure we have a valid account snapshot with adj_equity
        if self._account_snapshot is None:
            self._gate_account_snapshot_present = False
            self.log.warning(
                "Refusing to rebalance: no AccountSnapshotEvent received yet (need account.adj_equity for AUM)"
            )
            return False
        self._gate_account_snapshot_present = True
        aum = self._account_snapshot.account.adj_equity
        try:
            aum_f = float(aum) if aum is not None else 0.0
        except Exception:
            aum_f = 0.0
        if aum is None or aum_f <= 0.0:
            self.log.warning(
                "Refusing to rebalance: invalid account.adj_equity=%s in latest AccountSnapshotEvent",
                aum,
            )
            return False

        allocator: MultiSleeveAllocator = self.rm.get("multi_sleeve_allocator")
        if not allocator:
            self.log.error("MultiSleeveAllocator not found in RuntimeManager")
            raise RuntimeError("MultiSleeveAllocator not found")

        allocator_wants_rebalance = allocator.should_rebalance(now=now)
        self._gate_allocator_wants_rebalance = bool(allocator_wants_rebalance)
        self.log.debug(
            "Allocator rebalance check: allocator_wants_rebalance=%s",
            allocator_wants_rebalance,
        )

        self._gate_market_clock_present = self._market_clock is not None
        is_market_open_now = (
            self._market_clock.is_market_open if self._market_clock else False
        )
        self._gate_market_open_now = bool(is_market_open_now)
        self.log.debug("Market open now: %s", is_market_open_now)

        is_market_open_today = (
            self._market_clock.next_market_open.date() == now.date()
            if self._market_clock and self._market_clock.next_market_open
            else False
        )
        self._gate_market_open_today = bool(is_market_open_today)
        self.log.debug(
            "Market open today: %s; next_market_open_date=%s; now_date=%s",
            is_market_open_today,
            (
                self._market_clock.next_market_open.date()
                if self._market_clock and self._market_clock.next_market_open
                else None
            ),
            now.date(),
        )

        # Check market data freshness - require at least one BarsCheckedEvent received
        # for today's date.
        is_market_data_fresh = (
            self.state.last_market_data_ts is not None
            and self.state.last_market_data_ts.date() == now.date()
        )
        self._gate_market_data_fresh = bool(is_market_data_fresh)
        self.log.debug(
            "Market data freshness check: is_market_data_fresh=%s last_market_data_ts=%s",
            is_market_data_fresh,
            self.state.last_market_data_ts,
        )

        return (
            is_market_data_fresh
            and allocator_wants_rebalance
            and (is_market_open_now or is_market_open_today)
        )
        # return allocator_wants_rebalance  # TEMPORARY OVERRIDE FOR TESTING

    def _generate_rebalance_plan_request(
        self, now: Optional[datetime] = None
    ) -> "V2RebalancePlanRequestEvent":
        """Generate a RebalancePlanRequestEvent by:
        - Resetting MarketDataStore caches to ensure fresh data.
        - Invoking the precompute logic in MultiSleeveAllocator to prepare sleeves for rebalancing.
        - Bootstrapping sleeve state if needed (e.g., trend sleeve sector weights).
        - Fetching target weights from MultiSleeveAllocator.
        - Generate a unique rebalance ID.
        - Creating and returning a RebalancePlanRequestEvent with ID and weights.

        Args:
            now: Current time as tz-naive US/Eastern. If None, uses current system time.
        Returns:
            A RebalancePlanRequestEvent with the target weights.
        """
        if now is None:
            now_native = datetime.now().astimezone()
            now = to_canonical_eastern_naive(pd.Timestamp(now_native))

        if self._account_snapshot is None:
            raise RuntimeError(
                "Cannot generate rebalance plan: no AccountSnapshotEvent received yet"
            )
        aum = self._account_snapshot.account.adj_equity
        try:
            aum_f = float(aum) if aum is not None else 0.0
        except Exception:
            aum_f = 0.0
        if aum is None or aum_f <= 0.0:
            raise RuntimeError(
                f"Cannot generate rebalance plan: invalid account.adj_equity={aum}"
            )

        allocator: MultiSleeveAllocator = self.rm.get("multi_sleeve_allocator")
        if not allocator:
            self.log.error("MultiSleeveAllocator not found in RuntimeManager")
            raise RuntimeError("MultiSleeveAllocator not found")
        mds: MarketDataStore = self.rm.get("market_data_store")
        if not mds:
            self.log.error("MarketDataStore not found in RuntimeManager")
            raise RuntimeError("MarketDataStore not found")

        # Reset MDS caches to ensure fresh data
        mds.reset_memory_cache()
        self.log.debug("Reset MarketDataStore memory cache")

        # Precompute signals/scores
        end = now
        precompute_weeks = int(self.config.precompute_lookback_weeks)
        # Determine whether we need a one-time bootstrap.
        trend_sleeve = allocator.sleeves.get("trend")
        bootstrap_needed = (
            self.state.last_rebalance_ts is None
            and trend_sleeve is not None
            and trend_sleeve.get_last_rebalance_datetime() is None
        )
        if bootstrap_needed:
            # When bootstrapping, we need caches covering BOTH:
            #   (a) the bootstrap simulation window (bootstrap_lookback_weeks), and
            #   (b) the normal precompute window used for the live rebalance request.
            bootstrap_weeks = int(
                getattr(self.config, "bootstrap_lookback_weeks", 0) or 0
            )
            start = end - pd.Timedelta(weeks=(bootstrap_weeks + precompute_weeks))
        else:
            start = end - pd.Timedelta(weeks=precompute_weeks)
        self.log.debug(
            "Starting allocator precompute: start=%s end=%s",
            start,
            end,
        )
        precompute_rst = allocator.precompute(
            start=start,
            end=end,
        )
        self.log.debug(
            "Completed allocator precompute: start=%s end=%s result=%s",
            start,
            end,
            precompute_rst,
        )

        # Bootstrap sleeve state if needed
        if bootstrap_needed:
            try:
                self._bootstrap_weights(now=now.to_pydatetime())
            except Exception:
                # Never block live operation on bootstrap; log and continue.
                self.log.exception("Bootstrap failed; continuing without bootstrap")

        # Generate target weights
        as_of = now - pd.Timedelta(days=1)
        rebal_ctx = RebalanceContext(
            rebalance_ts=now.to_pydatetime(),
            aum=float(aum),
        )
        self.log.debug("Generating global target weights as_of=%s", as_of)
        weights, allocator_ctx = allocator.generate_global_target_weights(
            as_of=as_of,
            rebalance_ctx=rebal_ctx,
        )
        self.log.debug(
            "Generated target weights: %s allocator_ctx=%s",
            weights,
            allocator_ctx,
        )

        # Generate unique rebalance ID
        rebalance_id = self._generate_rebalance_id()
        self.log.debug("Generated rebalance ID: %s", rebalance_id)

        event = V2RebalancePlanRequestEvent(
            ts=now.timestamp(),
            rebalance_id=rebalance_id,
            weights=weights,
            source=self.name,
        )
        return event

    def _bootstrap_weights(self, now: Optional[datetime] = None) -> Dict[str, float]:
        """Bootstrap sleeve internal state for first-time live deployment.

        This is primarily intended to warm the trend sleeve's sector weights
        smoothing/hysteresis so day-1 live allocations don't start from a fully
        cold state.

        Constraints
        -----------
        - Does not emit any rebalance events / orders.
        - Avoids mutating allocator-level state (friction control baseline, last
          portfolio, etc.). Only sleeve-level state is advanced.

        Returns
        -------
        Dict[str, float]
            Latest bootstrapped trend sleeve stock weights (if any), else {}.
        """

        if now is None:
            now_native = datetime.now().astimezone()
            now_ts = to_canonical_eastern_naive(pd.Timestamp(now_native))
            now = now_ts.to_pydatetime()

        allocator: MultiSleeveAllocator = self.rm.get("multi_sleeve_allocator")
        if not allocator:
            self.log.error("MultiSleeveAllocator not found in RuntimeManager")
            raise RuntimeError("MultiSleeveAllocator not found")
        trend = allocator.sleeves.get("trend")
        if trend is None:
            self.log.info("No trend sleeve configured; skipping bootstrap")
            return {}
        if trend.get_last_rebalance_datetime() is not None:
            self.log.debug("Trend sleeve already has state; skipping bootstrap")
            return (
                getattr(getattr(trend, "state", None), "last_stock_weights", None) or {}
            )

        lookback_weeks = int(getattr(self.config, "bootstrap_lookback_weeks", 52) or 52)
        end_as_of = (pd.Timestamp(now) - pd.Timedelta(days=1)).normalize()
        start_boot = (end_as_of - pd.Timedelta(weeks=lookback_weeks)).normalize()
        if end_as_of <= start_boot:
            self.log.warning(
                "Invalid bootstrap window: start=%s end=%s; skipping bootstrap",
                start_boot.date(),
                end_as_of.date(),
            )
            return {}

        # Ensure sleeve isn't gated off during warm-up.
        regime = "bull"
        dummy_aum = 1_000_000.0

        self.log.info(
            "Bootstrapping trend sleeve via daily forward simulation [%s, %s]",
            start_boot.date(),
            end_as_of.date(),
        )

        last_weights: Dict[str, float] = {}

        steps = 0
        rebalances = 0
        daily_dates = pd.date_range(start=start_boot, end=end_as_of, freq="D")
        for dt in daily_dates:
            steps += 1

            dt_ts = pd.Timestamp(dt).to_pydatetime()
            try:
                if not trend.should_rebalance(dt_ts):
                    continue
            except Exception:
                # If the sleeve rejects the timestamp (e.g. due to timezone issues),
                # skip this day; bootstrap is best-effort.
                self.log.exception("Bootstrap should_rebalance failed at %s", dt.date())
                continue

            rebalances += 1
            rebal_ctx = RebalanceContext(
                rebalance_ts=pd.Timestamp(dt).to_pydatetime(),
                aum=float(dummy_aum),
            )
            # IMPORTANT: signals must use data up to t-1 (no lookahead).
            as_of = (pd.Timestamp(dt) - pd.Timedelta(days=1)).normalize()
            # Wide lookback for safety; caches from allocator.precompute are preferred.
            start_for_signals = (as_of - pd.Timedelta(days=730)).normalize()
            try:
                last_weights = trend.generate_target_weights_for_date(
                    as_of=as_of,
                    start_for_signals=start_for_signals,
                    regime=regime,
                    rebalance_ctx=rebal_ctx,
                )
            except Exception:
                self.log.exception("Bootstrap rebalance step failed for %s", dt.date())
                continue

        self.log.info(
            "Bootstrap complete (days=%d, rebalances=%d); trend last_rebalance_ts=%s",
            steps,
            rebalances,
            getattr(getattr(trend, "state", None), "last_rebalance_ts", None),
        )
        return last_weights

    def _check_should_cleanup_positions(self, now: Optional[datetime] = None) -> bool:
        """Check if a position cleanup should be triggered.
        A position cleanup should be triggered if:
        - Position cleanup is enabled in the ATConfig, AND
        - The market is currently open, or will be open later today, AND
        - Last position cleanup was more than the configured interval ago, AND
        - Last rebalance weights are available in state, AND
        - Account snapshot is available with positions.
        Note that the actual identification of residual positions is done during
        the generation of the PositionCleanupPlanRequestEvent.

        Args:
            now: Current time as tz-naive US/Eastern. If None, uses current system time.
        Returns:
            True if a position cleanup should be triggered, False otherwise.
        """
        if now is None:
            now_native = datetime.now().astimezone()
            now = to_canonical_eastern_naive(pd.Timestamp(now_native))

        # Check if position cleanup is enabled
        if not bool(getattr(self.config, "position_cleanup_enabled", False)):
            self.log.debug("Position cleanup disabled by config")
            return False
        # Check configured interval
        interval_days = int(
            getattr(self.config, "position_cleanup_interval_days", 0) or 0
        )
        if interval_days <= 0:
            self.log.warning(
                "Refusing to run position cleanup: invalid position_cleanup_interval_days=%s",
                interval_days,
            )
            return False

        # Require market clock available
        if self._market_clock is None:
            self.log.warning(
                "Refusing to run position cleanup: no MarketClockEvent received yet"
            )
            return False
        # Require market open now or later today.
        is_market_open_now = bool(getattr(self._market_clock, "is_market_open", False))
        next_open = getattr(self._market_clock, "next_market_open", None)
        is_market_open_today = False
        if is_market_open_now:
            is_market_open_today = True
        elif next_open is not None:
            is_market_open_today = next_open.date() == now.date()
        if not (is_market_open_now or is_market_open_today):
            self.log.debug(
                "Skipping position cleanup: market not open now and not opening later today (now=%s next_open=%s)",
                now,
                next_open,
            )
            return False

        # Require sufficient time since last position cleanup.
        last_cleanup_ts = getattr(self.state, "last_position_cleanup_ts", None)
        if last_cleanup_ts is not None:
            age_seconds = float(now.timestamp() - last_cleanup_ts.timestamp())
            min_age_seconds = float(interval_days) * 24 * 3600
            if age_seconds < min_age_seconds:
                self.log.debug(
                    "Skipping position cleanup: last cleanup too recent (age=%.0fs < min=%.0fs)",
                    max(0.0, age_seconds),
                    max(0.0, min_age_seconds),
                )
                return False

        # Require last rebalance weights (used to identify residuals vs. intended holdings).
        last_weights = getattr(self.state, "last_rebalance_weights", None)
        if not isinstance(last_weights, dict) or len(last_weights) == 0:
            self.log.warning(
                "Refusing to run position cleanup: missing last_rebalance_weights in state"
            )
            return False

        # Require a recent account snapshot with positions.
        if self._account_snapshot is None:
            self.log.warning(
                "Refusing to run position cleanup: no AccountSnapshotEvent received yet"
            )
            return False
        positions = getattr(self._account_snapshot, "positions", None) or []
        if len(positions) == 0:
            self.log.debug("Skipping position cleanup: account has no positions")
            return False

        return True

    def _generate_position_cleanup_plan_request(
        self, now: Optional[datetime] = None
    ) -> Optional[V2PositionCleanupPlanRequestEvent]:
        """Generate a position cleanup plan request (if needed).

        A position is considered a cleanup candidate when:
        - Its symbol is NOT present in the last confirmed rebalance weights (AT state), AND
        - Either |market_value| <= `config.position_cleanup_market_value_threshold`,
            OR |qty| <= `config.position_cleanup_qty_threshold`.

        This method only emits the *intent list*; actual order planning/execution happens
        downstream.

        Args:
            now: Current time as tz-naive US/Eastern. If None, uses current system time.
        Returns:
            A PositionCleanupPlanRequestEvent when at least one cleanup intent exists,
            otherwise None.
        """
        if now is None:
            now_native = datetime.now().astimezone()
            now = to_canonical_eastern_naive(pd.Timestamp(now_native))

        if self._account_snapshot is None:
            raise RuntimeError(
                "Cannot generate position cleanup plan: no AccountSnapshotEvent received yet"
            )
        positions: List[PositionSnapshot] | None = getattr(
            self._account_snapshot, "positions", None
        )
        if positions is None:
            raise RuntimeError(
                "Cannot generate position cleanup plan: account snapshot positions is None"
            )
        if len(positions) == 0:
            self.log.info("No positions in account snapshot; nothing to clean up")
            return None

        last_weights = getattr(self.state, "last_rebalance_weights", None)
        if not isinstance(last_weights, dict) or len(last_weights) == 0:
            raise RuntimeError(
                "Cannot generate position cleanup plan: missing last_rebalance_weights in state"
            )

        # Identify residual positions
        market_value_threshold = self.config.position_cleanup_market_value_threshold
        qty_threshold = self.config.position_cleanup_qty_threshold

        mv_thresh_d = to_decimal(market_value_threshold) or Decimal("0")
        qty_thresh_d = to_decimal(qty_threshold) or Decimal("0")
        intents: Dict[str, V2PositionCleanupIntent] = {}
        tickers_in_last_rebalance_weights = set(last_weights.keys())
        for pos in positions:
            # If symbol was in last rebalance weights, no cleanup
            if pos.symbol in tickers_in_last_rebalance_weights:
                self.log.debug(
                    "Position %s has target weight in last rebalance; skipping cleanup",
                    pos.symbol,
                )
                continue

            ticker = pos.symbol
            market_value_d = to_decimal(pos.market_value) or Decimal("0")
            qty_d = to_decimal(pos.qty) or Decimal("0")
            mv_below = abs(market_value_d) <= mv_thresh_d
            qty_below = abs(qty_d) <= qty_thresh_d
            should_cleanup = mv_below or qty_below
            self.log.debug(
                "Evaluating position %s for cleanup: market_value=%s qty=%s mv_below=%s qty_below=%s should_cleanup=%s",
                ticker,
                market_value_d,
                qty_d,
                mv_below,
                qty_below,
                should_cleanup,
            )

            if not should_cleanup:
                continue
            reasons: List[str] = []
            if mv_below:
                reasons.append("below_min_market_value")
            if qty_below:
                reasons.append("below_min_qty")
            intents[ticker] = V2PositionCleanupIntent(
                ticker=ticker,
                reason=",".join(reasons),
                observed_qty=qty_d,
                qty_threshold=qty_thresh_d,
                observed_market_value=market_value_d,
                market_value_threshold=mv_thresh_d,
            )

        self.log.debug(
            "Identified %d positions for cleanup: %s",
            len(intents),
            list(intents.keys()),
        )
        if len(intents) == 0:
            self.log.info("No residual positions detected; skipping cleanup request")
            return None

        cleanup_id = self._generate_position_cleanup_id()
        return V2PositionCleanupPlanRequestEvent(
            ts=now.timestamp(),
            request_id=cleanup_id,
            intents=intents,
            source=self.name,
        )

    @staticmethod
    def _generate_rebalance_id() -> str:
        """Generate a unique rebalance ID."""
        return f"rebalance-{uuid.uuid4()}"

    @staticmethod
    def _generate_position_cleanup_id() -> str:
        """Generate a unique position cleanup ID."""
        return f"position_cleanup-{uuid.uuid4()}"
