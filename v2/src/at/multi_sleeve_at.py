import asyncio
from datetime import datetime
import uuid
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional

import pandas as pd


_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from .base_at import BaseATService
from .config import ATConfig
from runtime_manager import RuntimeManager
from events.events import (
    AccountSnapshotEvent,
    BaseEvent,
    MarketClockEvent,
    RebalancePlanRequestEvent,
)
from events.event_bus import EventBus
from allocator.multi_sleeve_allocator import MultiSleeveAllocator
from market_data_store import MarketDataStore
from utils.tz import to_canonical_eastern_naive
from states.base_state import BaseState
from context.rebalance import RebalanceContext


class MultiSleeveATState(BaseState):
    STATE_KEY = "at.multi_sleeve"
    SCHEMA_VERSION = 1

    # Pending rebalance info
    pending_rebalance_ts: Optional[datetime] = None
    pending_rebalance_id: Optional[str] = None
    pending_rebalance_weights: Optional[Dict[str, float]] = None

    # Last confirmed rebalance info
    last_rebalance_ts: Optional[datetime] = None
    last_rebalance_id: Optional[str] = None
    last_rebalance_weights: Optional[Dict[str, float]] = None

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
        self._market_clock: Optional[MarketClockEvent] = None
        self._account_snapshot: Optional[AccountSnapshotEvent] = None

    async def _run_loop(self) -> None:
        """Background loop."""

        while self._running:
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
                    event = RebalancePlanRequestEvent(
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
                should_rebalance = await self._check_should_rebalance(now=now)
                self.log.debug("Rebalance check: should_rebalance=%s", should_rebalance)
                if should_rebalance:
                    event = await self._generate_rebalance_plan_request(now=now)
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
                    self.log.info(
                        "Emitted RebalancePlanRequestEvent: rebalance_id=%s target_weights=%s",
                        event.rebalance_id,
                        event.weights,
                    )

                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.log.exception("Error in MultiSleeveATService main loop")
                await asyncio.sleep(self._poll_interval_seconds)

    async def _handle_event(self, event: BaseEvent) -> None:
        self.log.debug(
            "AT event received: topic=%s type=%s source=%s ts=%s",
            getattr(event, "topic", None),
            type(event).__name__,
            getattr(event, "source", ""),
            getattr(event, "ts", None),
        )

        # Handle MarketClockEvent
        if isinstance(event, MarketClockEvent):
            self._market_clock = event
            self.log.debug(
                "Stored market clock: now=%s is_open=%s next_market_open=%s next_market_close=%s",
                event.now,
                event.is_market_open,
                event.next_market_open,
                event.next_market_close,
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

        # TODO: Handle RebalanceRequestConfirmationEvent

        self.log.debug(
            "Ignoring event: topic=%s type=%s source=%s ts=%s",
            getattr(event, "topic", None),
            type(event).__name__,
            getattr(event, "source", ""),
            getattr(event, "ts", None),
        )

    async def _check_should_rebalance(self, now: Optional[datetime] = None) -> bool:
        """Check if a rebalance should be triggered.
        A rebalance should be triggered if:
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

        if self._account_snapshot is None:
            self.log.warning(
                "Refusing to rebalance: no AccountSnapshotEvent received yet (need account.adj_equity for AUM)"
            )
            return False

        aum = getattr(self._account_snapshot.account, "adj_equity", None)
        if aum is None or not isinstance(aum, (int, float)) or float(aum) <= 0.0:
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
        self.log.debug(
            "Allocator rebalance check: allocator_wants_rebalance=%s",
            allocator_wants_rebalance,
        )

        is_market_open_now = (
            self._market_clock.is_market_open if self._market_clock else False
        )
        self.log.debug("Market open now: %s", is_market_open_now)

        is_market_open_today = (
            self._market_clock.next_market_open.date() == now.date()
            if self._market_clock and self._market_clock.next_market_open
            else False
        )
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

        return allocator_wants_rebalance and (
            is_market_open_now or is_market_open_today
        )
        # return allocator_wants_rebalance  # TEMPORARY OVERRIDE FOR TESTING

    async def _generate_rebalance_plan_request(
        self, now: Optional[datetime] = None
    ) -> "RebalancePlanRequestEvent":
        """Generate a RebalancePlanRequestEvent by:
        - Resetting MarketDataStore caches to ensure fresh data.
        - Invoking the precompute logic in MultiSleeveAllocator to prepare sleeves for rebalancing.
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
        aum = getattr(self._account_snapshot.account, "adj_equity", None)
        if aum is None or not isinstance(aum, (int, float)) or float(aum) <= 0.0:
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
        lookback_weeks = self.config.precompute_lookback_weeks
        end = now
        start = end - pd.Timedelta(weeks=lookback_weeks)
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

        event = RebalancePlanRequestEvent(
            ts=now.timestamp(),
            rebalance_id=rebalance_id,
            weights=weights,
            source=self.name,
        )
        return event

    @staticmethod
    def _generate_rebalance_id() -> str:
        """Generate a unique rebalance ID."""
        return f"rebalance-{uuid.uuid4()}"
