from __future__ import annotations

from abc import ABC, abstractmethod

from pathlib import Path
import sys
from typing import Optional

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.event_bus import EventBus
from events.events import BaseEvent, RebalancePlanRequestEvent, MarketClockEvent

from services.base_service import BaseService


class BaseEML(BaseService, ABC):
    """Base class for all Execution Market Link (EML) services.

    EML responsibilities:
    - Connect to a broker execution API.
    - Execute rebalance requests emitted by AT (currently `RebalancePlanRequestEvent`).
    - Consolidate and relay execution updates (orders/fills) back to the event bus.
    - Relay account state (cash/equity/positions) by publishing events.

    Non-responsibilities:
    - Signal generation / allocator logic
    - Market data acquisition (belongs to IML / MarketDataStore)
    """

    def __init__(
        self,
        bus: "EventBus",
        name: str = "EML",
    ):
        super().__init__(bus=bus, name=name)

        # Internal caches
        self._market_clock: Optional[MarketClockEvent] = None

    @abstractmethod
    async def _run_loop(self) -> None:
        """Main EML loop.

        Typical patterns:
        - keep broker connection alive / handle websocket streams
        - periodically refresh account state
        - poll broker for open order / fill updates
        """

        raise NotImplementedError

    async def _handle_event(self, event: BaseEvent) -> None:
        """Handle incoming events.

        Default behavior:
        - dispatch `RebalancePlanRequestEvent` to `execute_rebalance_plan()`
        - update internal market clock on `MarketClockEvent`
        - ignore other event types

        Concrete implementations may override this if they want to handle
        additional topics.
        """
        self.log.debug(
            "EML event received: topic=%s type=%s source=%s ts=%s",
            getattr(event, "topic", None),
            type(event).__name__,
            getattr(event, "source", ""),
            getattr(event, "ts", None),
        )

        # Handle RebalancePlanRequestEvent
        if isinstance(event, RebalancePlanRequestEvent):
            await self.execute_rebalance_plan(event)
            return

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

        # Default: ignore other events
        self.log.debug(
            "Ignoring event: topic=%s type=%s source=%s ts=%s",
            getattr(event, "topic", None),
            type(event).__name__,
            getattr(event, "source", ""),
            getattr(event, "ts", None),
        )

    @abstractmethod
    async def execute_rebalance_plan(self, event: RebalancePlanRequestEvent) -> None:
        """Execute a rebalance plan request.

        `MultiSleeveATService` emits `RebalancePlanRequestEvent(rebalance_id, weights)`.
        EML should translate target weights into broker orders and submit them.

        Implementations should publish downstream execution/account events
        (order updates, fill updates, updated positions, etc) via the bus.
        """

        raise NotImplementedError

    # Event emitters

    async def emit(self, event: BaseEvent) -> None:
        """Publish any event to the event bus."""

        await self.bus.publish(event)

    async def emit_order_event(self, event: BaseEvent) -> None:
        """Publish an order-related event.

        Note: the event's topic should be `Topic.ORDER`.
        """

        await self.bus.publish(event)

    async def emit_fill_event(self, event: BaseEvent) -> None:
        """Publish a fill-related event.

        Note: the event's topic should be `Topic.FILL`.
        """

        await self.bus.publish(event)

    async def emit_account_event(self, event: BaseEvent) -> None:
        """Publish an account/positions-related event.

        The specific event type is implementation-defined.
        """

        await self.bus.publish(event)
