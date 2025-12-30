from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging

from pathlib import Path
import sys

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.event_bus import EventBus, Subscription
from events.topic import Topic
from events.events import BaseEvent, RebalancePlanRequestEvent


class BaseEML(ABC):
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
        self.log = logging.getLogger(self.__class__.__name__)
        self.bus = bus
        self.name = name

        self._running = False

    async def run(self, sub: "Subscription") -> None:
        """Main entrypoint.
        - start a background loop task
        - consume bus events until STOP
        - delegate non-STOP events to `_handle_event`
        - always shutdown cleanly
        """

        self._running = True
        await self._on_startup()

        run_task = asyncio.create_task(self._run_loop(), name=f"{self.name}.loop")

        try:
            while True:
                e = await sub.next()
                try:
                    if e.topic == Topic.STOP:
                        break

                    await self._handle_event(e)
                except Exception:
                    self.log.exception("Error processing event in EML service")
                finally:
                    sub.task_done()
        finally:
            self._running = False
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            await self._on_shutdown()

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

        if isinstance(event, RebalancePlanRequestEvent):
            await self.execute_rebalance_plan(event)
            return

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

    # Lifecycle hooks

    async def _on_startup(self) -> None:
        """Optional initialization hook."""

        self.log.info(f"{self.name} starting up...")

    async def _on_shutdown(self) -> None:
        """Optional cleanup hook (close sockets, flush state)."""

        self.log.info(f"{self.name} stopping...")
