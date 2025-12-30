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


class BaseATService(ABC):
    """
    Base class for all Auto Trader (AT) services.
    """

    def __init__(
        self,
        bus: "EventBus",
        name: str = "AT",
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.bus = bus
        self.name = name

        self._running = False

    async def run(self, sub: "Subscription") -> None:
        """
        Main entrypoint.
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
                    self.log.exception("Error processing event in AT service")
                finally:
                    sub.task_done()
        finally:
            self._running = False
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            await self._on_shutdown()

    @abstractmethod
    async def _run_loop(self) -> None:
        """
        Main AT loop.
        """
        raise NotImplementedError

    @abstractmethod
    async def _handle_event(self, event: BaseEvent) -> None:
        """
        Handle incoming events.
        """
        raise NotImplementedError

    # Event emitters

    async def emit_rebalance_plan_request(
        self, plan_request: "RebalancePlanRequestEvent"
    ) -> None:
        await self.bus.publish(plan_request)

    # Lifecycle hooks

    async def _on_startup(self) -> None:
        """
        Optional initialization hook.
        """
        self.log.info(f"{self.name} starting up...")

    async def _on_shutdown(self) -> None:
        """
        Optional cleanup hook (close sockets, flush state).
        """
        self.log.info(f"{self.name} stopping...")
