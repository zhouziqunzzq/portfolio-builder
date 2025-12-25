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
from events.events import MarketClockEvent, NewBarsEvent


class BaseIMLService(ABC):
    """
    Base class for all Information Market Link (IML) services.

    Responsibilities:
    - Own market time & session state
    - Fetch or receive market data
    - Emit CLOCK / BAR (and optionally QUOTE) events
    - Exit cleanly on STOP broadcast

    Non-responsibilities:
    - Strategy logic
    - Portfolio state
    - Broker interaction
    """

    def __init__(
        self,
        bus: "EventBus",
        bar_interval: str,
        name: str = "IML",
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.bus = bus
        self.bar_interval = bar_interval
        self.name = name

        self._running = False

    async def run(self, sub: "Subscription") -> None:
        """
        Main entrypoint.

        Concrete implementations should implement:
          - _run_loop()  (polling / streaming loop)
        """
        self._running = True
        await self._on_startup()

        run_task = asyncio.create_task(self._run_loop(), name=f"{self.name}.loop")

        try:
            while True:
                e = await sub.next()
                sub.task_done()
                if e.topic == Topic.STOP:
                    break
        finally:
            self._running = False
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            await self._on_shutdown()

    @abstractmethod
    async def _run_loop(self) -> None:
        """
        Main IML loop.

        Typical patterns:
        - polling loop (sleep → fetch → emit)
        - streaming loop (await feed → emit)
        """
        raise NotImplementedError

    @abstractmethod
    async def get_market_clock(self) -> "MarketClockEvent":
        """
        Return authoritative market clock snapshot.
        Must NOT block for long.
        """
        raise NotImplementedError

    @abstractmethod
    async def check_new_bars(self) -> bool:
        """
        Fetch newly CLOSED bars since last call.
        Return True if new bars are available.
        Note: Bars will be separately fetched from common runtime -> market data store.
            This method may refresh market data store and return True to indicate new data is present.
        """
        raise NotImplementedError

    # Event emitters

    async def emit_market_clock(self, clock: "MarketClockEvent") -> None:
        await self.bus.publish(clock)

    async def emit_new_bars(self, new_bars_event: "NewBarsEvent") -> None:
        await self.bus.publish(new_bars_event)

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
