from abc import ABC, abstractmethod


from pathlib import Path
import sys

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.event_bus import EventBus
from events.events import MarketClockEvent, NewBarsEvent

from services.base_service import BaseService


class BaseIMLService(BaseService, ABC):
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
        super().__init__(bus=bus, name=name)
        self.bar_interval = bar_interval

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
