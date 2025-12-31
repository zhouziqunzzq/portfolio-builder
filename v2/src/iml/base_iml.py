from abc import ABC, abstractmethod


from pathlib import Path
import sys
from typing import Tuple

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.event_bus import EventBus
from events.events import MarketClockEvent, NewBarsEvent, BarsCheckedEvent

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
    async def check_new_bars(self) -> Tuple[bool, bool]:
        """
        Fetch newly CLOSED bars since last call.
        Note: Bars will be separately fetched from common runtime -> market data store.
        Returns:
            Tuple[bool, bool]: (has_new_bars, bars_checked)
        """
        raise NotImplementedError

    # Event emitters

    async def emit_market_clock(self, clock: "MarketClockEvent") -> None:
        await self.bus.publish(clock)

    async def emit_new_bars(self, new_bars_event: "NewBarsEvent") -> None:
        await self.bus.publish(new_bars_event)

    async def emit_bars_checked(self, bars_checked_event: "BarsCheckedEvent") -> None:
        await self.bus.publish(bars_checked_event)
