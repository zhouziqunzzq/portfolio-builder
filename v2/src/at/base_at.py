from abc import ABC, abstractmethod


from pathlib import Path
import sys

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.event_bus import EventBus
from events.events import BaseEvent, RebalancePlanRequestEvent

from services.base_service import BaseService


class BaseATService(BaseService, ABC):
    """
    Base class for all Auto Trader (AT) services.
    """

    def __init__(
        self,
        bus: "EventBus",
        name: str = "AT",
    ):
        super().__init__(bus=bus, name=name)

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
