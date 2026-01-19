from abc import ABC, abstractmethod
from pathlib import Path
import sys
from typing import Set

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.topic import Topic
from events.event_bus import EventBus
from events.events import (
    BaseEvent,
    RebalancePlanRequestEvent,
    PositionCleanupPlanRequestEvent,
)

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

    @property
    def subscription_topics(self) -> Set[Topic]:
        topics = {
            Topic.MARKET_CLOCK,  # All ATs need market clock updates
            Topic.ACCOUNT,  # All ATs need account updates
        }

        return super().subscription_topics.union(topics)

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

    async def emit_position_cleanup_plan_request(
        self, plan_request: "PositionCleanupPlanRequestEvent"
    ) -> None:
        await self.bus.publish(plan_request)
