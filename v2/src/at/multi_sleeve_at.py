import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Optional

import pandas as pd


_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from .base_at import BaseATService
from .config import ATConfig
from runtime_manager import RuntimeManager
from events.events import BaseEvent, MarketClockEvent
from events.event_bus import EventBus
from allocator.multi_sleeve_allocator import MultiSleeveAllocator
from utils.tz import to_canonical_eastern_naive


class MultiSleeveATService(BaseATService):
    """Minimal multi-sleeve AutoTrader service.

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

        # TODO: Define and add state

        # Internal caches
        self._market_clock: Optional[MarketClockEvent] = None

    async def _run_loop(self) -> None:
        """Background loop (currently idle)."""

        while self._running:
            try:
                now_native = datetime.now().astimezone()
                # Convert to tz-naive US/Eastern wall time
                now = to_canonical_eastern_naive(pd.Timestamp(now_native))
                should_rebalance = await self._check_should_rebalance(now=now)
                self.log.debug("Rebalance check: should_rebalance=%s", should_rebalance)
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

        if isinstance(event, MarketClockEvent):
            self._market_clock = event
            self.log.debug(
                "Stored market clock: now=%s is_open=%s next_market_open=%s next_market_close=%s",
                event.now,
                event.is_market_open,
                event.next_market_open,
                event.next_market_close,
            )

    async def _check_should_rebalance(self, now: Optional[datetime] = None) -> bool:
        """Check if a rebalance should be triggered.

        Currently a stub that always returns False.
        """
        if now is None:
            now_native = datetime.now().astimezone()
            now = to_canonical_eastern_naive(pd.Timestamp(now_native))

        allocator: MultiSleeveAllocator = self.rm.get("multi_sleeve_allocator")
        if not allocator:
            self.log.warning("MultiSleeveAllocator not found in RuntimeManager")
            return False

        return allocator.should_rebalance(now=now)
