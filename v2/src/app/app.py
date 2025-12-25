import logging
import asyncio
from dataclasses import dataclass
from pathlib import Path
import signal
import sys
import time

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from configs import AppConfig
from runtime_manager import RuntimeManager, RuntimeManagerOptions
from events.event_bus import EventBus, EventBusOptions
from events.events import BaseEvent
from events.topic import Topic
from iml.base_iml import BaseIMLService
from iml.alpaca_polling_iml import AlpacaPollingIMLService


class App:
    def __init__(
        self,
        config: AppConfig,
        runtime_manager_options: RuntimeManagerOptions = RuntimeManagerOptions(),
        event_bus_options: EventBusOptions = EventBusOptions(),
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.rm = RuntimeManager.from_app_config(
            config,
            options=runtime_manager_options,
        )
        self.event_bus = EventBus(
            per_subscriber_queue_size=event_bus_options.per_subscriber_queue_size,
            drop_if_full=event_bus_options.drop_if_full,
            broadcast_topics=event_bus_options.broadcast_topics,
        )
        # IML
        self.iml: BaseIMLService = AlpacaPollingIMLService(
            bus=self.event_bus,
            rm=self.rm,
            config=self.config.iml,
            # Alpaca API credentials loaded from env by default
        )
        # TODO: EML
        # TODO: AutoTrader

    def _setup_graceful_shutdown(self) -> asyncio.Event:
        self._stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, lambda: self._stop_event.set())
        return self._stop_event

    async def _handle_graceful_shutdown(
        self,
        tasks: list[asyncio.Task],
    ) -> None:
        if not tasks:
            return

        # Wait for OS shutdown signal, then publish STOP ONCE.
        await self._stop_event.wait()
        self.log.info("Shutdown signal received; cancelling tasks...")
        await self.event_bus.publish(
            BaseEvent(topic=Topic.STOP, ts=time.time(), source="APP")
        )

        # Let tasks drain/exit; then cancel if anything is stuck
        _, pending = await asyncio.wait(
            tasks, timeout=self.config.runtime.graceful_shutdown_timeout_secs
        )
        for t in pending:
            self.log.warning(f"Task {t.get_name()} did not exit in time; cancelling...")
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self):
        self.log.info("App started.")

        # Setup graceful shutdown handler
        self._setup_graceful_shutdown()

        # Initialize service tasks here
        tasks = [
            asyncio.create_task(
                self.iml.run(
                    sub=self.event_bus.subscribe(
                        topics={Topic.STOP},
                    )
                ),
                name="IML",
            ),
        ]

        # Handle graceful shutdown
        await self._handle_graceful_shutdown(tasks)

        self.log.info("App finished.")
