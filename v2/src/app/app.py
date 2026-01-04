import logging
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import sys
import time
from typing import Dict, List

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from configs import AppConfig
from runtime_manager import RuntimeManager, RuntimeManagerOptions
from states.state_manager import FileStateManager
from events.event_bus import EventBus, EventBusOptions, Subscription
from events.events import BaseEvent
from events.topic import Topic
from iml.base_iml import BaseIMLService
from iml.alpaca_polling_iml import AlpacaPollingIMLService
from eml.base_eml import BaseEML
from eml.alpaca_eml import AlpacaEMLService
from at.base_at import BaseATService
from at.multi_sleeve_at import MultiSleeveATService


def _must_init_otel_metrics() -> None:
    """Initialize OpenTelemetry metrics export if configured.

    This intentionally fails fast when OTLP export is configured but the
    OpenTelemetry SDK/exporter dependencies are missing.

    Environment variables:
    - OTEL_EXPORTER_OTLP_ENDPOINT: e.g. http://otelcol:4317
    - OTEL_SERVICE_NAME: optional; defaults to 'portfolio-builder-v2'
    """

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except Exception as e:
        raise RuntimeError(
            "OpenTelemetry metrics export is configured (OTEL_EXPORTER_OTLP_ENDPOINT is set) "
            "but required OpenTelemetry packages are missing or broken. "
            "Install: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-grpc"
        ) from e

    service_name = os.environ.get("OTEL_SERVICE_NAME", "portfolio-builder-v2")
    resource = Resource.create({"service.name": service_name})

    exporter = OTLPMetricExporter(endpoint=endpoint)
    reader = PeriodicExportingMetricReader(exporter)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)


class App:
    def __init__(
        self,
        config: AppConfig,
        runtime_manager_options: RuntimeManagerOptions = RuntimeManagerOptions(),
        event_bus_options: EventBusOptions = EventBusOptions(),
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = config

        _must_init_otel_metrics()

        # Construct RuntimeManager which constructs common infrastructures
        self.rm = RuntimeManager.from_app_config(
            config,
            options=runtime_manager_options,
        )
        # Event bus
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
        # EML
        self.eml: BaseEML = AlpacaEMLService(
            bus=self.event_bus,
            rm=self.rm,
            config=self.config.eml,
            # Alpaca API credentials loaded from env by default
        )
        # AutoTrader (AT)
        self.at: BaseATService = MultiSleeveATService(
            bus=self.event_bus,
            rm=self.rm,
            config=self.config.at,
        )

        # Construct StateManager last to make sure all stateful components are registered
        self.state_manager = FileStateManager(
            runtime_manager=self.rm,
            state_file=self.config.runtime.state_file,
        )
        self.state_persist_interval_secs = (
            self.config.runtime.state_persist_interval_secs
        )

    def _setup_graceful_shutdown(self) -> asyncio.Event:
        self._stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, lambda: self._stop_event.set())
        return self._stop_event

    async def _handle_graceful_shutdown(
        self,
        tasks: List[asyncio.Task],
        subscriptions: Dict[str, Subscription],
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

        # Cleanup subscriptions
        for svc, sub in subscriptions.items():
            self.log.debug(f"Closing subscription for {svc}...")
            await sub.close()

    async def _run_periodic_state_persistence(self, sub: "Subscription") -> None:
        """
        Periodically persist runtime state to disk.
        Exits cleanly on STOP event.
        """
        try:
            while True:
                try:
                    e = await asyncio.wait_for(
                        sub.next(),
                        timeout=self.state_persist_interval_secs,
                    )
                    sub.task_done()
                    if e.topic == Topic.STOP:
                        self.log.debug(
                            "STOP event received; exiting state persistence task."
                        )
                        break
                except asyncio.TimeoutError:
                    self.state_manager.save_state()
                    self.log.debug("Periodic state persistence completed.")
        except asyncio.CancelledError:
            self.log.debug("Periodic state persistence task cancelled.")
            pass

    async def run(self):
        self.log.info("App started.")

        # Load persisted state
        state_loaded = self.state_manager.load_state()
        if not state_loaded:
            self.log.info("No persisted state loaded; starting fresh.")
        else:
            self.log.info("Persisted state loaded successfully.")

        # Setup graceful shutdown handler
        self._setup_graceful_shutdown()

        # Initialize service tasks and subscriptions
        subs: Dict[str, Subscription] = {
            "StatePersistence": self.event_bus.subscribe(
                topics={Topic.STOP},
            ),
            "IML": self.event_bus.subscribe(
                topics={Topic.STOP},
            ),
            "EML": self.event_bus.subscribe(
                topics={
                    Topic.STOP,
                    Topic.MARKET_CLOCK,
                    Topic.REBALANCE_PLAN,
                },
            ),
            "AT": self.event_bus.subscribe(
                topics={
                    Topic.STOP,
                    Topic.MARKET_CLOCK,
                    Topic.BAR,
                    Topic.ACCOUNT,
                    Topic.REBALANCE_PLAN,
                },
            ),
        }
        tasks = [
            asyncio.create_task(
                self._run_periodic_state_persistence(
                    sub=subs["StatePersistence"],
                ),
                name="StatePersistence",
            ),
            asyncio.create_task(
                self.iml.run(
                    sub=subs["IML"],
                ),
                name="IML",
            ),
            asyncio.create_task(
                self.eml.run(
                    sub=subs["EML"],
                ),
                name="EML",
            ),
            asyncio.create_task(
                self.at.run(
                    sub=subs["AT"],
                ),
                name="AT",
            ),
        ]

        # Handle graceful shutdown
        await self._handle_graceful_shutdown(tasks, subscriptions=subs)

        # Persist state on shutdown
        self.state_manager.save_state()
        self.log.info("State saved on shutdown.")

        self.log.info("App finished.")
