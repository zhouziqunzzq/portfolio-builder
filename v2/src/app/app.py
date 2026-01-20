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
from events.event_bus import EventBus, EventBusOptions
from events.events import BaseEvent, StopEvent
from events.topic import Topic
from iml.base_iml import BaseIMLService
from iml.alpaca_polling_iml import AlpacaPollingIMLService
from eml.base_eml import BaseEML
from eml.portfolio_eml import PortfolioEMLService
from at.base_at import BaseATService
from at.multi_sleeve_at import MultiSleeveATService


def _must_init_otel_metrics() -> None:
    """Initialize OpenTelemetry metrics export if configured.

    This intentionally fails fast when OTLP export is configured but the
    OpenTelemetry SDK/exporter dependencies are missing.

    Environment variables (subset):
    - OTEL_EXPORTER_OTLP_ENDPOINT: e.g. otelcol:4317 (grpc) or http://otelcol:4317 (will be normalized)
    - OTEL_EXPORTER_OTLP_PROTOCOL: if set, must be 'grpc' (this code uses the gRPC exporter)
    - OTEL_METRICS_EXPORTER: if set to 'none', metrics export is disabled
    - OTEL_SERVICE_NAME: optional; defaults to 'portfolio-builder-v2'
    - OTEL_RESOURCE_ATTRIBUTES: optional comma-separated k=v pairs (e.g. deployment.environment=live)
    """

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    metrics_exporter_raw = os.environ.get("OTEL_METRICS_EXPORTER")
    metrics_exporter = (
        "otlp" if metrics_exporter_raw is None else metrics_exporter_raw.strip().lower()
    )
    if metrics_exporter in {"none"}:
        return
    if metrics_exporter in {""}:
        metrics_exporter = "otlp"
    if metrics_exporter not in {"otlp"}:
        raise RuntimeError(
            f"Unsupported OTEL_METRICS_EXPORTER={metrics_exporter!r}; this app only supports 'otlp' or 'none'."
        )

    protocol = (os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "grpc").strip().lower()
    if protocol not in {"grpc"}:
        raise RuntimeError(
            f"Unsupported OTEL_EXPORTER_OTLP_PROTOCOL={protocol!r}; this app only supports 'grpc'."
        )

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

    def _parse_resource_attributes(raw: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for part in (raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            out[k] = v
        return out

    service_name = os.environ.get("OTEL_SERVICE_NAME", "portfolio-builder-v2")
    attrs = _parse_resource_attributes(os.environ.get("OTEL_RESOURCE_ATTRIBUTES", ""))
    # Ensure service.name is set deterministically.
    attrs["service.name"] = service_name
    resource = Resource.create(attrs)

    endpoint_s = endpoint.strip()
    insecure: bool
    if endpoint_s.startswith("http://"):
        endpoint_s = endpoint_s[len("http://") :]
        insecure = True
    elif endpoint_s.startswith("https://"):
        endpoint_s = endpoint_s[len("https://") :]
        insecure = False
    else:
        # Within Docker networks we typically use insecure gRPC.
        insecure = True

    exporter = OTLPMetricExporter(endpoint=endpoint_s, insecure=insecure)
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
            # Market Data API credentials loaded from env by default
        )
        # EML
        self.eml: BaseEML = PortfolioEMLService(
            bus=self.event_bus,
            rm=self.rm,
            config=self.config.eml,
            # Broker API credentials loaded from env by default
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
    ) -> None:
        if not tasks:
            return

        # Wait for OS shutdown signal, then publish STOP ONCE.
        await self._stop_event.wait()
        self.log.info("Shutdown signal received; cancelling tasks...")
        await self.event_bus.publish(StopEvent(ts=time.time(), source="APP"))

        # Let tasks drain/exit; then cancel if anything is stuck
        _, pending = await asyncio.wait(
            tasks, timeout=self.config.runtime.graceful_shutdown_timeout_secs
        )
        for t in pending:
            self.log.warning(f"Task {t.get_name()} did not exit in time; cancelling...")
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Cleanup subscriptions
        await self.event_bus.close_all_subscriptions()

    async def _run_periodic_state_persistence(self) -> None:
        """
        Periodically persist runtime state to disk.
        Exits cleanly on STOP event.
        """
        # Subscribe to STOP
        sub = self.event_bus.subscribe(topics={Topic.SYSTEM_STOP})
        self.log.debug("State persistence task subscribed to STOP topic.")

        try:
            while True:
                try:
                    e = await asyncio.wait_for(
                        sub.next(),
                        timeout=self.state_persist_interval_secs,
                    )
                    sub.task_done()
                    if e.topic == Topic.SYSTEM_STOP:
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

        # Initialize service tasks
        # Note: Subscriptions are handled within each service / task.
        tasks = [
            asyncio.create_task(
                self._run_periodic_state_persistence(),
                name="StatePersistence",
            ),
            asyncio.create_task(
                self.iml.run(),
                name="IML",
            ),
            asyncio.create_task(
                self.eml.run(),
                name="EML",
            ),
            asyncio.create_task(
                self.at.run(),
                name="AT",
            ),
        ]

        # Handle graceful shutdown
        await self._handle_graceful_shutdown(tasks)

        # Persist state on shutdown
        self.state_manager.save_state()
        self.log.info("State saved on shutdown.")

        self.log.info("App finished.")
