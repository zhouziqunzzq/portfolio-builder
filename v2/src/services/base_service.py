from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import functools
import logging
import threading
from pathlib import Path
import sys
from typing import Any, Callable, Optional

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.event_bus import EventBus, Subscription
from events.topic import Topic
from events.events import BaseEvent


class BaseService(ABC):
    """Common base class for v2 services.

    Provides:
    - Common resources: log, bus, name, _running
    - A canonical run() loop that exits on STOP
    - Threadpool offload helpers that are tracked and joined on shutdown
    """

    def __init__(self, bus: "EventBus", name: str):
        self.log = logging.getLogger(self.__class__.__name__)
        self.bus = bus
        self.name = name

        self._running = False

        # Shutdown coordination for any blocking execution (threads).
        self._shutdown_event = threading.Event()

        # Track threadpool futures created by this service so we can await them
        # during shutdown ("join" semantics). These are asyncio Futures wrapping
        # executor work.
        self._thread_futures: set[asyncio.Future[Any]] = set()

    async def run(self, sub: "Subscription") -> None:
        """Main entrypoint.

        Pattern:
        - call startup hook
        - start background loop task
        - consume bus events until STOP
        - delegate non-STOP events to _handle_event
        - always shutdown cleanly and join in-flight threadpool tasks
        """

        self._running = True
        await self._startup()

        run_task = asyncio.create_task(self._run_loop(), name=f"{self.name}.loop")
        try:
            while True:
                e = await sub.next()
                try:
                    if e.topic == Topic.STOP:
                        break
                    await self._handle_event(e)
                except Exception:
                    self.log.exception("Error processing event in service")
                finally:
                    sub.task_done()
        finally:
            # Signal shutdown early so any in-flight blocking loops can abort.
            try:
                self._shutdown_event.set()
            except Exception:
                pass

            self._running = False
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            await self._shutdown()

    @abstractmethod
    async def _run_loop(self) -> None:
        """Background loop for the service."""

        raise NotImplementedError

    async def _handle_event(self, event: BaseEvent) -> None:
        """Handle incoming events.

        Default: ignore all non-STOP events.
        """

        _ = event
        return

    async def _startup(self) -> None:
        await self._on_startup()

    async def _shutdown(self) -> None:
        # Ensure shutdown is signaled even if called directly.
        try:
            self._shutdown_event.set()
        except Exception:
            pass

        # Invoke shutdown requested hook.
        # Note: invoke here because subclasses may create background tasks
        # in the threadpool that need to be joined below.
        await self._on_shutdown_requested()

        # "Join" any in-flight executor work started by this service.
        try:
            self.log.debug("Awaiting service thread tasks to finish...")
            await self._join_thread_futures(timeout_seconds=10.0)
        except Exception:
            self.log.exception(
                "Error while awaiting service thread tasks during shutdown"
            )

        await self._on_shutdown()

    def _shutdown_requested(self) -> bool:
        """Return True if the service is shutting down."""

        try:
            return self._shutdown_event.is_set()
        except Exception:
            return True

    # Lifecycle hooks

    async def _on_startup(self) -> None:
        """Hook called on service startup."""
        self.log.info(f"{self.name} starting up...")

    async def _on_shutdown(self) -> None:
        """Hook called on service shutdown."""
        self.log.info(f"{self.name} stopping...")

    async def _on_shutdown_requested(self) -> None:
        """Hook called when shutdown is requested (before actual shutdown)."""
        self.log.info(f"{self.name} shutdown requested...")

    # Threadpool helpers

    async def _run_in_thread(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Run a blocking function in the default threadpool and track it.

        Key property: awaiting is shielded so cancellation of the caller does not
        cancel the underlying executor work.
        """

        loop = asyncio.get_running_loop()
        call = functools.partial(fn, *args, **kwargs)
        fut: asyncio.Future[Any] = loop.run_in_executor(None, call)
        self._thread_futures.add(fut)

        # IMPORTANT:
        # If the awaiting task is cancelled (e.g. during shutdown), we still want
        # to keep tracking the underlying executor work so shutdown can "join"
        # it. Therefore we remove futures from the tracking set only when they
        # actually complete.
        fut.add_done_callback(lambda f: self._thread_futures.discard(f))
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            raise

    async def _join_thread_futures(self, *, timeout_seconds: float = 10.0) -> None:
        pending = [f for f in list(self._thread_futures) if not f.done()]
        if not pending:
            self.log.debug("No in-flight service thread tasks to await.")
            return

        self.log.info(
            "Awaiting %d in-flight service thread tasks (timeout=%.1fs)...",
            len(pending),
            float(timeout_seconds),
        )
        done, still_pending = await asyncio.wait(
            pending, timeout=float(timeout_seconds)
        )
        _ = done
        if still_pending:
            self.log.warning(
                "Timed out waiting for %d service thread tasks to finish; proceeding with shutdown.",
                len(still_pending),
            )
