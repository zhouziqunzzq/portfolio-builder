from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from .base_iml import BaseIMLService

from events.events import MarketClockEvent
from events.event_bus import EventBus
from runtime_manager import RuntimeManager


class AlpacaPollingIMLService(BaseIMLService):
    """Polling IML that uses Alpaca's clock endpoint.

    This is intentionally minimal:
    - Polls Alpaca market clock periodically
    - Emits `MarketClockEvent` to the event bus
    - Bars handling is intentionally not implemented yet
    """

    def __init__(
        self,
        bus: "EventBus",
        rm: "RuntimeManager",
        *,
        bar_interval: str = "1d",
        poll_interval_seconds: float = 30.0,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: Optional[bool] = None,
        name: str = "AlpacaPollingIML",
    ):
        super().__init__(bus=bus, bar_interval=bar_interval, name=name)
        self.rm = rm

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        self._poll_interval_seconds = float(poll_interval_seconds)

        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self._base_url = base_url or os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if paper is None:
            # Default to paper unless user explicitly overrides.
            env_paper = os.environ.get("ALPACA_PAPER")
            if env_paper is None:
                self._paper = True
            else:
                self._paper = env_paper.strip().lower() in {"1", "true", "yes", "y"}
        else:
            self._paper = bool(paper)

        self._trading = self._build_trading_client()
        self.log.info(
            f"Initialized AlpacaPollingIMLService (paper={self._paper}, base_url={self._base_url})"
        )

        # Register self to RuntimeManager for lifecycle management.
        self.rm.set("iml", self)
        self.rm.set("alpaca_polling_iml", self)  # alias

    def _build_trading_client(self):
        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY (or pass api_key/secret_key)."
            )

        try:
            from alpaca.trading.client import TradingClient
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'alpaca-py'. Install with `pip install alpaca-py`."
            ) from e

        return TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
            url_override=self._base_url,
        )

    async def _run_loop(self) -> None:
        """Poll Alpaca clock and publish `MarketClockEvent` periodically."""

        # Emit one immediately on startup.
        while self._running:
            try:
                clock_event = await self.get_market_clock()
                self.log.debug(
                    f"Polled Alpaca market clock: is_market_open={clock_event.is_market_open}, "
                    f"now={clock_event.now}, "
                    f"next_market_open={clock_event.next_market_open}, "
                    f"next_market_close={clock_event.next_market_close}"
                )
                await self.emit_market_clock(clock_event)
                self.log.debug(f"Sleeping for {self._poll_interval_seconds} seconds")
                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep running; transient API/network issues are expected.
                self.log.exception("Error polling Alpaca market clock")
                await asyncio.sleep(min(30.0, max(1.0, self._poll_interval_seconds)))

    async def get_market_clock(self) -> MarketClockEvent:
        """Fetch clock from Alpaca (runs sync client in a worker thread)."""

        # alpaca-py is synchronous; avoid blocking the event loop.
        c = await asyncio.to_thread(self._trading.get_clock)

        # Alpaca's model is typically: timestamp, is_open, next_open, next_close.
        now = getattr(c, "timestamp", None)
        is_open = bool(getattr(c, "is_open", False))
        next_open = getattr(c, "next_open", None)
        next_close = getattr(c, "next_close", None)

        return MarketClockEvent(
            ts=time.time(),
            source=self.name,
            now=now,
            is_market_open=is_open,
            next_market_open=next_open if not is_open else None,
            next_market_close=next_close if is_open else None,
        )

    async def check_new_bars(self) -> bool:
        # Intentionally ignored for now.
        return False
