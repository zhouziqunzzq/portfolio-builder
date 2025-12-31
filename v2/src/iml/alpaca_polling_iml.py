from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from utils.tz import to_canonical_eastern_naive

from .base_iml import BaseIMLService
from .config import IMLConfig

from events.events import MarketClockEvent, NewBarsEvent, BarsCheckedEvent
from events.event_bus import EventBus
from runtime_manager import RuntimeManager
from allocator.multi_sleeve_allocator import MultiSleeveAllocator
from market_data_store import MarketDataStore
from states.base_state import BaseState


@dataclass
class PollingIMLState(BaseState):
    STATE_KEY = "iml.polling"
    SCHEMA_VERSION = 1

    # Timestamp of last bar refresh
    last_bar_refresh_time: Optional[datetime] = None

    def to_payload(self) -> Dict[str, Any]:
        last = (
            to_canonical_eastern_naive(self.last_bar_refresh_time).to_pydatetime()
            if self.last_bar_refresh_time is not None
            else None
        )
        return {
            "last_bar_refresh_time": (last.isoformat() if last is not None else None),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PollingIMLState":
        last_bar_refresh_time_str = payload.get("last_bar_refresh_time")
        last_bar_refresh_time = (
            datetime.fromisoformat(last_bar_refresh_time_str)
            if last_bar_refresh_time_str is not None
            else None
        )

        if last_bar_refresh_time is not None:
            last_bar_refresh_time = to_canonical_eastern_naive(
                last_bar_refresh_time
            ).to_pydatetime()
        return cls(
            last_bar_refresh_time=last_bar_refresh_time,
        )

    @classmethod
    def empty(cls) -> "PollingIMLState":
        return cls()


class AlpacaPollingIMLService(BaseIMLService):
    """Polling IML that uses Alpaca's clock endpoint.

    This is intentionally minimal:
    - Polls Alpaca market clock periodically
    - Emits `MarketClockEvent` to the event bus
    - Bars handling is intentionally not implemented yet

    Timezone convention
    -------------------
    This service uses tz-naive US/Eastern timestamps when interacting with
    `MarketDataStore` daily bars (and when persisting `PollingIMLState`).
    """

    def __init__(
        self,
        bus: "EventBus",
        rm: "RuntimeManager",
        *,
        config: Optional[IMLConfig] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: Optional[bool] = None,
        name: str = "AlpacaPollingIML",
    ):
        super().__init__(bus=bus, bar_interval="1d", name=name)
        if config is None:
            self.log.warning(
                "No IMLConfig provided; using default configuration values"
            )
            config = IMLConfig()
        self.config = config
        self.bar_interval = config.bar_interval
        if self.bar_interval != "1d":
            raise ValueError("AlpacaPollingIMLService only supports '1d' bar interval")
        self.rm = rm

        if config.polling_interval_secs <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        self._poll_interval_seconds = float(config.polling_interval_secs)

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

        # State
        # This will be managed by StateManager externally.
        self.state: PollingIMLState = PollingIMLState()

        # Internal market clock cache
        self._last_market_clock: Optional[MarketClockEvent] = None

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
        """Main IML event loop."""

        # Emit one immediately on startup.
        while self._running:
            try:
                # Get current time in local timezone
                now: datetime = datetime.now().astimezone()

                # Check market clock
                clock_event = await self.get_market_clock()
                self.log.debug(
                    f"Polled Alpaca market clock: is_market_open={clock_event.is_market_open}, "
                    f"now={clock_event.now}, "
                    f"next_market_open={clock_event.next_market_open}, "
                    f"next_market_close={clock_event.next_market_close}"
                )
                self._last_market_clock = clock_event
                await self.emit_market_clock(clock_event)

                # Check for new bars
                has_new_bars, bars_checked = await self.check_new_bars(now=now)
                if has_new_bars:
                    new_bars_event = NewBarsEvent(
                        ts=time.time(),
                        source=self.name,
                    )
                    self.log.debug("New bars detected; emitting NewBarsEvent")
                    await self.emit_new_bars(new_bars_event)
                if bars_checked:
                    bars_checked_event = BarsCheckedEvent(
                        ts=time.time(),
                        source=self.name,
                    )
                    await self.emit_bars_checked(bars_checked_event)

                # Sleep until next poll
                self.log.debug(f"Sleeping for {self._poll_interval_seconds} seconds")
                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep running; transient API/network issues are expected.
                self.log.exception("Error in AlpacaPollingIMLService main loop")
                await asyncio.sleep(min(30.0, max(1.0, self._poll_interval_seconds)))

    async def get_market_clock(self) -> MarketClockEvent:
        """Fetch clock from Alpaca (runs sync client in a worker thread)."""

        # alpaca-py is synchronous; avoid blocking the event loop.
        c = await self._run_in_thread(self._trading.get_clock)

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

    async def check_new_bars(self, now: Optional[datetime] = None) -> Tuple[bool, bool]:
        if now is None:
            now = datetime.now().astimezone()

        # Canonicalize to match MarketDataStore's daily OHLCV convention:
        # tz-naive timestamps interpreted as US/Eastern.
        now = to_canonical_eastern_naive(now).to_pydatetime()

        # Check if bars should be fetched
        if not self._should_fetch_new_bars(now):
            return False, False

        # Grab the MarketDataStore and MultiSleeveAllocator
        mds: MarketDataStore = self.rm.get("market_data_store")
        if not mds:
            self.log.error(
                "MarketDataStore not found in RuntimeManager; cannot check new bars"
            )
            raise RuntimeError("MarketDataStore not found")
        allocator: MultiSleeveAllocator = self.rm.get("multi_sleeve_allocator")
        if not allocator:
            self.log.error(
                "MultiSleeveAllocator not found in RuntimeManager; cannot check new bars"
            )
            raise RuntimeError("MultiSleeveAllocator not found")

        # Get all tickers from allocator
        tickers = allocator.get_universe()
        self.log.debug(f"Checking new bars for universe of {len(tickers)} tickers")

        # Invoke MarketDataStore to refresh bars for these tickers
        end_ts = pd.Timestamp(now)
        start_ts = end_ts - pd.Timedelta(weeks=self.config.bar_fetch_lookback_weeks)
        start = start_ts.to_pydatetime()
        end = end_ts.to_pydatetime()
        has_new_bars = await self._run_in_thread(
            self._fetch_new_bars,
            mds,
            tickers,
            start,
            end,
            now=now,
        )

        return has_new_bars, True  # bars_checked=True

    def _should_fetch_new_bars(
        self,
        now: datetime,
    ) -> bool:
        """Check if ALL of the following conditions are met:
        - Market is NOT open (we only fetch closed daily bars when market is closed)
        - Enough time has passed since last bar refresh attempt
        """
        # Check market clock
        market_clock = self._last_market_clock
        if market_clock is None:
            self.log.warning(
                "No cached market clock; skipping new bars fetch for safety"
            )
            return False
        if market_clock.is_market_open:
            self.log.debug("Market is open; skipping new bars fetch")
            return False

        # Check last bar refresh attempt time
        if self.state.last_bar_refresh_time is not None:
            delta_secs = (now - self.state.last_bar_refresh_time).total_seconds()
            if delta_secs < self.config.bar_polling_interval_secs:
                self.log.debug(
                    f"Only {delta_secs} seconds since last bar refresh; "
                    f"waiting for {self.config.bar_polling_interval_secs} seconds"
                )
                return False

        # Market is closed and either never refreshed or enough time has passed
        return True

    def _fetch_new_bars(
        self,
        mds: MarketDataStore,
        tickers: List[str],
        start: datetime,
        end: datetime,
        now: Optional[datetime] = None,
    ) -> bool:
        """Fetch newly closed bars for the given tickers from MarketDataStore.
        Update last bar refresh time in state.
        Naively return True if new bars are found for any ticker.
        """
        # Allow graceful shutdown: if we are stopping, abort without mutating state.
        if self._shutdown_requested():
            self.log.info("Shutdown requested; aborting bar fetch")
            return False

        has_new_bars = False
        for ticker in tickers:
            if self._shutdown_requested():
                self.log.info(
                    "Shutdown requested; aborting bar fetch mid-loop (has_new_bars=%s so far)",
                    has_new_bars,
                )
                return False
            self.log.debug(
                f"Fetching OHLCV data for ticker {ticker} from {start.date()} to {end.date()}"
            )
            df = mds.get_ohlcv(
                ticker=ticker,
                start=start,
                end=end,
                interval=self.bar_interval,
                auto_adjust=True,
                local_only=False,  # force fetch from online
            )
            if df is None or df.empty:
                self.log.debug(f"No OHLCV data for ticker {ticker}")
                continue

            # Check if there are new bars since last refresh time
            last_refresh_time = self.state.last_bar_refresh_time
            if last_refresh_time is not None:
                new_bars = df[df.index > last_refresh_time]
            else:
                new_bars = df

            if not new_bars.empty:
                self.log.debug(
                    f"Found {len(new_bars)} new bars for ticker {ticker} since last refresh"
                )
                has_new_bars = True

        # Update last bar refresh time (only if not shutting down)
        if not self._shutdown_requested():
            self.state.last_bar_refresh_time = to_canonical_eastern_naive(
                now or datetime.now().astimezone()
            ).to_pydatetime()

        return has_new_bars
