from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

from algotrading.lib.types.market_data import Timeframe, TimeframeUnit


class BaseSessionCalendar(ABC):
    """
    Plug point for market calendars.
    You can implement this with pandas_market_calendars, exchange_calendars, or your own.

    Keep this tiny now; expand later.
    """

    @abstractmethod
    def session_date(self, ts: datetime) -> "date":
        """Return the trading session date for this timestamp (exchange tz)."""

    @abstractmethod
    def session_open(self, session: "date") -> datetime:
        """Session open timestamp for that session (exchange tz)."""

    @abstractmethod
    def session_close(self, session: "date") -> datetime:
        """Session close timestamp for that session (exchange tz)."""
