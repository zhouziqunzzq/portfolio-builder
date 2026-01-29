from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, date
from enum import Enum
from typing import Optional

from algotrading.lib.types.trading import InstrumentRef


class TimeframeUnit(str, Enum):
    SECOND = "s"
    MINUTE = "m"
    HOUR = "h"
    DAY = "d"


@dataclass(frozen=True, slots=True)
class Timeframe:
    """
    A bar timeframe.

    For DAY, `n` is typically 1, and bucketing should be session-aware.
    """

    n: int
    unit: TimeframeUnit

    def __str__(self) -> str:
        return f"{self.n}{self.unit.value}"

    @property
    def seconds(self) -> int:
        if self.unit == TimeframeUnit.SECOND:
            return self.n
        if self.unit == TimeframeUnit.MINUTE:
            return self.n * 60
        if self.unit == TimeframeUnit.HOUR:
            return self.n * 3600
        if self.unit == TimeframeUnit.DAY:
            # not a fixed number in market-session terms; return 86400 for utility only
            return self.n * 86400
        raise ValueError(f"Unknown unit: {self.unit}")


@dataclass(frozen=True, slots=True)
class BarKey:
    """
    Identity of a bar. start_ts is the bar start timestamp in exchange tz (or UTC if you normalize).
    """

    ref: InstrumentRef
    tf: Timeframe
    start_ts: datetime  # bar start


@dataclass(frozen=True, slots=True)
class BarBatchKey:
    """
    Identity of a batch of bars for cross-instrument sync.

    E.g. all 1m bars starting at 2024-01-01 09:30:00 for a set of instruments.
    """

    refs: tuple[InstrumentRef, ...]
    tf: Timeframe
    start_ts: datetime  # bar start


@dataclass(frozen=True, slots=True)
class OHLCVBar:
    """
    Minimal OHLCV. Add vwap, trades, etc later if needed.

    Convention: start_ts is bucket start (inclusive); end_ts optional if you want (exclusive).
    An OHLCVBar represents data for [start_ts, end_ts).
    """

    start_ts: datetime
    end_ts: Optional[datetime]
    o: float
    h: float
    l: float
    c: float
    v: float
