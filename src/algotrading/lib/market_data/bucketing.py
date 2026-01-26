from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from algotrading.lib.types.market_data import Timeframe, TimeframeUnit
from algotrading.lib.calendar.session import BaseSessionCalendar


def floor_time(
    ts: datetime, tf: Timeframe, *, calendar: Optional[BaseSessionCalendar] = None
) -> datetime:
    """
    Compute bucket start for a timestamp.

    - For s/m/h: standard floor to interval boundary.
    - For d: session-aware if calendar provided; otherwise floor to UTC midnight (not ideal for markets).
    """
    if ts.tzinfo is None:
        raise ValueError("ts must be timezone-aware")

    if tf.unit == TimeframeUnit.DAY:
        if calendar is None:
            # fallback: UTC midnight floor (acceptable for crypto; not for equities RTH)
            utc = ts.astimezone(timezone.utc)
            return datetime(utc.year, utc.month, utc.day, tzinfo=timezone.utc)
        sess = calendar.session_date(ts)
        return calendar.session_open(sess)

    # for s/m/h we can floor based on epoch seconds in the same tz
    seconds = tf.seconds
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % seconds)
    return datetime.fromtimestamp(floored, tz=ts.tzinfo)


def shift_timeframe(ts: datetime, tf: Timeframe, n: int) -> datetime:
    """
    Shift timestamp by n timeframes.

    NOTE: Intraday-only MVP. Do NOT use for session-aware daily/weekly bars.
    """
    if tf.unit == TimeframeUnit.DAY:
        raise ValueError(
            "shift_timeframe() is intraday-only; daily bars require session calendar."
        )

    seconds = tf.seconds
    return ts + timedelta(seconds=seconds * n)
