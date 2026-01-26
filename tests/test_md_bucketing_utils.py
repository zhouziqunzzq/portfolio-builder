from datetime import datetime, timezone, date

import pytest

from algotrading.lib.market_data.bucketing import floor_time
from algotrading.lib.market_data.bucketing import shift_timeframe
from algotrading.lib.types.market_data import Timeframe, TimeframeUnit


class DummyCal:
    def __init__(self, open_dt: datetime, sess_date: date):
        self._open = open_dt
        self._date = sess_date

    def session_date(self, ts: datetime) -> date:
        return self._date

    def session_open(self, session: date) -> datetime:
        return self._open

    def session_close(self, session: date) -> datetime:
        return self._open.replace(hour=16)


def test_floor_time_minute():
    # 1m
    # 12:07:34 -> 12:07:00
    ts = datetime(2021, 1, 1, 12, 7, 34, tzinfo=timezone.utc)
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    assert floor_time(ts, tf) == datetime(2021, 1, 1, 12, 7, 0, tzinfo=timezone.utc)

    # 15m
    # 12:07:34 -> 12:00:00
    ts = datetime(2021, 1, 1, 12, 7, 34, tzinfo=timezone.utc)
    tf = Timeframe(15, TimeframeUnit.MINUTE)
    assert floor_time(ts, tf) == datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_floor_time_hour():
    # 1h
    # 11:07:34 -> 11:00:00
    ts = datetime(2021, 1, 1, 11, 7, 34, tzinfo=timezone.utc)
    tf = Timeframe(1, TimeframeUnit.HOUR)
    assert floor_time(ts, tf) == datetime(2021, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

    # 2h
    # 12:07:34 -> 12:00:00
    ts = datetime(2021, 1, 1, 12, 7, 34, tzinfo=timezone.utc)
    tf = Timeframe(2, TimeframeUnit.HOUR)
    assert floor_time(ts, tf) == datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_floor_time_day_without_calendar():
    ts = datetime(2021, 1, 4, 16, 0, 0, tzinfo=timezone.utc)
    tf = Timeframe(1, TimeframeUnit.DAY)
    assert floor_time(ts, tf) == datetime(2021, 1, 4, 0, 0, 0, tzinfo=timezone.utc)


def test_floor_time_day_with_calendar():
    ts = datetime(2021, 1, 4, 16, 0, 0, tzinfo=timezone.utc)
    open_dt = datetime(2021, 1, 4, 9, 30, 0, tzinfo=timezone.utc)
    cal = DummyCal(open_dt, date(2021, 1, 4))
    tf = Timeframe(1, TimeframeUnit.DAY)
    assert floor_time(ts, tf, calendar=cal) == open_dt


def test_floor_time_naive_raises():
    ts = datetime(2021, 1, 1, 12, 0, 0)  # naive
    tf = Timeframe(1, TimeframeUnit.MINUTE)
    with pytest.raises(ValueError):
        floor_time(ts, tf)


def test_shift_seconds_and_minutes_and_days():
    ts = datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # shift by seconds
    tf_s = Timeframe(15, TimeframeUnit.SECOND)
    assert shift_timeframe(ts, tf_s, 1) == datetime(
        2021, 1, 1, 12, 0, 15, tzinfo=timezone.utc
    )
    assert shift_timeframe(ts, tf_s, -1) == datetime(
        2021, 1, 1, 11, 59, 45, tzinfo=timezone.utc
    )

    # shift by minutes
    tf_m = Timeframe(5, TimeframeUnit.MINUTE)
    assert shift_timeframe(ts, tf_m, 2) == datetime(
        2021, 1, 1, 12, 10, 0, tzinfo=timezone.utc
    )
    assert shift_timeframe(ts, tf_m, -1) == datetime(
        2021, 1, 1, 11, 55, 0, tzinfo=timezone.utc
    )

    # shift by hours
    tf_h = Timeframe(1, TimeframeUnit.HOUR)
    assert shift_timeframe(ts, tf_h, 3) == datetime(
        2021, 1, 1, 15, 0, 0, tzinfo=timezone.utc
    )
    assert shift_timeframe(ts, tf_h, -2) == datetime(
        2021, 1, 1, 10, 0, 0, tzinfo=timezone.utc
    )

    # shift by days - currently raises
    tf_d = Timeframe(1, TimeframeUnit.DAY)
    with pytest.raises(ValueError):
        shift_timeframe(ts, tf_d, 1)
    with pytest.raises(ValueError):
        shift_timeframe(ts, tf_d, -1)
