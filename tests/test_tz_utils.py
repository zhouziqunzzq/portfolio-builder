import pandas as pd
from algotrading.lib.utils.tz import as_eastern, as_utc


def test_as_utc_naive_timestamp():
    ts = pd.Timestamp("2025-12-29 12:00:00")  # naive
    converted = as_utc(ts)
    assert converted.tz is not None
    assert str(converted.tz) == "UTC"
    assert converted.isoformat().startswith("2025-12-29T12:00:00")


def test_as_utc_from_eastern():
    ts = pd.Timestamp("2025-12-29 07:00:00", tz="US/Eastern")
    converted = as_utc(ts)
    assert converted.tz is not None
    assert str(converted.tz) == "UTC"
    # 07:00 Eastern is 12:00 UTC during standard time (EST)
    assert converted.hour == 12


def test_as_eastern_naive_timestamp():
    ts = pd.Timestamp("2025-12-29 12:00:00")
    converted = as_eastern(ts)
    assert converted.tz is not None
    assert "US/Eastern" in str(converted.tz)


def test_as_eastern_from_utc():
    ts = pd.Timestamp("2025-12-29 12:00:00", tz="UTC")
    converted = as_eastern(ts)
    assert "US/Eastern" in str(converted.tz)


def test_as_eastern_handles_nonexistent_dst_time():
    # 2022-03-13 is the US/Eastern DST spring-forward day; 02:xx does not exist.
    ts = pd.Timestamp("2022-03-13 02:20:01.235228")
    converted = as_eastern(ts)
    assert converted.tz is not None
    assert "US/Eastern" in str(converted.tz)
    # With nonexistent='shift_forward', the time is shifted into the first valid hour.
    assert converted.hour == 3
    assert converted.minute == 20


def test_as_eastern_handles_ambiguous_dst_time():
    # 2022-11-06 is the US/Eastern DST fall-back day; 01:xx is ambiguous.
    ts = pd.Timestamp("2022-11-06 01:20:00")
    converted = as_eastern(ts)
    assert converted.tz is not None
    assert "US/Eastern" in str(converted.tz)
    # ambiguous=False chooses standard time (EST, UTC-5).
    assert int(converted.utcoffset().total_seconds()) == -5 * 60 * 60
