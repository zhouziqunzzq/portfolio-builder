import pandas as pd
from v2.src.utils.tz import as_utc, as_eastern


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
