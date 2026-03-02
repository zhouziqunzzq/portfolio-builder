from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from v2.src.sleeves.trend.trend_sleeve import TrendSleeve


class _FakeMDS:
    def __init__(self, frame: pd.DataFrame, local_only: bool = True):
        self.frame = frame
        self.local_only = local_only
        self.calls: list[dict] = []

    def get_ohlcv(self, **kwargs):
        self.calls.append(kwargs)
        return self.frame


def _make_sleeve_with_mds(mds) -> TrendSleeve:
    sleeve = TrendSleeve.__new__(TrendSleeve)
    sleeve.mds = mds
    return sleeve


def test_get_trading_calendar_raises_when_mds_missing():
    sleeve = TrendSleeve.__new__(TrendSleeve)
    sleeve.mds = None

    with pytest.raises(ValueError, match="mds is not set"):
        sleeve._get_trading_calendar("2026-02-01", "2026-02-10")


def test_get_trading_calendar_returns_empty_index_for_empty_ohlcv():
    empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    mds = _FakeMDS(empty_df)
    sleeve = _make_sleeve_with_mds(mds)

    out = sleeve._get_trading_calendar("2026-02-01", "2026-02-10")

    assert isinstance(out, pd.DatetimeIndex)
    assert len(out) == 0


def test_get_trading_calendar_filters_index_without_length_mismatch():
    idx = pd.date_range("2026-02-01", periods=18, freq="D")
    frame = pd.DataFrame(
        {
            "Open": range(18),
            "High": range(18),
            "Low": range(18),
            "Close": range(18),
            "Volume": [1_000_000] * 18,
        },
        index=idx,
    )
    mds = _FakeMDS(frame)
    sleeve = _make_sleeve_with_mds(mds)

    out = sleeve._get_trading_calendar("2026-02-02", "2026-02-18", interval="1d")

    expected = pd.date_range("2026-02-02", "2026-02-18", freq="D")
    assert out.equals(expected)


def test_get_trading_calendar_normalizes_daily_bounds_and_deduplicates_sorted_index():
    idx = pd.to_datetime(
        [
            "2026-02-03",
            "2026-02-01",
            "2026-02-02",
            "2026-02-02",
            "2026-02-04",
        ]
    )
    frame = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [1, 2, 3, 4, 5],
            "Low": [1, 2, 3, 4, 5],
            "Close": [1, 2, 3, 4, 5],
            "Volume": [100, 100, 100, 100, 100],
        },
        index=idx,
    )
    mds = _FakeMDS(frame, local_only=False)
    sleeve = _make_sleeve_with_mds(mds)

    out = sleeve._get_trading_calendar(
        start=datetime(2026, 2, 2, 12, 30),
        end=datetime(2026, 2, 3, 21, 0),
        interval="1d",
    )

    assert out.equals(pd.DatetimeIndex(["2026-02-02", "2026-02-03"]))
    assert len(mds.calls) == 1
    assert mds.calls[0]["start"] == pd.Timestamp("2026-02-02")
    assert mds.calls[0]["end"] == pd.Timestamp("2026-02-03")
    assert mds.calls[0]["local_only"] is False
