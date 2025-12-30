from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from v2.src.market_data_store import MarketDataStore


def _write_cached_daily_parquet(
    root: str,
    *,
    ticker: str,
    start: str = "2025-01-01",
    end: str = "2025-01-10",
) -> None:
    mds = MarketDataStore(data_root=root, local_only=True)

    idx = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1_000_000,
        },
        index=idx,
    )

    # Write directly into the expected cache path.
    path = mds._data_path(ticker, "1d", True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_get_ohlcv_accepts_tz_aware_start_end(tmp_path):
    # Arrange: cached data exists; no online fetch should be needed.
    _write_cached_daily_parquet(str(tmp_path), ticker="TEST")

    mds = MarketDataStore(data_root=str(tmp_path), local_only=False)

    # tz-aware timestamps (UTC) that would previously raise when compared to
    # tz-naive cached index boundaries.
    start = datetime(2025, 1, 3, 12, 0, tzinfo=timezone.utc)
    end = datetime(2025, 1, 7, 12, 0, tzinfo=timezone.utc)

    # Act
    df = mds.get_ohlcv(
        ticker="TEST",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        local_only=False,
    )

    # Assert
    assert not df.empty
    assert df.index.tz is None
    assert df.index.min() >= pd.Timestamp("2025-01-03")
    assert df.index.max() <= pd.Timestamp("2025-01-07")
