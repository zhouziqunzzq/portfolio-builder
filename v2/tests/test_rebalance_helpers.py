import pytest
import pandas as pd

from v2.src.sleeves.common.rebalance_helpers import should_rebalance


def test_should_rebalance_table_driven():
    """Table-driven tests for `should_rebalance` covering boolean results and errors."""

    cases = [
        {
            "name": "never_rebalanced",
            "last": None,
            "current": pd.Timestamp("2025-12-18"),
            "freq": "D",
            "expected": True,
        },
        {
            "name": "invalid_freq",
            "last": pd.Timestamp("2025-12-01"),
            "current": pd.Timestamp("2025-12-02"),
            "freq": "X",
            "raises": ValueError,
        },
        {
            "name": "current_before_last",
            "last": pd.Timestamp("2025-12-03"),
            "current": pd.Timestamp("2025-12-02"),
            "freq": "D",
            "raises": ValueError,
        },
        {
            "name": "daily_same_day",
            "last": pd.Timestamp("2025-12-18 10:00"),
            "current": pd.Timestamp("2025-12-18 17:00"),
            "freq": "D",
            "expected": False,
        },
        {
            "name": "daily_next_day",
            "last": pd.Timestamp("2025-12-18 10:00"),
            "current": pd.Timestamp("2025-12-19 09:00"),
            "freq": "D",
            "expected": True,
        },
        {
            "name": "weekly_same_week",
            "last": pd.Timestamp("2025-12-15"),
            "current": pd.Timestamp("2025-12-17"),
            "freq": "W",
            "expected": False,
        },
        {
            "name": "weekly_next_week",
            "last": pd.Timestamp("2025-12-15"),
            "current": pd.Timestamp("2025-12-22"),
            "freq": "W",
            "expected": True,
        },
        {
            "name": "weekly_cross_year",
            "last": pd.Timestamp("2024-12-30"),
            "current": pd.Timestamp("2025-01-02"),
            "freq": "W",
            "expected": False,
        },
        {
            "name": "monthly_same_month",
            "last": pd.Timestamp("2025-11-01"),
            "current": pd.Timestamp("2025-11-15"),
            "freq": "M",
            "expected": False,
        },
        {
            "name": "monthly_next_month",
            "last": pd.Timestamp("2025-11-01"),
            "current": pd.Timestamp("2025-12-01"),
            "freq": "M",
            "expected": True,
        },
    ]

    for case in cases:
        last = case["last"]
        current = case["current"]
        freq = case["freq"]

        if "raises" in case:
            with pytest.raises(case["raises"]):
                should_rebalance(last, current, freq)
            continue

        expected = case["expected"]
        result = should_rebalance(last, current, freq)
        try:
            assert result is expected
        except AssertionError as e:
            raise AssertionError(
                f"Case '{case['name']}' failed: expected {expected}, got {result}"
            ) from e
