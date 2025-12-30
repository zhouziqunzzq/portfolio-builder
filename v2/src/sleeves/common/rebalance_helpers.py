from typing import Optional
import pandas as pd

import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[2]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from utils.tz import as_eastern

VALID_REBALANCE_FREQS = {"D", "W", "M"}


def should_rebalance(
    last_rebalance_ts: pd.Timestamp | None,
    current_ts: pd.Timestamp,
    rebalance_freq: str,
) -> bool:
    """
    Determines whether a rebalance should occur based on the last rebalance timestamp,
    the current timestamp, and the desired rebalance frequency.

    Args:
        last_rebalance_ts (pd.Timestamp | None): Timestamp of the last rebalance.
        current_ts (pd.Timestamp): Current timestamp.
        rebalance_freq (str): Rebalance frequency ('D', 'W', 'M').

    Returns:
        bool: True if a rebalance should occur, False otherwise.
    """
    if rebalance_freq not in VALID_REBALANCE_FREQS:
        raise ValueError(f"Invalid rebalance frequency: {rebalance_freq}")

    if last_rebalance_ts is None:
        return True  # Always rebalance if never done before

    # Unify timestamps to Eastern Time for comparison
    last_rebalance_ts = as_eastern(last_rebalance_ts)
    current_ts = as_eastern(current_ts)

    if current_ts <= last_rebalance_ts:
        # We don't allow rebalancing "backwards" in time
        raise ValueError("current_ts must be after last_rebalance_ts")

    if rebalance_freq == "D":
        # Daily: Check if the date has changed
        return current_ts.date() > last_rebalance_ts.date()
    elif rebalance_freq == "W":
        # Weekly: Check if the ISO (year, week) has advanced. Use the ISO
        # calendar year+week pair rather than the regular year to correctly
        # handle weeks that cross year boundaries (e.g. 2024-12-30 -> 2025-01-02
        # can be the same ISO week).
        last_iso = last_rebalance_ts.isocalendar()
        curr_iso = current_ts.isocalendar()
        try:
            last_iso_year, last_iso_week = last_iso.year, last_iso.week
            curr_iso_year, curr_iso_week = curr_iso.year, curr_iso.week
        except AttributeError:
            # Fallback for tuple-like results
            last_iso_year, last_iso_week = last_iso[0], last_iso[1]
            curr_iso_year, curr_iso_week = curr_iso[0], curr_iso[1]

        return (curr_iso_year, curr_iso_week) > (last_iso_year, last_iso_week)
    elif rebalance_freq == "M":
        # Monthly: Check if the month has changed
        return (
            current_ts.month > last_rebalance_ts.month
            or current_ts.year > last_rebalance_ts.year
        )

    # Shouldn't reach here
    raise RuntimeError("Unhandled rebalance frequency case")


def infer_approx_rebalance_days(freq: str) -> int:
    """
    Infers the approximate number of days corresponding to a given rebalance frequency string.
    Args:
        freq (str): Rebalance frequency string (e.g., 'D', 'W', 'M').
    Returns:
        int: Approximate number of days for the given frequency.
    Raises:
        ValueError: If the frequency string is invalid.
    """
    if freq not in VALID_REBALANCE_FREQS:
        raise ValueError(f"Invalid rebalance frequency: {freq}")

    if not freq:
        return 21
    f = freq.upper().strip()

    freq_to_days = {
        "D": 1,
        "W": 7,
        "M": 30,
        # "Q": 90,
        # "A": 365,
        # "Y": 365,
    }
    if f in freq_to_days:
        return freq_to_days[f]

    # Should not reach here due to earlier validation
    raise ValueError(f"Cannot infer days for frequency: {freq}")


def get_closest_date_on_or_before(
    date: pd.Timestamp, dates: pd.DatetimeIndex
) -> pd.Timestamp:
    """
    Given a target date and a sorted DatetimeIndex, returns the closest date in the index
    that is on or before the target date.
    Args:
        date (pd.Timestamp): Target date.
        dates (pd.DatetimeIndex): Sorted index of dates.
    Returns:
        pd.Timestamp: Closest date on or before the target date.
    """
    return dates[dates <= date].max()


def get_row_by_closest_date(
    df: Optional[pd.DataFrame], d: pd.Timestamp
) -> Optional[pd.Series]:
    """
    Given a DataFrame with a DatetimeIndex and a target date, returns the row corresponding
    to the closest date on or before the target date.
    Args:
        df (Optional[pd.DataFrame]): DataFrame with DatetimeIndex.
        d (pd.Timestamp): Target date.
    Returns:
        Optional[pd.Series]: Row corresponding to the closest date, or None if not found.
    """
    if df is None or df.empty:
        return None
    # find the closest available date on or before d
    idx = get_closest_date_on_or_before(d, df.index)
    if idx is None:
        return None
    row = df.loc[idx]
    if row.isna().all():
        return None
    return row
