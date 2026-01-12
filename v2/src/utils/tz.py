import pandas as pd


def to_canonical_eastern_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert `ts` to tz-naive US/Eastern.

    Convention used across v2 for daily (and coarser) bars:
    - If `ts` is tz-aware, convert to US/Eastern and then drop tzinfo.
    - If `ts` is tz-naive, assume it already represents US/Eastern wall time.

    Returns a tz-naive pandas Timestamp.
    """
    out = pd.Timestamp(pd.to_datetime(ts))
    out = as_eastern(out)
    return out.tz_localize(None)


def as_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Converts a pandas Timestamp to UTC timezone."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def as_eastern(ts: pd.Timestamp) -> pd.Timestamp:
    """Converts a pandas Timestamp to US/Eastern timezone."""
    if ts.tzinfo is None:
        # Localizing a tz-naive timestamp can fail on DST transitions:
        # - spring-forward: non-existent local times (e.g. 02:20)
        # - fall-back: ambiguous local times (e.g. 01:20)
        # For v2's daily-oriented timestamp convention, we prefer a deterministic,
        # non-throwing mapping.
        return ts.tz_localize(
            "US/Eastern",
            # For US/Eastern DST spring-forward, the gap is 1 hour.
            # Using a timedelta preserves the wall-clock minutes/seconds.
            nonexistent=pd.Timedelta(hours=1),
            ambiguous=False,
        )
    return ts.tz_convert("US/Eastern")
