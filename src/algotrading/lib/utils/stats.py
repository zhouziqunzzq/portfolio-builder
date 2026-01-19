from __future__ import annotations
import numpy as np
import pandas as pd
from pyparsing import Iterable, Mapping


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mean = s.mean()
    std = s.std()
    if std is None or std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - mean) / std


def winsorized_zscore(
    s: pd.Series,
    clip: float | None = None,
) -> pd.Series:
    z = zscore(s)
    if clip is not None and clip > 0:
        z = z.clip(lower=-clip, upper=clip)
    return z


def zscore_matrix_column_wise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each row (date) across ALL tickers (column) (no sector grouping).

    df: DataFrame[Date x Ticker]
    """
    mean = df.mean(axis=1)  # mean for each row
    std = df.std(axis=1).replace(0, np.nan)  # std for each row
    return df.sub(mean, axis=0).div(std, axis=0)


def windorized_zscore_matrix_column_wise(
    df: pd.DataFrame,
    clip: float | None = None,
) -> pd.DataFrame:
    z = zscore_matrix_column_wise(df)
    if clip is not None and clip > 0:
        z = z.clip(lower=-clip, upper=clip)
    return z


# Apply sector mask: keep only rows where price >= min_price
#    or ADV >= threshold, etc.
def apply_boolean_mask(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """
    df: DataFrame indexed by ticker
    mask: Boolean Series of same index
    """
    keep = mask.reindex(df.index).fillna(False)
    return df.loc[keep]


def normalize_weights(w: pd.Series) -> pd.Series:
    total = w.sum()
    if total <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    return w / total


def sector_mean_snapshot(
    stock_scores: pd.Series,
    sector_map: Mapping[str, str],
    invalid_labels: Iterable[str] = ("", "Unknown"),
) -> pd.Series:
    """
    Single-date version:
      stock_scores: pd.Series[ticker] -> score
      sector_map:   ticker -> sector

    Returns pd.Series[sector] -> mean(score) over tickers in that sector.
    """
    if stock_scores.empty:
        return pd.Series(dtype=float)

    smap = pd.Series(sector_map)
    # restrict to tickers we have scores for
    smap = smap.reindex(stock_scores.index)

    # drop invalid / missing sectors
    mask_valid = smap.notna() & ~smap.astype(str).str.strip().isin(invalid_labels)
    if not mask_valid.any():
        return pd.Series(dtype=float)

    scores_valid = stock_scores[mask_valid]
    sectors_valid = smap[mask_valid]

    sector_scores = scores_valid.groupby(sectors_valid).mean()
    sector_scores.name = "sector_score"
    return sector_scores


def sector_mean_matrix(
    stock_score_mat: pd.DataFrame,
    sector_map: Mapping[str, str],
    invalid_labels: Iterable[str] = ("", "Unknown"),
) -> pd.DataFrame:
    """
    Vectorized sector scoring:

      stock_score_mat: Date x Ticker matrix of stock scores
      sector_map:      ticker -> sector

    Returns
    -------
    sector_scores: Date x Sector matrix, where each cell is the
    cross-sectional mean of stock_score for that sector on that date.

    NaN stock scores are ignored in the per-sector mean.
    """
    # Fast path: empty input -> preserve date index but no sectors
    if stock_score_mat.empty:
        return pd.DataFrame(index=stock_score_mat.index)

    # Build a Series mapping existing tickers -> sector (None if missing)
    sectors = pd.Series({
        ticker: sector_map.get(ticker, None) for ticker in stock_score_mat.columns
    })

    # Restrict to tickers with a valid sector label
    valid_cols = sectors.dropna().index.tolist()
    if not valid_cols:
        return pd.DataFrame(index=stock_score_mat.index)

    sub = stock_score_mat[valid_cols]
    sectors = sectors[valid_cols]

    # Avoid grouping along axis=1 directly; use transpose trick as in SignalEngine
    stacked = sub.copy()
    stacked.columns = pd.MultiIndex.from_arrays([sectors, stacked.columns])
    sector_scores = stacked.T.groupby(level=0).mean().T

    # Ensure we have the full original date index (fill missing dates with NaNs)
    sector_scores = sector_scores.reindex(stock_score_mat.index)

    return sector_scores
