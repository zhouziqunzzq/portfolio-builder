from __future__ import annotations
import numpy as np
import pandas as pd
from pyparsing import Iterable, Mapping


# ------------------------------------------------------------
# 1) Cross-sectional z-score
# ------------------------------------------------------------
def zscore_cross_sectional(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mean = s.mean()
    std = s.std()
    if std is None or std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


# ------------------------------------------------------------
# 2) Winsorized z-score
# ------------------------------------------------------------
def winsorized_zscore(
    s: pd.Series,
    clip: float | None = None,
) -> pd.Series:
    z = zscore_cross_sectional(s)
    if clip is not None and clip > 0:
        z = z.clip(lower=-clip, upper=clip)
    return z


# ------------------------------------------------------------
# 3) Apply sector mask: keep only rows where price >= min_price
#    or ADV >= threshold, etc.
# ------------------------------------------------------------
def apply_boolean_mask(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """
    df: DataFrame indexed by ticker
    mask: Boolean Series of same index
    """
    keep = mask.reindex(df.index).fillna(False)
    return df.loc[keep]


# ------------------------------------------------------------
# 4) Normalize weights to sum=1
# ------------------------------------------------------------
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
    if stock_score_mat.empty:
        # preserve date index, but no sectors
        return pd.DataFrame(index=stock_score_mat.index)

    # Map tickers -> sector
    smap = pd.Series(sector_map)
    # keep tickers that exist in the matrix
    smap = smap.reindex(stock_score_mat.columns)

    # drop invalid / missing sectors
    mask_valid = smap.notna() & ~smap.astype(str).str.strip().isin(invalid_labels)
    if not mask_valid.any():
        return pd.DataFrame(index=stock_score_mat.index)

    valid_tickers = smap.index[mask_valid]
    smap_valid = smap[mask_valid]

    # restrict matrix to valid tickers
    sub = stock_score_mat.reindex(columns=valid_tickers)

    # Long form: (date, ticker) -> score
    long = sub.stack(dropna=True)  # drop NaNs in scores
    if long.empty:
        return pd.DataFrame(index=stock_score_mat.index)

    long.index.set_names(["date", "ticker"], inplace=True)
    df_long = long.to_frame(name="score").reset_index()

    # attach sector
    df_long["sector"] = df_long["ticker"].map(smap_valid.to_dict())
    df_long = df_long.dropna(subset=["sector"])

    if df_long.empty:
        return pd.DataFrame(index=stock_score_mat.index)

    # group by (date, sector) -> mean score
    sector_scores = (
        df_long.groupby(["date", "sector"])["score"]
        .mean()
        .unstack("sector")
        .sort_index()
    )

    # Ensure we have the full original date index (fill missing dates with NaNs)
    sector_scores = sector_scores.reindex(stock_score_mat.index)

    return sector_scores
