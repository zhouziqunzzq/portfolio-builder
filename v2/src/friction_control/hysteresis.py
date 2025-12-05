import pandas as pd
import numpy as np


def apply_weight_hysteresis_row(
    w_prev: pd.Series,
    w_new: pd.Series,
    dw_min: float = 0.01,
    keep_cash: bool = True,
) -> pd.Series:
    """
    Apply hysteresis to weight changes between previous and new weights.

    Parameters
    ----------
    w_prev : pd.Series
        Previous weights indexed by asset.
    w_new : pd.Series
        New target weights indexed by asset.
    dw_min : float
        Minimum weight change threshold to trigger an update.
    keep_cash : bool
        If True, allow the sum of weights to be less than 1 (i.e., keep cash position).
        Will still normalize if total weight exceeds 1.

    Returns
    -------
    pd.Series
        Adjusted weights after applying hysteresis.
    """
    # Align indexes
    all_tickers = w_prev.index.union(w_new.index)
    w_prev = w_prev.reindex(all_tickers, fill_value=0.0)
    w_new = w_new.reindex(all_tickers, fill_value=0.0)

    # Use a small tolerance to avoid floating-point precision issues
    diff = (w_new - w_prev).abs()
    # Elementwise tolerance: eps * max(1, diff)
    tol = np.finfo(float).eps * diff.where(diff >= 1.0, 1.0)
    change_mask = (diff + tol) >= dw_min
    w_adj = w_prev.copy()
    w_adj[change_mask] = w_new[change_mask]

    # Normalize if total weight exceeds 1 or if not keeping cash
    total_weight = w_adj.sum()
    if (not keep_cash) or (total_weight > 1.0):
        w_adj /= total_weight

    return w_adj


def apply_weight_hysteresis_matrix(
    W: pd.DataFrame,
    dw_min: float = 0.01,
    keep_cash: bool = True,
) -> pd.DataFrame:
    """
    Apply hysteresis to a DataFrame of weights over multiple time periods.

    Parameters
    ----------
    W : pd.DataFrame
        DataFrame of weights with time index and asset columns.
    dw_min : float
        Minimum weight change threshold to trigger an update.
    keep_cash : bool
        If True, allow the sum of weights to be less than 1 (i.e., keep cash position).
        Will still normalize if total weight exceeds 1.

    Returns
    -------
    pd.DataFrame
        Adjusted weights after applying hysteresis.
    """
    W_adj = W.copy()

    for t in range(1, W.shape[0]):
        w_prev = W_adj.iloc[t - 1]
        w_new = W.iloc[t]
        w_adj = apply_weight_hysteresis_row(
            w_prev, w_new, dw_min=dw_min, keep_cash=keep_cash
        )
        W_adj.iloc[t] = w_adj

    return W_adj