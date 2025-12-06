import pandas as pd
import numpy as np
from typing import Optional


def apply_min_trade_notional_row(
    w_prev: pd.Series,
    w_new: pd.Series,
    portfolio_value: float,
    min_trade_abs: float = 10.0,
    min_trade_pct_of_aum: float = 0.0,
    keep_cash: bool = True,
) -> pd.Series:
    """
    Enforce a minimum notional trade size per name.

    If the absolute dollar change for a name is smaller than the threshold,
    we keep the previous weight instead of moving to the target.

    Parameters
    ----------
    w_prev : pd.Series
        Weights *before* this rebalance (actual live weights).
    w_new : pd.Series
        Desired target weights (e.g., after hysteresis).
    portfolio_value : float
        Current portfolio value (pre-trade AUM).
    min_trade_abs : float
        Minimum trade size in dollars. Example: 200.0 means
        we ignore trades smaller than $200.
    min_trade_pct_of_aum : float
        Optional additional threshold as a fraction of AUM.
        Example: 0.002 means 0.2% of AUM. The effective threshold
        will be max(min_trade_abs, min_trade_pct_of_aum * AUM).
    keep_cash : bool
        If True, allow the sum of weights to be less than 1 (i.e., keep cash position).
        Will still normalize if total weight exceeds 1.

    Returns
    -------
    w_eff : pd.Series
        Post-friction effective target weights.
    """
    # Edge case: if all thresholds <= 0, simply return target weights
    if (min_trade_abs <= 0) and (min_trade_pct_of_aum <= 0):
        return w_new

    # Align indices
    all_names = w_prev.index.union(w_new.index)
    w_prev = w_prev.reindex(all_names, fill_value=0.0)
    w_new = w_new.reindex(all_names, fill_value=0.0)

    aum = float(portfolio_value)

    # Compute effective threshold
    thresholds = []
    if min_trade_abs > 0:
        thresholds.append(min_trade_abs)
    if min_trade_pct_of_aum > 0:
        thresholds.append(min_trade_pct_of_aum * aum)
    min_trade = max(thresholds) if thresholds else 0.0

    if min_trade <= 0:
        # Nothing to do, just return target
        return w_new

    # Dollar exposure before/after
    dollar_prev = w_prev * aum
    dollar_tgt = w_new * aum
    dollar_diff = (dollar_tgt - dollar_prev).abs()

    # Only execute trades that are large enough. Add a tiny epsilon-based
    # tolerance to avoid floating-point rounding making a trade just miss
    # the threshold.
    # Tolerance: scale with larger of dollar_diff and the threshold, but ensure
    # a small absolute floor to cover typical floating rounding on decimal fractions.
    eps = np.finfo(float).eps
    tol = eps * np.maximum(dollar_diff, min_trade)
    tol = np.maximum(tol, 1e-12)
    execute_mask = (dollar_diff + tol) >= min_trade

    w_eff = w_prev.copy()
    w_eff[execute_mask] = w_new[execute_mask]

    # Renormalize to keep gross exposure sane if keep_cash is False or if total weight > 1
    if (not keep_cash) or (w_eff.sum() > 1.0):
        total = w_eff.sum()
        if total > 0:
            w_eff = w_eff / total

    # print(f"Portfolio Value: {portfolio_value}, Min Trade: {min_trade}")
    # print(
    #     f"trades<300={ (dollar_diff < 300).sum() }, "
    #     f"trades<500={ (dollar_diff < 500).sum() }, "
    #     f"trades>=500={ (dollar_diff >= 500).sum() }"
    # )

    return w_eff
