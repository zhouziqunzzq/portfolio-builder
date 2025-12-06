import pandas as pd
from typing import Tuple


def apply_min_holding_period_row(
    w_prev: pd.Series,
    w_proposed: pd.Series,
    holding_age: pd.Series,
    min_holding_rebalances: int,
    keep_cash: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply a minimum holding period constraint on a single rebalance row.

    Logic:
    - If a position is "young" (holding_age < min_holding_rebalances),
      and we are trying to SELL it (w_proposed < w_prev),
      then we BLOCK the sell and keep the previous weight.
    - Buys / adds are always allowed.
    - Positions that are fully closed have their holding_age reset to 0.
    - New positions start with holding_age = 1.

    Parameters
    ----------
    w_prev : pd.Series
        Previous *effective* weights (after friction) at the last rebalance.
    w_proposed : pd.Series
        Proposed weights for this rebalance (after hysteresis + min-notional).
    holding_age : pd.Series
        Current holding ages in rebalance steps (integers), indexed by ticker.
    min_holding_rebalances : int
        Minimum number of rebalance steps a position must be held
        before it can be reduced (sold/trimmed).
    keep_cash : bool
        If True, allow sum of weights <= 1 and only scale down if > 1.
        If False, normalize weights to sum to 1 if total > 0.

    Returns
    -------
    w_eff : pd.Series
        Effective weights after applying minimum holding period.
    holding_age_next : pd.Series
        Updated holding ages after this rebalance.
    """
    # If no min holding constraint, just pass through
    if min_holding_rebalances <= 0:
        # Update holding_age as if all sells/buys are allowed
        holding_age_next = holding_age.copy()

        opened = (w_prev == 0) & (w_proposed > 0)
        continued = (w_prev > 0) & (w_proposed > 0)
        closed = w_proposed == 0

        holding_age_next[opened] = 1
        holding_age_next[continued] = holding_age_next[continued] + 1
        holding_age_next[closed] = 0

        return w_proposed, holding_age_next

    # Ensure aligned indexes
    idx = w_prev.index.union(w_proposed.index).union(holding_age.index)
    w_prev = w_prev.reindex(idx).fillna(0.0)
    w_prop = w_proposed.reindex(idx).fillna(0.0)
    holding_age = holding_age.reindex(idx).fillna(0).astype(int)

    # Identify sells on young positions
    young = holding_age < min_holding_rebalances
    was_held = w_prev > 0
    is_selling = w_prop < w_prev

    lock_mask = young & was_held & is_selling

    # Apply lock: freeze at previous weights where sells are forbidden
    w_eff = w_prop.copy()
    if lock_mask.any():
        w_eff[lock_mask] = w_prev[lock_mask]

    # Renormalize respecting keep_cash
    total = w_eff.sum()
    if not keep_cash:
        if total > 0:
            w_eff = w_eff / total
    else:
        if total > 1.0:
            w_eff = w_eff / total

    # Update holding ages for next rebalance (based on final effective weights)
    holding_age_next = holding_age.copy()

    opened = (w_prev == 0) & (w_eff > 0)
    continued = (w_prev > 0) & (w_eff > 0)
    closed = w_eff == 0

    holding_age_next[opened] = 1
    holding_age_next[continued] = holding_age_next[continued] + 1
    holding_age_next[closed] = 0

    # print(f"[friction] locked sells for {lock_mask.sum()} tickers")
    return w_eff, holding_age_next
