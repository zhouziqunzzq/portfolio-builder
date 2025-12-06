import pandas as pd
from pandas.testing import assert_series_equal

from v2.src.friction_control.min_holding_period import apply_min_holding_period_row


def test_apply_min_holding_period_row_table_driven():
    """Table-driven tests for `apply_min_holding_period_row`.

    Each case includes previous effective weights, proposed weights, holding ages,
    minimum holding rebalances and expected effective weights and next ages.
    """

    cases = [
        {
            "name": "min_zero_passthrough_and_age_updates",
            "w_prev": pd.Series({"A": 0.5, "B": 0.0}),
            "w_proposed": pd.Series({"A": 0.4, "B": 0.1}),
            "holding_age": pd.Series({"A": 3, "B": 0}),
            "min_rebalances": 0,
            "keep_cash": True,
            "expected_w": pd.Series({"A": 0.4, "B": 0.1}),
            "expected_age": pd.Series({"A": 4, "B": 1}),
        },
        {
            "name": "young_sell_locked",
            "w_prev": pd.Series({"A": 0.5}),
            "w_proposed": pd.Series({"A": 0.4}),
            "holding_age": pd.Series({"A": 1}),
            "min_rebalances": 3,
            "keep_cash": True,
            "expected_w": pd.Series({"A": 0.5}),
            "expected_age": pd.Series({"A": 2}),
        },
        {
            "name": "old_position_can_sell",
            "w_prev": pd.Series({"A": 0.5}),
            "w_proposed": pd.Series({"A": 0.4}),
            "holding_age": pd.Series({"A": 4}),
            "min_rebalances": 3,
            "keep_cash": True,
            "expected_w": pd.Series({"A": 0.4}),
            "expected_age": pd.Series({"A": 5}),
        },
        {
            "name": "closed_resets_age",
            "w_prev": pd.Series({"A": 0.5}),
            "w_proposed": pd.Series({"A": 0.0}),
            "holding_age": pd.Series({"A": 5}),
            "min_rebalances": 2,
            "keep_cash": True,
            "expected_w": pd.Series({"A": 0.0}),
            "expected_age": pd.Series({"A": 0}),
        },
        {
            "name": "new_position_starts_age_1",
            "w_prev": pd.Series({"A": 0.0}),
            "w_proposed": pd.Series({"A": 0.1}),
            "holding_age": pd.Series({"A": 0}),
            "min_rebalances": 2,
            "keep_cash": True,
            "expected_w": pd.Series({"A": 0.1}),
            "expected_age": pd.Series({"A": 1}),
        },
        {
            "name": "keep_cash_false_normalizes",
            "w_prev": pd.Series({"A": 0.5, "B": 0.0}),
            "w_proposed": pd.Series({"A": 0.6, "B": 0.5}),
            "holding_age": pd.Series({"A": 2, "B": 0}),
            # use min_rebalances > 0 so we go through the main path (not the early return)
            "min_rebalances": 1,
            "keep_cash": False,
            "expected_w": pd.Series({"A": 0.6 / 1.1, "B": 0.5 / 1.1}),
            "expected_age": pd.Series({"A": 3, "B": 1}),
        },
    ]

    for case in cases:
        w_prev = case["w_prev"]
        w_prop = case["w_proposed"]
        holding_age = case["holding_age"]
        min_reb = case["min_rebalances"]
        keep_cash = case["keep_cash"]

        w_eff, holding_age_next = apply_min_holding_period_row(
            w_prev, w_prop, holding_age, min_reb, keep_cash=keep_cash
        )

        # align indices for comparison
        all_idx = w_prev.index.union(w_prop.index).union(holding_age.index)
        expected_w = case["expected_w"].reindex(all_idx, fill_value=0.0)
        expected_age = case["expected_age"].reindex(all_idx, fill_value=0).astype(int)

        try:
            assert_series_equal(w_eff.sort_index(), expected_w.sort_index(), atol=1e-12, rtol=1e-12)
            assert_series_equal(holding_age_next.sort_index(), expected_age.sort_index())
        except AssertionError as e:
            raise AssertionError(f"Case '{case['name']}' failed: {e}") from e
