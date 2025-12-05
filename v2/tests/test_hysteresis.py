import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal

from v2.src.friction_control.hysteresis import (
    apply_weight_hysteresis_row,
    apply_weight_hysteresis_matrix,
)


def test_apply_weight_hysteresis_row_table_driven():
    """Table-driven tests for `apply_weight_hysteresis_row`.

    Each case is a dict with inputs and the expected Series output.
    """

    cases = [
        {
            "name": "no_change_below_threshold",
            "w_prev": pd.Series({"A": 0.5, "B": 0.5}),
            "w_new": pd.Series({"A": 0.505, "B": 0.495}),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.5, "B": 0.5}),
        },
        {
            "name": "change_above_threshold",
            "w_prev": pd.Series({"A": 0.5, "B": 0.5}),
            "w_new": pd.Series({"A": 0.52, "B": 0.48}),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.52, "B": 0.48}),
        },
        {
            "name": "new_ticker_added",
            "w_prev": pd.Series({"A": 0.5}),
            "w_new": pd.Series({"A": 0.5, "B": 0.02}),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.5, "B": 0.02}),
        },
        {
            "name": "remove_to_zero",
            "w_prev": pd.Series({"A": 0.5, "B": 0.5}),
            "w_new": pd.Series({"A": 0.5, "B": 0.0}),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.5, "B": 0.0}),
        },
        {
            "name": "normalize_when_sum_gt_one",
            "w_prev": pd.Series({"A": 0.6, "B": 0.4}),
            "w_new": pd.Series({"A": 0.8, "B": 0.3}),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.8 / 1.1, "B": 0.3 / 1.1}),
        },
        {
            "name": "normalize_when_keep_cash_false",
            "w_prev": pd.Series({"A": 0.5, "B": 0.0}),
            "w_new": pd.Series({"A": 0.6, "B": 0.1}),
            "dw_min": 0.01,
            "keep_cash": False,
            "expected": pd.Series({"A": 0.6 / 0.7, "B": 0.1 / 0.7}),
        },
        {
            "name": "edge_threshold_equality_triggers_change",
            "w_prev": pd.Series({"A": 0.2}),
            "w_new": pd.Series({"A": 0.21}),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.21}),
        },
        {
            "name": "edge_threshold_set_to_zero_always_changes",
            "w_prev": pd.Series({"A": 0.2}),
            "w_new": pd.Series({"A": 0.2000000001}),
            "dw_min": 0.0,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.2000000001}),
        },
    ]

    for case in cases:
        w_prev = case["w_prev"]
        w_new = case["w_new"]
        dw_min = case["dw_min"]
        keep_cash = case["keep_cash"]

        result = apply_weight_hysteresis_row(
            w_prev, w_new, dw_min=dw_min, keep_cash=keep_cash
        )

        # Align expected to union of indexes to match function behavior
        all_idx = w_prev.index.union(w_new.index)
        expected = case["expected"].reindex(all_idx, fill_value=0.0)

        # Sort indices to avoid ordering-related diffs
        result_sorted = result.sort_index()
        expected_sorted = expected.sort_index()

        # Use a small tolerance for floating point comparisons
        try:
            assert_series_equal(result_sorted, expected_sorted, atol=1e-12, rtol=1e-12)
        except AssertionError as e:
            # Re-raise with the case name to make failures easier to identify
            raise AssertionError(f"Case '{case['name']}' failed: {e}") from e


def test_apply_weight_hysteresis_matrix_table_driven():
    """Table-driven tests for `apply_weight_hysteresis_matrix` using DataFrames.

    Cases are similar to the row tests but expressed as DataFrames.
    """
    cases = [
        {
            "name": "no_change_below_threshold",
            # row0 = previous weights; row1 = new targets (differences < dw_min -> keep prev)
            "W": pd.DataFrame(
                {"A": [0.5, 0.505], "B": [0.5, 0.495]},
                index=["2020-01-01", "2020-01-02"],
            ),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.DataFrame(
                {"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=["2020-01-01", "2020-01-02"]
            ),
        },
        {
            "name": "change_above_threshold",
            # row0 = previous; row1 = new targets (differences >= dw_min -> update)
            "W": pd.DataFrame(
                {"A": [0.5, 0.52], "B": [0.5, 0.48]}, index=["2020-01-01", "2020-01-02"]
            ),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.DataFrame(
                {"A": [0.5, 0.52], "B": [0.5, 0.48]}, index=["2020-01-01", "2020-01-02"]
            ),
        },
        {
            "name": "new_ticker_added",
            # row0 has only A non-zero; row1 introduces B -> B change >= dw_min -> update B
            "W": pd.DataFrame(
                {"A": [0.5, 0.5], "B": [0.0, 0.02]}, index=["2020-01-01", "2020-01-02"]
            ),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.DataFrame(
                {"A": [0.5, 0.5], "B": [0.0, 0.02]}, index=["2020-01-01", "2020-01-02"]
            ),
        },
        {
            "name": "normalize_when_sum_gt_one",
            # row1 sums to 1.1 and should be normalized
            "W": pd.DataFrame(
                {"A": [0.6, 0.8], "B": [0.4, 0.3]}, index=["2020-01-01", "2020-01-02"]
            ),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.DataFrame(
                {"A": [0.6, 0.8 / 1.1], "B": [0.4, 0.3 / 1.1]},
                index=["2020-01-01", "2020-01-02"],
            ),
        },
        {
            "name": "normalize_when_keep_cash_false",
            # keep_cash False forces normalization even when sum < 1
            "W": pd.DataFrame(
                {"A": [0.5, 0.6], "B": [0.0, 0.1]}, index=["2020-01-01", "2020-01-02"]
            ),
            "dw_min": 0.01,
            "keep_cash": False,
            "expected": pd.DataFrame(
                {"A": [0.5, 0.6 / 0.7], "B": [0.0, 0.1 / 0.7]},
                index=["2020-01-01", "2020-01-02"],
            ),
        },
        {
            "name": "edge_single_row",
            # Single row input should return unchanged
            "W": pd.DataFrame({"A": [0.3], "B": [0.7]}, index=["2020-01-01"]),
            "dw_min": 0.01,
            "keep_cash": True,
            "expected": pd.DataFrame({"A": [0.3], "B": [0.7]}, index=["2020-01-01"]),
        },
        {
            "name": "edge_threshold_set_to_zero_always_changes",
            "W": pd.DataFrame(
                {"A": [0.2, 0.2000000001]}, index=["2020-01-01", "2020-01-02"]
            ),
            "dw_min": 0.0,
            "keep_cash": True,
            "expected": pd.DataFrame(
                {"A": [0.2, 0.2000000001]}, index=["2020-01-01", "2020-01-02"]
            ),
        },
    ]

    for case in cases:
        W = case["W"]
        dw_min = case["dw_min"]
        keep_cash = case["keep_cash"]

        result = apply_weight_hysteresis_matrix(W, dw_min=dw_min, keep_cash=keep_cash)

        expected = case["expected"].reindex(
            index=result.index, columns=result.columns, fill_value=0.0
        )

        # Sort to avoid ordering issues
        result_sorted = result.sort_index(axis=0).sort_index(axis=1)
        expected_sorted = expected.sort_index(axis=0).sort_index(axis=1)

        try:
            assert_frame_equal(result_sorted, expected_sorted, atol=1e-12, rtol=1e-12)
        except AssertionError as e:
            raise AssertionError(f"Case '{case['name']}' failed: {e}") from e
