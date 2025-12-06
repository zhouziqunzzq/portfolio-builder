import pandas as pd
from pandas.testing import assert_series_equal

from v2.src.friction_control.min_trade_notional import apply_min_trade_notional_row


def test_min_trade_notional_row_table_driven():
    """Table-driven tests for `min_trade_notional_row`.

    Cases cover: no-threshold, below/at/equal-above threshold, pct-of-aum threshold,
    new ticker introduced, and normalization when `keep_cash` is False.
    """

    aum = 10_000.0

    cases = [
        {
            "name": "no_min_trade_returns_target",
            "w_prev": pd.Series({"A": 0.3}),
            "w_new": pd.Series({"A": 0.35}),
            "portfolio_value": aum,
            "min_trade_abs": 0.0,
            "min_trade_pct": 0.0,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.35}),
        },
        {
            "name": "below_threshold_keeps_prev",
            "w_prev": pd.Series({"A": 0.5}),
            # change = 0.0024 -> $24 < default $25
            "w_new": pd.Series({"A": 0.5024}),
            "portfolio_value": aum,
            "min_trade_abs": 25.0,
            "min_trade_pct": 0.0,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.5}),
        },
        {
            "name": "equal_threshold_executes",
            "w_prev": pd.Series({"A": 0.5}),
            # change = 0.0025 -> $25 == threshold; should execute (with eps tolerance)
            "w_new": pd.Series({"A": 0.5025}),
            "portfolio_value": aum,
            "min_trade_abs": 25.0,
            "min_trade_pct": 0.0,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.5025}),
        },
        {
            "name": "pct_of_aum_threshold_used",
            "w_prev": pd.Series({"A": 0.1}),
            # change = 0.005 -> $50; pct threshold 0.01*AUM = $100 so should NOT execute
            "w_new": pd.Series({"A": 0.105}),
            "portfolio_value": aum,
            "min_trade_abs": 25.0,
            "min_trade_pct": 0.01,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.1}),
        },
        {
            "name": "new_ticker_added_executes",
            "w_prev": pd.Series({"A": 0.5}),
            # B introduced with 2% -> $200 > $25 -> should execute
            "w_new": pd.Series({"A": 0.5, "B": 0.02}),
            "portfolio_value": aum,
            "min_trade_abs": 25.0,
            "min_trade_pct": 0.0,
            "keep_cash": True,
            "expected": pd.Series({"A": 0.5, "B": 0.02}),
        },
        {
            "name": "keep_cash_false_normalizes",
            "w_prev": pd.Series({"A": 0.5, "B": 0.0}),
            "w_new": pd.Series({"A": 0.6, "B": 0.1}),
            "portfolio_value": aum,
            "min_trade_abs": 25.0,
            "min_trade_pct": 0.0,
            "keep_cash": False,
            "expected": pd.Series({"A": 0.6 / 0.7, "B": 0.1 / 0.7}),
        },
    ]

    for case in cases:
        w_prev = case["w_prev"]
        w_new = case["w_new"]
        pv = case["portfolio_value"]
        min_trade_abs = case["min_trade_abs"]
        min_trade_pct = case["min_trade_pct"]
        keep_cash = case["keep_cash"]

        result = apply_min_trade_notional_row(
            w_prev,
            w_new,
            portfolio_value=pv,
            min_trade_abs=min_trade_abs,
            min_trade_pct_of_aum=min_trade_pct,
            keep_cash=keep_cash,
        )

        # Align expected index
        all_idx = w_prev.index.union(w_new.index)
        expected = case["expected"].reindex(all_idx, fill_value=0.0)

        try:
            assert_series_equal(result.sort_index(), expected.sort_index(), atol=1e-12, rtol=1e-12)
        except AssertionError as e:
            raise AssertionError(f"Case '{case['name']}' failed: {e}") from e
