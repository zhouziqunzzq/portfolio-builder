import pandas as pd
from pandas.testing import assert_series_equal

from v2.src.context.friction_control import FrictionControlContext
from v2.src.friction_control.friction_control_config import FrictionControlConfig
from v2.src.friction_control.friction_controller import FrictionController


def test_friction_controller_min_holding_period_blocks_sells_until_old_enough():
    cfg = FrictionControlConfig(
        hysteresis_dw_min=0.0,
        min_trade_notional_abs=0.0,
        min_trade_pct_of_aum=0.0,
        min_holding_rebalances=3,
    )
    fc = FrictionController(keep_cash=True, config=cfg)
    ctx = FrictionControlContext(aum=100_000.0)

    # Start with a young position in A
    fc.state.holding_age = pd.Series({"A": 1}, dtype=int)

    w_prev = {"A": 0.5, "B": 0.0}
    w_new = {"A": 0.4, "B": 0.1}

    w_eff_1 = fc.apply(w_prev=w_prev, w_new=w_new, ctx=ctx)
    expected_1 = pd.Series({"A": 0.5, "B": 0.1})
    assert_series_equal(pd.Series(w_eff_1).sort_index(), expected_1.sort_index(), atol=1e-12, rtol=1e-12)

    expected_age_1 = pd.Series({"A": 2, "B": 1}, dtype=int)
    assert_series_equal(fc.state.holding_age.sort_index(), expected_age_1.reindex(fc.state.holding_age.index, fill_value=0).sort_index())

    # Still too young to sell A
    w_prev_2 = w_eff_1
    w_new_2 = {"A": 0.0, "B": 0.1}
    w_eff_2 = fc.apply(w_prev=w_prev_2, w_new=w_new_2, ctx=ctx)
    expected_2 = pd.Series({"A": 0.5, "B": 0.1})
    assert_series_equal(pd.Series(w_eff_2).sort_index(), expected_2.sort_index(), atol=1e-12, rtol=1e-12)

    expected_age_2 = pd.Series({"A": 3, "B": 2}, dtype=int)
    assert_series_equal(fc.state.holding_age.sort_index(), expected_age_2.reindex(fc.state.holding_age.index, fill_value=0).sort_index())

    # Now old enough: allow sell
    w_prev_3 = w_eff_2
    w_new_3 = {"A": 0.0, "B": 0.1}
    w_eff_3 = fc.apply(w_prev=w_prev_3, w_new=w_new_3, ctx=ctx)
    expected_3 = pd.Series({"A": 0.0, "B": 0.1})
    assert_series_equal(pd.Series(w_eff_3).sort_index(), expected_3.sort_index(), atol=1e-12, rtol=1e-12)

    expected_age_3 = pd.Series({"A": 0, "B": 3}, dtype=int)
    assert_series_equal(fc.state.holding_age.sort_index(), expected_age_3.reindex(fc.state.holding_age.index, fill_value=0).sort_index())


def test_friction_controller_uses_ctx_aum_for_min_trade_notional():
    cfg = FrictionControlConfig(
        hysteresis_dw_min=0.0,
        min_trade_notional_abs=25.0,
        min_trade_pct_of_aum=0.0,
        min_holding_rebalances=0,
    )
    fc = FrictionController(keep_cash=True, config=cfg)

    ctx = FrictionControlContext(aum=10_000.0)

    w_prev = {"A": 0.5}
    w_new = {"A": 0.5024}  # $24 change at AUM=10k -> below $25 threshold

    w_eff = fc.apply(w_prev=w_prev, w_new=w_new, ctx=ctx)
    expected = pd.Series({"A": 0.5})
    assert_series_equal(pd.Series(w_eff).sort_index(), expected.sort_index(), atol=1e-12, rtol=1e-12)
