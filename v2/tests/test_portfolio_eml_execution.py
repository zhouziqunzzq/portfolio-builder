import time
from datetime import datetime
from decimal import Decimal

import pytest

from v2.src.eml.portfolio_eml import PortfolioEMLService
from v2.src.eml.rebalance_execution import (
    RebalanceBuySkipReason,
    RebalanceExecutionSkip,
    RebalanceExecutionStatus,
)
from v2.src.eml.config import EMLConfig
from v2.src.eml.state import PortfolioEMLState
from v2.src.events.event_bus import EventBus
from v2.src.events.events import (
    V2MarketClockEvent,
    V2RebalancePlanRequestEvent,
)
from v2.src.models import AccountSnapshot, PositionSnapshot
from trading_api.exceptions import (
    AuthError,
    InvalidOrder,
    OrderRejected,
    TemporaryUnavailable,
    UnsupportedOrderShape,
)

from v2.tests.fakes import FakeTradingAPI


def test_tradability_check_blocks_non_tradable():
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)
    trading.set_instrument("BBB", tradable=False)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False),
    )

    e = V2RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 0.5, "BBB": 0.5}
    )

    with pytest.raises(RuntimeError, match="Non-tradable"):
        svc._execute_rebalance_plan(e)


def test_sells_before_buys_and_min_order_size_filtering(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.next_order_fill_after = 2
    trading.set_instrument("AAA", tradable=True)
    trading.set_instrument("BBB", tradable=True)

    # Hold AAA ($1000), want to rotate to BBB.
    trading.set_positions(
        [PositionSnapshot(symbol="AAA", qty=10.0, market_value=1000.0)]
    )

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=True, min_order_size_notional=50.0),
    )

    # Avoid real sleeping in wait loop
    monkeypatch.setattr(time, "sleep", lambda _: None)

    e = V2RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 0.0, "BBB": 1.0}
    )
    svc._execute_rebalance_plan(e)

    assert len(trading.submitted) == 2
    assert trading.submitted[0]["side"] == "sell"
    assert trading.submitted[0]["symbol"] == "AAA"
    # Full liquidation prefers qty sells when possible
    assert trading.submitted[0]["notional"] is None
    assert trading.submitted[0]["qty"] is not None
    assert trading.submitted[1]["side"] == "buy"
    assert trading.submitted[1]["symbol"] == "BBB"


def test_near_zero_weights_are_ignored(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    # Weight is tiny -> should be ignored -> no orders
    e = V2RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 1e-12}
    )
    svc._execute_rebalance_plan(e)
    assert trading.submitted == []


def test_min_order_size_filters_small_orders(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)
    trading.set_instrument("BBB", tradable=True)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=50.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    # With equity=1000:
    # - AAA @ 10% => $100 (should trade)
    # - BBB @ 1%  => $10  (should be ignored due to min_order_size=50)
    e = V2RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 0.10, "BBB": 0.01}
    )
    svc._execute_rebalance_plan(e)

    assert len(trading.submitted) == 1
    assert trading.submitted[0]["side"] == "buy"
    assert trading.submitted[0]["symbol"] == "AAA"


@pytest.mark.parametrize("supports_notional_buys", [True, None])
def test_supported_or_unknown_notional_buy_capability_preserves_notional_order(
    monkeypatch, supports_notional_buys
):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument(
        "AAA", tradable=True, supports_notional_buys=supports_notional_buys
    )
    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = svc._execute_rebalance_plan(
        V2RebalancePlanRequestEvent(
            ts=time.time(), rebalance_id="r1", weights={"AAA": 0.5}
        )
    )

    assert result.status is RebalanceExecutionStatus.COMPLETED
    assert result.skips == ()
    assert len(trading.submitted) == 1
    assert trading.submitted[0]["notional"] is not None
    assert trading.submitted[0]["qty"] is None


def test_unsupported_notional_buy_uses_preflighted_whole_share_quantity(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument("AAA", tradable=True, supports_notional_buys=False)
    trading.set_preflight_cost("AAA", quantity=1, estimated_cost=100)
    trading.set_preflight_cost("AAA", quantity=5, estimated_cost=500)
    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = svc._execute_rebalance_plan(
        V2RebalancePlanRequestEvent(
            ts=time.time(), rebalance_id="r1", weights={"AAA": 0.5}
        )
    )

    assert result.status is RebalanceExecutionStatus.COMPLETED
    assert len(trading.submitted) == 1
    assert trading.submitted[0]["notional"] is None
    assert int(trading.submitted[0]["qty"]) == 5
    assert trading.actions.count("preflight_order") == 2
    assert trading.actions.count("get_instrument:AAA") == 1


def test_sub_share_fallback_is_skipped_and_later_buy_executes(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument("AAA", tradable=True, supports_notional_buys=False)
    trading.set_instrument("BBB", tradable=True, supports_notional_buys=True)
    trading.set_preflight_cost("AAA", quantity=1, estimated_cost=100)
    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = svc._execute_rebalance_plan(
        V2RebalancePlanRequestEvent(
            ts=time.time(),
            rebalance_id="r1",
            weights={"AAA": 0.05, "BBB": 0.5},
        )
    )

    assert result.status is RebalanceExecutionStatus.COMPLETED_WITH_SKIPS
    assert result.skips == (
        RebalanceExecutionSkip(
            symbol="AAA",
            desired_notional=Decimal("50.0"),
            reason=RebalanceBuySkipReason.BELOW_ONE_WHOLE_SHARE,
            estimated_unit_cost=Decimal("100"),
        ),
    )
    assert [order["symbol"] for order in trading.submitted] == ["BBB"]


def test_runtime_unsupported_notional_rejection_uses_quantity_fallback(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument("AAA", tradable=True, supports_notional_buys=None)
    trading.set_preflight_cost("AAA", quantity=1, estimated_cost=100)
    trading.set_preflight_cost("AAA", quantity=3, estimated_cost=300)
    trading.set_submit_error(
        "AAA",
        shape="notional",
        error=UnsupportedOrderShape("Amount orders are not allowed"),
    )
    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = svc._execute_rebalance_plan(
        V2RebalancePlanRequestEvent(
            ts=time.time(), rebalance_id="r1", weights={"AAA": 0.3}
        )
    )

    assert result.status is RebalanceExecutionStatus.COMPLETED
    assert len(trading.submitted) == 1
    assert trading.submitted[0]["notional"] is None
    assert int(trading.submitted[0]["qty"]) == 3


def test_quantity_preflight_rejection_skips_only_unsupported_symbol(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument("AAA", tradable=True, supports_notional_buys=False)
    trading.set_instrument("BBB", tradable=True, supports_notional_buys=True)
    trading.set_preflight_error(
        "AAA", shape="quantity", error=InvalidOrder("quantity is not allowed")
    )
    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = svc._execute_rebalance_plan(
        V2RebalancePlanRequestEvent(
            ts=time.time(),
            rebalance_id="r1",
            weights={"AAA": 0.3, "BBB": 0.4},
        )
    )

    assert result.status is RebalanceExecutionStatus.COMPLETED_WITH_SKIPS
    assert result.skips[0].reason is RebalanceBuySkipReason.QUANTITY_PREFLIGHT_REJECTED
    assert [order["symbol"] for order in trading.submitted] == ["BBB"]


def test_quantity_submission_rejection_skips_only_unsupported_symbol(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument("AAA", tradable=True, supports_notional_buys=False)
    trading.set_instrument("BBB", tradable=True, supports_notional_buys=True)
    trading.set_preflight_cost("AAA", quantity=1, estimated_cost=100)
    trading.set_preflight_cost("AAA", quantity=3, estimated_cost=300)
    trading.set_submit_error(
        "AAA", shape="quantity", error=InvalidOrder("quantity is not allowed")
    )
    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = svc._execute_rebalance_plan(
        V2RebalancePlanRequestEvent(
            ts=time.time(),
            rebalance_id="r1",
            weights={"AAA": 0.3, "BBB": 0.4},
        )
    )

    assert result.status is RebalanceExecutionStatus.COMPLETED_WITH_SKIPS
    assert result.skips[0].reason is RebalanceBuySkipReason.QUANTITY_SUBMISSION_REJECTED
    assert [order["symbol"] for order in trading.submitted] == ["BBB"]


@pytest.mark.parametrize(
    "error",
    [
        TemporaryUnavailable("try later"),
        AuthError("bad credentials"),
        OrderRejected("unknown rejection"),
        RuntimeError("unknown broker failure"),
    ],
)
def test_non_deterministic_quantity_preflight_error_remains_execution_failure(
    monkeypatch, error
):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument("AAA", tradable=True, supports_notional_buys=False)
    trading.set_preflight_error("AAA", shape="quantity", error=error)
    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(type(error), match=str(error)):
        svc._execute_rebalance_plan(
            V2RebalancePlanRequestEvent(
                ts=time.time(), rebalance_id="r1", weights={"AAA": 0.3}
            )
        )


def test_whole_share_fallback_honors_cash_buffer(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, cash=250.0, buying_power=250.0))
    trading.set_instrument("AAA", tradable=True, supports_notional_buys=False)
    trading.set_preflight_cost("AAA", quantity=1, estimated_cost=100)
    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(
            include_positions=False,
            min_order_size_notional=1.0,
            cash_buffer_abs=100.0,
        ),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = svc._execute_rebalance_plan(
        V2RebalancePlanRequestEvent(
            ts=time.time(), rebalance_id="r1", weights={"AAA": 0.5}
        )
    )

    assert result.status is RebalanceExecutionStatus.COMPLETED
    assert int(trading.submitted[0]["qty"]) == 1


def test_execute_pending_marks_state_executed(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state = PortfolioEMLState()
    svc.state.pending_rebalance_requests["r1"] = {
        "rebalance_id": "r1",
        "request_ts": time.time(),
        "weights": {"AAA": 1.0},
        "source": "test",
        "correlation_id": "",
    }

    # Pending execution is gated on a known, open market clock.
    svc._market_clock = V2MarketClockEvent(
        ts=time.time(),
        source="test",
        now=datetime.now(),
        is_market_open=True,
    )

    svc._execute_pending_rebalance_plans()
    assert "r1" not in svc.state.pending_rebalance_requests
    entry = next(
        x for x in svc.state.executed_rebalance_history if x.get("rebalance_id") == "r1"
    )
    assert entry["status"] == "completed"
    assert entry["execution_result"] == {"status": "completed", "skips": []}


def test_execute_pending_persists_completed_with_skips_result(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument("AAA", tradable=True, supports_notional_buys=False)
    trading.set_preflight_cost("AAA", quantity=1, estimated_cost=100)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state = PortfolioEMLState()
    svc.state.pending_rebalance_requests["r1"] = {
        "rebalance_id": "r1",
        "request_ts": time.time(),
        "weights": {"AAA": 0.05},
        "source": "test",
        "correlation_id": "",
    }
    svc._market_clock = V2MarketClockEvent(
        ts=time.time(),
        source="test",
        now=datetime.now(),
        is_market_open=True,
    )

    svc._execute_pending_rebalance_plans()

    assert "r1" not in svc.state.pending_rebalance_requests
    entry = next(
        x for x in svc.state.executed_rebalance_history if x.get("rebalance_id") == "r1"
    )
    assert entry["status"] == "completed_with_skips"
    assert entry["execution_result"]["status"] == "completed_with_skips"
    assert entry["execution_result"]["skips"][0]["reason"] == "below_one_whole_share"


def test_execute_pending_skips_when_market_clock_unknown(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state = PortfolioEMLState()
    svc.state.pending_rebalance_requests["r1"] = {
        "rebalance_id": "r1",
        "request_ts": time.time(),
        "weights": {"AAA": 1.0},
        "source": "test",
        "correlation_id": "",
    }

    # No market clock set => should skip execution.
    assert getattr(svc, "_market_clock", None) is None
    svc._execute_pending_rebalance_plans()

    assert "r1" in svc.state.pending_rebalance_requests
    assert svc.state.executed_rebalance_history == []
    assert trading.submitted == []


def test_execute_pending_skips_when_market_closed(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state = PortfolioEMLState()
    svc.state.pending_rebalance_requests["r1"] = {
        "rebalance_id": "r1",
        "request_ts": time.time(),
        "weights": {"AAA": 1.0},
        "source": "test",
        "correlation_id": "",
    }

    svc._market_clock = V2MarketClockEvent(
        ts=time.time(),
        source="test",
        now=datetime.now(),
        is_market_open=False,
        next_market_open=datetime.now(),
    )

    svc._execute_pending_rebalance_plans()

    assert "r1" in svc.state.pending_rebalance_requests
    assert svc.state.executed_rebalance_history == []
    assert trading.submitted == []


def test_execute_pending_retries_then_moves_to_failed(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(
            include_positions=False,
            min_order_size_notional=1.0,
            max_pending_rebalance_execution_retries=2,
        ),
    )

    svc.state = PortfolioEMLState()
    svc.state.pending_rebalance_requests["r1"] = {
        "rebalance_id": "r1",
        "request_ts": time.time(),
        "weights": {"AAA": 1.0},
        "source": "test",
        "correlation_id": "",
        "status": "pending",
        "execution_failures": 0,
    }

    svc._market_clock = V2MarketClockEvent(
        ts=time.time(),
        source="test",
        now=datetime.now(),
        is_market_open=True,
    )

    def _boom(_event):
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "_execute_rebalance_plan", _boom)

    # 1st attempt fails -> still pending
    svc._execute_pending_rebalance_plans()
    assert "r1" in svc.state.pending_rebalance_requests
    assert svc.state.pending_rebalance_requests["r1"].get("execution_failures") == 1
    assert svc.state.executed_rebalance_history == []
    assert svc.state.failed_rebalance_requests == []

    # 2nd attempt fails -> moved to failed, pending cleared
    svc._execute_pending_rebalance_plans()
    assert "r1" not in svc.state.pending_rebalance_requests
    assert svc.state.executed_rebalance_history == []
    assert len(svc.state.failed_rebalance_requests) == 1
    assert svc.state.failed_rebalance_requests[0].get("rebalance_id") == "r1"
    assert svc.state.failed_rebalance_requests[0].get("status") == "failed"


def test_execute_rebalance_plan_cancels_open_orders_before_submitting(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    e = V2RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 1.0}
    )
    svc._execute_rebalance_plan(e)

    assert trading.list_orders_called >= 1
    assert trading.cancel_order_called >= 1
    assert trading.submitted


def test_execute_rebalance_plan_cancels_open_orders_on_execution_error(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_instrument("AAA", tradable=True)
    trading.next_order_final_status = "rejected"
    trading.next_order_fill_after = 1

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    e = V2RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 1.0}
    )

    with pytest.raises(RuntimeError):
        svc._execute_rebalance_plan(e)

    # Cancel-all is attempted both before submitting and after an error.
    assert trading.list_orders_called >= 2
