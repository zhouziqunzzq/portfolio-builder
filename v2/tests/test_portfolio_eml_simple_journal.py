import asyncio
import time
from datetime import datetime
from decimal import Decimal

import pytest

from v2.src.eml.config import EMLConfig
from v2.src.eml.portfolio_eml import (
    AmbiguousOrderOutcome,
    PortfolioEMLService,
)
from v2.src.eml.rebalance_execution import ProcessedRebalanceOrder
from v2.src.eml.state import PortfolioEMLState
from v2.src.events.event_bus import EventBus
from v2.src.events.events import V2MarketClockEvent, V2RebalancePlanRequestEvent
from v2.src.models import AccountSnapshot, PositionSnapshot
from v2.tests.fakes import FakeTradingAPI
from models.trading import OrderState, OrderStatus
from trading_api.exceptions import InvalidOrder, TemporaryUnavailable


def _service(trading: FakeTradingAPI) -> PortfolioEMLService:
    return PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=False, min_order_size_notional=1.0),
    )


def _request(weights: dict[str, float]) -> V2RebalancePlanRequestEvent:
    return V2RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights=weights
    )


def test_filled_sell_is_not_repeated_after_later_buy_fails(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_positions(
        [PositionSnapshot(symbol="AAA", qty=10.0, market_value=1000.0)]
    )
    svc = _service(trading)
    event = _request({"AAA": 0.0, "BBB": 1.0})
    svc.state.remember_pending_rebalance_request(event)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    original_buys = svc._execute_buy_orders_blocking

    def fail_buys(*_args, **_kwargs):
        raise RuntimeError("later buy failed")

    monkeypatch.setattr(svc, "_execute_buy_orders_blocking", fail_buys)
    with pytest.raises(RuntimeError, match="later buy failed"):
        svc._execute_rebalance_plan(event)

    journal = svc.state.pending_rebalance_requests["r1"]["execution_journal"]
    assert len(journal) == 1
    assert journal[0]["symbol"] == "AAA"
    assert journal[0]["fill_kind"] == "full"

    monkeypatch.setattr(svc, "_execute_buy_orders_blocking", original_buys)
    svc._execute_rebalance_plan(event)
    assert [order["symbol"] for order in trading.submitted] == ["AAA", "BBB"]
    assert "list_positions" in trading.actions  # even with routine polling off


def test_whole_share_fallback_is_not_repeated_after_later_failure(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(
        AccountSnapshot(equity=1000.0, cash=1000.0, buying_power=1000.0)
    )
    trading.set_instrument("AAA", supports_notional_buys=False)
    trading.set_instrument("BBB", supports_notional_buys=True)
    trading.set_preflight_cost("AAA", quantity=1, estimated_cost=100)
    trading.set_preflight_cost("AAA", quantity=5, estimated_cost=500)
    trading.set_submit_error("BBB", shape="notional", error=InvalidOrder("bad"))
    svc = _service(trading)
    event = _request({"AAA": 0.5, "BBB": 0.5})
    svc.state.remember_pending_rebalance_request(event)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    with pytest.raises(InvalidOrder):
        svc._execute_rebalance_plan(event)
    journal = svc.state.pending_rebalance_requests["r1"]["execution_journal"]
    assert journal[0]["used_whole_share_fallback"] is True

    trading._submit_errors.clear()
    svc._execute_rebalance_plan(event)
    assert [order["symbol"] for order in trading.submitted] == ["AAA", "BBB"]


def test_journal_suppresses_opposite_side_and_round_trips():
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_positions(
        [PositionSnapshot(symbol="AAA", qty=20.0, market_value=2000.0)]
    )
    svc = _service(trading)
    event = _request({"AAA": 0.5})
    svc.state.remember_pending_rebalance_request(event)
    svc.state.record_processed_rebalance_order(
        ProcessedRebalanceOrder(
            rebalance_id="r1",
            symbol="AAA",
            side="buy",
            client_order_id="client-1",
            broker_order_id="broker-1",
            fill_kind="partial",
            used_whole_share_fallback=False,
            filled_qty=Decimal("2"),
            filled_notional=Decimal("200"),
            recorded_at="2026-09-23T00:00:00+00:00",
        )
    )

    svc.state = PortfolioEMLState.from_dict(svc.state.to_dict())
    svc._execute_rebalance_plan(event)
    assert trading.submitted == []
    assert svc.state.processed_rebalance_symbols("r1") == {"AAA"}


def test_timeout_partial_fill_cancels_remainder_and_returns_final_fill():
    trading = FakeTradingAPI()
    svc = _service(trading)
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def get_order(order_id: str) -> OrderState:
        status = (
            OrderStatus.CANCELED
            if order_id in trading.cancelled_ids
            else OrderStatus.PARTIALLY_FILLED
        )
        return OrderState(
            broker_order_id=order_id,
            status=status,
            filled_qty=Decimal("2"),
        )

    trading.get_order = get_order
    result = svc._wait_for_order_fill(
        "broker-1",
        timeout_seconds=0.5,
        poll_interval_seconds=0.6,
        sleep_fn=sleep,
        now_fn=lambda: clock[0],
    )
    assert result.status == OrderStatus.CANCELED
    assert result.filled_qty == Decimal("2")
    assert trading.cancelled_ids == ["broker-1"]


def test_partial_fill_is_journaled_and_not_retried(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    svc = _service(trading)
    event = _request({"AAA": 1.0})
    svc.state.remember_pending_rebalance_request(event)

    monkeypatch.setattr(
        svc,
        "_wait_for_order_fill",
        lambda order_id, **_kwargs: OrderState(
            broker_order_id=order_id,
            status=OrderStatus.CANCELED,
            filled_qty=Decimal("2"),
        ),
    )
    svc._execute_rebalance_plan(event)
    svc._execute_rebalance_plan(event)

    assert len(trading.submitted) == 1
    entry = svc.state.pending_rebalance_requests["r1"]["execution_journal"][0]
    assert entry["fill_kind"] == "partial"
    assert entry["filled_qty"] == "2"


def test_timed_out_order_without_confirmed_fill_is_ambiguous():
    trading = FakeTradingAPI()
    svc = _service(trading)
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def get_order(order_id: str) -> OrderState:
        status = (
            OrderStatus.CANCELED
            if order_id in trading.cancelled_ids
            else OrderStatus.NEW
        )
        return OrderState(broker_order_id=order_id, status=status)

    trading.get_order = get_order
    with pytest.raises(AmbiguousOrderOutcome, match="unknown fill quantity"):
        svc._wait_for_order_fill(
            "broker-1",
            timeout_seconds=0.5,
            poll_interval_seconds=0.6,
            sleep_fn=sleep,
            now_fn=lambda: clock[0],
        )


def test_timed_out_confirmed_zero_fill_can_retry():
    trading = FakeTradingAPI()
    svc = _service(trading)
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def get_order(order_id: str) -> OrderState:
        status = (
            OrderStatus.CANCELED
            if order_id in trading.cancelled_ids
            else OrderStatus.NEW
        )
        return OrderState(
            broker_order_id=order_id, status=status, filled_qty=Decimal("0")
        )

    trading.get_order = get_order
    with pytest.raises(TimeoutError, match="no fill"):
        svc._wait_for_order_fill(
            "broker-1",
            timeout_seconds=0.5,
            poll_interval_seconds=0.6,
            sleep_fn=sleep,
            now_fn=lambda: clock[0],
        )
    assert trading.cancelled_ids == ["broker-1"]


def test_ambiguous_notional_sell_does_not_try_quantity_fallback():
    trading = FakeTradingAPI()
    trading.set_submit_error(
        "AAA", shape="notional", error=TemporaryUnavailable("lost response")
    )
    svc = _service(trading)

    with pytest.raises(AmbiguousOrderOutcome):
        svc._submit_sell_market_order_prefer_notional(
            symbol="AAA", notional=100, qty_fallback=1
        )
    assert trading.actions.count("submit_order") == 1


def test_old_pending_request_loads_with_empty_journal():
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    svc = _service(trading)
    event = _request({"AAA": 1.0})
    svc.state.remember_pending_rebalance_request(event)
    old_payload = svc.state.to_payload()
    del old_payload["pending_rebalance_requests"]["r1"]["execution_journal"]
    svc.state = PortfolioEMLState.from_dict(
        {"state_key": "eml.portfolio", "schema_version": 2, "payload": old_payload}
    )
    svc._execute_rebalance_plan(event)
    assert [order["symbol"] for order in trading.submitted] == ["AAA"]


def test_ambiguous_submission_fails_without_automatic_retry(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))
    trading.set_submit_error(
        "AAA", shape="notional", error=TemporaryUnavailable("lost response")
    )
    svc = _service(trading)
    svc._market_clock = V2MarketClockEvent(
        ts=time.time(), source="test", now=datetime.now(), is_market_open=True
    )
    event = _request({"AAA": 1.0})
    svc.state.remember_pending_rebalance_request(event)

    svc._execute_pending_rebalance_plans()
    assert "r1" not in svc.state.pending_rebalance_requests
    assert len(svc.state.failed_rebalance_requests) == 1
    assert "manual review required" in svc.state.failed_rebalance_requests[0]["error"]

    asyncio.run(svc.execute_rebalance_plan(event))
    assert "r1" not in svc.state.pending_rebalance_requests
    assert len(svc.state.failed_rebalance_requests) == 1
