import time
from datetime import datetime
from decimal import Decimal

import pytest

from v2.src.eml.config import EMLConfig
from v2.src.eml.portfolio_eml import AmbiguousOrderOutcome, PortfolioEMLService
from v2.src.eml.rebalance_execution import (
    RebalanceBuySkipReason,
    RebalanceExecutionResult,
    RebalanceExecutionSkip,
    RebalanceExecutionStatus,
)
from v2.src.events.event_bus import EventBus
from v2.src.events.events import V2MarketClockEvent, V2RebalancePlanRequestEvent
from v2.tests.fakes import FakeTradingAPI
from models.trading import OrderState, OrderStatus


class RecordingCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, str]]] = []

    def add(self, value: int, attributes: dict[str, str]) -> None:
        self.calls.append((value, attributes))


def _service() -> PortfolioEMLService:
    return PortfolioEMLService(
        bus=EventBus(),
        trading_api=FakeTradingAPI(),
        config=EMLConfig(include_positions=False),
    )


def test_completed_rebalance_metrics_count_status_and_each_skip() -> None:
    svc = _service()
    outcomes = RecordingCounter()
    executed = RecordingCounter()
    skips = RecordingCounter()
    svc._rebalance_executions_counter = outcomes
    svc._executed_rebalance_count_counter = executed
    svc._rebalance_buy_skips_counter = skips

    svc._observe_completed_rebalance(
        RebalanceExecutionResult(status=RebalanceExecutionStatus.COMPLETED)
    )
    svc._observe_completed_rebalance(
        RebalanceExecutionResult(
            status=RebalanceExecutionStatus.COMPLETED_WITH_SKIPS,
            skips=(
                RebalanceExecutionSkip(
                    symbol="AAA",
                    desired_notional=Decimal("50"),
                    reason=RebalanceBuySkipReason.BELOW_ONE_WHOLE_SHARE,
                ),
                RebalanceExecutionSkip(
                    symbol="BBB",
                    desired_notional=Decimal("100"),
                    reason=RebalanceBuySkipReason.UNIT_COST_UNAVAILABLE,
                ),
            ),
        )
    )

    assert [attributes["status"] for _, attributes in outcomes.calls] == [
        "completed",
        "completed_with_skips",
    ]
    assert all(attributes["result"] == "success" for _, attributes in outcomes.calls)
    assert len(executed.calls) == 2
    assert [attributes["reason"] for _, attributes in skips.calls] == [
        "below_one_whole_share",
        "unit_cost_unavailable",
    ]
    assert all(
        set(attributes) == {"service", "reason"} for _, attributes in skips.calls
    )


def test_accepted_partial_fill_is_counted_only_once() -> None:
    svc = _service()
    partials = RecordingCounter()
    svc._rebalance_partial_fills_counter = partials
    svc._trading_api.get_order = lambda order_id: OrderState(
        broker_order_id=order_id,
        status=OrderStatus.CANCELED,
        filled_qty=Decimal("2"),
    )

    result = svc._wait_for_order_fill("broker-1", sleep_fn=lambda _: None)

    assert result.filled_qty == Decimal("2")
    assert partials.calls == [(1, {"service": svc.name, "terminal_status": "canceled"})]


def test_timed_out_partial_fill_counts_after_cancellation() -> None:
    svc = _service()
    partials = RecordingCounter()
    svc._rebalance_partial_fills_counter = partials
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def get_order(order_id: str) -> OrderState:
        status = (
            OrderStatus.CANCELED
            if order_id in svc._trading_api.cancelled_ids
            else OrderStatus.PARTIALLY_FILLED
        )
        return OrderState(
            broker_order_id=order_id, status=status, filled_qty=Decimal("2")
        )

    svc._trading_api.get_order = get_order
    svc._wait_for_order_fill(
        "broker-1",
        timeout_seconds=0.5,
        poll_interval_seconds=0.6,
        sleep_fn=sleep,
        now_fn=lambda: clock[0],
    )

    assert svc._trading_api.cancelled_ids == ["broker-1"]
    assert partials.calls == [(1, {"service": svc.name, "terminal_status": "canceled"})]


def test_zero_fill_does_not_count_as_accepted_partial() -> None:
    svc = _service()
    partials = RecordingCounter()
    svc._rebalance_partial_fills_counter = partials
    svc._trading_api.get_order = lambda order_id: OrderState(
        broker_order_id=order_id,
        status=OrderStatus.CANCELED,
        filled_qty=Decimal("0"),
    )

    with pytest.raises(RuntimeError, match="did not fill"):
        svc._wait_for_order_fill("broker-1", sleep_fn=lambda _: None)
    assert partials.calls == []


def test_manual_review_metric_uses_existing_failed_request_error() -> None:
    assert (
        PortfolioEMLService._count_manual_review_requests(
            [
                {"error": "manual review required: submission outcome unknown"},
                {"error": "max retries exceeded"},
                {"error": "manual review required: cancellation not confirmed"},
            ]
        )
        == 2
    )


def test_ambiguous_outcome_counts_manual_review_without_retry(monkeypatch) -> None:
    svc = _service()
    outcomes = RecordingCounter()
    skips = RecordingCounter()
    svc._rebalance_executions_counter = outcomes
    svc._rebalance_buy_skips_counter = skips
    svc._market_clock = V2MarketClockEvent(
        ts=time.time(), source="test", now=datetime.now(), is_market_open=True
    )
    event = V2RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 1.0}
    )
    svc.state.remember_pending_rebalance_request(event)

    def fail(_event: V2RebalancePlanRequestEvent) -> RebalanceExecutionResult:
        raise AmbiguousOrderOutcome("submission outcome unknown")

    monkeypatch.setattr(svc, "_execute_rebalance_plan", fail)
    svc._execute_pending_rebalance_plans()

    assert not svc.state.pending_rebalance_requests
    assert svc.state.failed_rebalance_requests[0]["execution_failures"] == 0
    assert outcomes.calls == [(1, {"result": "manual_review", "service": svc.name})]
    assert skips.calls == []
