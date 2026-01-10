import asyncio
import time
from datetime import datetime, timedelta

import pytest

from v2.src.eml.portfolio_eml import PortfolioEMLService
from v2.src.eml.config import EMLConfig
from v2.src.eml.state import EMLState

from v2.tests.fakes import FakeTradingAPI

# NOTE: `v2/src` code imports these modules as top-level packages (it mutates sys.path).
# Use the same import paths in tests to avoid duplicate module instances.
from events.event_bus import EventBus
from events.events import (
    MarketClockEvent,
    PositionCleanupIntent,
    PositionCleanupPlanRequestEvent,
)
from events.topic import Topic
from models import BrokerPosition


def _open_market_clock(now: datetime) -> MarketClockEvent:
    return MarketClockEvent(
        ts=now.timestamp(),
        source="unit",
        now=now,
        is_market_open=True,
        next_market_open=None,
        next_market_close=None,
    )


def test_execute_position_cleanup_plan_stores_pending_and_confirms():
    trading = FakeTradingAPI()
    bus = EventBus()
    sub = bus.subscribe(topics={Topic.POSITION_CLEANUP_PLAN})

    svc = PortfolioEMLService(
        bus=bus,
        trading_api=trading,
        config=EMLConfig(include_positions=True),
    )
    svc.state = EMLState.empty()

    req = PositionCleanupPlanRequestEvent(
        ts=time.time(),
        request_id="pc-1",
        intents={"AAA": PositionCleanupIntent(ticker="AAA", reason="unit")},
        source="unit",
    )

    async def _do():
        await svc.execute_position_cleanup_plan(req)
        evt = await sub.next()
        return evt

    evt = asyncio.run(_do())

    assert getattr(evt, "request_id", None) == "pc-1"
    assert getattr(evt, "topic", None) == Topic.POSITION_CLEANUP_PLAN
    assert "pc-1" in svc.state.pending_position_cleanup_requests


def test_pending_cleanup_cancelled_if_rebalance_executed_today():
    trading = FakeTradingAPI()
    bus = EventBus()

    svc = PortfolioEMLService(
        bus=bus,
        trading_api=trading,
        config=EMLConfig(include_positions=True),
    )
    svc.state = EMLState.empty()

    now = datetime.now()
    svc._market_clock = _open_market_clock(now)

    # Pending cleanup
    svc.state.pending_position_cleanup_requests["pc-1"] = {
        "request_id": "pc-1",
        "request_ts": time.time() - 10,
        "intents": {"AAA": {"ticker": "AAA", "reason": "unit"}},
        "source": "unit",
        "correlation_id": "",
        "status": "pending",
        "execution_failures": 0,
    }

    # Rebalance executed "today" -> should cancel cleanup.
    svc.state.executed_rebalance_history.append(
        {"rebalance_id": "r1", "executed_ts": time.time()}
    )

    svc._execute_pending_position_cleanup_plans()

    assert "pc-1" not in svc.state.pending_position_cleanup_requests
    assert len(svc.state.executed_position_cleanup_history) == 1
    assert svc.state.executed_position_cleanup_history[0].get("request_id") == "pc-1"
    assert svc.state.executed_position_cleanup_history[0].get("status") == "cancelled"
    assert trading.submitted == []


def test_pending_cleanup_executes_close_orders_for_long_and_short(monkeypatch, caplog):
    trading = FakeTradingAPI()
    trading.set_instrument("AAA", tradable=True)
    trading.set_instrument("BBB", tradable=True)

    # AAA long -> SELL qty; BBB short -> warn + skip buy-to-cover
    trading.set_positions(
        [
            BrokerPosition(symbol="AAA", qty=1.0, market_value=0.05),
            BrokerPosition(symbol="BBB", qty=-2.0, market_value=-0.08),
        ]
    )

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(include_positions=True, min_order_size=0.0),
    )
    svc.state = EMLState.empty()

    now = datetime.now()
    svc._market_clock = _open_market_clock(now)

    # Avoid real sleeping in wait loop if it ever happens.
    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state.pending_position_cleanup_requests["pc-1"] = {
        "request_id": "pc-1",
        "request_ts": time.time() - 10,
        "intents": {
            "AAA": {"ticker": "AAA", "reason": "unit"},
            "BBB": {"ticker": "BBB", "reason": "unit"},
        },
        "source": "unit",
        "correlation_id": "",
        "status": "pending",
        "execution_failures": 0,
    }

    svc._execute_pending_position_cleanup_plans()

    assert "pc-1" not in svc.state.pending_position_cleanup_requests
    assert any(
        x.get("request_id") == "pc-1" and x.get("status") == "executed"
        for x in svc.state.executed_position_cleanup_history
    )

    # cancel_orders pre-flight, then 1 order (sell only)
    assert trading.list_orders_called >= 1
    assert len(trading.submitted) == 1

    assert trading.submitted[0]["symbol"] == "AAA"
    assert str(trading.submitted[0]["side"]).lower().endswith("sell")
    assert float(trading.submitted[0]["qty"]) == pytest.approx(1.0)
    # No buy-to-cover for BBB since short position cleanup is explicitly not supported.

    assert any(
        "Residual short position detected during cleanup" in r.getMessage()
        for r in caplog.records
    )


def test_pending_cleanup_qty_safety_threshold_blocks_large_sells(monkeypatch):
    trading = FakeTradingAPI()
    trading.set_instrument("AAA", tradable=True)

    trading.set_positions(
        [
            # Too large for cleanup; should be blocked by EML safety gate.
            BrokerPosition(symbol="AAA", qty=2.0, market_value=10.0),
        ]
    )

    svc = PortfolioEMLService(
        bus=EventBus(),
        trading_api=trading,
        config=EMLConfig(
            include_positions=True,
            min_order_size=0.0,
            position_cleanup_max_abs_qty=1.0,
            max_pending_position_cleanup_execution_retries=5,
        ),
    )
    svc.state = EMLState.empty()

    now = datetime.now()
    svc._market_clock = _open_market_clock(now)

    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state.pending_position_cleanup_requests["pc-1"] = {
        "request_id": "pc-1",
        "request_ts": time.time() - 10,
        "intents": {
            "AAA": {"ticker": "AAA", "reason": "unit"},
        },
        "source": "unit",
        "correlation_id": "",
        "status": "pending",
        "execution_failures": 0,
    }

    svc._execute_pending_position_cleanup_plans()

    # Still pending and counted as a failure; no orders submitted.
    assert "pc-1" in svc.state.pending_position_cleanup_requests
    assert (
        svc.state.pending_position_cleanup_requests["pc-1"].get("execution_failures")
        == 1
    )
    assert trading.submitted == []
