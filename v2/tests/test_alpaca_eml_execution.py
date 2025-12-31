import time
from datetime import datetime

import pytest

from v2.src.eml.alpaca_eml import AlpacaEMLService
from v2.src.eml.config import EMLConfig
from v2.src.eml.state import EMLState
from v2.src.events.event_bus import EventBus
from v2.src.events.events import (
    BrokerAccount,
    BrokerPosition,
    MarketClockEvent,
    RebalancePlanRequestEvent,
)


class _FakeAsset:
    def __init__(self, *, tradable=True, status="active"):
        self._raw = {"tradable": tradable, "status": status}
        self.tradable = tradable
        self.status = status


class _FakeOrder:
    def __init__(self, order_id: str, status: str):
        self._raw = {"id": order_id, "status": status}
        self.id = order_id
        self.status = status


class _FakeTradingClient:
    def __init__(self):
        self._assets = {}
        self._account = BrokerAccount(equity=1000.0, adj_equity=1000.0)
        self._positions = []
        self.submitted = []  # list of dicts: {symbol, side, qty, notional}
        self._orders = {}  # id -> dict(status, polls)
        self._next_id = 1

        self.cancel_all_called = 0
        self.actions = []
        self.next_order_final_status = "filled"
        self.next_order_fill_after = 2

    def cancel_orders(self):
        self.cancel_all_called += 1
        self.actions.append("cancel_orders")
        return {"ok": True}

    def set_asset(self, symbol: str, *, tradable=True, status="active"):
        self._assets[str(symbol).upper()] = _FakeAsset(tradable=tradable, status=status)

    def set_account(self, account: BrokerAccount):
        self._account = account

    def set_positions(self, positions):
        self._positions = list(positions)

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return self._positions

    def get_positions(self):
        return self._positions

    def get_asset(self, symbol: str):
        sym = str(symbol).upper()
        if sym not in self._assets:
            raise RuntimeError("unknown asset")
        return self._assets[sym]

    def submit_order(self, order_req):
        # order_req is an alpaca-py MarketOrderRequest
        raw = getattr(order_req, "_raw", None)
        if isinstance(raw, dict):
            symbol = raw.get("symbol")
            side = raw.get("side")
            qty = raw.get("qty")
            notional = raw.get("notional")
        else:
            symbol = getattr(order_req, "symbol", None)
            side = getattr(order_req, "side", None)
            qty = getattr(order_req, "qty", None)
            notional = getattr(order_req, "notional", None)

        oid = f"O{self._next_id}"
        self._next_id += 1

        self.actions.append("submit_order")

        # Convert enums to readable values for asserts
        side_s = str(side).lower()
        if "buy" in side_s:
            side_s = "buy"
        elif "sell" in side_s:
            side_s = "sell"

        self.submitted.append(
            {
                "symbol": str(symbol).upper(),
                "side": side_s,
                "qty": qty,
                "notional": notional,
                "order_id": oid,
            }
        )

        self._orders[oid] = {
            "polls": 0,
            "fill_after": int(self.next_order_fill_after),
            "final": str(self.next_order_final_status),
        }
        return _FakeOrder(oid, status="new")

    def get_order_by_id(self, order_id: str):
        st = self._orders[str(order_id)]
        st["polls"] += 1
        if st["polls"] >= st["fill_after"]:
            return _FakeOrder(str(order_id), status=st["final"])
        return _FakeOrder(str(order_id), status="new")


def test_tradability_check_blocks_non_tradable():
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)
    trading.set_asset("BBB", tradable=False)

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=False),
    )

    e = RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 0.5, "BBB": 0.5}
    )

    with pytest.raises(RuntimeError, match="Non-tradable"):
        svc._execute_rebalance_plan(e)


def test_sells_before_buys_and_min_order_size_filtering(monkeypatch):
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)
    trading.set_asset("BBB", tradable=True)

    # Hold AAA ($1000), want to rotate to BBB.
    trading.set_positions([BrokerPosition(symbol="AAA", qty=10.0, market_value=1000.0)])

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=True, min_order_size=50.0),
    )

    # Avoid real sleeping in wait loop
    monkeypatch.setattr(time, "sleep", lambda _: None)

    e = RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 0.0, "BBB": 1.0}
    )
    svc._execute_rebalance_plan(e)

    assert len(trading.submitted) == 2
    assert trading.submitted[0]["side"] == "sell"
    assert trading.submitted[0]["symbol"] == "AAA"
    # Prefer notional sells when possible
    assert trading.submitted[0]["notional"] is not None
    assert trading.submitted[1]["side"] == "buy"
    assert trading.submitted[1]["symbol"] == "BBB"


def test_near_zero_weights_are_ignored(monkeypatch):
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=False, min_order_size=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    # Weight is tiny -> should be ignored -> no orders
    e = RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 1e-12}
    )
    svc._execute_rebalance_plan(e)
    assert trading.submitted == []


def test_min_order_size_filters_small_orders(monkeypatch):
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)
    trading.set_asset("BBB", tradable=True)

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=False, min_order_size=50.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    # With equity=1000:
    # - AAA @ 10% => $100 (should trade)
    # - BBB @ 1%  => $10  (should be ignored due to min_order_size=50)
    e = RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 0.10, "BBB": 0.01}
    )
    svc._execute_rebalance_plan(e)

    assert len(trading.submitted) == 1
    assert trading.submitted[0]["side"] == "buy"
    assert trading.submitted[0]["symbol"] == "AAA"


def test_execute_pending_marks_state_executed(monkeypatch):
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=False, min_order_size=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state = EMLState()
    svc.state.pending_rebalance_requests["r1"] = {
        "rebalance_id": "r1",
        "request_ts": time.time(),
        "weights": {"AAA": 1.0},
        "source": "test",
        "correlation_id": "",
    }

    # Pending execution is gated on a known, open market clock.
    svc._market_clock = MarketClockEvent(
        ts=time.time(),
        source="test",
        now=datetime.now(),
        is_market_open=True,
    )

    svc._execute_pending_rebalance_plans()
    assert "r1" not in svc.state.pending_rebalance_requests
    assert any(
        x.get("rebalance_id") == "r1" for x in svc.state.executed_rebalance_history
    )


def test_execute_pending_skips_when_market_clock_unknown(monkeypatch):
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=False, min_order_size=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state = EMLState()
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
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=False, min_order_size=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    svc.state = EMLState()
    svc.state.pending_rebalance_requests["r1"] = {
        "rebalance_id": "r1",
        "request_ts": time.time(),
        "weights": {"AAA": 1.0},
        "source": "test",
        "correlation_id": "",
    }

    svc._market_clock = MarketClockEvent(
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


def test_execute_rebalance_plan_cancels_open_orders_before_submitting(monkeypatch):
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=False, min_order_size=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    e = RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 1.0}
    )
    svc._execute_rebalance_plan(e)

    assert trading.cancel_all_called >= 1
    assert trading.submitted
    assert trading.actions[0] == "cancel_orders"


def test_execute_rebalance_plan_cancels_open_orders_on_execution_error(monkeypatch):
    trading = _FakeTradingClient()
    trading.set_asset("AAA", tradable=True)
    trading.next_order_final_status = "rejected"
    trading.next_order_fill_after = 1

    svc = AlpacaEMLService(
        bus=EventBus(),
        trading_client=trading,
        config=EMLConfig(include_positions=False, min_order_size=1.0),
    )

    monkeypatch.setattr(time, "sleep", lambda _: None)

    e = RebalancePlanRequestEvent(
        ts=time.time(), rebalance_id="r1", weights={"AAA": 1.0}
    )

    with pytest.raises(RuntimeError):
        svc._execute_rebalance_plan(e)

    # One cancel before submitting, one cancel after error.
    assert trading.cancel_all_called >= 2
