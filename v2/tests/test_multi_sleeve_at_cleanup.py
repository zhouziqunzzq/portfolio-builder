from __future__ import annotations

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
import inspect

import pytest

from v2.src.at.config import ATConfig
from v2.src.at.multi_sleeve_at import MultiSleeveATService
from v2.src.events.event_bus import EventBus
from v2.src.events.events import AccountSnapshotEvent, V2MarketClockEvent
from v2.src.models.broker import AccountSnapshot, PositionSnapshot


def _run(coro_or_fn):
    if inspect.iscoroutine(coro_or_fn):
        return asyncio.run(coro_or_fn)
    if inspect.iscoroutinefunction(coro_or_fn):
        return asyncio.run(coro_or_fn())
    if callable(coro_or_fn):
        return coro_or_fn()
    return coro_or_fn


class _DummyRM:
    def __init__(self):
        self._objects = {}

    def set(self, name: str, obj: object) -> None:
        self._objects[name] = obj

    def get(self, name: str) -> object:
        return self._objects.get(name)


def _make_service(*, config: ATConfig) -> MultiSleeveATService:
    bus = EventBus()
    rm = _DummyRM()
    return MultiSleeveATService(bus=bus, rm=rm, config=config)


def _set_market_clock_open_now(svc: MultiSleeveATService, now: datetime) -> None:
    svc._market_clock = V2MarketClockEvent(
        ts=now.timestamp(),
        now=now,
        is_market_open=True,
        next_market_open=None,
        next_market_close=None,
        source="unit",
    )


def _set_account_snapshot_with_positions(
    svc: MultiSleeveATService, now: datetime
) -> None:
    acct = AccountSnapshot(adj_equity=1000.0)
    pos = PositionSnapshot(symbol="SPY", qty=1.0, market_value=500.0)
    svc._account_snapshot = AccountSnapshotEvent(
        ts=now.timestamp(),
        account=acct,
        positions=[pos],
        source="unit",
    )


def _set_account_snapshot(
    svc: MultiSleeveATService, now: datetime, *, positions: list[PositionSnapshot]
) -> None:
    acct = AccountSnapshot(adj_equity=1000.0)
    svc._account_snapshot = AccountSnapshotEvent(
        ts=now.timestamp(),
        account=acct,
        positions=list(positions),
        source="unit",
    )


def test_cleanup_disabled_returns_false():
    svc = _make_service(config=ATConfig(position_cleanup_enabled=False))
    now = datetime(2026, 1, 7, 10, 0, 0)

    _set_market_clock_open_now(svc, now)
    _set_account_snapshot_with_positions(svc, now)
    svc.state.last_rebalance_weights = {"SPY": 1.0}

    assert _run(svc._check_should_cleanup_positions(now=now)) is False


def test_cleanup_requires_market_clock():
    svc = _make_service(config=ATConfig(position_cleanup_enabled=True))
    now = datetime(2026, 1, 7, 10, 0, 0)

    _set_account_snapshot_with_positions(svc, now)
    svc.state.last_rebalance_weights = {"SPY": 1.0}

    assert _run(svc._check_should_cleanup_positions(now=now)) is False


def test_cleanup_requires_market_open_now_or_today():
    svc = _make_service(config=ATConfig(position_cleanup_enabled=True))
    now = datetime(2026, 1, 7, 10, 0, 0)

    svc._market_clock = V2MarketClockEvent(
        ts=now.timestamp(),
        now=now,
        is_market_open=False,
        next_market_open=now + timedelta(days=1),
        next_market_close=None,
        source="unit",
    )
    _set_account_snapshot_with_positions(svc, now)
    svc.state.last_rebalance_weights = {"SPY": 1.0}

    assert _run(svc._check_should_cleanup_positions(now=now)) is False


def test_cleanup_respects_interval_days():
    svc = _make_service(
        config=ATConfig(
            position_cleanup_enabled=True,
            position_cleanup_interval_days=1,
        )
    )
    now = datetime(2026, 1, 7, 10, 0, 0)

    _set_market_clock_open_now(svc, now)
    _set_account_snapshot_with_positions(svc, now)
    svc.state.last_rebalance_weights = {"SPY": 1.0}
    svc.state.last_position_cleanup_ts = now - timedelta(hours=12)

    assert _run(svc._check_should_cleanup_positions(now=now)) is False


def test_cleanup_requires_last_rebalance_weights():
    svc = _make_service(config=ATConfig(position_cleanup_enabled=True))
    now = datetime(2026, 1, 7, 10, 0, 0)

    _set_market_clock_open_now(svc, now)
    _set_account_snapshot_with_positions(svc, now)

    assert _run(svc._check_should_cleanup_positions(now=now)) is False


def test_cleanup_requires_account_snapshot_positions():
    svc = _make_service(config=ATConfig(position_cleanup_enabled=True))
    now = datetime(2026, 1, 7, 10, 0, 0)

    _set_market_clock_open_now(svc, now)
    svc.state.last_rebalance_weights = {"SPY": 1.0}

    assert _run(svc._check_should_cleanup_positions(now=now)) is False


def test_cleanup_true_when_all_gates_pass():
    svc = _make_service(
        config=ATConfig(
            position_cleanup_enabled=True,
            position_cleanup_interval_days=1,
        )
    )
    now = datetime(2026, 1, 7, 10, 0, 0)

    _set_market_clock_open_now(svc, now)
    _set_account_snapshot_with_positions(svc, now)
    svc.state.last_rebalance_weights = {"SPY": 1.0}
    svc.state.last_position_cleanup_ts = now - timedelta(days=2)

    assert _run(svc._check_should_cleanup_positions(now=now)) is True


def test_generate_cleanup_raises_without_account_snapshot():
    svc = _make_service(config=ATConfig(position_cleanup_enabled=True))
    now = datetime(2026, 1, 7, 10, 0, 0)
    svc.state.last_rebalance_weights = {"SPY": 1.0}

    with pytest.raises(
        RuntimeError,
        match="Cannot generate position cleanup plan: no AccountSnapshotEvent received yet",
    ):
        _run(svc._generate_position_cleanup_plan_request(now=now))


def test_generate_cleanup_returns_none_when_no_positions():
    svc = _make_service(config=ATConfig(position_cleanup_enabled=True))
    now = datetime(2026, 1, 7, 10, 0, 0)
    svc.state.last_rebalance_weights = {"SPY": 1.0}
    _set_account_snapshot(svc, now, positions=[])

    assert _run(svc._generate_position_cleanup_plan_request(now=now)) is None


def test_generate_cleanup_raises_without_last_rebalance_weights():
    svc = _make_service(config=ATConfig(position_cleanup_enabled=True))
    now = datetime(2026, 1, 7, 10, 0, 0)
    _set_account_snapshot_with_positions(svc, now)

    with pytest.raises(
        RuntimeError,
        match="Cannot generate position cleanup plan: missing last_rebalance_weights in state",
    ):
        _run(svc._generate_position_cleanup_plan_request(now=now))


def test_generate_cleanup_skips_symbols_in_last_rebalance_weights():
    svc = _make_service(
        config=ATConfig(
            position_cleanup_enabled=True,
            position_cleanup_market_value_threshold=0.10,
            position_cleanup_qty_threshold=0.001,
        )
    )
    now = datetime(2026, 1, 7, 10, 0, 0)
    svc.state.last_rebalance_weights = {"SPY": 1.0}
    _set_account_snapshot(
        svc,
        now,
        positions=[PositionSnapshot(symbol="SPY", qty=0.0, market_value=0.0)],
    )

    assert _run(svc._generate_position_cleanup_plan_request(now=now)) is None


def test_generate_cleanup_builds_intents_for_residuals():
    svc = _make_service(
        config=ATConfig(
            position_cleanup_enabled=True,
            position_cleanup_market_value_threshold=0.10,
            position_cleanup_qty_threshold=0.001,
        )
    )
    now = datetime(2026, 1, 7, 10, 0, 0)
    svc.state.last_rebalance_weights = {"SPY": 1.0}

    _set_account_snapshot(
        svc,
        now,
        positions=[
            # In last weights -> never cleaned up, even if tiny
            PositionSnapshot(symbol="SPY", qty=0.0, market_value=0.0),
            # Residual by qty
            PositionSnapshot(symbol="AAA", qty=0.0005, market_value=10.0),
            # Residual by market value
            PositionSnapshot(symbol="BBB", qty=1.0, market_value=0.05),
            # Residual by both
            PositionSnapshot(symbol="DDD", qty=0.0, market_value=0.0),
            # Not residual
            PositionSnapshot(symbol="CCC", qty=1.0, market_value=10.0),
        ],
    )

    event = _run(svc._generate_position_cleanup_plan_request(now=now))
    assert event is not None
    assert set(event.intents.keys()) == {"AAA", "BBB", "DDD"}

    assert event.intents["AAA"].reason == "below_min_qty"
    assert event.intents["BBB"].reason == "below_min_market_value"
    assert event.intents["DDD"].reason == "below_min_market_value,below_min_qty"

    assert event.intents["AAA"].observed_qty == pytest.approx(Decimal("0.0005"))
    assert event.intents["AAA"].observed_market_value == pytest.approx(Decimal("10.0"))
