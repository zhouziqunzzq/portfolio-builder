import asyncio

from v2.src.eml.alpaca_eml import AlpacaEMLService
from v2.src.eml.config import EMLConfig
from v2.src.events.event_bus import EventBus


class _FakeTradingClient:
    def __init__(self):
        self.cancel_all_called = 0

    def cancel_orders(self):
        self.cancel_all_called += 1
        return {"ok": True}


def test_startup_cancels_open_orders_by_default():
    trading = _FakeTradingClient()
    svc = AlpacaEMLService(bus=EventBus(), trading_client=trading, config=EMLConfig())

    asyncio.run(svc._on_startup())
    assert trading.cancel_all_called == 1


def test_startup_cancel_can_be_disabled():
    trading = _FakeTradingClient()
    cfg = EMLConfig(cancel_open_orders_on_startup=False)
    svc = AlpacaEMLService(bus=EventBus(), trading_client=trading, config=cfg)

    asyncio.run(svc._on_startup())
    assert trading.cancel_all_called == 0


def test_shutdown_cancel_disabled_by_default():
    trading = _FakeTradingClient()
    svc = AlpacaEMLService(bus=EventBus(), trading_client=trading, config=EMLConfig())

    asyncio.run(svc._on_shutdown_requested())
    assert trading.cancel_all_called == 0


def test_shutdown_cancel_can_be_enabled():
    trading = _FakeTradingClient()
    cfg = EMLConfig(cancel_open_orders_on_shutdown=True)
    svc = AlpacaEMLService(bus=EventBus(), trading_client=trading, config=cfg)

    asyncio.run(svc._on_shutdown_requested())
    assert trading.cancel_all_called == 1
