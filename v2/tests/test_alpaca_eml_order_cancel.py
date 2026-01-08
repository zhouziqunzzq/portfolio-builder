import asyncio

from v2.src.eml.alpaca_eml import AlpacaEMLService
from v2.src.eml.config import EMLConfig
from v2.src.events.event_bus import EventBus

from v2.tests.fakes import FakeTradingClient


def test_startup_cancels_open_orders_by_default():
    trading = FakeTradingClient()
    svc = AlpacaEMLService(bus=EventBus(), trading_client=trading, config=EMLConfig())

    asyncio.run(svc._on_startup())
    assert trading.cancel_all_called == 1


def test_startup_cancel_can_be_disabled():
    trading = FakeTradingClient()
    cfg = EMLConfig(cancel_open_orders_on_startup=False)
    svc = AlpacaEMLService(bus=EventBus(), trading_client=trading, config=cfg)

    asyncio.run(svc._on_startup())
    assert trading.cancel_all_called == 0


def test_shutdown_cancel_disabled_by_default():
    trading = FakeTradingClient()
    svc = AlpacaEMLService(bus=EventBus(), trading_client=trading, config=EMLConfig())

    asyncio.run(svc._on_shutdown_requested())
    assert trading.cancel_all_called == 0


def test_shutdown_cancel_can_be_enabled():
    trading = FakeTradingClient()
    cfg = EMLConfig(cancel_open_orders_on_shutdown=True)
    svc = AlpacaEMLService(bus=EventBus(), trading_client=trading, config=cfg)

    asyncio.run(svc._on_shutdown_requested())
    assert trading.cancel_all_called == 1
