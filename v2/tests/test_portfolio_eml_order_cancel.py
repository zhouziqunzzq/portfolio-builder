import asyncio

from v2.src.eml.portfolio_eml import PortfolioEMLService
from v2.src.eml.config import EMLConfig
from v2.src.events.event_bus import EventBus

from v2.tests.fakes import FakeTradingAPI


def test_startup_cancels_open_orders_by_default():
    trading = FakeTradingAPI()
    svc = PortfolioEMLService(bus=EventBus(), trading_api=trading, config=EMLConfig())

    asyncio.run(svc._on_startup())
    assert trading.list_orders_called == 1


def test_startup_cancel_can_be_disabled():
    trading = FakeTradingAPI()
    cfg = EMLConfig(cancel_open_orders_on_startup=False)
    svc = PortfolioEMLService(bus=EventBus(), trading_api=trading, config=cfg)

    asyncio.run(svc._on_startup())
    assert trading.list_orders_called == 0


def test_shutdown_cancel_disabled_by_default():
    trading = FakeTradingAPI()
    svc = PortfolioEMLService(bus=EventBus(), trading_api=trading, config=EMLConfig())

    asyncio.run(svc._on_shutdown_requested())
    assert trading.list_orders_called == 0


def test_shutdown_cancel_can_be_enabled():
    trading = FakeTradingAPI()
    cfg = EMLConfig(cancel_open_orders_on_shutdown=True)
    svc = PortfolioEMLService(bus=EventBus(), trading_api=trading, config=cfg)

    asyncio.run(svc._on_shutdown_requested())
    assert trading.list_orders_called == 1
