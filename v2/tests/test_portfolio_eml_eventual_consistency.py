from decimal import Decimal

from v2.src.eml.config import EMLConfig
from v2.src.eml.portfolio_eml import PortfolioEMLService
from v2.src.events.event_bus import EventBus
from v2.src.models import AccountSnapshot
from v2.src.models.trading import InstrumentRef, OrderIntent, OrderSide

from v2.tests.fakes import FakeTradingAPI


def test_wait_for_order_fill_retries_on_order_not_found_yet():
    trading = FakeTradingAPI()
    trading.set_account(AccountSnapshot(equity=1000.0, adj_equity=1000.0))

    # Simulate eventual consistency: first 2 polls raise OrderNotFoundYet.
    trading.get_order_not_found_for_polls = 2
    # Ensure the order eventually fills.
    trading.next_order_fill_after = 4
    trading.next_order_final_status = "filled"

    svc = PortfolioEMLService(bus=EventBus(), trading_api=trading, config=EMLConfig())

    placed = trading.submit_order(
        OrderIntent(
            client_order_id="test-1",
            instrument=InstrumentRef(symbol="SPY"),
            side=OrderSide.BUY,
            notional=Decimal("10.00"),
        )
    )

    clock = {"t": 0.0}

    def now_fn() -> float:
        clock["t"] += 0.01
        return float(clock["t"])

    svc._wait_for_order_fill(
        placed.broker_order_id,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
        sleep_fn=lambda _: None,
        now_fn=now_fn,
    )

    # Sanity: we did poll multiple times (including not-found retries).
    assert trading.actions.count("get_order") >= 3
