from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from models.trading import (
    BrokerCapabilities,
    Instrument,
    InstrumentRef,
    OrderFilter,
    OrderIntent,
    OrderState,
    OrderStatus,
    PlacedOrder,
    PreflightOrderResult,
)
from trading_api.base import BaseSyncTradingAPI


@dataclass
class FakeAsset:
    tradable: bool = True
    status: str = "active"

    def __post_init__(self) -> None:
        # Mirror alpaca-py-ish shape used by some code paths.
        self._raw = {"tradable": self.tradable, "status": self.status}


@dataclass
class FakeOrder:
    id: str
    status: str

    def __post_init__(self) -> None:
        self._raw = {"id": self.id, "status": self.status}


class FakeTradingAPI(BaseSyncTradingAPI):
    """Test double for the broker-agnostic TradingAPI.

    This is intentionally shaped like `BaseSyncTradingAPI` (not alpaca-py), so
    EML unit tests can inject it directly without going through AlpacaTradingAPI.
    """

    name = "fake"

    def __init__(self):
        self._account: Any = None
        self._positions: List[Any] = []
        self._instruments: Dict[str, Instrument] = {}

        self.submitted: List[Dict[str, Any]] = []
        self.actions: List[str] = []

        self._orders: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 1

        self.next_order_final_status: str = "filled"
        self.next_order_fill_after: int = 1

        self.open_orders: List[OrderState] = [
            OrderState(broker_order_id="OPEN1", status=OrderStatus.OPEN)
        ]
        self.list_orders_called: int = 0
        self.cancel_order_called: int = 0
        self.cancelled_ids: List[str] = []

    # -----------------
    # Helpers for tests
    # -----------------

    def set_account(self, account: Any):
        self._account = account

    def set_positions(self, positions):
        self._positions = list(positions)

    def set_instrument(
        self,
        symbol: str,
        *,
        tradable: bool = True,
        fractionable: Optional[bool] = None,
    ) -> None:
        sym = str(symbol).upper()
        self._instruments[sym] = Instrument(
            instrument=InstrumentRef(symbol=sym),
            tradable=bool(tradable),
            fractionable=fractionable,
        )

    # -----------------
    # TradingAPI methods
    # -----------------

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            supports_notional_market_orders=True,
            supports_qty_market_orders=True,
            supports_fractional_qty=True,
            supports_notional_sells=False,
            supports_preflight=False,
        )

    def get_account(self):
        if self._account is None:
            raise RuntimeError("account not set")
        self.actions.append("get_account")
        return self._account

    def list_positions(self):
        self.actions.append("list_positions")
        return list(self._positions)

    def get_instrument(self, instrument: InstrumentRef) -> Instrument:
        sym = str(instrument.symbol).upper()
        self.actions.append(f"get_instrument:{sym}")
        if sym in self._instruments:
            return self._instruments[sym]
        # Default: unknown instruments are tradable.
        return Instrument(instrument=InstrumentRef(symbol=sym), tradable=True)

    def preflight_order(self, intent: OrderIntent) -> PreflightOrderResult:
        self.actions.append("preflight_order")
        return PreflightOrderResult(instrument=intent.instrument, raw={"noop": True})

    def submit_order(self, intent: OrderIntent) -> PlacedOrder:
        self.actions.append("submit_order")
        oid = f"O{self._next_id}"
        self._next_id += 1

        side_s = str(intent.side).lower()
        if "buy" in side_s:
            side_s = "buy"
        elif "sell" in side_s:
            side_s = "sell"

        self.submitted.append(
            {
                "symbol": str(intent.instrument.symbol).upper(),
                "side": side_s,
                "qty": intent.qty,
                "notional": intent.notional,
                "order_id": oid,
            }
        )

        self._orders[oid] = {
            "polls": 0,
            "fill_after": int(self.next_order_fill_after),
            "final": str(self.next_order_final_status),
        }

        return PlacedOrder(broker_order_id=oid, client_order_id=intent.client_order_id)

    def get_order(self, broker_order_id: str) -> OrderState:
        self.actions.append("get_order")
        st = self._orders[str(broker_order_id)]
        st["polls"] += 1
        if st["polls"] >= st["fill_after"]:
            final = str(st["final"]).strip().lower()
            if final == "filled":
                status = OrderStatus.FILLED
            elif final in {"canceled", "cancelled"}:
                status = OrderStatus.CANCELED
            elif final == "rejected":
                status = OrderStatus.REJECTED
            elif final == "expired":
                status = OrderStatus.EXPIRED
            else:
                status = OrderStatus.UNKNOWN
            return OrderState(broker_order_id=str(broker_order_id), status=status)

        return OrderState(broker_order_id=str(broker_order_id), status=OrderStatus.NEW)

    def list_orders(self, order_filter: OrderFilter) -> List[OrderState]:
        self.list_orders_called += 1
        self.actions.append("list_orders")
        requested_statuses = None
        if getattr(order_filter, "statuses", None) is not None:
            requested_statuses = set(order_filter.statuses or [])

        orders = list(self.open_orders)
        if requested_statuses is None:
            return orders
        return [o for o in orders if o.status in requested_statuses]

    def cancel_order(self, broker_order_id: str) -> None:
        self.cancel_order_called += 1
        self.cancelled_ids.append(str(broker_order_id))
        self.actions.append("cancel_order")
        self.open_orders = [
            o for o in self.open_orders if o.broker_order_id != str(broker_order_id)
        ]
