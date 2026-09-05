from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from models.trading import (
    BrokerCapabilities,
    InstrumentMeta,
    InstrumentRef,
    OrderFilter,
    OrderIntent,
    OrderState,
    OrderStatus,
    PlacedOrder,
    PreflightOrderResult,
)
from trading_api.base import BaseSyncTradingAPI
from trading_api.exceptions import OrderNotFoundYet


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
        self._instruments: Dict[str, InstrumentMeta] = {}
        self._preflight_costs: Dict[tuple[str, Decimal], Decimal] = {}
        self._preflight_errors: Dict[tuple[str, str], Exception] = {}
        self._submit_errors: Dict[tuple[str, str], Exception] = {}

        self.submitted: List[Dict[str, Any]] = []
        self.actions: List[str] = []

        self._orders: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 1

        self.next_order_final_status: str = "filled"
        self.next_order_fill_after: int = 1
        # If > 0, get_order will raise OrderNotFoundYet for the first N polls per order.
        self.get_order_not_found_for_polls: int = 0

        self.open_orders: List[OrderState] = [
            OrderState(broker_order_id="OPEN1", status=OrderStatus.OPEN)
        ]
        self.list_orders_called: int = 0
        self.cancel_order_called: int = 0
        self.cancelled_ids: List[str] = []
        self.supports_preflight: bool = False

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
        supports_notional_buys: Optional[bool] = None,
    ) -> None:
        sym = str(symbol).upper()
        self._instruments[sym] = InstrumentMeta(
            instrument=InstrumentRef(symbol=sym),
            tradable=bool(tradable),
            fractionable=fractionable,
            supports_notional_buys=supports_notional_buys,
        )

    def set_preflight_cost(
        self, symbol: str, *, quantity: Decimal | int, estimated_cost: Decimal | int
    ) -> None:
        self.supports_preflight = True
        self._preflight_costs[(str(symbol).upper(), Decimal(quantity))] = Decimal(
            estimated_cost
        )

    def set_preflight_error(self, symbol: str, *, shape: str, error: Exception) -> None:
        self.supports_preflight = True
        self._preflight_errors[(str(symbol).upper(), str(shape))] = error

    def set_submit_error(self, symbol: str, *, shape: str, error: Exception) -> None:
        self._submit_errors[(str(symbol).upper(), str(shape))] = error

    # -----------------
    # TradingAPI methods
    # -----------------

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            supports_notional_market_orders=True,
            supports_qty_market_orders=True,
            supports_fractional_qty=True,
            supports_notional_sells=False,
            supports_preflight=self.supports_preflight,
        )

    def get_account(self):
        if self._account is None:
            raise RuntimeError("account not set")
        self.actions.append("get_account")
        return self._account

    def list_positions(self):
        self.actions.append("list_positions")
        return list(self._positions)

    def get_instrument(self, instrument: InstrumentRef) -> InstrumentMeta:
        sym = str(instrument.symbol).upper()
        self.actions.append(f"get_instrument:{sym}")
        if sym in self._instruments:
            return self._instruments[sym]
        # Default: unknown instruments are tradable.
        return InstrumentMeta(instrument=InstrumentRef(symbol=sym), tradable=True)

    def preflight_order(self, intent: OrderIntent) -> PreflightOrderResult:
        self.actions.append("preflight_order")
        symbol = str(intent.instrument.symbol).upper()
        shape = "notional" if intent.notional is not None else "quantity"
        error = self._preflight_errors.get((symbol, shape))
        if error is not None:
            raise error
        estimated_cost = None
        if intent.qty is not None:
            estimated_cost = self._preflight_costs.get((symbol, intent.qty))
        return PreflightOrderResult(
            instrument=intent.instrument,
            estimated_cost=estimated_cost,
            raw={"noop": True},
        )

    def submit_order(self, intent: OrderIntent) -> PlacedOrder:
        self.actions.append("submit_order")
        oid = f"O{self._next_id}"
        self._next_id += 1

        side_s = str(intent.side).lower()
        if "buy" in side_s:
            side_s = "buy"
        elif "sell" in side_s:
            side_s = "sell"

        shape = "notional" if intent.notional is not None else "quantity"
        error = self._submit_errors.get((str(intent.instrument.symbol).upper(), shape))
        if error is not None:
            raise error

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

        if int(self.get_order_not_found_for_polls) > 0 and st["polls"] <= int(
            self.get_order_not_found_for_polls
        ):
            raise OrderNotFoundYet("order not visible yet")

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
