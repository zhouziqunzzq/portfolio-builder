from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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


class FakeTradingClient:
    """Test double for the Alpaca trading client.

    Supports:
    - cancel_orders
    - get_account / set_account
    - get_all_positions/get_positions / set_positions
    - get_asset / set_asset
    - submit_order
    - get_order_by_id

    Behavior knobs:
    - next_order_final_status: filled/rejected/etc
    - next_order_fill_after: number of polls before reaching final status
    """

    def __init__(self):
        self._assets: Dict[str, FakeAsset] = {}
        self._account: Any = None
        self._positions: List[Any] = []

        self.submitted: List[Dict[str, Any]] = []
        self.actions: List[str] = []
        self.cancel_all_called: int = 0

        self._orders: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 1

        self.next_order_final_status: str = "filled"
        self.next_order_fill_after: int = 1

    def cancel_orders(self):
        self.cancel_all_called += 1
        self.actions.append("cancel_orders")
        return {"ok": True}

    def set_asset(self, symbol: str, *, tradable: bool = True, status: str = "active"):
        self._assets[str(symbol).upper()] = FakeAsset(tradable=tradable, status=status)

    def set_account(self, account: Any):
        self._account = account

    def set_positions(self, positions):
        self._positions = list(positions)

    def get_account(self):
        if self._account is None:
            raise RuntimeError("account not set")
        return self._account

    def get_all_positions(self):
        return list(self._positions)

    def get_positions(self):
        return list(self._positions)

    def get_asset(self, symbol: str):
        sym = str(symbol).upper()
        if sym not in self._assets:
            raise RuntimeError("unknown asset")
        return self._assets[sym]

    def submit_order(self, order_req):
        self.actions.append("submit_order")

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

        # Convert enums/strings to readable values for asserts
        side_s = str(side).lower()
        if "buy" in side_s:
            side_s = "buy"
        elif "sell" in side_s:
            side_s = "sell"

        self.submitted.append(
            {
                "symbol": str(symbol).upper() if symbol is not None else None,
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

        return FakeOrder(id=oid, status="new")

    def get_order_by_id(self, order_id: str):
        st = self._orders[str(order_id)]
        st["polls"] += 1
        if st["polls"] >= st["fill_after"]:
            return FakeOrder(id=str(order_id), status=st["final"])
        return FakeOrder(id=str(order_id), status="new")
