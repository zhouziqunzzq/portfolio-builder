from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional


class InstrumentType(Enum):
    EQUITY = "equity"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    DAY = "day"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InstrumentRef:
    """Broker-agnostic instrument reference.

    Keep this minimal: Public.com requires an explicit instrument type for many calls;
    other brokers may ignore it.
    """

    symbol: str
    instrument_type: InstrumentType = InstrumentType.EQUITY


@dataclass(frozen=True)
class BrokerCapabilities:
    """Feature flags that guide execution decisions.

    Prefer explicit capabilities over try/except probing.
    """

    supports_notional_market_orders: bool = False
    supports_qty_market_orders: bool = False
    supports_fractional_qty: bool = False
    supports_notional_sells: bool = False
    supports_cancel_all_open_orders: bool = False
    has_preflight: bool = False
    eventual_consistency_after_submit: bool = False


@dataclass(frozen=True)
class Tradability:
    tradable: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class OrderIntent:
    """Normalized order intent emitted by execution logic.

    Exactly one of `qty` or `notional` should be set.
    """

    client_order_id: str
    instrument: InstrumentRef
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY

    qty: Optional[Decimal] = None
    notional: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if bool(self.qty is None) == bool(self.notional is None):
            raise ValueError("OrderIntent must specify exactly one of qty or notional")

        if self.order_type == OrderType.MARKET:
            if self.limit_price is not None:
                raise ValueError("Market orders cannot specify limit_price")
        elif self.order_type == OrderType.LIMIT:
            if self.limit_price is None:
                raise ValueError("Limit orders must specify limit_price")
            if self.limit_price <= 0:
                raise ValueError("limit_price must be > 0")
        else:
            raise ValueError(f"Unknown order_type: {self.order_type}")


@dataclass(frozen=True)
class PlacedOrder:
    broker_order_id: str
    client_order_id: str
    submitted_at: Optional[datetime] = None
    raw: Any = None


@dataclass(frozen=True)
class OrderState:
    broker_order_id: str
    status: OrderStatus

    filled_qty: Optional[Decimal] = None
    filled_notional: Optional[Decimal] = None
    avg_fill_price: Optional[Decimal] = None

    last_update_ts: Optional[float] = None
    raw: Any = None
