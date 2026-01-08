from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class BrokerAccount:
    """Normalized broker account snapshot.

    This is intentionally a small, stable set of fields needed by the app.
    Execution adapters (EML) should map broker-specific payloads into this model.
    """

    id: Optional[str] = None
    status: Optional[str] = None
    cash: Optional[Decimal] = None
    buying_power: Optional[Decimal] = None
    portfolio_value: Optional[Decimal] = None
    equity: Optional[Decimal] = None
    last_equity: Optional[Decimal] = None
    adj_equity: Optional[Decimal] = None  # Adjusted equity after cash buffers, if any


@dataclass(frozen=True)
class BrokerPosition:
    """Normalized broker position snapshot."""

    symbol: str
    qty: Optional[Decimal] = None
    market_value: Optional[Decimal] = None
    avg_entry_price: Optional[Decimal] = None
    side: Optional[str] = None
    unrealized_pl: Optional[Decimal] = None
