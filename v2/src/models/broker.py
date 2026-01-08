from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BrokerAccount:
    """Normalized broker account snapshot.

    This is intentionally a small, stable set of fields needed by the app.
    Execution adapters (EML) should map broker-specific payloads into this model.
    """

    id: Optional[str] = None
    status: Optional[str] = None
    cash: Optional[float] = None
    buying_power: Optional[float] = None
    portfolio_value: Optional[float] = None
    equity: Optional[float] = None
    last_equity: Optional[float] = None
    adj_equity: Optional[float] = None  # Adjusted equity after cash buffers, if any


@dataclass(frozen=True)
class BrokerPosition:
    """Normalized broker position snapshot."""

    symbol: str
    qty: Optional[float] = None
    market_value: Optional[float] = None
    avg_entry_price: Optional[float] = None
    side: Optional[str] = None
    unrealized_pl: Optional[float] = None
