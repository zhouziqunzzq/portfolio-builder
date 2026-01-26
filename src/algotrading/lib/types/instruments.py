from dataclasses import dataclass
from enum import Enum


class InstrumentType(Enum):
    EQUITY = "equity"


@dataclass(frozen=True, slots=True)
class InstrumentRef:
    """Universal instrument reference."""

    symbol: str
    instrument_type: InstrumentType = InstrumentType.EQUITY
