from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional


def to_decimal(v: Any) -> Optional[Decimal]:
    """Best-effort conversion to `Decimal`.

    Notes
    -----
    - Uses `Decimal(str(x))` for floats to avoid binary float artifacts.
    - Returns None for empty/invalid values.
    - Intended for boundary conversion (broker APIs, event payloads, state), not hot loops.
    """

    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    if isinstance(v, int):
        return Decimal(v)
    if isinstance(v, float):
        return Decimal(str(v))
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return Decimal(s)
        except Exception:
            return None
    # Fall back: try string conversion for other numeric-ish types.
    try:
        return Decimal(str(v))
    except Exception:
        return None
