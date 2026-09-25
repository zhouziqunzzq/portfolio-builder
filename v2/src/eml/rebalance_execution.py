from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Literal, Mapping, Optional, Tuple


class RebalanceExecutionStatus(str, Enum):
    """Terminal outcome of one rebalance execution attempt."""

    COMPLETED = "completed"
    COMPLETED_WITH_SKIPS = "completed_with_skips"


class RebalanceBuySkipReason(str, Enum):
    """Deterministic reasons why one rebalance buy was not submitted."""

    QUANTITY_ORDERS_UNSUPPORTED = "quantity_orders_unsupported"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    QUANTITY_PREFLIGHT_REJECTED = "quantity_preflight_rejected"
    UNIT_COST_UNAVAILABLE = "unit_cost_unavailable"
    BELOW_ONE_WHOLE_SHARE = "below_one_whole_share"
    QUANTITY_SUBMISSION_REJECTED = "quantity_submission_rejected"


@dataclass(frozen=True)
class RebalanceExecutionSkip:
    """One deterministic symbol-level skip produced during execution."""

    symbol: str
    desired_notional: Decimal
    reason: RebalanceBuySkipReason
    side: Literal["buy"] = "buy"
    estimated_unit_cost: Optional[Decimal] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side,
            "desired_notional": float(self.desired_notional),
            "reason": self.reason.value,
        }
        if self.estimated_unit_cost is not None:
            payload["estimated_unit_cost"] = float(self.estimated_unit_cost)
        return payload


@dataclass(frozen=True)
class RebalanceExecutionResult:
    """Structured outcome for one rebalance execution attempt."""

    status: RebalanceExecutionStatus
    skips: Tuple[RebalanceExecutionSkip, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "skips": [skip.to_payload() for skip in self.skips],
        }


@dataclass(frozen=True)
class ProcessedRebalanceOrder:
    """A confirmed fill that suppresses further trades for this request and symbol.

    This intentionally records successful work only. It does not try to recover
    orders across crashes or infer fills from position history.
    """

    rebalance_id: str
    symbol: str
    side: Literal["buy", "sell"]
    client_order_id: str
    broker_order_id: str
    fill_kind: Literal["full", "partial"]
    # True when an unsupported notional buy was filled as whole shares instead.
    used_whole_share_fallback: bool
    filled_qty: Optional[Decimal]
    filled_notional: Optional[Decimal]
    recorded_at: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "rebalance_id": self.rebalance_id,
            "symbol": self.symbol,
            "side": self.side,
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "fill_kind": self.fill_kind,
            "used_whole_share_fallback": self.used_whole_share_fallback,
            "filled_qty": str(self.filled_qty) if self.filled_qty is not None else None,
            "filled_notional": (
                str(self.filled_notional) if self.filled_notional is not None else None
            ),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProcessedRebalanceOrder":
        return cls(
            rebalance_id=str(payload["rebalance_id"]),
            symbol=str(payload["symbol"]),
            side=payload["side"],
            client_order_id=str(payload["client_order_id"]),
            broker_order_id=str(payload["broker_order_id"]),
            fill_kind=payload["fill_kind"],
            used_whole_share_fallback=bool(payload["used_whole_share_fallback"]),
            filled_qty=(
                Decimal(str(payload["filled_qty"]))
                if payload.get("filled_qty") is not None
                else None
            ),
            filled_notional=(
                Decimal(str(payload["filled_notional"]))
                if payload.get("filled_notional") is not None
                else None
            ),
            recorded_at=str(payload["recorded_at"]),
        )
