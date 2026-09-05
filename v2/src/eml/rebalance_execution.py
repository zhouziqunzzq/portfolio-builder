from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple


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
