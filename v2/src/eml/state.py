from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional

from events.events import (
    RebalancePlanRequestEvent,
)
from states.base_state import BaseState


class EMLState(BaseState):
    STATE_KEY = "eml.alpaca"
    SCHEMA_VERSION = 1

    # Pending rebalance requests (rebalance_id -> request payload)
    pending_rebalance_requests: Dict[str, Dict[str, Any]]

    # Executed rebalance requests ordered by execution timestamp (ascending)
    executed_rebalance_history: List[Dict[str, Any]]

    def __init__(
        self,
        *,
        pending_rebalance_requests: Optional[Dict[str, Dict[str, Any]]] = None,
        executed_rebalance_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.pending_rebalance_requests = dict(pending_rebalance_requests or {})
        self.executed_rebalance_history = list(executed_rebalance_history or [])
        self._sort_history_inplace()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "pending_rebalance_requests": self.pending_rebalance_requests,
            "executed_rebalance_history": self.executed_rebalance_history,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EMLState":
        pending = payload.get("pending_rebalance_requests")
        if pending is None:
            pending = {}
        if not isinstance(pending, dict):
            pending = {}

        executed = payload.get("executed_rebalance_history")
        if executed is None:
            executed = []
        if not isinstance(executed, list):
            executed = []

        # Defensive copy + minimal shape validation
        pending_out: Dict[str, Dict[str, Any]] = {}
        for k, v in pending.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, dict):
                continue
            pending_out[k] = dict(v)

        executed_out: List[Dict[str, Any]] = []
        for item in executed:
            if not isinstance(item, dict):
                continue
            executed_out.append(dict(item))

        return cls(
            pending_rebalance_requests=pending_out,
            executed_rebalance_history=executed_out,
        )

    @classmethod
    def empty(cls) -> "EMLState":
        return cls(pending_rebalance_requests={}, executed_rebalance_history=[])

    def has_pending_rebalance_request(self, rebalance_id: str) -> bool:
        return str(rebalance_id) in self.pending_rebalance_requests

    def remember_pending_rebalance_request(
        self, event: RebalancePlanRequestEvent
    ) -> None:
        self.pending_rebalance_requests[str(event.rebalance_id)] = {
            "rebalance_id": str(event.rebalance_id),
            "request_ts": float(event.ts),
            "weights": dict(event.weights or {}),
            "source": getattr(event, "source", ""),
            "correlation_id": getattr(event, "correlation_id", ""),
        }

    def mark_rebalance_executed(
        self,
        *,
        rebalance_id: str,
        executed_ts: Optional[float] = None,
    ) -> None:
        rid = str(rebalance_id)
        now_ts = float(executed_ts if executed_ts is not None else time.time())

        req = self.pending_rebalance_requests.pop(rid, None)
        if req is None:
            req = {"rebalance_id": rid}

        entry = {
            **dict(req),
            "rebalance_id": rid,
            "executed_ts": now_ts,
        }
        self.executed_rebalance_history.append(entry)
        self._sort_history_inplace()

    def _sort_history_inplace(self) -> None:
        def _key(x: Dict[str, Any]) -> float:
            v = x.get("executed_ts")
            try:
                return float(v)
            except Exception:
                return 0.0

        self.executed_rebalance_history.sort(key=_key)
