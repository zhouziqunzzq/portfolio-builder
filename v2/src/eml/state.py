from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional

from events.events import (
    PositionCleanupPlanRequestEvent,
    RebalancePlanRequestEvent,
)
from states.base_state import BaseState


class EMLState(BaseState):
    STATE_KEY = "eml.alpaca"
    SCHEMA_VERSION = 2

    # Pending rebalance requests (rebalance_id -> request payload)
    pending_rebalance_requests: Dict[str, Dict[str, Any]]
    # TODO: Implement handling for position cleanup requests
    pending_position_cleanup_requests: Dict[str, Dict[str, Any]]

    # Failed rebalance requests ordered by failure timestamp (ascending)
    failed_rebalance_requests: List[Dict[str, Any]]
    # TODO: Implement handling for position cleanup requests
    failed_position_cleanup_requests: List[Dict[str, Any]]

    # Executed rebalance requests ordered by execution timestamp (ascending)
    executed_rebalance_history: List[Dict[str, Any]]
    # TODO: Implement handling for position cleanup requests
    executed_position_cleanup_history: List[Dict[str, Any]]

    def __init__(
        self,
        *,
        pending_rebalance_requests: Optional[Dict[str, Dict[str, Any]]] = None,
        pending_position_cleanup_requests: Optional[Dict[str, Dict[str, Any]]] = None,
        failed_rebalance_requests: Optional[List[Dict[str, Any]]] = None,
        failed_position_cleanup_requests: Optional[List[Dict[str, Any]]] = None,
        executed_rebalance_history: Optional[List[Dict[str, Any]]] = None,
        executed_position_cleanup_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.pending_rebalance_requests = dict(pending_rebalance_requests or {})
        self.pending_position_cleanup_requests = dict(
            pending_position_cleanup_requests or {}
        )
        self.failed_rebalance_requests = list(failed_rebalance_requests or [])
        self.failed_position_cleanup_requests = list(
            failed_position_cleanup_requests or []
        )
        self.executed_rebalance_history = list(executed_rebalance_history or [])
        self.executed_position_cleanup_history = list(
            executed_position_cleanup_history or []
        )
        self._sort_history_inplace()
        self._sort_failed_inplace()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "pending_rebalance_requests": self.pending_rebalance_requests,
            "pending_position_cleanup_requests": self.pending_position_cleanup_requests,
            "failed_rebalance_requests": self.failed_rebalance_requests,
            "failed_position_cleanup_requests": self.failed_position_cleanup_requests,
            "executed_rebalance_history": self.executed_rebalance_history,
            "executed_position_cleanup_history": self.executed_position_cleanup_history,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EMLState":
        pending = payload.get("pending_rebalance_requests")
        if pending is None:
            pending = {}
        if not isinstance(pending, dict):
            pending = {}

        failed = payload.get("failed_rebalance_requests")
        if failed is None:
            failed = []
        if not isinstance(failed, list):
            failed = []

        executed = payload.get("executed_rebalance_history")
        if executed is None:
            executed = []
        if not isinstance(executed, list):
            executed = []

        pending_pc = payload.get("pending_position_cleanup_requests")
        if pending_pc is None:
            pending_pc = {}
        if not isinstance(pending_pc, dict):
            pending_pc = {}

        failed_pc = payload.get("failed_position_cleanup_requests")
        if failed_pc is None:
            failed_pc = []
        if not isinstance(failed_pc, list):
            failed_pc = []

        executed_pc = payload.get("executed_position_cleanup_history")
        if executed_pc is None:
            executed_pc = []
        if not isinstance(executed_pc, list):
            executed_pc = []

        # Defensive copy + minimal shape validation
        pending_out: Dict[str, Dict[str, Any]] = {}
        for k, v in pending.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, dict):
                continue
            pending_out[k] = dict(v)

        pending_pc_out: Dict[str, Dict[str, Any]] = {}
        for k, v in pending_pc.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, dict):
                continue
            pending_pc_out[k] = dict(v)

        failed_out: List[Dict[str, Any]] = []
        for item in failed:
            if not isinstance(item, dict):
                continue
            failed_out.append(dict(item))

        failed_pc_out: List[Dict[str, Any]] = []
        for item in failed_pc:
            if not isinstance(item, dict):
                continue
            failed_pc_out.append(dict(item))

        executed_out: List[Dict[str, Any]] = []
        for item in executed:
            if not isinstance(item, dict):
                continue
            executed_out.append(dict(item))

        executed_pc_out: List[Dict[str, Any]] = []
        for item in executed_pc:
            if not isinstance(item, dict):
                continue
            executed_pc_out.append(dict(item))

        return cls(
            pending_rebalance_requests=pending_out,
            pending_position_cleanup_requests=pending_pc_out,
            failed_rebalance_requests=failed_out,
            failed_position_cleanup_requests=failed_pc_out,
            executed_rebalance_history=executed_out,
            executed_position_cleanup_history=executed_pc_out,
        )

    @classmethod
    def empty(cls) -> "EMLState":
        return cls(
            pending_rebalance_requests={},
            pending_position_cleanup_requests={},
            failed_rebalance_requests=[],
            failed_position_cleanup_requests=[],
            executed_rebalance_history=[],
            executed_position_cleanup_history=[],
        )

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
            "status": "pending",
            "execution_failures": 0,
        }

    def increment_pending_rebalance_execution_failure(self, rebalance_id: str) -> int:
        rid = str(rebalance_id)
        req = self.pending_rebalance_requests.get(rid)
        if req is None:
            return 0

        cur = req.get("execution_failures", 0)
        try:
            cur_i = int(cur)
        except Exception:
            cur_i = 0

        nxt = cur_i + 1
        req["execution_failures"] = nxt
        return nxt

    def mark_rebalance_failed(
        self,
        *,
        rebalance_id: str,
        failed_ts: Optional[float] = None,
        error: str = "",
    ) -> Dict[str, Any]:
        rid = str(rebalance_id)
        now_ts = float(failed_ts if failed_ts is not None else time.time())

        req = self.pending_rebalance_requests.pop(rid, None)
        if req is None:
            req = {"rebalance_id": rid}

        entry = {
            **dict(req),
            "rebalance_id": rid,
            "status": "failed",
            "failed_ts": now_ts,
        }
        if error:
            entry["error"] = str(error)

        self.failed_rebalance_requests.append(entry)
        self._sort_failed_inplace()
        return entry

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

    # ------------------------------------------------------------------
    # Position cleanup request bookkeeping
    # ------------------------------------------------------------------

    def has_pending_position_cleanup_request(self, request_id: str) -> bool:
        return str(request_id) in self.pending_position_cleanup_requests

    def remember_pending_position_cleanup_request(
        self, event: PositionCleanupPlanRequestEvent
    ) -> None:
        def _json_decimal_str(v: Any) -> Optional[str]:
            """Convert common numeric inputs into a JSON-safe decimal string.

            Rationale: Python's stdlib JSON cannot serialize `Decimal`, and converting
            to float would lose precision.
            """

            if v is None:
                return None
            if isinstance(v, Decimal):
                return str(v)
            if isinstance(v, int):
                return str(v)
            if isinstance(v, float):
                # Best-effort: preserve the float's decimal string representation.
                return str(Decimal(str(v)))
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                try:
                    # Validate it is a decimal-like string.
                    Decimal(s)
                    return s
                except Exception:
                    return None
            return None

        intents = getattr(event, "intents", None)
        intents_out: Dict[str, Dict[str, Any]] = {}
        if isinstance(intents, dict):
            for sym, intent in intents.items():
                # `intent` is expected to be PositionCleanupIntent-like.
                ticker = str(getattr(intent, "ticker", sym) or sym)
                intents_out[str(sym)] = {
                    "ticker": ticker,
                    "reason": str(getattr(intent, "reason", "") or ""),
                    "observed_qty": _json_decimal_str(
                        getattr(intent, "observed_qty", None)
                    ),
                    "qty_threshold": _json_decimal_str(
                        getattr(intent, "qty_threshold", None)
                    ),
                    "observed_market_value": _json_decimal_str(
                        getattr(intent, "observed_market_value", None)
                    ),
                    "market_value_threshold": _json_decimal_str(
                        getattr(intent, "market_value_threshold", None)
                    ),
                }

        rid = str(getattr(event, "request_id", "") or "")
        if not rid:
            raise ValueError("PositionCleanupPlanRequestEvent missing request_id")

        self.pending_position_cleanup_requests[rid] = {
            "request_id": rid,
            "request_ts": float(getattr(event, "ts", time.time())),
            "intents": intents_out,
            "source": getattr(event, "source", ""),
            "correlation_id": getattr(event, "correlation_id", ""),
            "status": "pending",
            "execution_failures": 0,
        }

    def increment_pending_position_cleanup_execution_failure(
        self, request_id: str
    ) -> int:
        rid = str(request_id)
        req = self.pending_position_cleanup_requests.get(rid)
        if req is None:
            return 0

        cur = req.get("execution_failures", 0)
        try:
            cur_i = int(cur)
        except Exception:
            cur_i = 0

        nxt = cur_i + 1
        req["execution_failures"] = nxt
        return nxt

    def mark_position_cleanup_failed(
        self,
        *,
        request_id: str,
        failed_ts: Optional[float] = None,
        error: str = "",
    ) -> Dict[str, Any]:
        rid = str(request_id)
        now_ts = float(failed_ts if failed_ts is not None else time.time())

        req = self.pending_position_cleanup_requests.pop(rid, None)
        if req is None:
            req = {"request_id": rid}

        entry = {
            **dict(req),
            "request_id": rid,
            "status": "failed",
            "failed_ts": now_ts,
        }
        if error:
            entry["error"] = str(error)

        self.failed_position_cleanup_requests.append(entry)
        self._sort_failed_inplace()
        return entry

    def mark_position_cleanup_executed(
        self,
        *,
        request_id: str,
        executed_ts: Optional[float] = None,
        status: str = "executed",
        note: str = "",
    ) -> None:
        rid = str(request_id)
        now_ts = float(executed_ts if executed_ts is not None else time.time())

        req = self.pending_position_cleanup_requests.pop(rid, None)
        if req is None:
            req = {"request_id": rid}

        entry = {
            **dict(req),
            "request_id": rid,
            "status": str(status or "executed"),
            "executed_ts": now_ts,
        }
        if note:
            entry["note"] = str(note)

        self.executed_position_cleanup_history.append(entry)
        self._sort_history_inplace()

    def _sort_failed_inplace(self) -> None:
        def _key(x: Dict[str, Any]) -> float:
            v = x.get("failed_ts")
            try:
                return float(v)
            except Exception:
                return 0.0

        self.failed_rebalance_requests.sort(key=_key)

        def _key_pc(x: Dict[str, Any]) -> float:
            v = x.get("failed_ts")
            try:
                return float(v)
            except Exception:
                return 0.0

        self.failed_position_cleanup_requests.sort(key=_key_pc)

    def _sort_history_inplace(self) -> None:
        def _key(x: Dict[str, Any]) -> float:
            v = x.get("executed_ts")
            try:
                return float(v)
            except Exception:
                return 0.0

        self.executed_rebalance_history.sort(key=_key)

        def _key_pc(x: Dict[str, Any]) -> float:
            v = x.get("executed_ts")
            try:
                return float(v)
            except Exception:
                return 0.0

        self.executed_position_cleanup_history.sort(key=_key_pc)
