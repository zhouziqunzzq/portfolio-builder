from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
import logging

from .base_state import BaseState, StateSerializationError

import sys

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from algotrading.lib.state.manager import BaseStateManager
from algotrading.lib.state.file_store import FileStateStore
from runtime_manager import RuntimeManager


@dataclass
class FileStateManager(BaseStateManager):
    """Filesystem-backed manager that persists all managed state in one JSON file."""

    runtime_manager: RuntimeManager
    state_file: Path

    # Internal knobs
    file_schema_version: int = 1
    backup_suffix: str = ".bak"
    tmp_suffix: str = ".tmp"

    def __init__(
        self,
        runtime_manager: RuntimeManager,
        *,
        state_file: Optional[str | Path] = None,
        skip_self_check: bool = False,  # Useful for tests
    ) -> None:
        self.runtime_manager = runtime_manager
        self.log = logging.getLogger(self.__class__.__name__)

        if state_file is None:
            cfg = getattr(runtime_manager, "app_config", None)
            runtime_cfg = getattr(cfg, "runtime", None) if cfg is not None else None
            cfg_state_file = (
                getattr(runtime_cfg, "state_file", None)
                if runtime_cfg is not None
                else None
            )
            state_file = cfg_state_file

        if state_file is None:
            raise ValueError(
                "runtime.state_file is None; configure AppConfig.runtime.state_file "
                "or pass state_file=... to FileStateManager"
            )

        if isinstance(state_file, str) and not state_file.strip():
            raise ValueError(
                "runtime.state_file is empty; provide a valid path for state persistence"
            )

        self.state_file = Path(state_file)
        self._store = FileStateStore(
            state_file=self.state_file,
            file_schema_version=self.file_schema_version,
            backup_suffix=self.backup_suffix,
            tmp_suffix=self.tmp_suffix,
        )

        # Fail fast if RuntimeManager wiring is incomplete.
        if not skip_self_check:
            self._self_check_managed_objects()

    def _self_check_managed_objects(self) -> None:
        """Validate that RuntimeManager exposes all managed objects and BaseState."""

        missing: list[str] = []
        bad_state: list[str] = []

        for name in sorted(self.managed_names()):
            try:
                obj = self._get_stateful_object(name)
            except Exception:
                missing.append(name)
                continue

            try:
                _ = self._get_state(obj)
            except Exception:
                bad_state.append(name)

        if missing or bad_state:
            parts: list[str] = []
            if missing:
                parts.append(f"missing objects for: {missing}")
            if bad_state:
                parts.append(f"missing/invalid .state for: {bad_state}")

            raise ValueError(
                "RuntimeManager is not wired for FileStateManager ("
                + "; ".join(parts)
                + ")"
            )

    # ---------------------------
    # Managed objects
    # ---------------------------

    def managed_names(self) -> set[str]:
        return {"trend", "defensive", "sideways_base", "allocator", "iml", "eml", "at"}

    def _aliases(self) -> Dict[str, str]:
        return {
            # canonical
            "trend": "trend",
            "defensive": "defensive",
            "sideways_base": "sideways_base",
            "allocator": "allocator",
            "iml": "iml",
            "eml": "eml",
            "at": "at",
            # common variants
            "trend_sleeve": "trend",
            "defensive_sleeve": "defensive",
            "sideways_base_sleeve": "sideways_base",
            "multi_sleeve_allocator": "allocator",
            "alpaca_polling_iml": "iml",
            "portfolio_eml": "eml",
            "multi_sleeve_at": "at",
        }

    def _normalize_names(self, names: Optional[Sequence[str]]) -> list[str]:
        if names is None:
            return sorted(self.managed_names())

        alias = self._aliases()
        out: list[str] = []
        for n in names:
            key = alias.get(str(n), None)
            if key is None:
                raise KeyError(
                    f"Unknown managed name '{n}'. Known: {sorted(alias.keys())}"
                )
            out.append(key)

        # unique but stable order
        seen: set[str] = set()
        uniq: list[str] = []
        for k in out:
            if k in seen:
                continue
            seen.add(k)
            uniq.append(k)
        return uniq

    def _get_stateful_object(self, name: str) -> Any:
        # Obtain references from RuntimeManager
        # (We intentionally use rm.get(...) to avoid tight coupling.)
        # Apply aliasing if applicable.
        alias = self._aliases()
        if name in alias:
            name = alias[name]
        return self.runtime_manager.get(name)

    def _get_state(self, obj: Any) -> BaseState:
        st = getattr(obj, "state", None)
        if not isinstance(st, BaseState):
            raise TypeError(
                f"Managed object {type(obj).__name__} has no BaseState .state"
            )
        return st

    # ---------------------------
    # Public API
    # ---------------------------

    def save_state(self, names: Optional[Sequence[str]] = None) -> None:
        selected = self._normalize_names(names)

        # Load existing so partial writes can merge (when names != all)
        existing = self._store.load_blob() or {}
        states_existing = existing.get("states") if isinstance(existing, dict) else None
        if not isinstance(states_existing, dict):
            states_existing = {}

        states_out: Dict[str, Any] = dict(states_existing)

        for name in selected:
            obj = self._get_stateful_object(name)
            state = self._get_state(obj)
            states_out[name] = state.to_dict()

        out: Dict[str, Any] = {
            "file_schema_version": int(self.file_schema_version),
            "states": states_out,
        }

        self._store.save_blob(out)

    def load_state(self, names: Optional[Sequence[str]] = None) -> bool:
        selected = self._normalize_names(names)

        blob = self._store.load_blob()
        if blob is None:
            # If neither the main nor backup exists, treat as "no state yet".
            if not self.state_file.exists() and not self._store._backup_file().exists():
                return False
            raise StateSerializationError(
                f"Failed to read/parse state file (or backup): {self.state_file}"
            )

        if not isinstance(blob, dict):
            raise StateSerializationError(
                f"State file root must be an object/dict: {self.state_file}"
            )

        states = blob.get("states")
        if not isinstance(states, dict):
            raise StateSerializationError(
                f"State file missing 'states' dict: {self.state_file}"
            )

        # Load requested states
        for name in selected:
            raw_state = states.get(name)
            if raw_state is None:
                self.log.warning(
                    f"State file missing required state '{name}': {self.state_file}; skipping load for this state"
                )
                continue

            obj = self._get_stateful_object(name)
            current_state = self._get_state(obj)
            state_cls = type(current_state)

            if not isinstance(raw_state, Mapping):
                raise StateSerializationError(
                    f"State '{name}' must be a mapping/dict in: {self.state_file}"
                )

            try:
                loaded = state_cls.from_dict(raw_state)
            except Exception as e:
                raise StateSerializationError(
                    f"Failed to deserialize state '{name}' ({state_cls.__name__}) from {self.state_file}"
                ) from e

            setattr(obj, "state", loaded)

        return True

    def reset_state(self, names: Optional[Sequence[str]] = None) -> None:
        selected = self._normalize_names(names)
        for name in selected:
            obj = self._get_stateful_object(name)
            state = self._get_state(obj)
            setattr(obj, "state", type(state).empty())

        # Persist the reset state to disk (so restarts match the reset intent).
        self.save_state(names=selected)
