from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import json
import os
import shutil

from .base_state import BaseState, StateSerializationError

import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from runtime_manager import RuntimeManager


StateBlob = Dict[str, Any]


class BaseStateManager(ABC):
    """Public interface for runtime state persistence.

    A StateManager owns *persistence* and *reset* semantics for a set of
    stateful runtime objects (sleeves, allocators, etc).

    Each stateful object is expected to expose a `.state` attribute that is a
    `BaseState` instance.
    """

    @abstractmethod
    def managed_names(self) -> set[str]:
        """Canonical names of managed objects (e.g. {'trend', 'allocator'})."""

    @abstractmethod
    def save_state(self, names: Optional[Sequence[str]] = None) -> None:
        """Persist state for `names` (or all if None)."""

    @abstractmethod
    def load_state(self, names: Optional[Sequence[str]] = None) -> bool:
        """Load persisted state into live objects.

        Returns True if a state file was found and successfully loaded.
        """

    @abstractmethod
    def reset_state(self, names: Optional[Sequence[str]] = None) -> None:
        """Reset state for `names` (or all if None)."""


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
    ) -> None:
        self.runtime_manager = runtime_manager

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

    # ---------------------------
    # Managed objects
    # ---------------------------

    def managed_names(self) -> set[str]:
        return {"trend", "defensive", "sideways_base", "allocator"}

    def _aliases(self) -> Dict[str, str]:
        return {
            # canonical
            "trend": "trend",
            "defensive": "defensive",
            "sideways_base": "sideways_base",
            "allocator": "allocator",
            # common variants
            "trend_sleeve": "trend",
            "defensive_sleeve": "defensive",
            "sideways_base_sleeve": "sideways_base",
            "multi_sleeve_allocator": "allocator",
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
        rm = self.runtime_manager
        if name == "trend":
            return rm.get("trend_sleeve")
        if name == "defensive":
            return rm.get("defensive_sleeve")
        if name == "sideways_base":
            return rm.get("sideways_base_sleeve")
        if name == "allocator":
            return rm.get("multi_sleeve_allocator")
        raise KeyError(name)

    def _get_state(self, obj: Any) -> BaseState:
        st = getattr(obj, "state", None)
        if not isinstance(st, BaseState):
            raise TypeError(
                f"Managed object {type(obj).__name__} has no BaseState .state"
            )
        return st

    # ---------------------------
    # IO helpers
    # ---------------------------

    def _backup_file(self) -> Path:
        return self.state_file.with_suffix(self.state_file.suffix + self.backup_suffix)

    def _tmp_file(self) -> Path:
        return self.state_file.with_suffix(self.state_file.suffix + self.tmp_suffix)

    def _atomic_write_json(self, path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = self._tmp_file()
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        # Backup old file (best-effort)
        if path.exists():
            try:
                shutil.copy2(path, self._backup_file())
            except Exception:
                pass

        os.replace(tmp, path)

    def _read_json(self, path: Path) -> Optional[StateBlob]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _read_state_blob(self) -> Optional[StateBlob]:
        blob = self._read_json(self.state_file)
        if blob is not None:
            return blob

        # Fallback to backup if main is corrupted/missing
        bak = self._backup_file()
        blob = self._read_json(bak)
        return blob

    # ---------------------------
    # Public API
    # ---------------------------

    def save_state(self, names: Optional[Sequence[str]] = None) -> None:
        selected = self._normalize_names(names)

        # Load existing so partial writes can merge (when names != all)
        existing = self._read_state_blob() or {}
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

        self._atomic_write_json(self.state_file, out)

    def load_state(self, names: Optional[Sequence[str]] = None) -> bool:
        selected = self._normalize_names(names)
        blob = self._read_state_blob()
        if blob is None:
            # If neither the main nor backup exists, treat as "no state yet".
            if not self.state_file.exists() and not self._backup_file().exists():
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

        # Load requested states; require each one to be present and valid.
        for name in selected:
            raw_state = states.get(name)
            if raw_state is None:
                raise StateSerializationError(
                    f"State file missing required state '{name}': {self.state_file}"
                )

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
