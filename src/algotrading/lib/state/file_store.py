from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import json
import os
import shutil

from .errors import StateSerializationError


JsonObject = Dict[str, Any]
StateBlob = Dict[str, Any]


@dataclass
class FileStateStore:
    """Filesystem-backed store that persists state in one JSON file."""

    state_file: Path
    file_schema_version: int = 1
    backup_suffix: str = ".bak"
    tmp_suffix: str = ".tmp"

    def __post_init__(self) -> None:
        self.state_file = Path(self.state_file)

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

    # ---------------------------
    # Public API
    # ---------------------------

    def load_blob(self, *, strict: bool = False) -> Optional[StateBlob]:
        primary_exists = self.state_file.exists()
        blob = self._read_json(self.state_file)
        if blob is not None:
            return blob

        backup = self._backup_file()
        backup_exists = backup.exists()
        blob = self._read_json(backup)
        if blob is not None:
            return blob

        if strict and (primary_exists or backup_exists):
            raise StateSerializationError(
                f"Failed to read/parse state file (or backup): {self.state_file}"
            )

        return None

    def save_blob(self, blob: Mapping[str, Any]) -> None:
        self._atomic_write_json(self.state_file, blob)

    def load_states(self) -> Dict[str, Any]:
        blob = self.load_blob()
        if blob is None:
            return {}

        if not isinstance(blob, dict):
            raise StateSerializationError(
                f"State file root must be an object/dict: {self.state_file}"
            )

        states = blob.get("states")
        if not isinstance(states, dict):
            raise StateSerializationError(
                f"State file missing 'states' dict: {self.state_file}"
            )

        return dict(states)

    def save_states(self, states: Mapping[str, Any]) -> None:
        blob: Dict[str, Any] = {
            "file_schema_version": int(self.file_schema_version),
            "states": dict(states),
        }
        self.save_blob(blob)
