from __future__ import annotations

from pathlib import Path
import json

from algotrading.lib.state.file_store import FileStateStore


def test_file_state_store_atomic_write_creates_file(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    store = FileStateStore(state_file=state_file)

    store.save_states({"alpha": {"x": 1}})

    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["file_schema_version"] == 1
    assert data["states"]["alpha"]["x"] == 1


def test_file_state_store_creates_backup_on_overwrite(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    store = FileStateStore(state_file=state_file)

    store.save_states({"alpha": {"x": 1}})
    store.save_states({"alpha": {"x": 2}})

    backup = state_file.with_suffix(state_file.suffix + ".bak")
    assert backup.exists()


def test_file_state_store_fallback_to_backup_on_corrupt_primary(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    store = FileStateStore(state_file=state_file)

    store.save_states({"alpha": {"x": 1}})
    store.save_states({"alpha": {"x": 2}})

    state_file.write_text("{", encoding="utf-8")

    blob = store.load_blob()
    assert blob is not None
    assert blob["states"]["alpha"]["x"] == 1


def test_file_state_store_missing_files_return_none_and_empty(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    store = FileStateStore(state_file=state_file)

    assert store.load_blob() is None
    assert store.load_states() == {}
