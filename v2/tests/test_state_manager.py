from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from v2.src.states.base_state import BaseState
from v2.src.states.state_manager import FileStateManager


@dataclass
class _TinyState(BaseState):
    STATE_KEY = "tiny"
    SCHEMA_VERSION = 1

    x: int = 0

    def to_payload(self):
        return {"x": int(self.x)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]):
        return cls(x=int(payload.get("x", 0)))

    @classmethod
    def empty(cls):
        return cls(x=0)


class _Obj:
    def __init__(self, x: int):
        self.state = _TinyState(x=x)


class _FakeRuntimeCfg:
    def __init__(self, state_file: str | None):
        self.state_file = state_file


class _FakeAppCfg:
    def __init__(self, state_file: str | None):
        self.runtime = _FakeRuntimeCfg(state_file)


class _FakeRuntimeManager:
    def __init__(self, mapping: dict[str, object], *, state_file: str | None):
        self._mapping = dict(mapping)
        self.app_config = _FakeAppCfg(state_file)

    def get(self, name: str) -> object:
        return self._mapping[name]


def test_file_state_manager_save_load_roundtrip(tmp_path: Path):
    trend = _Obj(1)
    defensive = _Obj(2)
    sideways = _Obj(3)
    allocator = _Obj(4)

    state_file = tmp_path / "state.json"
    rm = _FakeRuntimeManager(
        {
            "trend_sleeve": trend,
            "defensive_sleeve": defensive,
            "sideways_base_sleeve": sideways,
            "multi_sleeve_allocator": allocator,
        },
        state_file=str(state_file),
    )

    sm = FileStateManager(rm)
    sm.save_state()
    assert state_file.exists()

    # mutate
    trend.state.x = 10
    defensive.state.x = 20

    loaded = sm.load_state()
    assert loaded is True

    assert trend.state.x == 1
    assert defensive.state.x == 2
    assert sideways.state.x == 3
    assert allocator.state.x == 4


def test_file_state_manager_subset_reset(tmp_path: Path):
    trend = _Obj(5)
    defensive = _Obj(6)
    sideways = _Obj(7)
    allocator = _Obj(8)

    state_file = tmp_path / "state.json"
    rm = _FakeRuntimeManager(
        {
            "trend_sleeve": trend,
            "defensive_sleeve": defensive,
            "sideways_base_sleeve": sideways,
            "multi_sleeve_allocator": allocator,
        },
        state_file=str(state_file),
    )

    sm = FileStateManager(rm)
    sm.save_state()

    sm.reset_state(names=["trend"])

    assert trend.state.x == 0
    # Others unchanged
    assert defensive.state.x == 6
    assert sideways.state.x == 7
    assert allocator.state.x == 8


def test_file_state_manager_creates_backup_on_overwrite(tmp_path: Path):
    trend = _Obj(1)
    defensive = _Obj(2)
    sideways = _Obj(3)
    allocator = _Obj(4)

    state_file = tmp_path / "state.json"
    rm = _FakeRuntimeManager(
        {
            "trend_sleeve": trend,
            "defensive_sleeve": defensive,
            "sideways_base_sleeve": sideways,
            "multi_sleeve_allocator": allocator,
        },
        state_file=str(state_file),
    )

    sm = FileStateManager(rm)
    sm.save_state()

    # change and re-save
    trend.state.x = 99
    sm.save_state()

    backup = state_file.with_suffix(state_file.suffix + ".bak")
    assert backup.exists()


def test_file_state_manager_load_missing_file_returns_false(tmp_path: Path):
    trend = _Obj(1)
    defensive = _Obj(2)
    sideways = _Obj(3)
    allocator = _Obj(4)

    state_file = tmp_path / "state.json"
    rm = _FakeRuntimeManager(
        {
            "trend_sleeve": trend,
            "defensive_sleeve": defensive,
            "sideways_base_sleeve": sideways,
            "multi_sleeve_allocator": allocator,
        },
        state_file=str(state_file),
    )

    sm = FileStateManager(rm)
    assert sm.load_state() is False


def test_file_state_manager_unknown_name_raises(tmp_path: Path):
    trend = _Obj(1)
    defensive = _Obj(2)
    sideways = _Obj(3)
    allocator = _Obj(4)

    state_file = tmp_path / "state.json"
    rm = _FakeRuntimeManager(
        {
            "trend_sleeve": trend,
            "defensive_sleeve": defensive,
            "sideways_base_sleeve": sideways,
            "multi_sleeve_allocator": allocator,
        },
        state_file=str(state_file),
    )

    sm = FileStateManager(rm)
    with pytest.raises(KeyError):
        sm.save_state(names=["nope"])


def test_file_state_manager_requires_state_file():
    trend = _Obj(1)
    defensive = _Obj(2)
    sideways = _Obj(3)
    allocator = _Obj(4)

    rm = _FakeRuntimeManager(
        {
            "trend_sleeve": trend,
            "defensive_sleeve": defensive,
            "sideways_base_sleeve": sideways,
            "multi_sleeve_allocator": allocator,
        },
        state_file=None,
    )

    with pytest.raises(ValueError, match=r"runtime\.state_file is None"):
        FileStateManager(rm)
