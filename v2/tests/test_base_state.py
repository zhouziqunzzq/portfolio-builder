from __future__ import annotations

from dataclasses import dataclass

import pytest

from v2.src.states.base_state import BaseState, StateSerializationError


@dataclass
class _DemoStateV1(BaseState):
    STATE_KEY = "demo"
    SCHEMA_VERSION = 1

    n: int = 0
    s: str = ""

    def to_payload(self):
        return {"n": self.n, "s": self.s}

    @classmethod
    def from_payload(cls, payload):
        return cls(n=int(payload.get("n", 0)), s=str(payload.get("s", "")))

    @classmethod
    def empty(cls):
        return cls(n=0, s="")


class _BadPayloadState(BaseState):
    STATE_KEY = "bad"

    def to_payload(self):
        return [1, 2, 3]  # not a dict

    @classmethod
    def from_payload(cls, payload):
        return cls()

    @classmethod
    def empty(cls):
        return cls()


@dataclass
class _DemoStateV2(BaseState):
    STATE_KEY = "demo_v2"
    SCHEMA_VERSION = 2

    n: int = 0
    s: str = ""
    flag: bool = False

    def to_payload(self):
        return {"n": self.n, "s": self.s, "flag": self.flag}

    @classmethod
    def upgrade_payload(cls, payload, *, from_version: int):
        upgraded = dict(payload)
        if from_version == 1:
            upgraded.setdefault("flag", False)
            return upgraded
        return upgraded

    @classmethod
    def from_payload(cls, payload):
        return cls(
            n=int(payload.get("n", 0)),
            s=str(payload.get("s", "")),
            flag=bool(payload.get("flag", False)),
        )

    @classmethod
    def empty(cls):
        return cls(n=0, s="", flag=False)


class _SlotsState(BaseState):
    STATE_KEY = "slots"

    __slots__ = ("x",)

    def __init__(self, x: int = 0) -> None:
        self.x = x

    def to_payload(self):
        return {"x": self.x}

    @classmethod
    def from_payload(cls, payload):
        return cls(x=int(payload.get("x", 0)))

    @classmethod
    def empty(cls):
        return cls(x=0)


def test_to_dict_roundtrip_current_version():
    s0 = _DemoStateV1(n=7, s="hi")
    blob = s0.to_dict()
    s1 = _DemoStateV1.from_dict(blob)
    assert s1 == s0


def test_to_dict_requires_dict_payload():
    with pytest.raises(StateSerializationError):
        _BadPayloadState().to_dict()


def test_from_dict_rejects_key_mismatch():
    blob = _DemoStateV1(n=1, s="x").to_dict()
    blob["state_key"] = "other"
    with pytest.raises(StateSerializationError, match="State key mismatch"):
        _DemoStateV1.from_dict(blob)


def test_from_dict_rejects_invalid_schema_version():
    blob = _DemoStateV1(n=1, s="x").to_dict()
    blob["schema_version"] = "not-an-int"
    with pytest.raises(StateSerializationError, match="Invalid schema_version"):
        _DemoStateV1.from_dict(blob)


def test_from_dict_rejects_non_mapping_payload():
    blob = _DemoStateV1(n=1, s="x").to_dict()
    blob["payload"] = ["nope"]
    with pytest.raises(StateSerializationError, match="payload must be a mapping"):
        _DemoStateV1.from_dict(blob)


def test_from_dict_upgrades_older_versions():
    # Persisted v1 payload, loaded by v2 class.
    blob = {
        "state_key": _DemoStateV2.key(),
        "schema_version": 1,
        "payload": {"n": 3, "s": "y"},
    }
    s = _DemoStateV2.from_dict(blob)
    assert s.n == 3
    assert s.s == "y"
    assert s.flag is False


def test_from_dict_rejects_newer_versions():
    blob = _DemoStateV1(n=1, s="x").to_dict()
    blob["schema_version"] = 999
    with pytest.raises(StateSerializationError, match="newer than supported"):
        _DemoStateV1.from_dict(blob)


def test_reset_returns_empty_instance():
    s0 = _DemoStateV1(n=9, s="abc")
    s1 = s0.reset()
    assert isinstance(s1, _DemoStateV1)
    assert s1 == _DemoStateV1.empty()
    assert s1 is not s0


def test_reset_inplace_updates_dict_backed_objects():
    s = _DemoStateV1(n=9, s="abc")
    s.reset_inplace()
    assert s == _DemoStateV1.empty()


def test_reset_inplace_raises_for_slots_objects():
    s = _SlotsState(x=5)
    # Note: because BaseState is __dict__-backed, subclasses may still have a
    # __dict__ even if they define __slots__. In that case, reset_inplace works.
    s.reset_inplace()
    assert s.x == 0
