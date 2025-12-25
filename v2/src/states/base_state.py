from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Mapping, Type, TypeVar


JsonObject = Dict[str, Any]
TState = TypeVar("TState", bound="BaseState")


class StateSerializationError(ValueError):
    """Raised when persisted state cannot be parsed/validated."""


class BaseState(ABC):
    """Base contract for persisted runtime state.

    Any state object that should be managed by a `StateStore` must implement this
    interface.

    Design goals:
      - Explicit, stable key per state type (`STATE_KEY`).
      - Schema versioning with an upgrade hook.
      - JSON-serializable payloads (no pickles).

    Notes
    -----
    - `to_payload()` MUST return a JSON-serializable mapping.
      (e.g. timestamps as ISO strings, pandas objects converted to builtins)
    - `from_payload()` should accept payloads from older schema versions via
      `upgrade_payload()`.
    """

    STATE_KEY: ClassVar[str]
    SCHEMA_VERSION: ClassVar[int] = 1

    # ----------------------------
    # Identity / versioning
    # ----------------------------
    @classmethod
    def key(cls) -> str:
        key = getattr(cls, "STATE_KEY", None)
        return str(key) if key else cls.__name__

    @classmethod
    def schema_version(cls) -> int:
        return int(getattr(cls, "SCHEMA_VERSION", 1))

    # ----------------------------
    # Serialization
    # ----------------------------
    @abstractmethod
    def to_payload(self) -> JsonObject:
        """Return a JSON-serializable payload for persistence."""

    @classmethod
    @abstractmethod
    def from_payload(cls: Type[TState], payload: Mapping[str, Any]) -> TState:
        """Reconstruct state from the *current* schema version payload."""

    @classmethod
    def upgrade_payload(
        cls, payload: Mapping[str, Any], *, from_version: int
    ) -> JsonObject:
        """Upgrade payload from an older schema version.

        Implementations may override this if they change persisted structure.
        Default behavior assumes backward compatibility (no transformation).
        """

        _ = from_version
        return dict(payload)

    def to_dict(self) -> JsonObject:
        """Serialize including metadata for safe round-tripping."""

        payload = self.to_payload()
        if not isinstance(payload, dict):
            raise StateSerializationError(
                f"{self.__class__.__name__}.to_payload() must return a dict"
            )

        return {
            "state_key": self.key(),
            "schema_version": self.schema_version(),
            "payload": payload,
        }

    @classmethod
    def from_dict(cls: Type[TState], data: Mapping[str, Any]) -> TState:
        """Deserialize from `to_dict()` output (including version upgrade)."""

        if not isinstance(data, Mapping):
            raise StateSerializationError("State must be a mapping")

        state_key = data.get("state_key")
        if state_key is not None and str(state_key) != cls.key():
            raise StateSerializationError(
                f"State key mismatch: expected={cls.key()} got={state_key}"
            )

        raw_version = data.get("schema_version", None)
        try:
            version = (
                int(raw_version) if raw_version is not None else cls.schema_version()
            )
        except Exception as e:
            raise StateSerializationError(
                f"Invalid schema_version: {raw_version}"
            ) from e

        raw_payload = data.get("payload", {})
        if raw_payload is None:
            raw_payload = {}
        if not isinstance(raw_payload, Mapping):
            raise StateSerializationError("payload must be a mapping")

        payload: JsonObject
        if version == cls.schema_version():
            payload = dict(raw_payload)
        elif version < cls.schema_version():
            payload = cls.upgrade_payload(raw_payload, from_version=version)
            if not isinstance(payload, dict):
                raise StateSerializationError("upgrade_payload must return a dict")
        else:
            # Forward-incompatible: code is older than persisted state.
            raise StateSerializationError(
                f"State schema_version={version} is newer than supported={cls.schema_version()}"
            )

        return cls.from_payload(payload)

    # ----------------------------
    # Reset
    # ----------------------------
    @classmethod
    @abstractmethod
    def empty(cls: Type[TState]) -> TState:
        """Return a fresh/default instance used for reset semantics."""

    def reset(self: TState) -> TState:
        """Return a reset (fresh) state instance.

        This is intentionally *not* in-place: most callers (state stores, runners)
        should just replace the reference on the owning component.
        """

        return type(self).empty()

    def reset_inplace(self) -> None:
        """Best-effort in-place reset.

        Use only when other code holds a reference to this state object and
        replacing the reference is inconvenient.

        The default implementation supports mutable, `__dict__`-backed objects.
        Dataclasses with `slots=True` or frozen instances may not support this.
        """

        fresh = type(self).empty()
        did_reset = False

        # 1) Reset dict-backed attributes
        if hasattr(self, "__dict__") and hasattr(fresh, "__dict__"):
            self.__dict__.clear()
            self.__dict__.update(fresh.__dict__)
            did_reset = True

        # 2) Reset slot-backed attributes (common for dataclasses with slots=True)
        for cls in type(self).mro():
            if "__slots__" not in getattr(cls, "__dict__", {}):
                continue
            slots = cls.__dict__.get("__slots__")
            if slots is None:
                continue

            if isinstance(slots, str):
                slot_names = [slots]
            else:
                try:
                    slot_names = list(slots)
                except TypeError:
                    slot_names = []

            for name in slot_names:
                if name in ("__dict__", "__weakref__"):
                    continue
                if not hasattr(fresh, name):
                    continue
                try:
                    setattr(self, name, getattr(fresh, name))
                    did_reset = True
                except Exception:
                    # best-effort; keep going
                    continue

        if did_reset:
            return

        raise NotImplementedError(
            f"{type(self).__name__} does not support reset_inplace(); use reset()"
        )
