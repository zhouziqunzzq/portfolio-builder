from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Sequence


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
