from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Generic, Optional, TypeVar

from algotrading.lib.eventing.md_events import (
    BarCompleted,
    BarClosed,
)
from algotrading.lib.types.instruments import InstrumentRef
from algotrading.lib.types.market_data import Timeframe

MarketDataEvent = BarCompleted | BarClosed

InputT = TypeVar("InputT", bound="BaseAlphaInput")
OutputT = TypeVar("OutputT", bound="BaseAlphaOutput")
BufferT = TypeVar("BufferT")


@dataclass(frozen=True)
class BaseAlphaConfig(ABC):
    """Base config for alpha implementations."""

    kind: str
    # TODO: add more fields


@dataclass(frozen=True)
class SingleInstrumentAlphaConfig(BaseAlphaConfig):
    """Config for alphas bound to a single instrument/timeframe."""

    ref: InstrumentRef
    tf: Timeframe


@dataclass(frozen=True)
class BaseAlphaInput(ABC):
    """Marker base class for alpha inputs."""

    pass


@dataclass(frozen=True)
class MarketDataAlphaInput(BaseAlphaInput):
    """Alpha input derived from a market data event."""

    event: MarketDataEvent


@dataclass(frozen=True)
class BaseAlphaOutput(ABC):
    """Marker base class for alpha outputs."""

    pass


@dataclass(frozen=True)
class ScalarAlphaOutput(BaseAlphaOutput):
    """Scalar alpha output with a warmup-ready flag."""

    value: float
    is_ready: bool


class BaseAlpha(ABC, Generic[InputT, OutputT, BufferT]):
    """Event-driven alpha base with optional ring buffer storage."""

    def __init__(
        self,
        config: BaseAlphaConfig,
        buffer_size: Optional[int] = None,
    ) -> None:
        self.config = config
        self._buffer: Deque[BufferT] = (
            deque(maxlen=buffer_size) if buffer_size is not None else deque()
        )
        self._last_output: Optional[OutputT] = None

    @property
    def buffer(self) -> Deque[BufferT]:
        """Expose the internal buffer for inspection/testing."""

        return self._buffer

    @property
    def last_output(self) -> Optional[OutputT]:
        """Return the most recent output, if any."""

        return self._last_output

    def _append_buffer(self, value: BufferT) -> None:
        self._buffer.append(value)

    def _set_last_output(self, output: OutputT) -> OutputT:
        self._last_output = output
        return output

    def reset(self) -> None:
        """Clear internal state and cached output."""

        self._buffer.clear()
        self._last_output = None

    @abstractmethod
    def ready(self) -> bool:
        """Whether the alpha has enough data to emit ready outputs."""

        raise NotImplementedError

    @abstractmethod
    def update(self, alpha_input: InputT) -> OutputT:
        """Process an input event and return the latest output."""

        raise NotImplementedError
