from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Literal, Optional
from collections import deque

from algotrading.lib.eventing.md_events import BarClosed, BarCompleted

from .base import (
    BaseAlpha,
    BaseAlphaOutput,
    MarketDataAlphaInput,
    SingleInstrumentAlphaConfig,
)
from .ema import EMAAlpha, EMAAlphaConfig
from .sma import SMAAlpha, SMAAlphaConfig

MAType = Literal["sma", "ema"]


@dataclass(frozen=True)
class MACDAlphaConfig(SingleInstrumentAlphaConfig):
    """Configuration for MACD alpha."""

    kind: str = field(default="macd", init=False)
    ma_type: MAType = "ema"
    fast_window: int = 12
    slow_window: int = 26
    signal_window: int = 9


@dataclass(frozen=True)
class MACDAlphaOutput(BaseAlphaOutput):
    """MACD output with signal and component averages.

    The signal is the moving average of the MACD line (fast - slow), using the
    same `ma_type` and `signal_window` configured for the alpha.
    """

    macd: float
    signal: float
    fast: float
    slow: float
    is_ready: bool


class MACDAlpha(BaseAlpha[MarketDataAlphaInput, MACDAlphaOutput, float]):
    """Event-driven MACD alpha over bar close prices."""

    def __init__(self, config: MACDAlphaConfig) -> None:
        if config.fast_window <= 0 or config.slow_window <= 0:
            raise ValueError("MACD windows must be > 0")
        if config.signal_window <= 0:
            raise ValueError("MACD signal_window must be > 0")
        if config.fast_window >= config.slow_window:
            raise ValueError("MACD fast_window must be < slow_window")

        super().__init__(config=config, buffer_size=None)
        self.config = config
        self._signal_buffer: Deque[float] = deque(maxlen=config.signal_window)
        self._signal_ema: Optional[float] = None
        self._signal_alpha = 2.0 / (config.signal_window + 1)

        if config.ma_type == "sma":
            self._fast_alpha = SMAAlpha(
                SMAAlphaConfig(ref=config.ref, tf=config.tf, window=config.fast_window)
            )
            self._slow_alpha = SMAAlpha(
                SMAAlphaConfig(ref=config.ref, tf=config.tf, window=config.slow_window)
            )
        elif config.ma_type == "ema":
            self._fast_alpha = EMAAlpha(
                EMAAlphaConfig(ref=config.ref, tf=config.tf, window=config.fast_window)
            )
            self._slow_alpha = EMAAlpha(
                EMAAlphaConfig(ref=config.ref, tf=config.tf, window=config.slow_window)
            )
        else:
            raise ValueError(f"Unsupported MA type: {config.ma_type}")

    def update(self, alpha_input: MarketDataAlphaInput) -> MACDAlphaOutput:
        """Update MACD state with a bar event and emit the latest values."""

        event = alpha_input.event
        if not isinstance(event, (BarCompleted, BarClosed)):
            raise TypeError(f"Unsupported market data event: {type(event)!r}")
        if event.key.ref != self.config.ref or event.key.tf != self.config.tf:
            raise ValueError("Event does not match MACD config ref/tf")

        fast_out = self._fast_alpha.update(alpha_input)
        slow_out = self._slow_alpha.update(alpha_input)

        if not (self._fast_alpha.ready() and self._slow_alpha.ready()):
            return self._set_last_output(
                MACDAlphaOutput(
                    macd=float("nan"),
                    signal=float("nan"),
                    fast=fast_out.value,
                    slow=slow_out.value,
                    is_ready=False,
                )
            )

        macd_value = fast_out.value - slow_out.value
        signal_value = self._update_signal(macd_value)
        if signal_value is None:
            return self._set_last_output(
                MACDAlphaOutput(
                    macd=macd_value,
                    signal=float("nan"),
                    fast=fast_out.value,
                    slow=slow_out.value,
                    is_ready=False,
                )
            )
        return self._set_last_output(
            MACDAlphaOutput(
                macd=macd_value,
                signal=signal_value,
                fast=fast_out.value,
                slow=slow_out.value,
                is_ready=True,
            )
        )

    def ready(self) -> bool:
        """True once both the averages and signal are ready."""

        return (
            self._fast_alpha.ready()
            and self._slow_alpha.ready()
            and len(self._signal_buffer) >= self.config.signal_window
        )

    def reset(self) -> None:
        """Clear internal state and cached output."""

        super().reset()
        self._fast_alpha.reset()
        self._slow_alpha.reset()
        self._signal_buffer.clear()
        self._signal_ema = None

    def _update_signal(self, macd_value: float) -> Optional[float]:
        self._signal_buffer.append(macd_value)
        if len(self._signal_buffer) < self.config.signal_window:
            return None

        if self.config.ma_type == "sma":
            return sum(self._signal_buffer) / self.config.signal_window

        if self._signal_ema is None:
            self._signal_ema = sum(self._signal_buffer) / self.config.signal_window
        else:
            self._signal_ema = (
                self._signal_alpha * macd_value
                + (1.0 - self._signal_alpha) * self._signal_ema
            )
        return self._signal_ema
