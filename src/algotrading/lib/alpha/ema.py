from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from algotrading.lib.eventing.md_events import BarClosed, BarCompleted

from .base import (
    BaseAlpha,
    MarketDataAlphaInput,
    ScalarAlphaOutput,
    SingleInstrumentAlphaConfig,
)


@dataclass(frozen=True)
class EMAAlphaConfig(SingleInstrumentAlphaConfig):
    """Configuration for exponential moving average alpha."""

    kind: str = field(default="ema", init=False)
    window: int = 20

    def id(self) -> str:
        return f"{super().id()}_window={self.window}"


class EMAAlpha(BaseAlpha[MarketDataAlphaInput, ScalarAlphaOutput, float]):
    """Event-driven EMA alpha over bar close prices."""

    def __init__(self, config: EMAAlphaConfig) -> None:
        if config.window <= 0:
            raise ValueError("EMA window must be > 0")
        super().__init__(config=config, buffer_size=config.window)
        self.config = config
        self._ema: Optional[float] = None
        self._alpha = 2.0 / (config.window + 1)

    def update(self, alpha_input: MarketDataAlphaInput) -> ScalarAlphaOutput:
        """Update EMA state with a bar event and emit the latest value."""

        event = alpha_input.event
        if not isinstance(event, (BarCompleted, BarClosed)):
            raise TypeError(f"Unsupported market data event: {type(event)!r}")
        if event.key.ref != self.config.ref or event.key.tf != self.config.tf:
            raise ValueError("Event does not match EMA config ref/tf")

        price = float(event.bar.c)
        self._append_buffer(price)

        if not self.ready():
            return self._set_last_output(
                ScalarAlphaOutput(
                    updated_ts=alpha_input.ts,
                    value=float("nan"),
                    is_ready=False,
                ),
                updated_ts=alpha_input.ts,
            )

        if self._ema is None:
            self._ema = sum(self.buffer) / self.config.window
        else:
            self._ema = self._alpha * price + (1.0 - self._alpha) * self._ema

        return self._set_last_output(
            ScalarAlphaOutput(
                updated_ts=alpha_input.ts,
                value=self._ema,
                is_ready=True,
            ),
            updated_ts=alpha_input.ts,
        )

    def ready(self) -> bool:
        """True once the EMA window is fully populated."""

        return len(self.buffer) >= self.config.window

    def reset(self) -> None:
        """Clear internal state and cached output."""

        super().reset()
        self._ema = None
