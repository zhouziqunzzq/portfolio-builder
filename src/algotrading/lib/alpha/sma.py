from __future__ import annotations

from dataclasses import dataclass, field

from algotrading.lib.eventing.md_events import BarClosed, BarCompleted

from .base import (
    BaseAlpha,
    MarketDataAlphaInput,
    ScalarAlphaOutput,
    SingleInstrumentAlphaConfig,
)


@dataclass(frozen=True)
class SMAAlphaConfig(SingleInstrumentAlphaConfig):
    """Configuration for simple moving average alpha."""

    kind: str = field(default="sma", init=False)
    window: int = 20

    def id(self) -> str:
        return f"{super().id()}_window={self.window}"


class SMAAlpha(BaseAlpha[MarketDataAlphaInput, ScalarAlphaOutput, float]):
    """Event-driven SMA alpha over bar close prices."""

    def __init__(self, config: SMAAlphaConfig) -> None:
        if config.window <= 0:
            raise ValueError("SMA window must be > 0")
        super().__init__(config=config, buffer_size=config.window)
        self.config = config

    def update(self, alpha_input: MarketDataAlphaInput) -> ScalarAlphaOutput:
        """Update SMA state with a bar event and emit the latest value."""

        event = alpha_input.event
        if not isinstance(event, (BarCompleted, BarClosed)):
            raise TypeError(f"Unsupported market data event: {type(event)!r}")
        if event.key.ref != self.config.ref or event.key.tf != self.config.tf:
            raise ValueError("Event does not match SMA config ref/tf")

        self._append_buffer(float(event.bar.c))

        if not self.ready():
            return self._set_last_output(
                ScalarAlphaOutput(
                    updated_ts=alpha_input.ts,
                    value=float("nan"),
                    is_ready=False,
                ),
                updated_ts=alpha_input.ts,
            )

        sma_value = sum(self.buffer) / self.config.window
        return self._set_last_output(
            ScalarAlphaOutput(
                updated_ts=alpha_input.ts,
                value=sma_value,
                is_ready=True,
            ),
            updated_ts=alpha_input.ts,
        )

    def ready(self) -> bool:
        """True once the SMA window is fully populated."""

        return len(self.buffer) >= self.config.window
