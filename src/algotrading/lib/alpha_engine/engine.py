from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Type

from algotrading.lib.alpha.base import (
    BaseAlpha,
    BaseAlphaConfig,
    BaseAlphaOutput,
    MarketDataAlphaInput,
    MarketDataEvent,
    SingleInstrumentAlphaConfig,
)
from algotrading.lib.types.instruments import InstrumentRef
from algotrading.lib.types.market_data import Timeframe

from .base import AlphaKey, BaseAlphaEngine


@dataclass(frozen=True)
class _AlphaGroupKey:
    ref: InstrumentRef
    tf: Timeframe


class AlphaEngine(BaseAlphaEngine):
    """Event-driven alpha router grouped by (instrument, timeframe)."""

    def __init__(self) -> None:
        self._groups: Dict[_AlphaGroupKey, Dict[Type[BaseAlpha], BaseAlpha]] = {}

    def subscribe(
        self,
        ref: InstrumentRef,
        tf: Timeframe,
        alpha_type: Type[BaseAlpha],
        config: BaseAlphaConfig,
    ) -> BaseAlpha:
        if isinstance(config, SingleInstrumentAlphaConfig):
            if config.ref != ref or config.tf != tf:
                raise ValueError("Alpha config ref/tf does not match subscription")

        key = _AlphaGroupKey(ref=ref, tf=tf)
        group = self._groups.setdefault(key, {})
        if alpha_type in group:
            return group[alpha_type]

        instance = alpha_type(config)
        group[alpha_type] = instance
        return instance

    def update(self, event: MarketDataEvent) -> Dict[AlphaKey, BaseAlphaOutput]:
        group_key = _AlphaGroupKey(ref=event.key.ref, tf=event.key.tf)
        group = self._groups.get(group_key)
        if not group:
            return {}

        outputs: Dict[AlphaKey, BaseAlphaOutput] = {}
        alpha_input = MarketDataAlphaInput(event=event)
        for alpha_type, alpha in group.items():
            outputs[
                AlphaKey(ref=group_key.ref, tf=group_key.tf, alpha_type=alpha_type)
            ] = alpha.update(alpha_input)
        return outputs

    def ready(self, key: AlphaKey) -> bool:
        alpha = self._get_alpha(key)
        return alpha.ready() if alpha is not None else False

    def get(self, key: AlphaKey) -> Optional[BaseAlphaOutput]:
        alpha = self._get_alpha(key)
        return alpha.last_output if alpha is not None else None

    def keys(self) -> Iterable[AlphaKey]:
        for group_key, group in self._groups.items():
            for alpha_type in group.keys():
                yield AlphaKey(
                    ref=group_key.ref,
                    tf=group_key.tf,
                    alpha_type=alpha_type,
                )

    def reset(self) -> None:
        for group in self._groups.values():
            for alpha in group.values():
                alpha.reset()

    def _get_alpha(self, key: AlphaKey) -> Optional[BaseAlpha]:
        group = self._groups.get(_AlphaGroupKey(ref=key.ref, tf=key.tf))
        if not group:
            return None
        return group.get(key.alpha_type)
