from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Type

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

from .base import AlphaKey, AlphaView, BaseAlphaEngine


@dataclass(frozen=True)
class _AlphaGroupKey:
    ref: InstrumentRef
    tf: Timeframe


class AlphaEngine(BaseAlphaEngine):
    """Event-driven alpha router grouped by (instrument, timeframe)."""

    def __init__(self) -> None:
        self._groups: Dict[_AlphaGroupKey, Dict[Tuple[Type[BaseAlpha], str], BaseAlpha]] = {}

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
        alpha_id = config.id()
        group = self._groups.setdefault(key, {})
        instance_key = (alpha_type, alpha_id)
        if instance_key in group:
            return group[instance_key]

        instance = alpha_type(config)
        group[instance_key] = instance
        return instance

    def update(self, event: MarketDataEvent) -> Dict[AlphaKey, BaseAlphaOutput]:
        group_key = _AlphaGroupKey(ref=event.key.ref, tf=event.key.tf)
        group = self._groups.get(group_key)
        if not group:
            return {}

        outputs: Dict[AlphaKey, BaseAlphaOutput] = {}
        ts = event.key.start_ts
        alpha_input = MarketDataAlphaInput(event=event, ts=ts)
        for (alpha_type, alpha_id), alpha in group.items():
            outputs[
                AlphaKey(
                    ref=group_key.ref,
                    tf=group_key.tf,
                    alpha_type=alpha_type,
                    alpha_id=alpha_id,
                )
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
            for alpha_type, alpha_id in group.keys():
                yield AlphaKey(
                    ref=group_key.ref,
                    tf=group_key.tf,
                    alpha_type=alpha_type,
                    alpha_id=alpha_id,
                )

    def reset(self) -> None:
        for group in self._groups.values():
            for alpha in group.values():
                alpha.reset()

    def get_view(self, keys: Iterable[AlphaKey]) -> AlphaView:
        outputs: Dict[AlphaKey, Optional[BaseAlphaOutput]] = {}
        for key in keys:
            outputs[key] = self.get(key)
        return AlphaView(outputs=outputs)

    def _get_alpha(self, key: AlphaKey) -> Optional[BaseAlpha]:
        group = self._groups.get(_AlphaGroupKey(ref=key.ref, tf=key.tf))
        if not group:
            return None
        return group.get((key.alpha_type, key.alpha_id))
