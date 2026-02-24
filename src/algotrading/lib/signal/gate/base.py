import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

from algotrading.lib.alpha_engine import AlphaKey, AlphaView


class BaseDecisionGate(ABC):
    """Base class for decision gates that control the flow of trading signals.
    A decision gate outputs a boolean value indicating whether a predictor should be invoked to generate a trading signal.
    """

    def __init__(self, alpha_keys: Iterable[AlphaKey]):
        super().__init__()

        self.log = logging.getLogger(self.__class__.__name__)
        # alpha_keys identify set of Alphas based on which the gate makes its decision.
        # This typically equals to the set of Alphas consumed by the downstream predictor.
        self.alpha_keys = alpha_keys

    @abstractmethod
    def should_emit(self, view: AlphaView, latest_event_ts: datetime) -> bool:
        """Determine whether the gate conditions are met based on the latest alpha outputs and event timestamp."""
        raise NotImplementedError
