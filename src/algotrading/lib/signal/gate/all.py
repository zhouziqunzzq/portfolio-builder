import datetime
from typing import Iterable

from .base import BaseDecisionGate

from algotrading.lib.alpha import ScalarAlphaOutput
from algotrading.lib.alpha_engine import AlphaKey, AlphaView


class AllNewerThanDecisionGate(BaseDecisionGate):
    """Gate that returns True iff.:
    - All alphas are ready, and
    - All alpha outputs are not older than the latest event timestamp."""

    def __init__(self, alpha_keys: Iterable[AlphaKey]):
        super().__init__(alpha_keys)

    def should_emit(self, view: AlphaView, latest_event_ts: datetime) -> bool:
        for key in self.alpha_keys:
            # Check alpha readiness
            output = view.get(key)
            if output is None:  # Alpha not defined or not ready
                self.log.debug(f"Gate blocked: missing output for {key}")
                return False
            if isinstance(output, ScalarAlphaOutput) and not output.is_ready:
                self.log.debug(f"Gate blocked: alpha {key} is not ready")
                return False

            # Check output freshness
            if output.updated_ts < latest_event_ts:
                self.log.debug(
                    f"Gate blocked: output for {key} is stale (updated at {output.updated_ts}, event ts {latest_event_ts})"
                )
                return False

            self.log.debug(
                f"Output for {key} is ready and fresh (updated at {output.updated_ts}, event ts {latest_event_ts})"
            )

        self.log.debug("Gate passed: all outputs are ready and fresh")
        return True
