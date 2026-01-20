from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from models import AccountSnapshot, PositionSnapshot
from models.trading import (
    BrokerCapabilities,
    InstrumentRef,
    OrderIntent,
    OrderState,
    OrderFilter,
    PreflightOrderResult,
    PlacedOrder,
    Instrument,
)


class BaseTradingAPI(ABC):
    """Broker adapter contract used by EML.

    This contract is intentionally narrow: it only includes primitives EML needs
    for execution + safety.
    """

    name: str
    log = logging.getLogger(__name__)

    @abstractmethod
    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def list_positions(self) -> List[PositionSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def get_instrument(self, instrument: InstrumentRef) -> Instrument:
        raise NotImplementedError

    @abstractmethod
    def preflight_order(self, intent: OrderIntent) -> PreflightOrderResult:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> PlacedOrder:
        raise NotImplementedError

    @abstractmethod
    def get_order(self, broker_order_id: str) -> OrderState:
        raise NotImplementedError

    @abstractmethod
    def list_orders(self, order_filter: OrderFilter) -> List[OrderState]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> None:
        raise NotImplementedError


class BaseSyncTradingAPI(BaseTradingAPI, ABC):
    """Marker base class for polling-based adapters."""


class BaseAsyncTradingAPI(BaseTradingAPI, ABC):
    """Optional extension for streaming-based adapters.

    Even async adapters should still implement all sync primitives for startup,
    recovery, and safety operations.
    """

    @abstractmethod
    def subscribe_order_updates(self) -> None:
        """Start streaming order updates."""

        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
