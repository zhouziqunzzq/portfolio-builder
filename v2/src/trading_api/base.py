from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from models import BrokerAccount, BrokerPosition
from models.trading import (
    BrokerCapabilities,
    InstrumentRef,
    OrderIntent,
    OrderState,
    PlacedOrder,
    Tradability,
)


class BaseTradingAPI(ABC):
    """Broker adapter contract used by EML.

    This contract is intentionally narrow: it only includes primitives EML needs
    for execution + safety.
    """

    name: str

    @abstractmethod
    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> BrokerAccount:
        raise NotImplementedError

    @abstractmethod
    def list_positions(self) -> List[BrokerPosition]:
        raise NotImplementedError

    @abstractmethod
    def check_tradable(self, instrument: InstrumentRef) -> Tradability:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> PlacedOrder:
        raise NotImplementedError

    @abstractmethod
    def get_order(self, broker_order_id: str) -> OrderState:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def cancel_all_open_orders(self) -> None:
        """Best-effort cancel of open orders."""

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
