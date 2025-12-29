from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import traceback
from typing import Dict, List, Optional, Set
from datetime import datetime

import numpy as np
import pandas as pd


# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[2]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from universe_manager import UniverseManager
from market_data_store import MarketDataStore
from signal_engine import SignalEngine
from vec_signal_engine import VectorizedSignalEngine
from context.rebalance import RebalanceContext


class BaseSleeve(ABC):
    """
    Base class for all sleeves.
    """

    def __init__(
        self,
        market_data_store: Optional[MarketDataStore] = None,
        universe_manager: Optional[UniverseManager] = None,
        signal_engine: Optional[SignalEngine] = None,
        vectorized_signal_engine: Optional[VectorizedSignalEngine] = None,
    ):
        self.universe_manager = universe_manager
        self.market_data_store = market_data_store
        self.signal_engine = signal_engine
        self.vectorized_signal_engine = vectorized_signal_engine

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    @abstractmethod
    def get_universe(self, as_of: Optional[datetime | str] = None) -> Set[str]:
        """
        Get the universe, i.e. all tickers tradable for the sleeve.
        If as_of is provided, get the universe as-of that date. Otherwise, return the
        universe of all time (including tickers no longer in the index).
        """
        raise NotImplementedError("get_universe() must be implemented by subclasses")

    # ------------------------------------------------------------------
    # Rebalancer
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_target_weights_for_date(
        self,
        as_of: datetime | str,
        start_for_signals: datetime | str,
        regime: str = "bull",
        rebalance_ctx: Optional[RebalanceContext] = None,
    ) -> Dict[str, float]:
        """
        Generate target weights for the sleeve as of the given date.

        Args:
            as_of: The date of the market data up to which sleeve is allowed to use for generating weights.
            start_for_signals: The start date for signal calculations.
            regime: The primary regime at the moment (e.g. "bull", "bear", etc.).
            rebalance_ctx: Optional RebalanceContext providing additional context.

        Returns:
            A dictionary mapping tickers to their target weights.
        """
        raise NotImplementedError(
            "generate_target_weights_for_date() must be implemented by subclasses"
        )

    @abstractmethod
    def should_rebalance(
        self,
        now: datetime | str,
    ) -> bool:
        """
        Determine whether the sleeve should rebalance at the given datetime.

        Args:
            now: The current datetime to evaluate.

        Returns:
            A boolean indicating whether to rebalance.
        """
        raise NotImplementedError(
            "should_rebalance() must be implemented by subclasses"
        )

    # ------------------------------------------------------------------
    # Precompute for vectorized signals generation
    # ------------------------------------------------------------------
    @abstractmethod
    def precompute(
        self,
        start: datetime | str,
        end: datetime | str,
        sample_dates: Optional[List[datetime | str]] = None,
        warmup_buffer: Optional[int] = None,  # in days
    ) -> pd.DataFrame:
        """
        Precompute any necessary data for vectorized signal generation. Stateful sleeves
        should store the precomputed data to its internal state for later use.

        Args:
            start: The start date for precomputation.
            end: The end date for precomputation.
            sample_dates: Optional list of specific dates to sample.
            warmup_buffer: Optional warmup buffer in days (if applicable).
        Returns:
            A DataFrame containing precomputed data.
        """
        raise NotImplementedError("precompute() must be implemented by subclasses")
