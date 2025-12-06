from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .friction_control_config import FrictionControlConfig
from .hysteresis import apply_weight_hysteresis_row
from .min_trade_notional import apply_min_trade_notional_row

import pandas as pd
import numpy as np


class FrictionController:
    """
    Applies friction controls (e.g. hysteresis, min trade notional) to target weights.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        initial_value: float = 100_000.0,
        keep_cash: bool = True,
        config: Optional[FrictionControlConfig] = None,
    ):
        """
        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of prices with dates as index and tickers as columns.
        weights : pd.DataFrame
            DataFrame of target weights with dates as index and tickers as columns.
            Note that this only includes weights on rebalance dates.
        initial_value : float
            Initial portfolio value.
        keep_cash : bool
            If True, allow the sum of weights to be less than 1 (i.e., keep cash position).
            Will still normalize if total weight exceeds 1.
        config : Optional[FrictionControlConfig]
            Configuration for friction controls. If None, default configuration is used.
        """
        self.initial_value = initial_value
        self.keep_cash = keep_cash
        self.config = config if config is not None else FrictionControlConfig()
        # Make copies to avoid mutating external DataFrames
        self.prices = prices.copy()
        self.weights = weights.copy()
        self.weights_raw = weights.copy()
        # Calculate relevant dates
        self.price_dates = self.prices.index.tolist()
        self.weights_dates = self.weights_raw.index.tolist()
        self.rebalance_dates_set = set(self.weights_dates).intersection(
            set(self.price_dates)
        )
        self.rebalance_dates = sorted(self.rebalance_dates_set)
        # Ensure columns are aligned & uppercase tickers
        self.prices.columns = [c.upper() for c in self.prices.columns]
        self.weights.columns = [c.upper() for c in self.weights.columns]
        # Align tickers: keep intersection
        common_cols = sorted(set(self.prices.columns) & set(self.weights.columns))
        self.prices = self.prices[common_cols]
        self.weights = self.weights[common_cols]
        # Align date index: use prices index as master
        self.prices = self.prices.sort_index()
        self.weights = self.weights.reindex(self.prices.index).ffill().fillna(0.0)
        # Pre-calculate returns
        self.returns = self.prices.pct_change(fill_method=None).fillna(0.0)

    def apply_all(self) -> pd.DataFrame:
        """
        Apply all friction controls to the weights DataFrame.
        """
        aum = self.initial_value
        W_final_vals = []  # list of pd.Series; final weights for rebalance dates
        w_prev = None  # previous effective weights on last rebalance date

        for date in self.price_dates:
            w_t = self.weights.loc[date]

            if date in self.rebalance_dates_set:
                print(f"Applying friction controls on rebalance date {date.date()}")
                # On rebalance date, apply friction controls
                if w_prev is None:
                    # First rebalance, no previous weights
                    w_eff = w_t.copy()
                else:
                    # Apply hysteresis
                    w_hyst = apply_weight_hysteresis_row(
                        w_prev,
                        w_t,
                        dw_min=self.config.hysteresis_dw_min,
                        keep_cash=self.keep_cash,
                    )
                    # Apply min trade notional
                    w_eff = apply_min_trade_notional_row(
                        w_prev,
                        w_hyst,
                        portfolio_value=aum,
                        min_trade_abs=self.config.min_trade_notional_abs,
                        min_trade_pct_of_aum=self.config.min_trade_pct_of_aum,
                        keep_cash=self.keep_cash,
                    )
                W_final_vals.append(w_eff.copy())
                w_prev = w_eff
            else:
                # Non-rebalance date, carry forward previous weights
                w_eff = w_prev if w_prev is not None else w_t

            # Update AUM for next day
            daily_return = (self.returns.loc[date] * w_eff).sum()
            aum *= 1.0 + daily_return

        # print(W_final_vals)
        W_eff = pd.DataFrame(W_final_vals, index=self.rebalance_dates)
        return W_eff.reindex(self.weights_dates).fillna(0.0)
