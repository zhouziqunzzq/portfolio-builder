from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Mapping
from .friction_control_config import FrictionControlConfig
from .hysteresis import apply_weight_hysteresis_row
from .min_trade_notional import apply_min_trade_notional_row
from .min_holding_period import apply_min_holding_period_row

# Make v2/src importable by adding it to sys.path. This allows using
# direct module imports (e.g. `from universe_manager import ...`) rather
# than referencing the `src.` package namespace.
import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from context.friction_control import FrictionControlContext

import pandas as pd
import numpy as np


class FrictionControllerVec:
    """
    Vectorized Friction Controller.
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
        # Print some warning if rebalance dates were dropped due to missing prices
        dropped_dates = set(self.weights_dates) - self.rebalance_dates_set
        if dropped_dates:
            print(
                f"[FrictionController] Warning: Dropped {len(dropped_dates)} rebalance dates due to missing price data: {sorted(dropped_dates)}"
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
        # Holding age state (rebalance steps)
        holding_age = pd.Series(0, index=self.weights.columns, dtype=int)

        for date in self.price_dates:
            w_t = self.weights.loc[date]

            if date in self.rebalance_dates_set:
                # print(f"Applying friction controls on rebalance date {date.date()}")
                # On rebalance date, apply friction controls
                if w_prev is None:
                    # First rebalance, no previous weights
                    w_eff = w_t.copy()
                    opened = w_eff > 0
                    holding_age[opened] = 1
                    holding_age[~opened] = 0
                else:
                    # Apply hysteresis
                    w_hyst = apply_weight_hysteresis_row(
                        w_prev,
                        w_t,
                        dw_min=self.config.hysteresis_dw_min,
                        keep_cash=self.keep_cash,
                    )
                    # Apply min trade notional
                    w_after_notional = apply_min_trade_notional_row(
                        w_prev,
                        w_hyst,
                        portfolio_value=aum,
                        min_trade_abs=self.config.min_trade_notional_abs,
                        min_trade_pct_of_aum=self.config.min_trade_pct_of_aum,
                        keep_cash=self.keep_cash,
                    )
                    # Apply min holding period
                    w_eff, holding_age = apply_min_holding_period_row(
                        w_prev=w_prev,
                        w_proposed=w_after_notional,
                        holding_age=holding_age,
                        min_holding_rebalances=self.config.min_holding_rebalances,
                        keep_cash=self.keep_cash,
                    )
                    # Debug: print intermediate weights
                    # print(
                    #     f"  Raw target weights: {', '.join([f'{t}:{v:.4f}' for t,v in w_t.items() if v > 0.0])}"
                    # )
                    # print(
                    #     f"  Previous effective weights: "
                    #     f"{', '.join([f'{t}:{v:.4f}' for t,v in w_prev.items() if v > 0.0])}"
                    # )
                    # print(
                    #     f"  Post-hysteresis weights: "
                    #     f"{', '.join([f'{t}:{v:.4f}' for t,v in w_hyst.items() if v > 0.0])}"
                    # )
                    # print(
                    #     f"  Post-min-trade-notional weights: "
                    #     f"{', '.join([f'{t}:{v:.4f}' for t,v in w_after_notional.items() if v > 0.0])}"
                    # )
                    # print(
                    #     f"  Post-min-holding-period weights: "
                    #     f"{', '.join([f'{t}:{v:.4f}' for t,v in w_eff.items() if v > 0.0])}"
                    # )
                    # print(
                    #     f"  Updated holding ages: "
                    #     f"{', '.join([f'{t}:{v}' for t,v in holding_age.items() if v > 0])}"
                    # )
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


@dataclass
class FrictionControllerState:
    holding_age: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))


def _as_weight_series(w: Optional[object]) -> pd.Series:
    """Coerce weights to a float pd.Series; None -> empty series.

    Normalizes ticker labels to uppercase when they are strings.
    """
    if w is None:
        s = pd.Series(dtype=float)
    elif isinstance(w, pd.Series):
        s = w.copy()
    elif isinstance(w, Mapping):
        s = pd.Series(dict(w), dtype=float)
    else:
        raise TypeError(
            "Weights must be a mapping like Dict[str, float] (or a pd.Series); "
            f"got {type(w)!r}"
        )

    if len(s.index) > 0:
        s.index = [i.upper() if isinstance(i, str) else i for i in s.index]
    return s.astype(float).fillna(0.0)


class FrictionController:
    """Stateful (non-vectorized) friction controller.

    Intended for live / step-by-step use: caller provides previous and current
    target weights each rebalance, and the controller returns the effective
    weights after applying all configured frictions.
    """

    def __init__(
        self,
        keep_cash: bool = True,
        config: Optional[FrictionControlConfig] = None,
        state: Optional[FrictionControllerState] = None,
    ):
        self.keep_cash = keep_cash
        self.config = config if config is not None else FrictionControlConfig()
        self.state = state if state is not None else FrictionControllerState()

    def apply(
        self,
        w_prev: Optional[Dict[str, float]],
        w_new: Dict[str, float],
        ctx: FrictionControlContext,
    ) -> Dict[str, float]:
        """Apply friction controls for a single rebalance step.

        Parameters
        ----------
        w_prev : Optional[Dict[str, float]]
            Previous effective weights (post-friction) from the prior rebalance.
            If None, treated as all-cash / zero positions.
        w_new : Dict[str, float]
            Proposed new target weights for the current rebalance.
        ctx : FrictionControlContext
            Context for friction controls (currently only includes `aum`).

        Returns
        -------
        Dict[str, float]
            Effective weights after applying hysteresis, min trade notional,
            and min holding period.
        """
        w_prev_s = _as_weight_series(w_prev)
        w_new_s = _as_weight_series(w_new)

        idx = w_prev_s.index.union(w_new_s.index).union(self.state.holding_age.index)
        w_prev_s = w_prev_s.reindex(idx).fillna(0.0)
        w_new_s = w_new_s.reindex(idx).fillna(0.0)
        holding_age = self.state.holding_age.reindex(idx).fillna(0).astype(int)

        w_hyst = apply_weight_hysteresis_row(
            w_prev_s,
            w_new_s,
            dw_min=self.config.hysteresis_dw_min,
            keep_cash=self.keep_cash,
        )

        w_after_notional = apply_min_trade_notional_row(
            w_prev_s,
            w_hyst,
            portfolio_value=float(ctx.aum),
            min_trade_abs=self.config.min_trade_notional_abs,
            min_trade_pct_of_aum=self.config.min_trade_pct_of_aum,
            keep_cash=self.keep_cash,
        )

        w_eff, holding_age_next = apply_min_holding_period_row(
            w_prev=w_prev_s,
            w_proposed=w_after_notional,
            holding_age=holding_age,
            min_holding_rebalances=self.config.min_holding_rebalances,
            keep_cash=self.keep_cash,
        )

        self.state.holding_age = holding_age_next
        return {str(k): float(v) for k, v in w_eff.items()}
