from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class PortfolioBacktester:
    """
    Simple vectorized backtester for a weight-based multi-asset portfolio.

    - prices: Date x Ticker (Close)
    - weights: Date x Ticker (target weights on those dates)
    """

    prices: pd.DataFrame
    weights: pd.DataFrame
    trading_days_per_year: int = 252
    initial_value: float = 100_000.0
    cost_per_turnover: float = 0.0  # e.g. 0.001 = 10 bps per full turnover

    def __post_init__(self):
        # Make copies to avoid mutating external DataFrames
        self.prices = self.prices.copy()
        self.weights = self.weights.copy()

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

    # ------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------
    def _compute_returns(self) -> pd.DataFrame:
        """
        Instrument returns from prices: simple pct change, NaN -> 0.
        """
        # Explicit fill_method=None to avoid deprecated implicit forward-fill
        rets = self.prices.pct_change(fill_method=None).fillna(0.0)
        return rets

    def _compute_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """
        Daily turnover = 0.5 * sum(|w_t - w_{t-1}|).
        0.5 because buying+selling is double-counted otherwise.
        """
        dw = weights.diff().abs()
        turnover = 0.5 * dw.sum(axis=1)
        turnover.iloc[0] = 0.0
        return turnover

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Run the backtest.
        Returns DataFrame with columns:
        - 'gross_return'     : portfolio return before costs
        - 'cost'             : transaction cost impact
        - 'portfolio_return' : net return after costs
        - 'equity'           : equity curve from net returns
        - 'turnover'         : daily turnover
        - 'weight_sum'       : equity exposure (1 - weight_sum = cash)
        """
        rets = self._compute_returns()

        # Raw portfolio returns before costs
        gross_rets = (self.weights * rets).sum(axis=1)

        # Turnover and exposure
        turnover = self._compute_turnover(self.weights)
        weight_sum = self.weights.sum(axis=1)

        # Transaction costs: proportional to turnover
        # cost_per_turnover is interpreted as cost when turnover = 1.0 (i.e. 100% of portfolio traded)
        cost = self.cost_per_turnover * turnover

        # Net returns after costs
        net_rets = gross_rets - cost

        # Equity curve from net returns
        equity = self.initial_value * (1.0 + net_rets).cumprod()

        result = pd.DataFrame(
            {
                "gross_return": gross_rets,
                "cost": cost,
                "portfolio_return": net_rets,
                "equity": equity,
                "turnover": turnover,
                "weight_sum": weight_sum,
            },
            index=self.prices.index,
        )
        return result

    def stats(
        self,
        result: Optional[pd.DataFrame] = None,
        auto_warmup: bool = True,
        warmup_days: int = 0,
    ) -> Dict[str, float | pd.Timestamp | None]:
        """
        Compute basic performance stats from result of `run()`.

        Parameters
        ----------
        result : DataFrame, optional
            Output of self.run(). If None, run() is called internally.
        auto_warmup : bool, default True
            If True, automatically drop the initial period where the portfolio
            is effectively not invested (weight_sum <= 1%).
        warmup_days : int, default 0
            If > 0, drop the first N days from the stats window
            *after* any auto_warmup trimming.
        """
        if result is None:
            result = self.run()

        df = result.copy()

        # -------- auto warmup: start when we actually have exposure --------
        if auto_warmup and "weight_sum" in df.columns:
            live_mask = df["weight_sum"] > 0.01  # >1% invested
            if live_mask.any():
                first_live_idx = live_mask.idxmax()
                df = df.loc[first_live_idx:]

        # -------- optional fixed warmup days --------
        if warmup_days > 0 and len(df) > warmup_days:
            df = df.iloc[warmup_days:]

        port_rets = df["portfolio_return"]

        # Guard: if we trimmed too much
        if len(port_rets) == 0:
            return {
                "CAGR": np.nan,
                "Volatility": np.nan,
                "Sharpe": np.nan,
                "MaxDrawdown": np.nan,
                "AvgDailyTurnover": np.nan,
                "EffectiveStart": None,
                "EffectiveEnd": None,
            }

        # Effective window
        eff_start = df.index[0]
        eff_end = df.index[-1]

        # ------ CAGR ------
        n_days = len(port_rets)
        total_return = (1.0 + port_rets).prod()
        cagr = total_return ** (self.trading_days_per_year / n_days) - 1.0

        # ------ Vol + Sharpe (correct) ------
        mean_daily = port_rets.mean()
        std_daily = port_rets.std()
        vol = std_daily * np.sqrt(self.trading_days_per_year)

        sharpe = (
            mean_daily / std_daily * np.sqrt(self.trading_days_per_year)
            if std_daily > 0
            else np.nan
        )

        # ------ Max drawdown ------
        equity = df["equity"]
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_dd = drawdown.min()

        # ------ Turnover ------
        avg_turnover = df["turnover"].mean()

        return {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "MaxDrawdown": max_dd,
            "AvgDailyTurnover": avg_turnover,
            "EffectiveStart": eff_start,
            "EffectiveEnd": eff_end,
        }
