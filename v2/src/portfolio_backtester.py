from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class PortfolioBacktester:
    """
    Simple vectorized backtester for a weight-based multi-asset portfolio.

    - prices: Date x Ticker (Open or Close prices on those dates)
    - weights: Date x Ticker (target weights on those dates)

    Parameters
    ----------
    trading_days_per_year : int
        Used for annualizing returns/vol.
    initial_value : float
        Starting portfolio value.
    cost_per_turnover : float
        Transaction cost as a *rate* per unit of turnover.
        Example: 0.001 = 10 bps per full (100%) turnover.
    bid_ask_bps_per_side : float
        Additional bid-ask cost per side, in basis points.
        For example, 5 bps per side ~ 10 bps round-trip.
        This is also applied proportional to turnover.
    risk_free_rate_annual : float
        Annualized risk-free rate (e.g. 0.04 for 4%).
        Used to compute excess-return Sharpe.
    """

    prices: pd.DataFrame
    weights: pd.DataFrame
    trading_days_per_year: int = 252
    initial_value: float = 100_000.0

    # Cost model
    cost_per_turnover: float = 0.0  # e.g. commissions, fees
    bid_ask_bps_per_side: float = 0.0  # additional spread cost knob

    # Risk-free rate (annualized) for Sharpe
    risk_free_rate_annual: float = 0.0

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

    def _compute_drawdown_stats(
        self, equity: pd.Series
    ) -> Dict[str, float | pd.Timestamp | None]:
        # running max
        running_max = equity.cummax()

        # drawdowns
        dd = (equity - running_max) / running_max

        # trough date
        trough_date = dd.idxmin()
        trough_value = equity.loc[trough_date]

        # the peak is the last time running_max == equity BEFORE trough
        peak_date = equity.loc[:trough_date].idxmax()
        peak_value = equity.loc[peak_date]

        # find recovery: first date AFTER trough when equity regains peak
        recovery_candidates = equity.loc[trough_date:].index[
            equity.loc[trough_date:] >= peak_value
        ]
        recovery_date = recovery_candidates[0] if len(recovery_candidates) > 0 else None

        return {
            "mdd": dd.min(),
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": recovery_date,
            "days_in_drawdown": (
                (recovery_date - peak_date).days if recovery_date else None
            ),
        }

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def run(self, apply_weights_to: str = "next") -> pd.DataFrame:
        """
        Run the backtest.

        Parameters
        ----------
        apply_weights_to : {"prev", "same", "next"}, default "next"
            Controls which return interval the weights at date t are applied to.

            - "prev": weights[t] * return[t]   where return[t] is (P[t]/P[t-1]-1)
                    => interprets weights as being in force over (t-1 -> t).
                    (Generally WRONG if you trade at t.)

            - "same": weights[t] * return[t]   same as "prev" in this implementation
                    (kept for readability / future extension)

            - "next": weights[t] * return_next[t] where return_next[t] is (P[t+1]/P[t]-1)
                    => interprets weights as being established at t (after trading at t)
                        and earning (t -> t+1). This is what you want if your weights
                        matrix is indexed by the trade/execution date (e.g., trade at t open/close).

        Returns DataFrame with columns:
        - 'gross_return'     : portfolio return before costs
        - 'cost'             : transaction cost impact (rate)
        - 'portfolio_return' : net return after costs
        - 'equity'           : equity curve from net returns
        - 'turnover'         : daily turnover
        - 'weight_sum'       : equity exposure (1 - weight_sum = cash)
        """
        mode = (apply_weights_to or "next").lower().strip()
        if mode not in ("prev", "same", "next"):
            raise ValueError(
                f"apply_weights_to must be one of {{'prev','same','next'}}, got: {apply_weights_to}"
            )

        # ------------------------------------------------------------
        # Returns
        # ------------------------------------------------------------
        rets = self._compute_returns()  # default: P[t]/P[t-1]-1, indexed by t

        if mode in ("prev", "same"):
            aligned_rets = rets
        else:
            # "next": make return at t represent (t -> t+1)
            # Use shift(-1) so weights[t] earn the next interval.
            aligned_rets = rets.shift(-1).fillna(0.0)

        # Raw portfolio returns before costs
        gross_rets = (self.weights * aligned_rets).sum(axis=1)

        # ------------------------------------------------------------
        # Turnover and exposure
        # ------------------------------------------------------------
        turnover = self._compute_turnover(self.weights)
        weight_sum = self.weights.sum(axis=1)

        # ------------------------------------------------------------
        # Costs
        # ------------------------------------------------------------
        spread_rate = 2.0 * self.bid_ask_bps_per_side / 10_000.0
        effective_cost_rate = self.cost_per_turnover + spread_rate
        cost = effective_cost_rate * turnover

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

        # ------ Vol + Sharpe with risk-free adjustment ------
        # Convert annual risk-free rate to daily
        rf_daily = (1.0 + self.risk_free_rate_annual) ** (
            1.0 / self.trading_days_per_year
        ) - 1.0

        excess_rets = port_rets - rf_daily

        mean_daily_excess = excess_rets.mean()
        std_daily = excess_rets.std()

        vol = std_daily * np.sqrt(self.trading_days_per_year)

        sharpe = (
            mean_daily_excess / std_daily * np.sqrt(self.trading_days_per_year)
            if std_daily > 0
            else np.nan
        )

        # ------ Skewness of returns ------
        # Measure asymmetry of the net (post-cost) portfolio return distribution.
        # Use pandas' sample skew (Fisher's definition, unbiased) on the
        # portfolio returns within the effective window.
        skewness = float(port_rets.skew()) if len(port_rets) > 0 else np.nan

        # ------ Kurtosis of returns ------
        # Measure the tailedness of the net (post-cost) portfolio return distribution.
        # Use pandas' kurtosis (excess kurtosis by default) on the portfolio returns
        # within the effective window.
        kurtosis = float(port_rets.kurtosis()) if len(port_rets) > 0 else np.nan

        # ------ Max drawdown and detailed drawdown stats ------
        equity = df["equity"]
        dd_stats = self._compute_drawdown_stats(equity)
        max_dd = dd_stats.get("mdd")

        # ------ Turnover ------
        avg_turnover = df["turnover"].mean()

        # ------ Total costs (monetary) ------
        # If the run() result included a 'cost' column (daily cost as a
        # return-rate), approximate the monetary cost on each day as
        # cost_rate * equity_{t-1} (previous day's equity). The very first
        # day's previous equity is approximated with `initial_value`.
        total_cost_amount = np.nan
        if "cost" in df.columns and "equity" in df.columns:
            prev_equity = df["equity"].shift(1).fillna(self.initial_value)
            cost_amounts = df["cost"] * prev_equity
            total_cost_amount = float(cost_amounts.sum())

        return {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "MaxDrawdown": max_dd,
            # Detailed drawdown info (may be None)
            "DDPeakDate": dd_stats.get("peak_date"),
            "DDTroughDate": dd_stats.get("trough_date"),
            "DDRecoveryDate": dd_stats.get("recovery_date"),
            "DaysInDrawdown": dd_stats.get("days_in_drawdown"),
            "AvgDailyTurnover": avg_turnover,
            "TotalCost": total_cost_amount,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "InitialEquity": float(self.initial_value),
            "EffectiveStart": eff_start,
            "EffectiveEnd": eff_end,
        }
