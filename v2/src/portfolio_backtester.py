from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from market_data_store import MarketDataStore


@dataclass
class PortfolioBacktesterConfig:
    """
    Configuration for the PortfolioBacktester.

    Parameters
    ----------
    execution_mode : str
        Execution mode for price data: "open_to_open", "close_to_close", etc.
        The only mode supported currently is "open_to_open".
        - "open_to_open": assumes trades are executed at the open price of day t,
          and the returns earned are from open_t to open_{t+1}.
          Note: To avoid lookahead bias, the weights at day t should be based
          on information available BEFORE the open of day t,
          e.g., calculated using prior day's close.
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

    execution_mode: Optional[str] = "open_to_open"
    trading_days_per_year: Optional[int] = 252
    initial_value: Optional[float] = 100_000.0

    # Cost model
    cost_per_turnover: Optional[float] = 0.0  # e.g. commissions, fees
    bid_ask_bps_per_side: Optional[float] = 0.0  # additional spread cost knob

    # Risk-free rate (annualized) for Sharpe
    risk_free_rate_annual: Optional[float] = 0.0


@dataclass
class PortfolioBacktester:
    """
    Simple vectorized backtester for a weight-based multi-asset portfolio.

    Parameters
    ----------
    market_data_store : MarketDataStore, optional
        Market data store for price data access.
    """

    market_data_store: Optional[MarketDataStore] = None

    @staticmethod
    def _prepare_prices_and_weights(
        prices: pd.DataFrame, weights: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align and clean prices and weights DataFrames.
        Parameters
        ----------
        prices : DataFrame
            Price data: Date x Ticker
        weights : DataFrame
            Weights data: Date x Ticker
        Returns
        -------
        tuple of (prices, weights) DataFrames, aligned on dates and tickers.
        Missing weights are forward-filled and then filled with 0.0.
        """
        # Make copies to avoid mutating external DataFrames
        prices = prices.copy()
        weights = weights.copy()

        # Ensure columns are aligned & uppercase tickers
        prices.columns = [c.upper() for c in prices.columns]
        weights.columns = [c.upper() for c in weights.columns]
        # Align tickers: keep intersection
        common_cols = sorted(set(prices.columns) & set(weights.columns))
        prices = prices[common_cols]
        weights = weights[common_cols]

        # Debug: report missing dates
        missing = weights.index.difference(prices.index)
        if not missing.empty:
            print(
                "[PortfolioBacktester] WARNING: missing exec prices for rebalance dates:",
                list(missing[:10]),
                "count=",
                len(missing),
            )

        # Align date index: use prices index as master
        prices = prices.sort_index()
        weights = weights.reindex(prices.index).ffill().fillna(0.0)

        return prices, weights

    # ------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------
    @staticmethod
    def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Instrument returns from prices: simple pct change, NaN -> 0.
        """
        rets = prices.pct_change(fill_method=None).fillna(0.0)
        return rets

    @staticmethod
    def _compute_turnover(weights: pd.DataFrame) -> pd.Series:
        """
        Daily turnover = 0.5 * sum(|w_t - w_{t-1}|).
        0.5 because buying+selling is double-counted otherwise.
        """
        dw = weights.diff().abs()
        turnover = 0.5 * dw.sum(axis=1)
        turnover.iloc[0] = 0.0
        return turnover

    @staticmethod
    def _compute_drawdown_stats(
        equity: pd.Series,
    ) -> Dict[str, float | pd.Timestamp | None]:
        # running max
        running_max = equity.cummax()

        # drawdowns
        dd = (equity - running_max) / running_max

        # trough date
        trough_date = dd.idxmin()
        # trough_value = equity.loc[trough_date]

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
    def run_vec(
        self,
        weights: pd.DataFrame = None,
        config: Optional[PortfolioBacktesterConfig] = None,
        start: Optional[datetime | str] = None,
        end: Optional[datetime | str] = None,
    ) -> pd.DataFrame:
        """
        Run the vectorized backtest.
        Steps:
        - Fetch prices for all tickers in weights, aligned on dates.
        - Compute daily returns according to execution mode:
            - "open_to_open": weights at day t earn returns from open_t to open_{t+1}.
        - Given daily returns and weights, compute:
            - Gross portfolio returns before costs
            - Turnover
            - Transaction costs
            - Net portfolio returns after costs
            - Equity curve from net returns

        Parameters
        ----------
        weights: DataFrame
            Weights data: Date x Ticker.
        config : PortfolioBacktesterConfig, optional
            Backtester configuration. If None, default config is used.
        start : datetime or str, optional
            Start date for backtest (inclusive). If None, use first date in weights.
        end : datetime or str, optional
            End date for backtest (inclusive). If None, use last date in weights.

        Returns DataFrame with columns:
        - 'gross_return'     : portfolio return before costs
        - 'cost'             : transaction cost impact (rate)
        - 'portfolio_return' : net return after costs
        - 'equity'           : equity curve from net returns
        - 'turnover'         : daily turnover
        - 'weight_sum'       : equity exposure (1 - weight_sum = cash)
        """
        config = config or PortfolioBacktesterConfig()
        mode = config.execution_mode.lower().strip()
        if mode not in ("open_to_open"):
            raise ValueError(f"Unsupported execution_mode: {config.execution_mode}")

        # Collect all tickers appeared in weights
        tickers = sorted([c.upper() for c in weights.columns])
        if tickers is None or len(tickers) == 0:
            raise ValueError("Weights DataFrame must have at least one ticker column.")

        # Determine start and end dates
        if start is None:
            start = weights.index[0]
        if end is None:
            end = weights.index[-1]

        # Assemble prices for all tickers
        prices = self.market_data_store.get_ohlcv_matrix(
            tickers=tickers,
            start=start,
            end=end,
            field=self._ohlcv_field_from_execution_mode(config.execution_mode),
            interval="1d",
            auto_adjust=True,
        )

        # Prepare prices and weights
        prices, weights = self._prepare_prices_and_weights(prices, weights)

        # Returns
        rets = self._compute_returns(prices)  # default: P[t]/P[t-1]-1, indexed by t
        if mode in ("open_to_open"):
            # Weights at day t earn returns from open_t to open_{t+1}.
            # Use shift(-1) so weights[t] earn the next interval.
            aligned_rets = rets.shift(-1).fillna(0.0)

        # Raw portfolio returns before costs
        gross_rets = (weights * aligned_rets).sum(axis=1)

        # Turnover and exposure
        turnover = self._compute_turnover(weights)
        weight_sum = weights.sum(axis=1)

        # Costs
        spread_rate = 2.0 * config.bid_ask_bps_per_side / 10_000.0
        effective_cost_rate = config.cost_per_turnover + spread_rate
        cost = effective_cost_rate * turnover

        # Net returns after costs
        net_rets = gross_rets - cost

        # Equity curve from net returns
        equity = config.initial_value * (1.0 + net_rets).cumprod()

        result = pd.DataFrame(
            {
                "gross_return": gross_rets,
                "cost": cost,
                "portfolio_return": net_rets,
                "equity": equity,
                "turnover": turnover,
                "weight_sum": weight_sum,
            },
            index=prices.index,
        )
        return result

    def stats(
        self,
        result: pd.DataFrame,
        auto_warmup: bool = True,
        warmup_days: int = 0,
        config: Optional[PortfolioBacktesterConfig] = None,
    ) -> Dict[str, float | pd.Timestamp | None]:
        """
        Compute basic performance stats from result of `run()`.

        Parameters
        ----------
        result : DataFrame
            Output of self.run().
        auto_warmup : bool, default True
            If True, automatically drop the initial period where the portfolio
            is effectively not invested (weight_sum <= 1%).
        warmup_days : int, default 0
            If > 0, drop the first N days from the stats window
            *after* any auto_warmup trimming.
        """
        config = config or PortfolioBacktesterConfig()
        if result is None:
            raise ValueError("result DataFrame must be provided for stats calculation.")

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
        cagr = total_return ** (config.trading_days_per_year / n_days) - 1.0

        # ------ Vol + Sharpe with risk-free adjustment ------
        # Convert annual risk-free rate to daily
        rf_daily = (1.0 + config.risk_free_rate_annual) ** (
            1.0 / config.trading_days_per_year
        ) - 1.0

        excess_rets = port_rets - rf_daily

        mean_daily_excess = excess_rets.mean()
        std_daily = excess_rets.std()

        vol = std_daily * np.sqrt(config.trading_days_per_year)

        sharpe = (
            mean_daily_excess / std_daily * np.sqrt(config.trading_days_per_year)
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
            prev_equity = df["equity"].shift(1).fillna(config.initial_value)
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
            "InitialEquity": float(config.initial_value),
            "EffectiveStart": eff_start,
            "EffectiveEnd": eff_end,
        }

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    @staticmethod
    def _ohlcv_field_from_execution_mode(execution_mode: str) -> str:
        mode = execution_mode.lower().strip()
        if mode in ("open_to_open", "close_to_close"):
            return "Open" if "open" in mode else "Close"
        else:
            raise ValueError(f"Unsupported execution_mode: {execution_mode}")
