from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional, Dict, Tuple, ClassVar
import logging

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_ROOT_SRC = Path(__file__).resolve().parent
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from market_data_store import MarketDataStore
from context.rebalance import RebalanceContext


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
    # Class logger (excluded from dataclass fields via ClassVar)
    log: ClassVar[logging.Logger] = logging.getLogger("PortfolioBacktester")

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
            PortfolioBacktester.log.warning(
                "missing exec prices for rebalance dates=%s count=%d",
                list(missing[:10]),
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
    def run_vectorized(
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
            interval="1d",  # backtest at daily frequency
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

    def run_iterative(
        self,
        start: datetime | str,
        end: datetime | str,
        universe: List[str],
        gen_weights_fn: Callable[
            [datetime | str, RebalanceContext],  # Inputs: as_of date, rebalance context
            tuple[Dict[str, float], Dict[str, Any]],  # Outputs: weights dict, info dict
        ],
        config: Optional[PortfolioBacktesterConfig] = None,
    ) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, Dict[str, Any]]]:
        """Run an *iterative* (non-vectorized) daily backtest.

        This method mirrors :meth:`run_vectorized` as closely as possible, but instead
        of taking a pre-built weight matrix it calls ``gen_weights_fn`` on each
        trading day to obtain that day's target weights.

        Execution model
        ---------------
        Only ``execution_mode="open_to_open"`` is supported.

        - Trades are assumed executed at the *open* of day ``t``.
        - The portfolio then earns returns from ``open_t`` to ``open_{t+1}``.
        - The last available date has no next open; its instrument returns are
          treated as 0.0 (same behavior as the vectorized ``shift(-1).fillna(0)``).

        Lookahead avoidance
        -------------------
        ``gen_weights_fn`` is called with ``as_of`` equal to the prior trading day
        when available (otherwise ``t - 1 calendar day`` for the first row). This
        is intended to prevent using information from day ``t`` to set weights
        for trades executed at the open of day ``t``.

        Costs and turnover
        ------------------
        Turnover and costs match :meth:`run_vec`:

        - ``turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|`` (first day = 0.0)
        - ``effective_cost_rate = cost_per_turnover + 2 * bid_ask_bps_per_side / 10_000``
        - ``cost_t = effective_cost_rate * turnover_t``
        - ``portfolio_return_t = gross_return_t - cost_t``

        Parameters
        ----------
        start, end : datetime | str
            Inclusive start/end of the backtest window.
        universe : list[str]
            Universe of tickers to fetch prices for. Tickers are uppercased.
        gen_weights_fn : Callable[[datetime | str, RebalanceContext], (dict, dict)]
            Callback invoked once per trading day.

            It receives:
            - ``as_of``: the date whose information should be used to form weights
            - ``rebalance_ctx``: a :class:`RebalanceContext` with
              ``rebalance_ts`` set to the current trading date and ``aum`` set to
              the portfolio equity entering the day.

            It must return:
            - weights: ``{ticker -> weight}`` (tickers uppercased; missing tickers
              are treated as 0)
            - info: arbitrary context dict captured and returned to the caller
        config : PortfolioBacktesterConfig, optional
            Backtest configuration (costs, initial equity, execution_mode, etc.).

        Returns
        -------
        (result, contexts) : (pd.DataFrame, dict[pd.Timestamp, dict[str, Any]])
            ``result`` mirrors :meth:`run_vectorized` exactly, with columns:
            - ``gross_return``
            - ``cost``
            - ``portfolio_return``
            - ``equity``
            - ``turnover``
            - ``weight_sum``

            ``contexts`` maps each trading date (``pd.Timestamp``) to the ``info``
            dict returned by ``gen_weights_fn`` for that date.
        """
        config = config or PortfolioBacktesterConfig()
        mode = config.execution_mode.lower().strip()
        if mode not in ("open_to_open"):
            raise ValueError(f"Unsupported execution_mode: {config.execution_mode}")

        if self.market_data_store is None:
            raise ValueError("market_data_store must be provided to run backtests.")

        # Universe tickers
        tickers = sorted({str(t).upper() for t in (universe or []) if str(t).strip()})
        if len(tickers) == 0:
            raise ValueError("Universe must contain at least one ticker.")

        # Build exec-price matrix for specified universe and date range
        prices = self.market_data_store.get_ohlcv_matrix(
            tickers=tickers,
            start=start,
            end=end,
            field=self._ohlcv_field_from_execution_mode(config.execution_mode),
            interval="1d",
            auto_adjust=True,
        )
        if prices is None or prices.empty:
            raise ValueError(
                "No price data returned for requested universe/date range."
            )
        prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
        prices = prices.sort_index()
        prices.columns = [str(c).upper() for c in prices.columns]

        # Ensure we only use tickers we have prices for
        tickers = [t for t in tickers if t in prices.columns]
        if len(tickers) == 0:
            raise ValueError("No overlapping tickers between universe and price data.")
        prices = prices[tickers]

        dates = pd.DatetimeIndex(prices.index)
        if len(dates) == 0:
            raise ValueError("No trading dates available for requested range.")

        # Cost knobs (mirror run_vec)
        spread_rate = 2.0 * float(config.bid_ask_bps_per_side) / 10_000.0
        effective_cost_rate = float(config.cost_per_turnover) + spread_rate

        contexts: Dict[pd.Timestamp, Dict[str, Any]] = {}

        # State
        prev_weights = np.zeros(len(tickers), dtype=float)
        prev_equity = float(config.initial_value)

        gross_returns: list[float] = []
        costs: list[float] = []
        net_returns: list[float] = []
        equities: list[float] = []
        turnovers: list[float] = []
        weight_sums: list[float] = []

        # Iterate over each trading day, using open_t -> open_{t+1} returns.
        for i, ts in enumerate(dates):
            # as_of: use prior trading day when available; otherwise use ts - 1 calendar day
            # IMPORTANT: by ensuring as_of < ts, lookahead bias should be avoided.
            if i > 0:
                as_of = dates[i - 1]
            else:
                as_of = ts - pd.Timedelta(days=1)

            rebalance_ctx = RebalanceContext(
                rebalance_ts=ts,
                aum=prev_equity,
            )

            w_dict, info = gen_weights_fn(as_of, rebalance_ctx)
            contexts[pd.Timestamp(ts)] = info

            w_up = {str(k).upper(): float(v) for k, v in (w_dict or {}).items()}
            w_vec = np.array([w_up.get(t, 0.0) for t in tickers], dtype=float)
            w_vec = np.nan_to_num(w_vec, nan=0.0, posinf=0.0, neginf=0.0)

            if i == 0:
                turnover = 0.0
            else:
                turnover = 0.5 * float(np.abs(w_vec - prev_weights).sum())

            # Instrument returns for this interval.
            # Returns from open_t to open_{t+1}
            if i < len(dates) - 1:
                p0 = prices.loc[ts].to_numpy(dtype=float)
                p1 = prices.loc[dates[i + 1]].to_numpy(dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    inst_ret = np.where(p0 > 0.0, (p1 / p0) - 1.0, 0.0)
                inst_ret = np.nan_to_num(inst_ret, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                inst_ret = np.zeros(len(tickers), dtype=float)

            gross_ret = float(np.dot(w_vec, inst_ret))
            cost = float(effective_cost_rate * turnover)
            net_ret = float(gross_ret - cost)

            equity = float(prev_equity * (1.0 + net_ret))

            gross_returns.append(gross_ret)
            costs.append(cost)
            net_returns.append(net_ret)
            equities.append(equity)
            turnovers.append(turnover)
            weight_sums.append(float(w_vec.sum()))

            prev_weights = w_vec
            prev_equity = equity

        result = pd.DataFrame(
            {
                "gross_return": pd.Series(gross_returns, index=dates, dtype=float),
                "cost": pd.Series(costs, index=dates, dtype=float),
                "portfolio_return": pd.Series(net_returns, index=dates, dtype=float),
                "equity": pd.Series(equities, index=dates, dtype=float),
                "turnover": pd.Series(turnovers, index=dates, dtype=float),
                "weight_sum": pd.Series(weight_sums, index=dates, dtype=float),
            },
            index=dates,
        )

        # Match run_vectorized: index should reflect the exec price index
        result = result.reindex(prices.index)
        return result, contexts

    def stats(
        self,
        result: pd.DataFrame,
        auto_warmup: bool = True,
        warmup_days: int = 0,
        config: Optional[PortfolioBacktesterConfig] = None,
    ) -> Dict[str, float | pd.Timestamp | None]:
        """
        Compute basic performance stats from result of the backtest run.

        Parameters
        ----------
        result : DataFrame
            Output of self.run_iterative() or self.run_vectorized().
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
