import pandas as pd
import numpy as np

def compute_benchmark(prices, start, end, price_col: str = "Close", rf: float = 0.0, trading_days: int = 252):
    """
    Compute simple benchmark statistics from a price series or DataFrame.

    Parameters
    - prices: pd.DataFrame or pd.Series containing price data (DatetimeIndex).
    - start, end: slice bounds (datetime-like) to limit the period.
    - price_col: if `prices` is a DataFrame, the column name to use (default 'Close').
    - rf: risk-free rate (annual) used in Sharpe calculation.
    - trading_days: number of trading days per year for annualization (default 252).

    Returns a dict with keys: 'CAGR', 'Volatility', 'Sharpe', 'MaxDrawdown', 'returns', 'equity'
    where 'returns' and 'equity' are pandas Series indexed by date.
    """
    # Accept either a Series or DataFrame
    if isinstance(prices, pd.DataFrame):
        if price_col not in prices.columns:
            raise ValueError(f"price_col '{price_col}' not found in DataFrame")
        price_ser = prices[price_col].loc[start:end].astype(float).copy()
    else:
        # assume Series
        price_ser = prices.loc[start:end].astype(float).copy()

    # Ensure datetime index and sorted
    price_ser.index = pd.to_datetime(price_ser.index)
    price_ser = price_ser.sort_index()

    # Compute simple returns and equity curve
    ret = price_ser.pct_change().fillna(0)
    equity = (1 + ret).cumprod()

    # Guard against too-short windows
    if len(equity) < 2:
        cagr = float('nan')
        vol = float('nan')
        sharpe = float('nan')
        maxdd = float('nan')
    else:
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        if years <= 0:
            cagr = float('nan')
        else:
            # final equity^(1/years)-1
            try:
                cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
            except Exception:
                cagr = float('nan')

        vol = ret.std() * np.sqrt(trading_days)
        sharpe = ((ret.mean() * trading_days) - rf) / vol if vol and not np.isnan(vol) else float('nan')

        # Max drawdown
        peak = equity.cummax()
        dd = (equity - peak) / peak
        maxdd = dd.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": maxdd,
        "returns": ret,
        "equity": equity,
    }
