import numpy as np
import pandas as pd

from v2.src.portfolio_backtester import PortfolioBacktester, PortfolioBacktesterConfig
from v2.src.context.rebalance import RebalanceContext


class _StubMDS:
    def __init__(self, prices: pd.DataFrame):
        self._prices = prices.copy()

    def get_ohlcv_matrix(
        self,
        tickers,
        start,
        end,
        field=None,
        interval="1d",
        auto_adjust=True,
        local_only=None,
    ):
        # Ignore field/interval for tests; return the provided matrix clipped to dates.
        df = self._prices.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df.sort_index()
        start_ts = pd.to_datetime(start).normalize()
        end_ts = pd.to_datetime(end).normalize()
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        cols = [str(c).upper() for c in df.columns]
        df.columns = cols
        want = [str(t).upper() for t in tickers]
        return df[want]


def test_run_matches_run_vec_for_same_weights():
    # 4 trading days, 2 tickers. Simple deterministic open-to-open moves.
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]).normalize()
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0, 121.0],  # +10%, +10%, 0%
            "BBB": [200.0, 190.0, 180.5, 189.525],  # -5%, -5%, +5%
        },
        index=dates,
    )

    mds = _StubMDS(prices)
    bt = PortfolioBacktester(market_data_store=mds)

    config = PortfolioBacktesterConfig(
        execution_mode="open_to_open",
        initial_value=100_000.0,
        cost_per_turnover=0.001,
        bid_ask_bps_per_side=5.0,
    )

    # Weights that rebalance on 2024-01-03.
    weights_map = {
        pd.Timestamp("2024-01-02"): {"AAA": 1.0},
        pd.Timestamp("2024-01-03"): {"BBB": 1.0},
        pd.Timestamp("2024-01-04"): {"BBB": 1.0},
        pd.Timestamp("2024-01-05"): {"BBB": 1.0},
    }

    def gen_weights_fn(as_of, rebalance_ctx: RebalanceContext):
        # Use rebalance date (execution date) to pick weights.
        ts = pd.to_datetime(rebalance_ctx.rebalance_ts).normalize()
        return weights_map.get(ts, {}), {"picked_for": ts}

    # Non-vec run (generates weights per trading day)
    result_run, ctx = bt.run_iterative(
        start=dates[0],
        end=dates[-1],
        universe=["AAA", "BBB"],
        gen_weights_fn=gen_weights_fn,
        config=config,
    )

    # Build weights df for run_vec
    w_df = pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"])
    for dt, w in weights_map.items():
        for k, v in w.items():
            w_df.at[pd.Timestamp(dt), k] = float(v)

    result_vec = bt.run_vectorized(weights=w_df, config=config, start=dates[0], end=dates[-1])

    # Same columns and index
    assert list(result_run.columns) == list(result_vec.columns)
    assert result_run.index.equals(result_vec.index)

    # Values should match closely
    for col in result_vec.columns:
        a = result_run[col].to_numpy(dtype=float)
        b = result_vec[col].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12), f"Mismatch in column {col}"

    # Context collected for each trading day
    assert len(ctx) == len(dates)
