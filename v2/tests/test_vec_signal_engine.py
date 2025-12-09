import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from v2.src.vec_signal_engine import VectorizedSignalEngine


def test_get_ts_momentum_table_driven():
    """Table-driven tests for `VectorizedSignalEngine.get_ts_momentum`.

    Each case contains a price matrix and lookbacks. Expected outputs are
    computed using the equivalent pandas operations so the test verifies the
    engine's behavior against the same logic implemented directly.
    """

    cases = [
        {
            "name": "basic_two_tickers",
            "price": pd.DataFrame(
                {
                    "A": [100.0, 110.0, 121.0, 133.1, 146.41],
                    "B": [100.0, 90.0, 99.0, 108.9, 119.79],
                },
                index=pd.date_range("2020-01-01", periods=5, freq="B"),
            ),
            "lookbacks": [2, 3],
        },
        {
            "name": "empty_price",
            "price": pd.DataFrame(),
            "lookbacks": [2],
        },
    ]

    engine = VectorizedSignalEngine(universe=None, mds=None)

    for case in cases:
        price = case["price"]
        lookbacks = case["lookbacks"]

        result = engine.get_ts_momentum(price, lookbacks)

        # Build expected results using the same pandas operations
        expected = {}
        if price.empty:
            for w in lookbacks:
                expected[w] = pd.DataFrame(index=price.index)
        else:
            rets = price.pct_change(fill_method=None)
            print(f"Returns:\n{rets}\n")
            for w in lookbacks:
                mu = rets.rolling(w).mean()
                sigma = rets.rolling(w).std()
                ts_raw = mu.div(sigma).replace([np.inf, -np.inf], np.nan)
                expected[w] = ts_raw
                print(f"Lookback {w}:")
                print(f"mu:\n{mu}\nsigma:\n{sigma}\nts_raw:\n{ts_raw}\n")

        # Ensure the windows returned match
        assert set(result.keys()) == set(expected.keys())

        for w in lookbacks:
            res_df = result[w]
            exp_df = expected[w]

            # Reindex to expected shape to avoid ordering issues
            res_df = res_df.reindex(index=exp_df.index, columns=exp_df.columns)

            try:
                assert_frame_equal(res_df, exp_df, atol=1e-12, rtol=1e-12)
            except AssertionError as e:
                raise AssertionError(
                    f"Case '{case['name']}' failed for window {w}: {e}"
                ) from e
