import shutil
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from market_data_store import MarketDataStore   # import your class

def main():
    # --- 1. Prepare a temporary data directory ---
    test_dir = Path("./test_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)  # wipe old test directory
    test_dir.mkdir()

    # --- 2. Initialize the store ---
    store = MarketDataStore(data_root=str(test_dir))

    ticker = "AAPL"
    end = datetime.today() - timedelta(days=1)      # yesterday
    start = end - timedelta(days=365)  # 1 year ago

    print("\n=== First Fetch (should download from yfinance) ===")
    df1 = store.get_ohlcv(ticker, start, end, interval="1d")
    print(df1.head())
    print(df1.tail())
    print(f"Rows returned: {len(df1)}")
    print(f"Columns: {df1.columns.tolist()}")

    # Check that data is saved locally
    data_path = test_dir / "ohlcv" / "1d" / ticker / "data.parquet"
    meta_path = test_dir / "ohlcv" / "1d" / ticker / "meta.json"
    print("\nChecking files...")
    print(f"data.parquet exists? {data_path.exists()}")
    print(f"meta.json exists? {meta_path.exists()}")

    # --- 3. Second fetch should hit ONLY local cache ---
    print("\n=== Second Fetch (should NOT download again) ===")
    df2 = store.get_ohlcv(ticker, start, end, interval="1d")
    print(f"Rows returned: {len(df2)}")
    print("First few rows:")
    print(df2.head())

    # --- 4. Validate consistency ---
    assert isinstance(df2, pd.DataFrame)
    assert len(df1) == len(df2)
    assert df1.equals(df2), "Cached data does not match downloaded data!"

    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    main()
