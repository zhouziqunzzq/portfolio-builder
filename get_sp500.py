import requests
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

resp = requests.get(url, headers=headers)
resp.raise_for_status()  # clearer error if something is wrong

tables = pd.read_html(resp.text)
df = tables[1]  # second table is the constituents on this page

# Keep original Wikipedia symbol for reference
df = df.rename(columns={
    "Symbol": "WikiSymbol",
    "Security": "Name",
    "GICS Sector": "GICS Sector",
    "GICS Sub-Industry": "GICS Sub-Industry",
})

def to_yf_symbol(sym: str) -> str:
    """
    Normalize a ticker from Wikipedia to a yfinance-compatible symbol.
    Example: 'BRK.B' -> 'BRK-B', 'BF.B' -> 'BF-B'
    """
    s = str(sym).strip().upper()
    # Generic rule: replace dot with dash for class shares
    s = s.replace(".", "-")
    return s

# Add yfinance-normalized ticker and expose it as 'Symbol' for the universe
df["Symbol"] = df["WikiSymbol"].apply(to_yf_symbol)

# Optional: ensure Symbol is uppercase (already done in to_yf_symbol)
df["Symbol"] = df["Symbol"].str.upper()

# Save to CSV in the format UniverseManager expects
df_out = df[["Symbol", "Name", "GICS Sector", "GICS Sub-Industry"]]
df_out.to_csv("./data/universe/sp500_constituents.csv", index=False)

print(df_out.head())
