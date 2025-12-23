"""Alpaca IML (Information Market Link) example script.

Demonstrates informational, read-only uses of the modern `alpaca-py` SDK.

Reads credentials from environment variables:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL` (optional; defaults to the paper API endpoint)

Examples:
  python playgrounds/alpaca/alpaca_iml.py --account
  python playgrounds/alpaca/alpaca_iml.py --price AAPL
  python playgrounds/alpaca/alpaca_iml.py --bars AAPL 1D 10

This script is read-only and will not place orders.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestQuoteRequest,
        StockLatestTradeRequest,
    )
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
except Exception:
    print("Missing dependency: install with `pip install alpaca-py`", file=sys.stderr)
    raise


def read_env():
    return {
        "api_key": os.environ.get("ALPACA_API_KEY"),
        "secret_key": os.environ.get("ALPACA_SECRET_KEY"),
        "base_url": os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        ),
        # For many accounts, explicitly setting `IEX` is required.
        # Supported values include: iex, sip
        "data_feed": os.environ.get("ALPACA_DATA_FEED", "iex"),
    }


def parse_data_feed(feed_str: str | None):
    if not feed_str:
        return None
    s = feed_str.strip().lower()
    if s == "iex":
        return DataFeed.IEX
    if s == "sip":
        return DataFeed.SIP
    return None


def default_time_window(timeframe: TimeFrame, limit: int):
    # Some Alpaca endpoints can return empty results with only `limit`.
    # Providing a time window makes results more deterministic.
    end = datetime.utcnow()
    if timeframe == TimeFrame.Day:
        start = end - timedelta(days=max(30, limit * 2))
    elif timeframe == TimeFrame.Week:
        start = end - timedelta(days=max(180, limit * 7 * 2))
    else:
        # Minute or others
        start = end - timedelta(minutes=max(120, limit * 3))
    return start, end


def build_client(creds):
    if not creds["api_key"] or not creds["secret_key"]:
        raise RuntimeError(
            "Environment variables ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
        )
    trading = TradingClient(
        api_key=creds["api_key"],
        secret_key=creds["secret_key"],
        paper=True,
        url_override=creds["base_url"],
    )
    # Historical data client used for bars and latest-price fallbacks
    data = StockHistoricalDataClient(creds["api_key"], creds["secret_key"])
    return trading, data


def account_info(trading: TradingClient):
    acct = trading.get_account()
    out = getattr(acct, "_raw", None)
    if out is None:
        out = {
            "id": getattr(acct, "id", None),
            "status": getattr(acct, "status", None),
            "cash": getattr(acct, "cash", None),
            "buying_power": getattr(acct, "buying_power", None),
            "portfolio_value": getattr(acct, "portfolio_value", None),
        }
    return out


def list_positions(trading: TradingClient):
    try:
        pos_list = trading.get_all_positions()
    except Exception:
        try:
            pos_list = trading.get_positions()
        except Exception:
            pos_list = []
    out = []
    for p in pos_list:
        raw = getattr(p, "_raw", None)
        if raw is None:
            raw = {
                "symbol": getattr(p, "symbol", None),
                "qty": getattr(p, "qty", None),
                "market_value": getattr(p, "market_value", None),
                "avg_entry_price": getattr(p, "avg_entry_price", None),
            }
        out.append(raw)
    return out


def latest_price(
    data: StockHistoricalDataClient,
    symbol: str,
    source: str = "bar",
):
    """Return a "latest price" snapshot using one of: trade | quote | bar."""

    source = (source or "bar").strip().lower()
    feed = parse_data_feed(read_env().get("data_feed"))

    if source == "trade":
        req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=feed)
        trades = data.get_stock_latest_trade(req)
        t = trades[symbol]
        return {
            "symbol": symbol,
            "source": "trade",
            "price": getattr(t, "price", None),
            "t": getattr(t, "timestamp", None),
        }

    if source == "quote":
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=feed)
        quotes = data.get_stock_latest_quote(req)
        q = quotes[symbol]
        bid = getattr(q, "bid_price", None)
        ask = getattr(q, "ask_price", None)
        mid = (bid + ask) / 2 if (bid is not None and ask is not None) else None
        return {
            "symbol": symbol,
            "source": "quote",
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "t": getattr(q, "timestamp", None),
        }

    if source != "bar":
        raise ValueError("source must be one of: bar, trade, quote")

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        limit=1,
        feed=feed,
    )
    barset = data.get_stock_bars(req)
    bars = barset[symbol]
    if not bars:
        # Retry with a time window
        start, end = default_time_window(TimeFrame.Minute, 1)
        req2 = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=1,
            feed=feed,
            start=start,
            end=end,
        )
        barset = data.get_stock_bars(req2)
        bars = barset[symbol]
        if not bars:
            raise RuntimeError(
                f"No bars returned for {symbol}. Try setting ALPACA_DATA_FEED=iex (or sip if you have it)."
            )
    b = bars[-1]
    return {
        "symbol": symbol,
        "source": "bar",
        "close": getattr(b, "c", getattr(b, "close", None)),
        "t": getattr(b, "t", None),
    }


def historical_bars(
    data: StockHistoricalDataClient, symbol: str, timeframe="1D", limit=100
):
    tf = parse_timeframe(timeframe)
    feed = parse_data_feed(read_env().get("data_feed"))
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        limit=limit,
        feed=feed,
    )
    try:
        barset = data.get_stock_bars(req)
        bars = barset[symbol]
        print(f"Retrieved {len(bars)} bars for {symbol} via alpaca-py SDK")
        if not bars:
            start, end = default_time_window(tf, limit)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                limit=limit,
                feed=feed,
                start=start,
                end=end,
            )
            barset = data.get_stock_bars(req)
            bars = barset[symbol]
        return [
            {
                "t": getattr(b, "t", None),
                "o": getattr(b, "o", getattr(b, "open", None)),
                "h": getattr(b, "h", getattr(b, "high", None)),
                "l": getattr(b, "l", getattr(b, "low", None)),
                "c": getattr(b, "c", getattr(b, "close", None)),
                "v": getattr(b, "v", getattr(b, "volume", None)),
            }
            for b in bars
        ]
    except AttributeError as e:
        # alpaca-py can sometimes return `None` entries in the raw bars payload,
        # which breaks BarSet parsing (inside the SDK). Re-request as raw JSON
        # and filter out nulls.
        creds = read_env()
        raw_client = StockHistoricalDataClient(
            creds["api_key"],
            creds["secret_key"],
            raw_data=True,
        )
        raw = raw_client.get_stock_bars(req)
        print(f"Retrieved raw bars for {symbol} via alpaca-py raw client")
        print(f"raw: {raw}")

        # raw response shapes vary; handle the common ones.
        bars_raw = None
        if isinstance(raw, dict):
            if symbol in raw and isinstance(raw.get(symbol), list):
                bars_raw = raw[symbol]
            elif isinstance(raw.get("bars"), dict) and symbol in raw["bars"]:
                bars_raw = raw["bars"][symbol]
            elif isinstance(raw.get("bars"), list):
                bars_raw = raw["bars"]

        if bars_raw is None:
            raise RuntimeError("Unexpected raw bars response format") from e

        out = []
        for b in bars_raw:
            if not b:
                continue
            # dict keys tend to be short ('t','o','h','l','c','v')
            if isinstance(b, dict):
                out.append(
                    {
                        "t": b.get("t"),
                        "o": b.get("o"),
                        "h": b.get("h"),
                        "l": b.get("l"),
                        "c": b.get("c"),
                        "v": b.get("v"),
                    }
                )
            else:
                # Extremely defensive fallback
                out.append(
                    {
                        "t": getattr(b, "t", None),
                        "o": getattr(b, "o", None),
                        "h": getattr(b, "h", None),
                        "l": getattr(b, "l", None),
                        "c": getattr(b, "c", None),
                        "v": getattr(b, "v", None),
                    }
                )

        return out


def list_assets(trading: TradingClient, status="active"):
    assets = trading.get_all_assets()
    out = []
    for a in assets:
        if status and getattr(a, "status", None) != status:
            continue
        out.append(
            getattr(
                a,
                "_raw",
                {
                    "symbol": getattr(a, "symbol", None),
                    "status": getattr(a, "status", None),
                },
            )
        )
    return out


def clock(trading: TradingClient):
    try:
        c = trading.get_clock()
        return getattr(
            c,
            "_raw",
            {
                "is_open": getattr(c, "is_open", None),
                "next_open": getattr(c, "next_open", None),
                "next_close": getattr(c, "next_close", None),
            },
        )
    except Exception:
        return {"error": "clock endpoint not available"}


def parse_timeframe(tf_str: str):
    s = tf_str.strip().lower()
    if "min" in s:
        return TimeFrame.Minute
    if "day" in s or s == "1d":
        return TimeFrame.Day
    if "week" in s or s == "1w":
        return TimeFrame.Week
    return TimeFrame.Day


def pretty_print(obj):
    print(json.dumps(obj, default=str, indent=2))


def build_arg_parser():
    p = argparse.ArgumentParser(description="Alpaca IML examples (read-only)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--account", action="store_true", help="Get account summary")
    group.add_argument(
        "--positions", action="store_true", help="List current positions"
    )
    group.add_argument(
        "--price", nargs=1, metavar="SYMBOL", help="Get latest price for SYMBOL"
    )
    group.add_argument(
        "--bars",
        nargs=3,
        metavar=("SYMBOL", "TIMEFRAME", "LIMIT"),
        help="Get historical bars: SYMBOL TIMEFRAME LIMIT",
    )
    group.add_argument(
        "--assets", action="store_true", help="List assets (active by default)"
    )
    group.add_argument(
        "--clock", action="store_true", help="Get market clock (open/close)"
    )

    p.add_argument(
        "--price-source",
        choices=("bar", "trade", "quote"),
        default="bar",
        help="How to define 'latest price' when using --price (default: bar)",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    creds = read_env()
    trading, data = build_client(creds)

    if args.account:
        pretty_print(account_info(trading))
        return

    if args.positions:
        pretty_print(list_positions(trading))
        return

    if args.price:
        symbol = args.price[0].upper()
        pretty_print(latest_price(data, symbol, source=args.price_source))
        return

    if args.bars:
        symbol, timeframe, limit = args.bars
        pretty_print(historical_bars(data, symbol.upper(), timeframe, int(limit)))
        return

    if args.assets:
        pretty_print(list_assets(trading))
        return

    if args.clock:
        pretty_print(clock(trading))
        return


if __name__ == "__main__":
    main()
