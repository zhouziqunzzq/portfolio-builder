"""Alpaca EML (Execution Market Link) example script.

Demonstrates basic execution/trading flows using the modern `alpaca-py` SDK.

Credentials are read from environment variables (no secrets in code):
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL` (optional; defaults to paper endpoint)

Safety:
- By default, orders are NOT submitted. Use `--submit` to actually place an order.

Examples:
  # Preview a buy market order (no submission)
  python playgrounds/alpaca/alpaca_eml.py --market AAPL --side buy --qty 1

  # Submit a buy market order
  python playgrounds/alpaca/alpaca_eml.py --market AAPL --side buy --qty 1 --submit

  # Submit a sell market order
  python playgrounds/alpaca/alpaca_eml.py --market AAPL --side sell --qty 1 --submit

  # Check order status
  python playgrounds/alpaca/alpaca_eml.py --order-status <ORDER_ID>

    # Cancel an order
    python playgrounds/alpaca/alpaca_eml.py --cancel-order <ORDER_ID>

This script is intentionally minimal and focuses on core execution operations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
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
        # Keep this explicit; we default to paper endpoint.
        "paper": os.environ.get("ALPACA_PAPER", "true").strip().lower()
        in {"1", "true", "yes"},
    }


def build_trading_client(creds):
    if not creds["api_key"] or not creds["secret_key"]:
        raise RuntimeError(
            "Environment variables ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
        )

    # `paper=True` is a safety guard; `url_override` lets us be explicit.
    return TradingClient(
        api_key=creds["api_key"],
        secret_key=creds["secret_key"],
        paper=creds["paper"],
        url_override=creds["base_url"],
    )


def pretty_print(obj):
    print(json.dumps(obj, default=str, indent=2))


def to_raw(model):
    raw = getattr(model, "_raw", None)
    if raw is not None:
        return raw
    # Best-effort fallback.
    out = {}
    for k in dir(model):
        if k.startswith("_"):
            continue
        try:
            v = getattr(model, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def preview_market_order(
    symbol: str, side: str, qty: float | None, notional: float | None, tif: str
):
    symbol = symbol.strip().upper()
    side_norm = side.strip().lower()
    tif_norm = tif.strip().lower()
    return {
        "type": "market",
        "symbol": symbol,
        "side": side_norm,
        "qty": qty,
        "notional": notional,
        "time_in_force": tif_norm,
        "note": "Preview only (use --submit to place).",
    }


def submit_market_order(
    trading: TradingClient,
    symbol: str,
    side: str,
    qty: float | None,
    notional: float | None,
    tif: str,
):
    symbol = symbol.strip().upper()
    side_enum = OrderSide.BUY if side.strip().lower() == "buy" else OrderSide.SELL
    tif_norm = tif.strip().lower()
    if tif_norm == "day":
        tif_enum = TimeInForce.DAY
    elif tif_norm == "gtc":
        tif_enum = TimeInForce.GTC
    else:
        raise ValueError("--tif must be one of: day, gtc")

    order_req = MarketOrderRequest(
        symbol=symbol,
        side=side_enum,
        time_in_force=tif_enum,
        qty=qty,
        notional=notional,
    )

    order = trading.submit_order(order_req)
    return to_raw(order)


def get_order_status(trading: TradingClient, order_id: str):
    order = trading.get_order_by_id(order_id)
    return to_raw(order)


def list_recent_orders(trading: TradingClient, limit: int):
    # Keep it simple: fetch orders and show the newest N.
    orders = trading.get_orders(
        filter=GetOrdersRequest(
            limit=limit,
        ),
    )
    return [to_raw(o) for o in orders]


def cancel_order(trading: TradingClient, order_id: str):
    trading.cancel_order_by_id(order_id)
    return {"canceled": True, "order_id": order_id}


def list_open_orders(trading: TradingClient, limit: int):
    orders = trading.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            limit=limit,
        ),
    )
    return [to_raw(o) for o in orders]


def cancel_all_open_orders(trading: TradingClient):
    # Alpaca endpoint cancels ALL open orders.
    result = trading.cancel_orders()
    # result is a list of CancelOrderResponse (or raw data if configured).
    return getattr(result, "_raw", result)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Alpaca EML examples (execution/trading)")
    group = p.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--market",
        metavar="SYMBOL",
        help="Preview/submit a MARKET order for SYMBOL",
    )
    group.add_argument(
        "--order-status",
        metavar="ORDER_ID",
        help="Fetch status for ORDER_ID",
    )
    group.add_argument(
        "--cancel-order",
        metavar="ORDER_ID",
        help="Cancel ORDER_ID",
    )
    group.add_argument(
        "--cancel-all-open",
        action="store_true",
        help="Cancel ALL open orders (requires --submit)",
    )
    group.add_argument(
        "--list-orders",
        action="store_true",
        help="List recent orders",
    )

    p.add_argument(
        "--side",
        choices=("buy", "sell"),
        default="buy",
        help="Order side for --market (default: buy)",
    )
    qty_group = p.add_mutually_exclusive_group()
    qty_group.add_argument(
        "--qty",
        type=float,
        help="Quantity for --market (shares). Mutually exclusive with --notional.",
    )
    qty_group.add_argument(
        "--notional",
        type=float,
        help="Notional for --market (USD). Mutually exclusive with --qty.",
    )
    p.add_argument(
        "--tif",
        choices=("day", "gtc"),
        default="day",
        help="Time in force for --market (default: day)",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Actually perform the action (default is preview only)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for --list-orders (default: 10)",
    )
    return p


def main():
    args = build_arg_parser().parse_args()

    creds = read_env()
    trading = build_trading_client(creds)

    if args.order_status:
        pretty_print(get_order_status(trading, args.order_status))
        return

    if args.cancel_order:
        pretty_print(cancel_order(trading, args.cancel_order))
        return

    if args.cancel_all_open:
        # Preview first to avoid accidental mass-cancel.
        open_orders = list_open_orders(trading, limit=max(50, args.limit))
        if not args.submit:
            pretty_print(
                {
                    "action": "cancel_all_open",
                    "submit": False,
                    "open_order_count": len(open_orders),
                    "open_order_ids": [o.get("id") for o in open_orders if isinstance(o, dict)],
                    "note": "Preview only. Re-run with --submit to cancel all open orders.",
                }
            )
            return

        pretty_print(
            {
                "action": "cancel_all_open",
                "submit": True,
                "open_order_count": len(open_orders),
                "cancel_results": cancel_all_open_orders(trading),
            }
        )
        return

    if args.list_orders:
        pretty_print(list_recent_orders(trading, args.limit))
        return

    # --market
    symbol = args.market
    if args.qty is None and args.notional is None:
        raise SystemExit("For --market you must provide either --qty or --notional")
    if args.qty is not None and args.qty <= 0:
        raise SystemExit("--qty must be > 0")
    if args.notional is not None and args.notional <= 0:
        raise SystemExit("--notional must be > 0")

    preview = preview_market_order(symbol, args.side, args.qty, args.notional, args.tif)
    if not args.submit:
        pretty_print(preview)
        return

    # Submit
    result = submit_market_order(
        trading=trading,
        symbol=symbol,
        side=args.side,
        qty=args.qty,
        notional=args.notional,
        tif=args.tif,
    )
    pretty_print(result)


if __name__ == "__main__":
    main()
