from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


def _parse_bool(v: Optional[str], *, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise ValueError(
            f"Missing required env var {name}. Create .env.alpaca from .env.alpaca.example"
        )
    return v


def _pretty(obj: Any) -> str:
    try:
        return str(asdict(obj))
    except Exception:
        return str(obj)


def _confirm(prompt: str) -> bool:
    ans = input(f"{prompt} (yes/no): ").strip().lower()
    return ans == "yes"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual Alpaca API tests for AlpacaTradingAPI (connects to real broker)."
    )
    parser.add_argument(
        "--env-file",
        default=".env.alpaca",
        help="Path to env file (default: .env.alpaca)",
    )
    parser.add_argument(
        "--place-order",
        action="store_true",
        help="Actually submit a test order (requires interactive confirmation)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Seconds between order status polls (default: 2)",
    )
    parser.add_argument(
        "--poll-timeout-seconds",
        type=float,
        default=30.0,
        help="Max seconds to poll for fills (default: 30)",
    )
    args = parser.parse_args()

    # Ensure `v2/src` is importable when running this file directly.
    #
    # IMPORTANT: this file lives next to `trading_api/alpaca.py`.
    # If the script directory is on `sys.path` (it is, when running as
    # `python .../alpaca_manual_tests.py`), it can shadow the external
    # `alpaca` package from alpaca-py, causing:
    #   "No module named 'alpaca.trading'; 'alpaca' is not a package"
    repo_root = Path(__file__).resolve().parents[3]
    v2_src = repo_root / "v2" / "src"
    script_dir = Path(__file__).resolve().parent

    cleaned: list[str] = []
    for p in list(sys.path):
        try:
            rp = Path(p).resolve()
        except Exception:
            cleaned.append(p)
            continue
        if rp == script_dir:
            continue
        cleaned.append(p)
    sys.path[:] = cleaned

    if str(v2_src) in sys.path:
        sys.path.remove(str(v2_src))
    sys.path.insert(0, str(v2_src))

    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path
    if not env_path.exists():
        alt = repo_root / ".env.alpaca"
        if alt.exists():
            env_path = alt

    if env_path.exists():
        # Keep existing env vars as higher priority.
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"Loaded env from: {env_path}")
    else:
        print("No .env.alpaca found; using current process env only.")

    api_key = _require_env("ALPACA_API_KEY")
    secret_key = _require_env("ALPACA_SECRET_KEY")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    paper = _parse_bool(os.environ.get("ALPACA_PAPER"), default=True)

    from trading_api.alpaca import AlpacaTradingAPI
    from models.trading import InstrumentRef, OrderIntent, OrderSide

    api = AlpacaTradingAPI(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url,
        paper=paper,
    )

    print(f"Broker: {api.name}")
    print(f"Capabilities: {_pretty(api.capabilities())}")

    acct = api.get_account()
    print(f"Account: {_pretty(acct)}")

    positions = api.list_positions()
    print(f"Positions ({len(positions)}):")
    for p in positions:
        print(f"  - {_pretty(p)}")

    symbol = os.environ.get("ALPACA_TEST_SYMBOL", "SPY").strip()
    inst = api.get_instrument(InstrumentRef(symbol=symbol))
    print(f"Instrument({symbol}): {_pretty(inst)}")

    # No-op preflight
    pf = api.preflight_order(
        OrderIntent(
            client_order_id=str(uuid.uuid4()),
            instrument=InstrumentRef(symbol=symbol),
            side=OrderSide.BUY,
            notional=Decimal("1.00"),
        )
    )
    print(f"Preflight(no-op): {_pretty(pf)}")

    if not args.place_order:
        print("Skipping order placement (pass --place-order to enable).")
        return 0

    test_side = os.environ.get("ALPACA_TEST_SIDE", "buy").strip().lower()
    if test_side not in {"buy", "sell"}:
        raise ValueError("ALPACA_TEST_SIDE must be 'buy' or 'sell'")

    client_order_id = str(uuid.uuid4())

    # BUY: use notional by default (safer). SELL: use qty by default.
    notional_s = os.environ.get("ALPACA_TEST_NOTIONAL", "5.00").strip()
    qty_s = os.environ.get("ALPACA_TEST_QTY", "1").strip()

    intent_kwargs: Dict[str, Any] = dict(
        client_order_id=client_order_id,
        instrument=InstrumentRef(symbol=symbol),
        side=OrderSide.BUY if test_side == "buy" else OrderSide.SELL,
    )

    if test_side == "buy":
        intent_kwargs["notional"] = Decimal(notional_s)
    else:
        intent_kwargs["qty"] = Decimal(qty_s)

    intent = OrderIntent(**intent_kwargs)

    print("\nAbout to submit Alpaca order:")
    print(f"  paper={paper} base_url={base_url}")
    print(f"  intent={_pretty(intent)}")

    if not paper:
        print("WARNING: ALPACA_PAPER is false; this may trade LIVE.")

    if not _confirm("Do you want to submit this order"):
        print("Order submission cancelled.")
        return 0

    placed = api.submit_order(intent)
    print(f"Submitted: {_pretty(placed)}")

    start = time.time()
    last_state = None
    while True:
        state = api.get_order(placed.broker_order_id)
        last_state = state
        print(f"OrderState: {_pretty(state)}")

        # state.status is an enum; compare via value and accept string fallbacks
        try:
            status_v = getattr(state.status, "value", str(state.status)).lower()
        except Exception:
            status_v = str(state.status).lower()
        if status_v in {"filled", "canceled", "cancelled", "rejected", "expired"}:
            break

        if time.time() - start > float(args.poll_timeout_seconds):
            print("Polling timeout reached.")
            break
        time.sleep(float(args.poll_seconds))

    if last_state is None:
        return 0

    try:
        status_v = getattr(last_state.status, "value", str(last_state.status)).lower()
    except Exception:
        status_v = str(last_state.status).lower()

    if status_v in {"open", "new", "accepted", "partially_filled"}:
        if _confirm("Order still open. Cancel it"):
            if _confirm("Confirm cancel order"):
                api.cancel_order(placed.broker_order_id)
                print("Cancel requested.")
            else:
                print("Cancel confirmation declined.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
