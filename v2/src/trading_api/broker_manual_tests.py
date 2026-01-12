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


def _require_env_any(*names: str) -> str:
    for name in names:
        v = os.environ.get(name)
        if v:
            return v
    raise ValueError(
        "Missing required env var. Tried: "
        + ", ".join(names)
        + ". Create .env.broker from .env.broker.example"
    )


def _pretty(obj: Any) -> str:
    try:
        return str(asdict(obj))
    except Exception:
        return str(obj)


def _confirm(prompt: str) -> bool:
    ans = input(f"{prompt} (yes/no): ").strip().lower()
    return ans == "yes"


def _load_env(*, env_file: str, repo_root: Path) -> Path | None:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path

    # Convenience fallbacks for running from repo root.
    candidates = [
        env_path,
        repo_root / env_path.name,
    ]

    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)
            return p
    return None


def _env_for_broker(
    broker: str, key: str, default: Optional[str] = None
) -> Optional[str]:
    # Prefer BROKER_* shared vars, then broker-specific vars.
    shared = os.environ.get(f"BROKER_{key}")
    if shared is not None and str(shared).strip() != "":
        return shared

    prefix = "ALPACA" if broker == "alpaca" else "PUBLICDOTCOM"
    v = os.environ.get(f"{prefix}_{key}")
    if v is not None and str(v).strip() != "":
        return v
    return default


def _build_api(*, broker: str):
    from trading_api.alpaca import AlpacaTradingAPI
    from trading_api.base import BaseSyncTradingAPI
    from trading_api.publicdotcom import PublicDotComTradingAPI

    api: BaseSyncTradingAPI
    if broker == "alpaca":
        api_key = _require_env_any("ALPACA_API_KEY")
        secret_key = _require_env_any("ALPACA_SECRET_KEY")
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        paper = _parse_bool(os.environ.get("ALPACA_PAPER"), default=True)
        api = AlpacaTradingAPI(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            paper=paper,
        )
        return api

    if broker == "publicdotcom":
        # Support both the canonical PUBLICDOTCOM_* vars and the SDK's older names.
        api_secret_key = _require_env_any(
            "PUBLICDOTCOM_API_SECRET_KEY", "API_SECRET_KEY"
        )
        default_account_number = _require_env_any(
            "PUBLICDOTCOM_DEFAULT_ACCOUNT_NUMBER", "DEFAULT_ACCOUNT_NUMBER"
        )
        base_url = os.environ.get("PUBLICDOTCOM_BASE_URL")
        api = PublicDotComTradingAPI(
            api_secret_key=api_secret_key,
            default_account_number=default_account_number,
            base_url=base_url,
        )
        return api

    raise ValueError(f"Unsupported broker: {broker}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manual broker API tests for BaseSyncTradingAPI adapters (connects to real broker)."
        )
    )
    parser.add_argument(
        "--broker",
        choices=["alpaca", "publicdotcom"],
        default=os.environ.get("BROKER", "alpaca").strip().lower(),
        help="Which broker adapter to test (default: alpaca; env: BROKER)",
    )
    parser.add_argument(
        "--env-file",
        default=".env.broker",
        help="Path to env file (default: .env.broker)",
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
    # `python .../broker_manual_tests.py`), it can shadow the external
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

    loaded = _load_env(env_file=args.env_file, repo_root=repo_root)
    if loaded is not None:
        print(f"Loaded env from: {loaded}")
    else:
        print("No .env file found; using current process env only.")

    broker = str(args.broker).strip().lower()
    api = _build_api(broker=broker)

    from models.trading import InstrumentRef, OrderIntent, OrderSide, OrderStatus
    from trading_api.exceptions import OrderNotFoundYet

    print(f"Broker: {api.name} (selected={broker})")
    print(f"Capabilities: {_pretty(api.capabilities())}")

    acct = api.get_account()
    print(f"Account: {_pretty(acct)}")

    positions = api.list_positions()
    print(f"Positions ({len(positions)}):")
    for p in positions:
        print(f"  - {_pretty(p)}")

    symbol = (_env_for_broker(broker, "TEST_SYMBOL", "SPY") or "SPY").strip()
    inst = api.get_instrument(InstrumentRef(symbol=symbol))
    print(f"Instrument({symbol}): {_pretty(inst)}")

    test_side = (_env_for_broker(broker, "TEST_SIDE", "buy") or "buy").strip().lower()
    if test_side not in {"buy", "sell"}:
        raise ValueError("TEST_SIDE must be 'buy' or 'sell'")

    client_order_id = str(uuid.uuid4())

    # BUY: use notional by default (safer). SELL: use qty by default.
    notional_s = (_env_for_broker(broker, "TEST_NOTIONAL", "5.00") or "5.00").strip()
    qty_s = (_env_for_broker(broker, "TEST_QTY", "1") or "1").strip()

    intent_kwargs: Dict[str, Any] = dict(
        client_order_id=client_order_id,
        instrument=InstrumentRef(symbol=symbol),
        side=OrderSide.BUY if test_side == "buy" else OrderSide.SELL,
    )

    caps = api.capabilities()
    if test_side == "buy":
        if not caps.supports_notional_market_orders:
            intent_kwargs["qty"] = Decimal(qty_s)
        else:
            intent_kwargs["notional"] = Decimal(notional_s)
    else:
        # Prefer qty for sells; notional sells are often unsupported.
        if caps.supports_qty_market_orders:
            intent_kwargs["qty"] = Decimal(qty_s)
        elif caps.supports_notional_sells:
            intent_kwargs["notional"] = Decimal(notional_s)
        else:
            raise ValueError("Broker adapter does not support sell market orders")

    intent = OrderIntent(**intent_kwargs)

    # Preflight (if supported) using the exact same intent we would submit.
    if api.capabilities().supports_preflight:
        pf = api.preflight_order(intent)
        print(f"Preflight: {_pretty(pf)}")
    else:
        print("Preflight: not supported by this broker adapter.")

    if not args.place_order:
        print("Skipping order placement (pass --place-order to enable).")
        return 0

    print("\nAbout to submit broker order:")
    print(f"  broker={broker}")
    print(f"  intent={_pretty(intent)}")
    if broker == "alpaca" and not _parse_bool(
        os.environ.get("ALPACA_PAPER"), default=True
    ):
        print("WARNING: ALPACA_PAPER is false; this may trade LIVE.")

    if not _confirm("Do you want to submit this order"):
        print("Order submission cancelled.")
        return 0

    placed = api.submit_order(intent)
    print(f"Submitted: {_pretty(placed)}")

    start = time.time()
    last_state = None
    while True:
        try:
            state = api.get_order(placed.broker_order_id)
        except OrderNotFoundYet:
            # Some brokers (notably Public.com) are eventually consistent: immediately
            # after order placement, the order may not be queryable yet.
            if time.time() - start > float(args.poll_timeout_seconds):
                print("Polling timeout reached (order not visible yet).")
                break
            print("Order not found yet; will retry...")
            time.sleep(float(args.poll_seconds))
            continue
        last_state = state
        print(f"OrderState: {_pretty(state)}")

        st = state.status
        if st in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }:
            break

        if time.time() - start > float(args.poll_timeout_seconds):
            print("Polling timeout reached.")
            break
        time.sleep(float(args.poll_seconds))

    if last_state is None:
        return 0

    if last_state.status in {
        OrderStatus.OPEN,
        OrderStatus.NEW,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
    }:
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
