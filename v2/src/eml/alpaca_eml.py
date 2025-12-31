from __future__ import annotations

import asyncio
from datetime import datetime
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from decimal import Decimal, ROUND_DOWN


from pathlib import Path
import sys

_ROOT_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(_ROOT_SRC))

from events.event_bus import EventBus
from events.events import (
    AccountSnapshotEvent,
    BrokerAccount,
    BrokerPosition,
    RebalancePlanRequestEvent,
    RebalancePlanConfirmationEvent,
    MarketClockEvent,
)
from runtime_manager import RuntimeManager
from states.base_state import BaseState

from .base_eml import BaseEML
from .config import EMLConfig
from .state import EMLState


class EMLShutdownRequested(Exception):
    """Raised internally to abort blocking execution during shutdown."""

    pass


class AlpacaEMLService(BaseEML):
    """Alpaca Execution Market Link (EML).

    Current scope:
    - Periodically fetch account + (optionally) positions from Alpaca
    - Publish an `AccountSnapshotEvent` to the event bus

    Future scope:
    - Translate `RebalancePlanRequestEvent` into broker orders
    - Publish order/fill updates to the bus
    """

    def __init__(
        self,
        bus: "EventBus",
        rm: Optional["RuntimeManager"] = None,
        *,
        config: Optional[EMLConfig] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: Optional[bool] = None,
        trading_client: Optional[Any] = None,
        name: str = "AlpacaEML",
    ):
        super().__init__(bus=bus, name=name)

        if config is None:
            self.log.warning("No EMLConfig provided; using default configuration")
            config = EMLConfig()
        self._validate_config(config)
        self.config = config

        if self.config.polling_interval_secs <= 0:
            raise ValueError("polling_interval_secs must be > 0")
        self._poll_interval_seconds = float(self.config.polling_interval_secs)

        self.rm = rm
        if self.rm is not None:
            # Register self to RuntimeManager for lifecycle management.
            self.rm.set("eml", self)
            self.rm.set("alpaca_eml", self)  # alias

        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self._base_url = base_url or os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if paper is None:
            env_paper = os.environ.get("ALPACA_PAPER")
            if env_paper is None:
                self._paper = True
            else:
                self._paper = env_paper.strip().lower() in {"1", "true", "yes", "y"}
        else:
            self._paper = bool(paper)

        self._injected_trading_client = trading_client is not None

        if trading_client is not None:
            self._trading = trading_client
        else:
            self._trading = self._build_trading_client()
        self.log.info(
            "Initialized AlpacaEMLService (paper=%s, base_url=%s)",
            self._paper,
            self._base_url,
        )

        # State
        # This will be managed by StateManager externally.
        self.state = EMLState()

    def _build_trading_client(self):
        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY (or pass api_key/secret_key)."
            )

        try:
            from alpaca.trading.client import TradingClient
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'alpaca-py'. Install with `pip install alpaca-py`."
            ) from e

        return TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
            url_override=self._base_url,
        )

    async def _on_startup(self) -> None:
        await super()._on_startup()

        if not getattr(self.config, "cancel_open_orders_on_startup", True):
            self.log.info("Startup open-order cancel disabled by config")
            return

        # Best-effort safety cleanup. This is intentionally not fatal.
        try:
            await self._run_in_thread(self._cancel_all_open_orders)
        except Exception:
            self.log.exception("Failed to cancel open orders on startup (continuing)")

    async def _on_shutdown_requested(self) -> None:
        # Best-effort safety cleanup on shutdown. This is intentionally not fatal.
        if getattr(self.config, "cancel_open_orders_on_shutdown", False):
            try:
                await self._run_in_thread(self._cancel_all_open_orders)
            except Exception:
                self.log.exception(
                    "Failed to cancel open orders on shutdown (continuing)"
                )

        await super()._on_shutdown_requested()

    async def _run_loop(self) -> None:
        while self._running:
            try:
                # Fetch account + positions
                account = await self._run_in_thread(self._get_account)
                positions: List[BrokerPosition] = []
                if self.config.include_positions:
                    positions = await self._run_in_thread(self._list_positions)
                event = AccountSnapshotEvent(
                    ts=time.time(),
                    source=self.name,
                    account=account,
                    positions=positions,
                )
                self.log.debug(
                    "Fetched account snapshot: account=%s positions=%s",
                    account,
                    positions,
                )
                await self.emit_account_event(event)

                # Execute any pending rebalance plans (blocking per plan, off the event loop thread)
                await self._run_in_thread(self._execute_pending_rebalance_plans)

                # GC execution history (best-effort; keep state from growing unbounded)
                self._gc_execution_history()

                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep running; transient API/network issues are expected.
                self.log.exception("Error in AlpacaEMLService main loop")
                await asyncio.sleep(min(30.0, max(1.0, self._poll_interval_seconds)))

    def _cancel_all_open_orders(self) -> None:
        """Cancel all currently-open orders at the broker (best-effort)."""

        # Preferred: Alpaca endpoint cancels all open orders.
        if hasattr(self._trading, "cancel_orders"):
            self.log.info("Canceling all open orders...")
            res = self._trading.cancel_orders()
            self.log.info("Cancel-all-open request completed: result=%s", res)
            return

        # Fallback: list open orders then cancel individually.
        open_ids: List[str] = []
        try:
            open_orders = self._list_open_orders_best_effort(limit=500)
            for o in open_orders:
                oid = self._extract_order_id(o)
                if oid:
                    open_ids.append(oid)
        except Exception:
            self.log.exception("Failed listing open orders")
            open_ids = []

        if not open_ids:
            self.log.info("No open orders found to cancel")
            return

        if not hasattr(self._trading, "cancel_order_by_id"):
            raise RuntimeError(
                "Trading client does not support cancel_orders or cancel_order_by_id"
            )

        self.log.info("Canceling %d open orders...", len(open_ids))
        for oid in open_ids:
            try:
                self._trading.cancel_order_by_id(oid)
            except Exception:
                self.log.exception("Failed canceling open order: order_id=%s", oid)

    def _list_open_orders_best_effort(self, *, limit: int = 100) -> List[Any]:
        """Best-effort retrieval of open orders.

        Uses alpaca-py filters when available; otherwise returns an empty list.
        """
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
        except Exception:
            return []

        if not hasattr(self._trading, "get_orders"):
            return []

        orders = self._trading.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=int(limit))
        )
        try:
            return list(orders)
        except Exception:
            return []

    @staticmethod
    def _extract_order_id(order_obj: Any) -> Optional[str]:
        raw = getattr(order_obj, "_raw", None)
        if isinstance(raw, dict) and raw.get("id"):
            return str(raw.get("id"))
        oid = getattr(order_obj, "id", None)
        if oid:
            return str(oid)
        return None

    async def execute_rebalance_plan(self, event: RebalancePlanRequestEvent) -> None:
        """Execute a rebalance plan request.
        This function simply records the pending rebalance request in state, and
        emits a RebalancePlanConfirmationEvent. Actual execution of the rebalance
        plan (placing orders) is handled asynchronously in the main loop. See `_run_loop()`
        and `_execute_pending_rebalance_plans()`.

        Args:
            event: RebalancePlanRequestEvent
        Returns:
            None
        """
        # Track pending rebalance requests in persisted state, and send back confirmation.
        now_ts = time.time()
        try:
            if self.state.has_pending_rebalance_request(event.rebalance_id):
                self.log.info(
                    "RebalancePlanRequestEvent already pending; ignoring duplicate: rebalance_id=%s",
                    event.rebalance_id,
                )
            else:
                self.state.remember_pending_rebalance_request(event)
            # We still send back confirmation even if duplicate.
            confirmation_event = RebalancePlanConfirmationEvent(
                ts=now_ts,
                rebalance_id=event.rebalance_id,
                confirmed_ts=now_ts,
                source=self.name,
            )
            await self.emit(confirmation_event)
            self.log.info(
                "Published RebalancePlanConfirmationEvent: rebalance_id=%s",
                event.rebalance_id,
            )
        except Exception:
            self.log.exception(
                "Failed to store pending rebalance request in state: rebalance_id=%s",
                getattr(event, "rebalance_id", None),
            )

        # Note: Actual execution of the rebalance plan (placing orders) is handled asynchronously
        # in the main loop, to avoid blocking the event handler.

    def _execute_pending_rebalance_plans(self) -> None:
        pending = self.state.pending_rebalance_requests
        if not pending:
            return

        # Process oldest-first for determinism.
        items: List[Tuple[str, Dict[str, Any]]] = []
        for rebalance_id, payload in dict(pending).items():
            if not isinstance(payload, dict):
                continue
            items.append((str(rebalance_id), dict(payload)))

        def _key(item: Tuple[str, Dict[str, Any]]) -> float:
            v = item[1].get("request_ts")
            try:
                return float(v)
            except Exception:
                return 0.0

        items.sort(key=_key)

        for rebalance_id, payload in items:
            # Another thread / loop iteration might have handled it already.
            if not self.state.has_pending_rebalance_request(rebalance_id):
                continue

            try:
                event = self._rebalance_request_from_state(payload)
                self._execute_rebalance_plan(event)
                self.state.mark_rebalance_executed(rebalance_id=rebalance_id)
                self.log.info(
                    "Rebalance executed successfully: rebalance_id=%s",
                    rebalance_id,
                )
            except EMLShutdownRequested:
                # Quiet exit on shutdown; leave pending state intact.
                self.log.info(
                    "Shutdown requested; aborting pending rebalance execution: rebalance_id=%s",
                    rebalance_id,
                )
                return
            except Exception:
                # Best-effort: keep processing other plans; do not mark as executed.
                self.log.exception(
                    "Failed executing pending rebalance plan: rebalance_id=%s",
                    rebalance_id,
                )

    def _execute_rebalance_plan(self, event: RebalancePlanRequestEvent) -> None:
        # NOTE:
        # In this repo, v2 code often runs with `v2/src` injected onto `sys.path`.
        # Depending on how code is invoked (tests vs runners), the same dataclass
        # may be imported under different module names (e.g. `events.events` vs
        # `v2.src.events.events`), which makes `isinstance()` brittle.
        # For execution we only require a minimal event shape.
        rebalance_id = getattr(event, "rebalance_id", None)
        weights = getattr(event, "weights", None)
        if rebalance_id is None or weights is None:
            raise TypeError(
                "event must have attributes 'rebalance_id' and 'weights' (RebalancePlanRequestEvent-like)"
            )
        self.log.info(
            "Executing rebalance plan: rebalance_id=%s weights=%s",
            rebalance_id,
            weights,
        )

        # 1) Normalize target weights
        target_weights = self._normalize_target_weights(weights)
        self.log.debug(
            "Normalized target weights: rebalance_id=%s target_weights=%s",
            rebalance_id,
            target_weights,
        )

        # 2) Fetch current account + positions
        account = self._get_account()
        positions = self._list_positions() if self.config.include_positions else []

        equity = self._get_effective_equity(account)
        if equity <= 0:
            raise RuntimeError(f"Invalid account equity for execution: {equity}")
        pos_by_symbol = self._positions_by_symbol(positions)

        self.log.debug(
            "Account equity=%.2f positions=%s",
            equity,
            pos_by_symbol,
        )

        # 3) Compute desired notional deltas
        deltas = self._compute_target_deltas(
            target_weights=target_weights,
            equity=equity,
            positions_by_symbol=pos_by_symbol,
        )
        self.log.debug(
            "Computed target deltas: rebalance_id=%s deltas=%s",
            rebalance_id,
            deltas,
        )

        # 4) Build sell/buy orders subject to min-order size and float-noise thresholds
        sells, buys = self._build_market_orders(
            deltas=deltas,
            positions_by_symbol=pos_by_symbol,
            min_order_size=float(self.config.min_order_size),
        )

        if not sells and not buys:
            self.log.info(
                "No executable orders after filtering; treating as executed: rebalance_id=%s",
                rebalance_id,
            )
            return

        self.log.debug(
            "Built market orders: rebalance_id=%s sells=%s buys=%s",
            rebalance_id,
            sells,
            buys,
        )

        # 5) Sanity check tradability for all tickers in final plan
        self._assert_symbols_tradable([o["symbol"] for o in (sells + buys)])
        self.log.debug(
            "All symbols in rebalance plan are tradable: rebalance_id=%s",
            rebalance_id,
        )

        # 6) Execute sells first (cash generation), then buys; block until filled
        self._execute_orders_blocking(sells)
        self._execute_orders_blocking(buys)

    # ----------------------------
    # Helpers (testable)
    # ----------------------------

    @staticmethod
    def _normalize_symbol(symbol: Any) -> str:
        return str(symbol).strip().upper()

    @classmethod
    def _normalize_target_weights(
        cls,
        weights: Mapping[str, float],
        *,
        weight_epsilon: float = 1e-10,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not weights:
            return out
        for k, v in dict(weights).items():
            sym = cls._normalize_symbol(k)
            try:
                w = float(v)
            except Exception:
                continue
            if abs(w) <= weight_epsilon:
                continue
            out[sym] = w
        return out

    @staticmethod
    def _get_effective_equity(account: BrokerAccount) -> float:
        # Prefer adjusted equity (cash buffer), then equity, then portfolio_value.
        for v in (account.adj_equity, account.equity, account.portfolio_value):
            try:
                if v is None:
                    continue
                vf = float(v)
                if vf > 0:
                    return vf
            except Exception:
                continue
        return 0.0

    @classmethod
    def _positions_by_symbol(
        cls, positions: Iterable[BrokerPosition]
    ) -> Dict[str, BrokerPosition]:
        out: Dict[str, BrokerPosition] = {}
        for p in positions or []:
            sym = cls._normalize_symbol(getattr(p, "symbol", ""))
            if not sym:
                continue
            out[sym] = p
        return out

    @staticmethod
    def _compute_target_deltas(
        *,
        target_weights: Mapping[str, float],
        equity: float,
        positions_by_symbol: Mapping[str, BrokerPosition],
    ) -> Dict[str, Dict[str, float]]:
        # Returns per-symbol: current_value, target_value, delta_value.
        symbols = set(positions_by_symbol.keys()) | set(target_weights.keys())
        out: Dict[str, Dict[str, float]] = {}

        for sym in symbols:
            w = float(target_weights.get(sym, 0.0) or 0.0)
            target_value = equity * w

            p = positions_by_symbol.get(sym)
            mv = 0.0
            if p is not None:
                try:
                    mv = float(p.market_value or 0.0)
                except Exception:
                    mv = 0.0
            delta = target_value - mv
            out[sym] = {
                "current_value": mv,
                "target_value": target_value,
                "delta_value": delta,
            }

        return out

    @classmethod
    def _build_market_orders(
        cls,
        *,
        deltas: Mapping[str, Mapping[str, float]],
        positions_by_symbol: Mapping[str, BrokerPosition],
        min_order_size: float,
        dollar_epsilon: float = 1e-6,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        sells: List[Dict[str, Any]] = []
        buys: List[Dict[str, Any]] = []

        min_abs = max(0.0, float(min_order_size))

        for sym, info in dict(deltas).items():
            try:
                dv = float(info.get("delta_value", 0.0) or 0.0)
            except Exception:
                continue
            if abs(dv) <= dollar_epsilon:
                continue

            if dv < 0:
                # SELL: prefer notional if broker supports it, with qty fallback.
                if abs(dv) < min_abs:
                    continue
                p = positions_by_symbol.get(sym)
                if p is None:
                    continue

                desired_notional = cls._round_usd(abs(dv))
                if desired_notional is None:
                    continue

                # Cap notional sells to current position market value (best-effort).
                try:
                    mv = float(p.market_value or 0.0)
                except Exception:
                    mv = 0.0
                if mv > 0:
                    mv_cap = cls._round_usd(mv)
                    if mv_cap is not None:
                        desired_notional = min(desired_notional, mv_cap)

                if desired_notional < min_abs:
                    continue

                qty_fallback = cls._estimate_qty_for_notional_sell(
                    p, notional=float(desired_notional)
                )
                if qty_fallback is None or qty_fallback <= 0:
                    continue

                sells.append(
                    {
                        "symbol": sym,
                        "side": "sell",
                        # Try a notional sell first if supported.
                        "notional": desired_notional,
                        # Fallback if broker/API rejects notional sells.
                        "qty_fallback": qty_fallback,
                    }
                )
            else:
                # BUY: use notional market orders for simplicity
                notional = cls._round_usd(dv)
                if notional is None:
                    continue
                if notional < min_abs:
                    continue
                buys.append(
                    {
                        "symbol": sym,
                        "side": "buy",
                        "qty": None,
                        "notional": notional,
                    }
                )

        # Deterministic order
        sells.sort(key=lambda x: x["symbol"])
        buys.sort(key=lambda x: x["symbol"])
        return sells, buys

    @staticmethod
    def _estimate_qty_for_notional_sell(
        position: BrokerPosition, *, notional: float
    ) -> Optional[float]:
        # Approximate qty to sell from position market_value/qty.
        # If we can't compute a reasonable unit price, fall back to selling full qty.
        try:
            qty = float(position.qty or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            return None

        try:
            mv = float(position.market_value or 0.0)
        except Exception:
            mv = 0.0

        if mv > 0:
            px = mv / qty
            if px > 0:
                est_qty = notional / px
                # Never sell more than we hold.
                return min(qty, max(0.0, est_qty))

        return qty

    def _assert_symbols_tradable(self, symbols: Iterable[str]) -> None:
        # Prefer alpaca enums when available, but allow injected clients in tests.
        try:
            from alpaca.trading.enums import AssetStatus
        except Exception:
            AssetStatus = None

        unique = []
        seen = set()
        for s in symbols:
            sym = self._normalize_symbol(s)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            unique.append(sym)

        not_tradable: List[str] = []
        for sym in unique:
            try:
                asset = self._trading.get_asset(sym)
            except Exception:
                self.log.debug(
                    "Failed to fetch asset info for symbol=%s; assumes not tradable",
                    sym,
                )
                not_tradable.append(sym)
                continue
            self.log.debug("Fetched asset info for symbol=%s: %s", sym, asset)

            raw = getattr(asset, "_raw", None)
            if isinstance(raw, dict):
                tradable = raw.get("tradable")
                status = raw.get("status")
            else:
                tradable = getattr(asset, "tradable", None)
                status = getattr(asset, "status", None)

            if tradable is False:
                self.log.debug("Symbol not tradable: %s", sym)
                not_tradable.append(sym)
                continue

            if status is not None:
                if AssetStatus is not None:
                    try:
                        if status != AssetStatus.ACTIVE:
                            self.log.debug(
                                "Symbol not active: %s; status=%s", sym, status
                            )
                            not_tradable.append(sym)
                            continue
                    except Exception:
                        pass
                else:
                    # Best-effort string check
                    if str(status).lower() != "active":
                        self.log.debug("Symbol not active: %s; status=%s", sym, status)
                        not_tradable.append(sym)
                        continue

        if not_tradable:
            raise RuntimeError(
                f"Non-tradable symbols in rebalance plan: {sorted(not_tradable)}"
            )

    def _execute_orders_blocking(self, orders: List[Dict[str, Any]]) -> None:
        if not orders:
            return

        for order in orders:
            if self._shutdown_requested():
                raise EMLShutdownRequested("shutdown requested")

            symbol = self._normalize_symbol(order.get("symbol"))
            side = str(order.get("side")).strip().lower()
            if side not in {"buy", "sell"}:
                raise ValueError(f"Invalid order side: {side}")

            if side == "sell":
                order_id = self._submit_sell_market_order_prefer_notional(
                    symbol=symbol,
                    notional=order.get("notional"),
                    qty_fallback=order.get("qty_fallback"),
                )
            else:
                order_id = self._submit_market_order(
                    symbol=symbol,
                    side=side,
                    qty=None,
                    notional=order.get("notional"),
                )
            self.log.info(
                "Submitted market order: symbol=%s side=%s order_id=%s",
                symbol,
                side,
                order_id,
            )

            self.log.info(
                "Waiting for order fill: symbol=%s side=%s order_id=%s",
                symbol,
                side,
                order_id,
            )
            self._wait_for_order_fill(
                order_id,
                timeout_seconds=float(self.config.wait_for_order_fill_timeout_secs),
            )
            self.log.info(
                "Order filled: symbol=%s side=%s order_id=%s",
                symbol,
                side,
                order_id,
            )

    def _submit_sell_market_order_prefer_notional(
        self,
        *,
        symbol: str,
        notional: Any = None,
        qty_fallback: Any = None,
    ) -> str:
        """Submit a SELL market order.

        Prefers notional sells when available, but will fall back to qty if the
        broker/API rejects notional sells.
        """

        if notional is not None:
            try:
                return self._submit_market_order(
                    symbol=symbol,
                    side="sell",
                    qty=None,
                    notional=notional,
                )
            except Exception:
                self.log.warning(
                    "Notional sell rejected; falling back to qty sell: symbol=%s notional=%s",
                    symbol,
                    notional,
                    exc_info=True,
                )

        if qty_fallback is None:
            raise RuntimeError(
                f"Cannot submit sell order for {symbol}: no qty fallback available"
            )

        return self._submit_market_order(
            symbol=symbol,
            side="sell",
            qty=qty_fallback,
            notional=None,
        )

    def _submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: Any = None,
        notional: Any = None,
    ) -> str:
        if qty is not None and notional is not None:
            raise ValueError(
                "Market order must specify either qty or notional, not both"
            )

        if notional is not None:
            notional = self._round_usd(notional)
            if notional is None or notional <= 0:
                raise ValueError(f"Invalid notional: {notional}")

        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
            order_req = MarketOrderRequest(
                symbol=symbol,
                side=side_enum,
                time_in_force=TimeInForce.DAY,
                qty=qty,
                notional=notional,
            )
        except Exception as e:
            # Unit tests may inject a fake trading client without installing alpaca-py.
            if not getattr(self, "_injected_trading_client", False):
                raise RuntimeError(
                    "Missing dependency 'alpaca-py'. Install with `pip install alpaca-py`."
                ) from e

            class _FallbackMarketOrderRequest:
                def __init__(self, *, symbol: str, side: str, qty: Any, notional: Any):
                    self.symbol = symbol
                    self.side = side
                    self.qty = qty
                    self.notional = notional
                    self._raw = {
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "notional": notional,
                    }

            order_req = _FallbackMarketOrderRequest(
                symbol=symbol,
                side=side,
                qty=qty,
                notional=notional,
            )

        self.log.debug(
            "Submitting market order: symbol=%s side=%s qty=%s notional=%s order_req=%s",
            symbol,
            side,
            qty,
            notional,
            order_req,
        )
        submitted = self._trading.submit_order(order_req)
        raw = getattr(submitted, "_raw", None)
        if isinstance(raw, dict) and raw.get("id"):
            return str(raw.get("id"))

        oid = getattr(submitted, "id", None)
        if oid:
            return str(oid)
        raise RuntimeError("Alpaca submit_order returned no order id")

    def _wait_for_order_fill(
        self,
        order_id: str,
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        # Prefer alpaca enums when available, but allow injected clients in tests.
        try:
            from alpaca.trading.enums import OrderStatus
        except Exception:
            OrderStatus = None

        start = float(now_fn())

        while True:
            if self._shutdown_requested():
                raise EMLShutdownRequested("shutdown requested")

            if float(now_fn()) - start > float(timeout_seconds):
                raise TimeoutError(
                    f"Timed out waiting for order fill: order_id={order_id}"
                )

            o = self._trading.get_order_by_id(order_id)
            raw = getattr(o, "_raw", None)
            if isinstance(raw, dict) and raw.get("status") is not None:
                status = raw.get("status")
            else:
                status = getattr(o, "status", None)

            self.log.debug(
                "Polled order status: order_id=%s status=%s",
                order_id,
                status,
            )

            status_s = str(status).strip().lower()
            if OrderStatus is not None:
                try:
                    if status == OrderStatus.FILLED:
                        return
                    if status in {
                        OrderStatus.CANCELED,
                        OrderStatus.REJECTED,
                        OrderStatus.EXPIRED,
                    }:
                        raise RuntimeError(
                            f"Order did not fill (status={status}): order_id={order_id}"
                        )
                except Exception:
                    # fall back to string checks
                    pass

            if status_s == "filled":
                return
            if status_s in {"canceled", "cancelled", "rejected", "expired"}:
                raise RuntimeError(
                    f"Order did not fill (status={status_s}): order_id={order_id}"
                )

            sleep_fn(float(poll_interval_seconds))

    @staticmethod
    def _round_usd(v: Any) -> Optional[float]:
        """Round a USD notional to 2 decimals (down), as required by Alpaca."""
        try:
            d = Decimal(str(v))
        except Exception:
            return None
        if d.is_nan():
            return None
        if d <= 0:
            return None
        out = d.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        try:
            return float(out)
        except Exception:
            return None

    @staticmethod
    def _rebalance_request_from_state(
        payload: Mapping[str, Any],
    ) -> RebalancePlanRequestEvent:
        rebalance_id = str(payload.get("rebalance_id") or "")
        if not rebalance_id:
            raise ValueError("Invalid pending rebalance payload: missing rebalance_id")

        ts = payload.get("request_ts")
        try:
            ts_f = float(ts)
        except Exception:
            ts_f = time.time()

        weights = payload.get("weights")
        if weights is None:
            weights = {}
        if not isinstance(weights, Mapping):
            weights = {}

        return RebalancePlanRequestEvent(
            ts=ts_f,
            rebalance_id=rebalance_id,
            weights=dict(weights),
            source=str(payload.get("source") or ""),
            correlation_id=str(payload.get("correlation_id") or ""),
        )

    def _gc_execution_history(self, *, now_ts: Optional[float] = None) -> None:
        """Discard executed rebalance history entries older than max_execution_history_days."""

        max_days_int = self.config.max_execution_history_days
        if max_days_int <= 0:
            self.log.debug(
                "EMLConfig: max_execution_history_days <= 0; skipping execution history GC."
            )
            return

        now = float(now_ts if now_ts is not None else time.time())
        cutoff = now - (max_days_int * 86400.0)
        hist = self.state.executed_rebalance_history

        before = len(hist)
        kept: List[Dict[str, Any]] = []
        for item in hist:
            if not isinstance(item, dict):
                continue
            ts = item.get("executed_ts")
            try:
                ts_f = float(ts)
            except Exception:
                ts_f = 0.0
            if ts_f >= cutoff:
                kept.append(item)
            else:
                self.log.debug(
                    "GC'ing executed rebalance history entry: rebalance_id=%s executed_ts=%s",
                    item.get("rebalance_id"),
                    item.get("executed_ts"),
                )

        if len(kept) != before:
            kept.sort(key=lambda x: float(x.get("executed_ts", 0.0) or 0.0))
            self.state.executed_rebalance_history = kept
            self.log.info(
                "GC'd executed rebalance history: removed=%d kept=%d cutoff=%s",
                before - len(kept),
                len(kept),
                datetime.fromtimestamp(cutoff).isoformat(),
            )

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None
        return None

    def _get_account(self) -> BrokerAccount:
        acct = self._trading.get_account()
        return BrokerAccount(
            id=acct.id,
            status=acct.status,
            cash=self._to_float(acct.cash),
            buying_power=self._to_float(acct.buying_power),
            portfolio_value=self._to_float(acct.portfolio_value),
            equity=self._to_float(acct.equity),
            last_equity=self._to_float(getattr(acct, "last_equity", None)),
            adj_equity=self._get_equity_adj(
                self._to_float(getattr(acct, "equity", None))
            ),
        )

    def _list_positions(self) -> List[BrokerPosition]:
        try:
            pos_list = self._trading.get_all_positions()
        except Exception:
            try:
                pos_list = self._trading.get_positions()
            except Exception:
                pos_list = []

        out: List[BrokerPosition] = []
        for p in pos_list:
            symbol = p.symbol
            if not symbol:
                continue
            out.append(
                BrokerPosition(
                    symbol=str(symbol),
                    qty=self._to_float(p.qty),
                    market_value=self._to_float(p.market_value),
                    avg_entry_price=self._to_float(p.avg_entry_price),
                    side=p.side,
                    unrealized_pl=self._to_float(p.unrealized_pl),
                )
            )

        return out

    @staticmethod
    def _validate_config(config: EMLConfig) -> None:
        if config.cash_buffer_pct is not None and config.cash_buffer_abs is not None:
            raise ValueError(
                "EMLConfig: cash_buffer_pct and cash_buffer_abs are mutually exclusive; only one may be set."
            )

    def _get_equity_adj(self, equity_abs: Optional[float]) -> Optional[float]:
        """Compute adjusted equity after applying cash buffer settings.

        Returns None if equity_abs is None.
        """
        if equity_abs is None:
            return None

        if self.config.cash_buffer_pct is not None:
            buffer_amt = equity_abs * self.config.cash_buffer_pct
            return max(0.0, equity_abs - buffer_amt)

        if self.config.cash_buffer_abs is not None:
            buffer_amt = self.config.cash_buffer_abs
            return max(0.0, equity_abs - buffer_amt)

        return equity_abs
