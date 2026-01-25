from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.enums import AssetStatus
from alpaca.trading.enums import PositionSide as AlpacaPositionSide
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OrderRequest,
)
from alpaca.trading.models import Order as AlpacaOrder

from models import AccountSnapshot, PositionSnapshot, PositionSide
from models.trading import (
    BrokerCapabilities,
    InstrumentMeta,
    InstrumentRef,
    OrderFilter,
    OrderIntent,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    PreflightOrderResult,
    PlacedOrder,
    TimeInForce,
)
from trading_api.base import BaseSyncTradingAPI
from trading_api.exceptions import (
    AuthError,
    BrokerApiError,
    InvalidOrder,
    NotTradable,
    OrderNotFoundYet,
    OrderRejected,
    RateLimited,
    TemporaryUnavailable,
)
from utils.decimals import to_decimal


class AlpacaTradingAPI(BaseSyncTradingAPI):
    """Alpaca broker adapter.

    This is a thin wrapper around `alpaca-py`'s `TradingClient` that exposes
    the narrow `BaseSyncTradingAPI` surface expected by the execution layer.
    """

    name = "alpaca"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: Optional[bool] = None,
        name: str = "AlpacaTradingAPI",
        list_orders_limit: int = 500,
    ):
        self.name = name

        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self._base_url = base_url or os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self._list_orders_limit = int(list_orders_limit)

        if paper is None:
            env_paper = os.environ.get("ALPACA_PAPER")
            if env_paper is None:
                self._paper = True
            else:
                self._paper = str(env_paper).strip().lower() in {"1", "true", "yes"}
        else:
            self._paper = bool(paper)

        if not self._api_key or not self._secret_key:
            raise AuthError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        self._trading = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
            url_override=self._base_url,
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> BrokerCapabilities:
        # Alpaca supports market orders with qty and (for buys) notional.
        # Notional sells may be rejected depending on account/asset; treat as unsupported.
        return BrokerCapabilities(
            supports_notional_market_orders=True,
            supports_qty_market_orders=True,
            supports_fractional_qty=True,
            supports_notional_sells=False,
            supports_preflight=False,
        )

    # ------------------------------------------------------------------
    # Account / Positions
    # ------------------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        try:
            acct = self._trading.get_account()
            equity = to_decimal(getattr(acct, "equity", None))
            return AccountSnapshot(
                id=getattr(acct, "id", None),
                status=getattr(acct, "status", None),
                cash=to_decimal(getattr(acct, "cash", None)),
                buying_power=to_decimal(getattr(acct, "buying_power", None)),
                portfolio_value=to_decimal(getattr(acct, "portfolio_value", None)),
                equity=equity,
                last_equity=to_decimal(getattr(acct, "last_equity", None)),
                adj_equity=None,  # Handled by EML
            )
        except Exception as e:
            raise self._map_exception(e) from e

    def list_positions(self) -> List[PositionSnapshot]:
        try:
            pos_list = self._trading.get_all_positions()
            out: List[PositionSnapshot] = []
            for p in pos_list:
                symbol = p.symbol
                if not symbol:
                    continue
                out.append(
                    PositionSnapshot(
                        symbol=str(symbol),
                        qty=to_decimal(p.qty),
                        market_value=to_decimal(p.market_value),
                        avg_entry_price=to_decimal(p.avg_entry_price),
                        side=(
                            PositionSide.LONG
                            if p.side == AlpacaPositionSide.LONG
                            else PositionSide.SHORT
                        ),
                        unrealized_pnl=to_decimal(p.unrealized_pl),
                    )
                )

            return out
        except Exception as e:
            raise self._map_exception(e) from e

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def get_instrument(self, instrument: InstrumentRef) -> InstrumentMeta:
        symbol = self._normalize_symbol(instrument.symbol)
        try:
            asset = self._trading.get_asset(symbol)
        except Exception as e:
            raise self._map_exception(e) from e

        tradable = asset.tradable
        fractionable = asset.fractionable
        status = asset.status
        if status != AssetStatus.ACTIVE:
            self.log.debug(
                f"Instrument {asset.symbol} status is {status}, marking as not tradable"
            )
            tradable = False

        return InstrumentMeta(
            instrument=InstrumentRef(
                symbol=asset.symbol, instrument_type=instrument.instrument_type
            ),
            tradable=bool(tradable) if tradable is not None else None,
            fractionable=bool(fractionable) if fractionable is not None else None,
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def preflight_order(self, intent: OrderIntent) -> PreflightOrderResult:
        # Alpaca does not provide a reliable preflight endpoint (fees/commissions
        # are not deterministically returned upfront via alpaca-py). Treat as a no-op.
        return PreflightOrderResult(instrument=intent.instrument, raw={"noop": True})

    def submit_order(self, intent: OrderIntent) -> PlacedOrder:
        symbol = self._normalize_symbol(intent.instrument.symbol)
        try:
            order_req = self._build_order_request(intent=intent, symbol=symbol)
            submitted = self._trading.submit_order(order_req)
            broker_order_id = submitted.id
            if not broker_order_id:
                raise BrokerApiError("Alpaca submit_order returned no order id")
            self.log.debug(
                f"Submitted Alpaca order {broker_order_id} for intent {intent}"
            )
            return PlacedOrder(
                broker_order_id=str(broker_order_id),
                client_order_id=intent.client_order_id,
                submitted_at=submitted.submitted_at,
                raw=submitted,
            )
        except BrokerApiError:
            raise
        except Exception as e:
            raise self._map_exception(e) from e

    def get_order(self, broker_order_id: str) -> OrderState:
        try:
            o = self._trading.get_order_by_id(str(broker_order_id))
        except Exception as e:
            mapped = self._map_exception(e)
            if isinstance(mapped, BrokerApiError) and self._looks_like_not_found(e):
                raise OrderNotFoundYet("order not visible yet") from e
            raise mapped from e

        return self._to_order_state(o)

    def list_orders(self, order_filter: OrderFilter) -> List[OrderState]:
        try:
            requested_statuses = set(order_filter.statuses or [])

            openish = {
                OrderStatus.NEW,
                OrderStatus.ACCEPTED,
                OrderStatus.OPEN,
                OrderStatus.PARTIALLY_FILLED,
            }
            closedish = {
                OrderStatus.FILLED,
                OrderStatus.CANCELED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
            }

            query_status = QueryOrderStatus.ALL
            if requested_statuses:
                if requested_statuses.issubset(openish):
                    query_status = QueryOrderStatus.OPEN
                elif requested_statuses.issubset(closedish):
                    query_status = QueryOrderStatus.CLOSED

            orders = self._trading.get_orders(
                filter=GetOrdersRequest(
                    status=query_status, limit=self._list_orders_limit
                )
            )

            out: List[OrderState] = []
            for o in orders or []:
                st = self._to_order_state(o)
                if (
                    requested_statuses is not None
                    and st.status not in requested_statuses
                ):
                    self.log.debug(
                        "Skipping order %s with status %s not in filter %s",
                        st.broker_order_id,
                        st.status,
                        requested_statuses,
                    )
                    continue
                out.append(st)
            return out
        except Exception as e:
            raise self._map_exception(e) from e

    def cancel_order(self, broker_order_id: str) -> None:
        try:
            self._trading.cancel_order_by_id(str(broker_order_id))
            return
        except Exception as e:
            raise self._map_exception(e) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        # Repo convention: uppercase and replace dots with dashes for yfinance/alpaca.
        return str(symbol).strip().upper().replace(".", "-")

    @staticmethod
    def _round_usd(v: Decimal) -> Decimal:
        return Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def _build_order_request(self, *, intent: OrderIntent, symbol: str) -> OrderRequest:
        """Create an alpaca-py order request."""
        qty = intent.qty
        notional = intent.notional

        if notional is not None:
            notional = self._round_usd(notional)
            self.log.debug(f"Rounded notional from {intent.notional} to {notional}")
            if notional <= 0:
                raise InvalidOrder(f"Invalid notional: {notional}")

        if qty is not None:
            try:
                qty_d = Decimal(str(qty))
            except Exception:
                raise InvalidOrder(f"Invalid qty: {qty}")
            if qty_d <= 0:
                raise InvalidOrder(f"Invalid qty: {qty}")

        side = AlpacaSide.BUY if intent.side == OrderSide.BUY else AlpacaSide.SELL
        tif = AlpacaTIF.DAY
        if intent.time_in_force != TimeInForce.DAY:
            raise InvalidOrder(f"Unsupported time_in_force: {intent.time_in_force}")

        if intent.order_type == OrderType.MARKET:
            return MarketOrderRequest(
                symbol=symbol,
                side=side,
                time_in_force=tif,
                qty=float(qty) if qty is not None else None,
                notional=float(notional) if notional is not None else None,
                client_order_id=intent.client_order_id,
            )

        if intent.order_type == OrderType.LIMIT:
            if intent.limit_price is None:
                raise InvalidOrder("Limit orders require limit_price")
            limit_price = self._round_usd(intent.limit_price)
            self.log.debug(
                f"Rounded limit_price from {intent.limit_price} to {limit_price}"
            )
            return LimitOrderRequest(
                symbol=symbol,
                side=side,
                time_in_force=tif,
                qty=float(qty) if qty is not None else None,
                limit_price=float(limit_price),
                client_order_id=intent.client_order_id,
            )

        raise InvalidOrder(f"Unsupported order_type: {intent.order_type}")

    def _to_order_state(self, order_obj: AlpacaOrder) -> OrderState:
        status_raw = order_obj.status
        filled_qty_raw = order_obj.filled_qty
        filled_avg_price_raw = order_obj.filled_avg_price
        updated_at_raw = order_obj.updated_at or order_obj.filled_at
        broker_order_id = order_obj.id

        status = self._map_order_status(status_raw)
        filled_qty = to_decimal(filled_qty_raw)
        avg_fill_price = to_decimal(filled_avg_price_raw)
        filled_notional = None
        if filled_qty is not None and avg_fill_price is not None:
            try:
                filled_notional = Decimal(str(filled_qty)) * Decimal(
                    str(avg_fill_price)
                )
            except Exception:
                self.log.debug(
                    f"Failed to compute filled_notional for order {broker_order_id}"
                )
                filled_notional = None
        last_update_ts = self._parse_dt_to_ts(updated_at_raw)

        return OrderState(
            broker_order_id=str(broker_order_id) if broker_order_id is not None else "",
            status=status,
            filled_qty=filled_qty,
            filled_notional=filled_notional,
            avg_fill_price=avg_fill_price,
            last_update_ts=last_update_ts,
            raw=order_obj,
        )

    @staticmethod
    def _parse_dt(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            s = str(value).strip()
            if not s:
                return None
            # Handle common ISO-8601 with Z suffix.
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            return dt
        except Exception:
            return None

    @classmethod
    def _parse_dt_to_ts(cls, value: Any) -> Optional[float]:
        dt = cls._parse_dt(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            return float(dt.timestamp())
        except Exception:
            return None

    @staticmethod
    def _map_order_status(value: Any) -> OrderStatus:
        if value is None:
            return OrderStatus.UNKNOWN

        if isinstance(value, AlpacaOrderStatus):
            value = value.value
        s = str(value).strip().lower()
        if s in {"new"}:
            return OrderStatus.NEW
        if s in {"accepted"}:
            return OrderStatus.ACCEPTED
        if s in {"pending_new"}:
            return OrderStatus.OPEN
        if s in {"partially_filled"}:
            return OrderStatus.PARTIALLY_FILLED
        if s in {"filled"}:
            return OrderStatus.FILLED
        if s in {"canceled", "cancelled"}:
            return OrderStatus.CANCELED
        if s in {"rejected"}:
            return OrderStatus.REJECTED
        if s in {"expired"}:
            return OrderStatus.EXPIRED
        return OrderStatus.UNKNOWN

    @staticmethod
    def _looks_like_not_found(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "not found" in msg or "404" in msg

    @staticmethod
    def _map_exception(exc: Exception) -> BrokerApiError:
        # Best-effort mapping based on common fields.
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)

        if status_code in {401, 403}:
            return AuthError(str(exc) or "unauthorized")
        if status_code == 429:
            retry_after = None
            try:
                retry_after_hdr = getattr(
                    getattr(exc, "response", None), "headers", {}
                ).get("Retry-After")
                if retry_after_hdr is not None:
                    retry_after = float(retry_after_hdr)
            except Exception:
                retry_after = None
            return RateLimited(
                str(exc) or "rate limited", retry_after_seconds=retry_after
            )
        if status_code is not None and 500 <= int(status_code) <= 599:
            return TemporaryUnavailable(str(exc) or "temporary unavailable")
        if status_code == 400:
            return InvalidOrder(str(exc) or "invalid order")
        if status_code == 422:
            return OrderRejected(str(exc) or "order rejected")

        # Some Alpaca errors embed codes in the message.
        msg = str(exc).lower()
        if "rate limit" in msg or "too many requests" in msg:
            return RateLimited(str(exc) or "rate limited")
        if "unauthorized" in msg or "forbidden" in msg:
            return AuthError(str(exc) or "unauthorized")
        if "not tradable" in msg or "asset is not tradable" in msg:
            return NotTradable(str(exc) or "not tradable")

        return BrokerApiError(str(exc) or exc.__class__.__name__)
