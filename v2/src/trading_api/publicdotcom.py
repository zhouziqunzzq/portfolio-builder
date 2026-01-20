from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, List, Optional

from public_api_sdk import (
    ApiKeyAuthConfig,
    InstrumentType as PublicInstrumentType,
    OrderExpirationRequest,
    OrderInstrument,
    OrderSide as PublicOrderSide,
    OrderStatus as PublicOrderStatus,
    OrderType as PublicOrderType,
    PreflightRequest,
    PublicApiClient,
    PublicApiClientConfiguration,
    TimeInForce as PublicTimeInForce,
)
from public_api_sdk.exceptions import (
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from public_api_sdk.models.instrument import Trading as PublicTrading
from public_api_sdk.models.order import Order as PublicOrder

from models import AccountSnapshot, PositionSnapshot
from models.trading import (
    BrokerCapabilities,
    Instrument,
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


class PublicDotComTradingAPI(BaseSyncTradingAPI):
    """Public.com broker adapter.

    Thin wrapper around the official `public_api_sdk.PublicApiClient` exposing the
    narrow `BaseSyncTradingAPI` surface expected by the execution layer.
    """

    name = "publicdotcom"

    def __init__(
        self,
        *,
        api_secret_key: Optional[str] = None,
        default_account_number: Optional[str] = None,
        base_url: Optional[str] = None,
        name: str = "PublicDotComTradingAPI",
    ):
        self.name = name

        self._api_secret_key = (
            api_secret_key
            or os.environ.get("PUBLICDOTCOM_API_SECRET_KEY")
            or os.environ.get("API_SECRET_KEY")
        )
        self._default_account_number = (
            default_account_number
            or os.environ.get("PUBLICDOTCOM_DEFAULT_ACCOUNT_NUMBER")
            or os.environ.get("DEFAULT_ACCOUNT_NUMBER")
        )
        self._base_url = base_url or os.environ.get("PUBLICDOTCOM_BASE_URL")

        if not self._api_secret_key:
            raise AuthError(
                "Missing Public.com credentials. Set API_SECRET_KEY (or PUBLICDOTCOM_API_SECRET_KEY)."
            )
        if not self._default_account_number:
            raise AuthError(
                "Missing Public.com default account number. Set DEFAULT_ACCOUNT_NUMBER (or PUBLICDOTCOM_DEFAULT_ACCOUNT_NUMBER)."
            )

        self._client = PublicApiClient(
            ApiKeyAuthConfig(api_secret_key=self._api_secret_key),
            config=PublicApiClientConfiguration(
                default_account_number=self._default_account_number,
                base_url=self._base_url,
            ),
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> BrokerCapabilities:
        # Public.com supports both notional ($ amount) and quantity-based equity orders,
        # and provides a preflight calculation endpoint.
        return BrokerCapabilities(
            supports_notional_market_orders=True,
            supports_qty_market_orders=True,
            supports_fractional_qty=True,
            supports_notional_sells=True,
            supports_preflight=True,
        )

    # ------------------------------------------------------------------
    # Account / Positions
    # ------------------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        try:
            p = self._client.get_portfolio()

            # Total equity/portfolio value: sum by asset type.
            total_equity: Optional[Decimal]
            try:
                total_equity = sum((e.value for e in p.equity), Decimal("0"))
            except Exception:
                total_equity = None

            cash = p.buying_power.cash_only_buying_power
            buying_power = p.buying_power.buying_power
            return AccountSnapshot(
                id=p.account_id,
                status=(
                    str(p.account_type.value) if p.account_type is not None else None
                ),
                cash=cash,
                buying_power=buying_power,
                portfolio_value=total_equity,
                equity=total_equity,
                last_equity=None,
                adj_equity=None,  # handled by EML
            )
        except Exception as e:
            raise self._map_exception(e) from e

    def list_positions(self) -> List[PositionSnapshot]:
        try:
            # Note: The docstring for get_portfolio().positions states that only non-IRA
            # accounts are supported, but in practice it seems to work for IRA accounts too.
            p = self._client.get_portfolio()
            out: List[PositionSnapshot] = []
            for pos in p.positions or []:
                symbol = self._normalize_symbol(pos.instrument.symbol)
                if not symbol:
                    continue
                out.append(
                    PositionSnapshot(
                        symbol=symbol,
                        qty=pos.quantity,
                        market_value=pos.current_value,
                        avg_entry_price=(
                            pos.cost_basis.unit_cost if pos.cost_basis else None
                        ),
                        side=None,
                        unrealized_pnl=(
                            pos.cost_basis.gain_value if pos.cost_basis else None
                        ),
                    )
                )
            return out
        except Exception as e:
            raise self._map_exception(e) from e

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def get_instrument(self, instrument: InstrumentRef) -> Instrument:
        symbol = self._normalize_symbol(instrument.symbol)

        try:
            inst = self._client.get_instrument(
                symbol=symbol,
                instrument_type=self._to_public_instrument_type(
                    instrument.instrument_type
                ),
            )
        except Exception as e:
            mapped = self._map_exception(e)
            if isinstance(mapped, OrderNotFoundYet):
                # Instrument lookup is not an eventual-consistency case.
                raise NotTradable(str(e) or "instrument not found") from e
            raise mapped from e

        tradable = inst.trading != PublicTrading.DISABLED
        fractionable = inst.fractional_trading == PublicTrading.BUY_AND_SELL

        return Instrument(
            instrument=InstrumentRef(
                symbol=inst.instrument.symbol,
                instrument_type=instrument.instrument_type,
            ),
            tradable=bool(tradable),
            fractionable=bool(fractionable),
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def preflight_order(self, intent: OrderIntent) -> PreflightOrderResult:
        symbol = self._normalize_symbol(intent.instrument.symbol)
        try:
            req = self._build_preflight_request(intent=intent, symbol=symbol)
            resp = self._client.perform_preflight_calculation(req)

            est_fees = None
            if resp.regulatory_fees is not None:
                try:
                    est_fees = sum(
                        (
                            resp.regulatory_fees.sec_fee,
                            resp.regulatory_fees.taf_fee,
                            resp.regulatory_fees.orf_fee,
                            resp.regulatory_fees.exchange_fee,
                            resp.regulatory_fees.occ_fee,
                            resp.regulatory_fees.cat_fee,
                        ),
                        Decimal("0"),
                    )
                except Exception:
                    est_fees = None

            return PreflightOrderResult(
                instrument=intent.instrument,
                estimated_commission=resp.estimated_commission,
                estimated_fees=est_fees,
                estimated_cost=resp.estimated_cost,
                estimated_proceeds=resp.estimated_proceeds,
                raw=resp,
            )
        except Exception as e:
            raise self._map_exception(e) from e

    def submit_order(self, intent: OrderIntent) -> PlacedOrder:
        symbol = self._normalize_symbol(intent.instrument.symbol)
        try:
            req = self._build_order_request(intent=intent, symbol=symbol)
            new_order = self._client.place_order(req)
            broker_order_id = new_order.order_id
            if not broker_order_id:
                raise BrokerApiError("Public.com place_order returned no order id")
            self.log.debug(
                "Submitted Public.com order %s for intent %s", broker_order_id, intent
            )
            return PlacedOrder(
                broker_order_id=str(broker_order_id),
                client_order_id=intent.client_order_id,
                submitted_at=None,
                raw=new_order,
            )
        except BrokerApiError:
            raise
        except Exception as e:
            raise self._map_exception(e) from e

    def get_order(self, broker_order_id: str) -> OrderState:
        try:
            o = self._client.get_order(order_id=str(broker_order_id))
        except Exception as e:
            mapped = self._map_exception(e)
            if isinstance(mapped, OrderNotFoundYet):
                raise OrderNotFoundYet("order not visible yet") from e
            raise mapped from e

        return self._to_order_state(o)

    def list_orders(self, order_filter: OrderFilter) -> List[OrderState]:
        # Best-effort: portfolio endpoint provides current orders.
        try:
            requested_statuses = set(order_filter.statuses or [])
            p = self._client.get_portfolio()

            out: List[OrderState] = []
            for o in p.orders or []:
                st = self._to_order_state(o)
                if requested_statuses and st.status not in requested_statuses:
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
            self._client.cancel_order(order_id=str(broker_order_id))
        except Exception as e:
            raise self._map_exception(e) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        # Repo convention: uppercase and replace dots with dashes.
        return str(symbol).strip().upper().replace(".", "-")

    @staticmethod
    def _round_usd(v: Decimal) -> Decimal:
        return Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    @staticmethod
    def _to_public_instrument_type(_: Any) -> PublicInstrumentType:
        # v2 execution currently only trades equities.
        return PublicInstrumentType.EQUITY

    @staticmethod
    def _to_public_side(side: OrderSide) -> PublicOrderSide:
        return PublicOrderSide.BUY if side == OrderSide.BUY else PublicOrderSide.SELL

    @staticmethod
    def _to_public_tif(tif: TimeInForce) -> PublicTimeInForce:
        if tif != TimeInForce.DAY:
            raise InvalidOrder(f"Unsupported time_in_force: {tif}")
        return PublicTimeInForce.DAY

    def _build_preflight_request(
        self, *, intent: OrderIntent, symbol: str
    ) -> PreflightRequest:
        qty = intent.qty
        notional = intent.notional

        if notional is not None:
            notional = self._round_usd(notional)
            self.log.debug("Rounded notional from %s to %s", intent.notional, notional)
            if notional <= 0:
                raise InvalidOrder(f"Invalid notional: {notional}")

        if qty is not None:
            try:
                qty_d = Decimal(str(qty))
            except Exception:
                raise InvalidOrder(f"Invalid qty: {qty}")
            if qty_d <= 0:
                raise InvalidOrder(f"Invalid qty: {qty}")

        order_type = (
            PublicOrderType.MARKET
            if intent.order_type == OrderType.MARKET
            else PublicOrderType.LIMIT
        )
        limit_price = None
        if intent.order_type == OrderType.LIMIT:
            limit_price = (
                self._round_usd(intent.limit_price) if intent.limit_price else None
            )
            self.log.debug(
                "Rounded limit_price from %s to %s", intent.limit_price, limit_price
            )
            if limit_price is None:
                raise InvalidOrder("Limit orders require limit_price")

        return PreflightRequest(
            instrument=OrderInstrument(symbol=symbol, type=PublicInstrumentType.EQUITY),
            order_side=self._to_public_side(intent.side),
            order_type=order_type,
            expiration=OrderExpirationRequest(
                time_in_force=self._to_public_tif(intent.time_in_force)
            ),
            quantity=qty,
            amount=notional,
            limit_price=limit_price,
        )

    def _build_order_request(self, *, intent: OrderIntent, symbol: str):
        from public_api_sdk import OrderRequest

        qty = intent.qty
        notional = intent.notional

        if notional is not None:
            notional = self._round_usd(notional)
            self.log.debug("Rounded notional from %s to %s", intent.notional, notional)
            if notional <= 0:
                raise InvalidOrder(f"Invalid notional: {notional}")

        if qty is not None:
            try:
                qty_d = Decimal(str(qty))
            except Exception:
                raise InvalidOrder(f"Invalid qty: {qty}")
            if qty_d <= 0:
                raise InvalidOrder(f"Invalid qty: {qty}")

        order_type = (
            PublicOrderType.MARKET
            if intent.order_type == OrderType.MARKET
            else PublicOrderType.LIMIT
        )
        limit_price = None
        if intent.order_type == OrderType.LIMIT:
            limit_price = (
                self._round_usd(intent.limit_price) if intent.limit_price else None
            )
            self.log.debug(
                "Rounded limit_price from %s to %s", intent.limit_price, limit_price
            )
            if limit_price is None:
                raise InvalidOrder("Limit orders require limit_price")

        return OrderRequest(
            order_id=intent.client_order_id,
            instrument=OrderInstrument(symbol=symbol, type=PublicInstrumentType.EQUITY),
            order_side=self._to_public_side(intent.side),
            order_type=order_type,
            expiration=OrderExpirationRequest(
                time_in_force=self._to_public_tif(intent.time_in_force)
            ),
            quantity=qty,
            amount=notional,
            limit_price=limit_price,
        )

    def _to_order_state(self, order_obj: PublicOrder) -> OrderState:
        status = self._map_order_status(order_obj.status)

        filled_qty = order_obj.filled_quantity
        avg_fill_price = order_obj.average_price
        filled_notional = None
        if filled_qty is not None and avg_fill_price is not None:
            try:
                filled_notional = Decimal(filled_qty) * Decimal(avg_fill_price)
            except Exception:
                filled_notional = None

        last_update_ts = self._parse_dt_to_ts(
            order_obj.closed_at or order_obj.created_at
        )

        return OrderState(
            broker_order_id=str(order_obj.order_id),
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
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
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

        if isinstance(value, PublicOrderStatus):
            s = str(value.value).strip().upper()
        else:
            s = str(value).strip().upper()

        if s == "NEW":
            return OrderStatus.NEW
        if s == "PARTIALLY_FILLED":
            return OrderStatus.PARTIALLY_FILLED
        if s in {"PENDING_CANCEL", "PENDING_REPLACE"}:
            return OrderStatus.OPEN
        if s in {"CANCELLED", "QUEUED_CANCELLED", "REPLACED"}:
            return OrderStatus.CANCELED
        if s == "FILLED":
            return OrderStatus.FILLED
        if s == "REJECTED":
            return OrderStatus.REJECTED
        if s == "EXPIRED":
            return OrderStatus.EXPIRED
        return OrderStatus.UNKNOWN

    @staticmethod
    def _looks_like_not_found(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "not found" in msg or "404" in msg

    @staticmethod
    def _map_exception(exc: Exception) -> BrokerApiError:
        if isinstance(exc, AuthenticationError):
            return AuthError(str(exc) or "unauthorized")
        if isinstance(exc, RateLimitError):
            return RateLimited(str(exc) or "rate limited")
        if isinstance(exc, ServerError):
            return TemporaryUnavailable(str(exc) or "temporary unavailable")
        if isinstance(exc, ValidationError):
            return InvalidOrder(str(exc) or "invalid order")
        if isinstance(exc, NotFoundError):
            # Used for get_order (eventual consistency) and some lookups.
            return OrderNotFoundYet(str(exc) or "not found")

        if isinstance(exc, APIError):
            code = exc.status_code
            if code in {401, 403}:
                return AuthError(str(exc) or "unauthorized")
            if code == 429:
                return RateLimited(str(exc) or "rate limited")
            if code is not None and 500 <= int(code) <= 599:
                return TemporaryUnavailable(str(exc) or "temporary unavailable")
            if code == 400:
                return InvalidOrder(str(exc) or "invalid order")
            if code == 422:
                return OrderRejected(str(exc) or "order rejected")
            if code == 404:
                return OrderNotFoundYet(str(exc) or "not found")

        msg = str(exc).lower()
        if "rate limit" in msg or "too many" in msg:
            return RateLimited(str(exc) or "rate limited")
        if "unauthorized" in msg or "forbidden" in msg:
            return AuthError(str(exc) or "unauthorized")
        if "not tradable" in msg:
            return NotTradable(str(exc) or "not tradable")
        if "not found" in msg or "404" in msg:
            return OrderNotFoundYet(str(exc) or "not found")

        return BrokerApiError(str(exc) or exc.__class__.__name__)
