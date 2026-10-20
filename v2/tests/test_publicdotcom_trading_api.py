from decimal import Decimal
import logging
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from public_api_sdk import InstrumentType as PublicInstrumentType
from public_api_sdk.exceptions import (
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from public_api_sdk.models.instrument import TradingPermission as PublicTrading


V2_SRC = Path(__file__).resolve().parents[1] / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from models.trading import (  # noqa: E402
    InstrumentMeta,
    InstrumentRef,
    OrderFilter,
    OrderIntent,
    OrderSide,
)
from trading_api.exceptions import (  # noqa: E402
    AuthError,
    BrokerApiError,
    InvalidOrder,
    NotTradable,
    OrderNotFoundYet,
    RateLimited,
    TemporaryUnavailable,
    OrderRejected,
    UnsupportedOrderShape,
)
from trading_api.publicdotcom import PublicDotComTradingAPI  # noqa: E402


class _InstrumentClient:
    def __init__(self, *, fractional_trading: PublicTrading):
        self._instrument = SimpleNamespace(
            instrument=SimpleNamespace(symbol="DELL"),
            trading=PublicTrading.BUY_AND_SELL,
            fractional_trading=fractional_trading,
        )

    def get_instrument(self, *, symbol, instrument_type):
        assert symbol == "DELL"
        assert instrument_type == PublicInstrumentType.EQUITY
        return self._instrument


def _adapter_with_client(client) -> PublicDotComTradingAPI:
    adapter = object.__new__(PublicDotComTradingAPI)
    adapter._client = client
    adapter.log = logging.getLogger("test-public-adapter")
    return adapter


def test_instrument_notional_buy_capability_defaults_to_unknown():
    metadata = InstrumentMeta(instrument=InstrumentRef(symbol="SPY"))

    assert metadata.supports_notional_buys is None


@pytest.mark.parametrize(
    ("fractional_trading", "expected"),
    [
        (PublicTrading.BUY_AND_SELL, True),
        (PublicTrading.LIQUIDATION_ONLY, False),
        (PublicTrading.DISABLED, False),
    ],
)
def test_public_instrument_uses_fractional_buy_permission_as_notional_buy_signal(
    fractional_trading, expected
):
    adapter = _adapter_with_client(
        _InstrumentClient(fractional_trading=fractional_trading)
    )

    metadata = adapter.get_instrument(InstrumentRef(symbol="dell"))

    assert metadata.fractionable is expected
    assert metadata.supports_notional_buys is expected


@pytest.mark.parametrize(
    "exc",
    [
        ValidationError("Amount orders are not allowed"),
        APIError(
            "order rejected",
            status_code=422,
            response_data={"detail": "Notional orders are not supported"},
        ),
    ],
)
def test_public_maps_unsupported_amount_order_to_specific_exception(exc):
    mapped = PublicDotComTradingAPI._map_exception(exc, operation="submit_order")

    assert isinstance(mapped, UnsupportedOrderShape)
    assert isinstance(mapped, InvalidOrder)


def test_public_preserves_generic_validation_and_rejection_mapping():
    invalid = PublicDotComTradingAPI._map_exception(
        ValidationError("quantity must be positive"), operation="submit_order"
    )
    rejected = PublicDotComTradingAPI._map_exception(
        APIError("order rejected", status_code=422), operation="submit_order"
    )

    assert type(invalid) is InvalidOrder
    assert type(rejected) is OrderRejected


class _FailingClient:
    """SDK stub: all calls fail before returning a response, with no network."""

    def __init__(self, error):
        self.error = error
        self.calls = []

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            self.calls.append(name)
            raise self.error

        return fail


def _invoke(adapter, operation):
    if operation in {"preflight_order", "submit_order"}:
        intent = OrderIntent(
            client_order_id="b8cb89ed-7191-4e46-9dc3-5016f2f7caae",
            instrument=InstrumentRef(symbol="SPY"),
            side=OrderSide.BUY,
            qty=Decimal("1"),
        )
        return getattr(adapter, operation)(intent)
    if operation == "get_instrument":
        return adapter.get_instrument(InstrumentRef(symbol="SPY"))
    if operation == "list_orders":
        return adapter.list_orders(OrderFilter())
    if operation in {"get_order", "cancel_order"}:
        return getattr(adapter, operation)("test-only")
    return getattr(adapter, operation)()


OPERATIONS = (
    "get_account",
    "list_positions",
    "get_instrument",
    "list_orders",
    "get_order",
    "preflight_order",
    "submit_order",
    "cancel_order",
)
READ_OPERATIONS = OPERATIONS[:5]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("payload_only", [False, True])
def test_wrapped_upstream_timeout_is_transient_at_every_entrypoint(
    operation, payload_only
):
    detail = "Unable to read upstream response with status 408"
    original = ValidationError(
        "Bad request" if payload_only else detail,
        status_code=400,
        response_data={"message": detail} if payload_only else {},
    )
    client = _FailingClient(original)
    adapter = _adapter_with_client(client)

    with pytest.raises(TemporaryUnavailable) as caught:
        _invoke(adapter, operation)

    assert caught.value.__cause__ is original
    assert str(caught.value).startswith(f"{operation}:")
    # Classification alone must not replay any request, especially a submission.
    assert len(client.calls) == 1


@pytest.mark.parametrize("operation", READ_OPERATIONS)
@pytest.mark.parametrize(
    "error",
    [
        ValidationError("Invalid account input", status_code=400),
        APIError("Invalid account input", status_code=400),
        APIError("Invalid account input", status_code=422),
        ValidationError("Amount orders are not allowed", status_code=400),
    ],
)
def test_read_validation_never_becomes_order_rejection(operation, error):
    adapter = _adapter_with_client(_FailingClient(error))
    with pytest.raises(BrokerApiError) as caught:
        _invoke(adapter, operation)
    assert type(caught.value) is BrokerApiError
    assert caught.value.__cause__ is error
    assert str(caught.value).startswith(f"{operation}:")


@pytest.mark.parametrize(
    "operation", ["preflight_order", "submit_order", "cancel_order"]
)
@pytest.mark.parametrize(
    "error, expected",
    [
        (ValidationError("quantity must be positive", status_code=400), InvalidOrder),
        (APIError("quantity must be positive", status_code=400), InvalidOrder),
        (APIError("order rejected", status_code=422), OrderRejected),
        (
            ValidationError("Amount orders are not allowed", status_code=400),
            UnsupportedOrderShape,
        ),
    ],
)
def test_real_order_failures_remain_terminal_at_entrypoints(operation, error, expected):
    adapter = _adapter_with_client(_FailingClient(error))
    with pytest.raises(expected) as caught:
        _invoke(adapter, operation)
    assert type(caught.value) is expected
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "error, expected",
    [
        (AuthenticationError("unauthorized"), AuthError),
        (APIError("forbidden", status_code=403), AuthError),
        (RateLimitError("too many requests", retry_after=7), RateLimited),
        (APIError("rate limited", status_code=429), RateLimited),
        (ServerError("unavailable", status_code=503), TemporaryUnavailable),
        (APIError("request timeout", status_code=408), TemporaryUnavailable),
        (APIError("gateway timeout", status_code=504), TemporaryUnavailable),
        (
            APIError(
                "Unable to read upstream response with status 503", status_code=400
            ),
            TemporaryUnavailable,
        ),
    ],
)
def test_transport_auth_and_throttling_classification(error, expected):
    mapped = PublicDotComTradingAPI._map_exception(error, operation="get_account")
    assert type(mapped) is expected
    if isinstance(error, RateLimitError):
        assert mapped.retry_after_seconds == 7


@pytest.mark.parametrize(
    "message",
    [
        "Invalid quantity 408",
        "Invalid order 504",
        "Request timeout setting is invalid",
        "Unable to read upstream response with status 400",
        "Unable to read upstream response with status 4080",
    ],
)
def test_unrelated_validation_is_not_made_retryable(message):
    error = ValidationError(message, status_code=400)
    mapped = PublicDotComTradingAPI._map_exception(error, operation="submit_order")
    assert type(mapped) is InvalidOrder


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("get_account", BrokerApiError),
        ("list_positions", BrokerApiError),
        ("list_orders", BrokerApiError),
        ("get_order", OrderNotFoundYet),
        ("get_instrument", NotTradable),
        ("cancel_order", OrderNotFoundYet),
    ],
)
def test_not_found_keeps_lookup_specific_semantics(operation, expected):
    adapter = _adapter_with_client(_FailingClient(NotFoundError("not found")))
    with pytest.raises(expected) as caught:
        _invoke(adapter, operation)
    assert type(caught.value) is expected


def test_local_preflight_validation_is_not_erased():
    adapter = _adapter_with_client(_FailingClient(AssertionError("must not call SDK")))
    intent = OrderIntent(
        client_order_id="b8cb89ed-7191-4e46-9dc3-5016f2f7caae",
        instrument=InstrumentRef(symbol="SPY"),
        side=OrderSide.BUY,
        qty=Decimal("0"),
    )
    with pytest.raises(InvalidOrder, match="Invalid qty"):
        adapter.preflight_order(intent)
    assert not adapter._client.calls


def test_actual_sdk_http_400_envelope_maps_account_timeout():
    # Exercise SDK classification too, without opening a connection or using credentials.
    from public_api_sdk.api_client import ApiClient
    from requests import Response

    response = Response()
    response.status_code = 400
    response._content = (
        b'{"message":"Unable to read upstream response with status 408"}'
    )
    sdk = object.__new__(ApiClient)
    with pytest.raises(ValidationError) as sdk_error:
        sdk._handle_response(response)

    adapter = _adapter_with_client(_FailingClient(sdk_error.value))
    with pytest.raises(TemporaryUnavailable) as caught:
        adapter.get_account()
    assert caught.value.__cause__ is sdk_error.value
    assert sdk_error.value.status_code == 400
