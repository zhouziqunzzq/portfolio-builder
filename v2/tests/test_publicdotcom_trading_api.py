from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from public_api_sdk import InstrumentType as PublicInstrumentType
from public_api_sdk.exceptions import APIError, ValidationError
from public_api_sdk.models.instrument import TradingPermission as PublicTrading


V2_SRC = Path(__file__).resolve().parents[1] / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from models.trading import InstrumentMeta, InstrumentRef  # noqa: E402
from trading_api.exceptions import (  # noqa: E402
    InvalidOrder,
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
    mapped = PublicDotComTradingAPI._map_exception(exc)

    assert isinstance(mapped, UnsupportedOrderShape)
    assert isinstance(mapped, InvalidOrder)


def test_public_preserves_generic_validation_and_rejection_mapping():
    invalid = PublicDotComTradingAPI._map_exception(
        ValidationError("quantity must be positive")
    )
    rejected = PublicDotComTradingAPI._map_exception(
        APIError("order rejected", status_code=422)
    )

    assert type(invalid) is InvalidOrder
    assert type(rejected) is OrderRejected
