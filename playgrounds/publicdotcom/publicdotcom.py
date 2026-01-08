import os
from decimal import Decimal
import uuid
import time

from public_api_sdk import (
    AccountType,
    ApiKeyAuthConfig,
    HistoryRequest,
    InstrumentsRequest,
    InstrumentType,
    Trading,
    PublicApiClient,
    PublicApiClientConfiguration,
    OrderExpirationRequest,
    OrderInstrument,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderRequest,
    PreflightRequest,
    TimeInForce,
    WaitTimeoutError,
)

from dotenv import load_dotenv

# load env variables from .env file
load_dotenv()


def example_accounts(cli: PublicApiClient):
    accounts = cli.get_accounts()
    print(f"Accounts: {accounts.accounts}")

    p = cli.get_portfolio()  # default account
    print(f"Default account portfolio: {p}")


def example_instruments(cli: PublicApiClient):
    instrument = public_api_client.get_instrument(
        symbol="SPY",
        instrument_type=InstrumentType.EQUITY,
    )
    print(f"Instrument SPY: {instrument}")


def example_market_data(cli: PublicApiClient):
    quotes = public_api_client.get_quotes(
        [
            OrderInstrument(
                symbol="SPY",
                type=InstrumentType.EQUITY,
            )
        ],
        # account_id is optional if `default_account_number` is set
        # account_id=account_id,
    )
    print(f"Quotes for SPY: {quotes}")


def example_preflight_order(cli: PublicApiClient):
    print("Performing preflight calculation...")
    preflight_request = PreflightRequest(
        instrument=OrderInstrument(
            symbol="SPY",
            type=InstrumentType.EQUITY,
        ),
        order_side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        expiration=OrderExpirationRequest(
            time_in_force=TimeInForce.DAY,
        ),
        amount=Decimal("8.80"),
    )
    preflight_response = public_api_client.perform_preflight_calculation(
        preflight_request,  # using default account
    )
    print(f"Preflight response: {preflight_response}\n\n")


def example_place_order(cli: PublicApiClient):
    order_request = OrderRequest(
        order_id=str(uuid.uuid4()),
        instrument=OrderInstrument(
            symbol="SPY",
            type=InstrumentType.EQUITY,
        ),
        order_side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        expiration=OrderExpirationRequest(
            time_in_force=TimeInForce.DAY,
        ),
        amount=Decimal("8.80"),
    )

    # Require confirmation from user before placing real order
    print(f"Order request: {order_request}")
    user_input = input("Do you want to place the order? (yes/no): ")
    if user_input.lower() != "yes":
        print("Order placement cancelled.")
        return

    print("Placing order...")

    new_order = public_api_client.place_order(
        order_request,  # using default account
    )
    print(f"Order response: {new_order}\n\n")

    # Get order status and details
    # According to the docs, after placing an order,
    # it may take some time for the order to be indexed / registered due to eventual consistency.
    # https://public.com/api/docs/resources/order-placement/get-order
    # Wait for 2s before getting the status/details
    print("Waiting for 2 seconds before checking order status...")
    time.sleep(2)
    order_status = new_order.get_status()
    print(f"Order status: {order_status}")
    order_details = new_order.get_details()
    print(f"Order details: {order_details}")

    # Wait for 3s and get the status again
    print("Waiting for 3 seconds before checking status again...")
    time.sleep(3)
    order_status = new_order.get_status()
    print(f"Order status after 3 seconds: {order_status}")

    # Cancel the order if it's still open
    if order_status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
        print("Cancelling the order...")
        new_order.cancel()
        # Cancel is async, wait for cancellation to be confirmed
        try:
            new_order.wait_for_status(
                target_status=OrderStatus.CANCELLED,
                timeout=10,
            )
            print("Order cancelled.")
        except WaitTimeoutError:
            print("Timeout waiting for order to be cancelled.")
            print("Please fix manually if needed.")


if __name__ == "__main__":
    api_secret_key = os.getenv("API_SECRET_KEY")
    if not api_secret_key:
        raise ValueError("API_SECRET_KEY not set in environment variables")
    default_account_number = os.getenv("DEFAULT_ACCOUNT_NUMBER")
    if not default_account_number:
        raise ValueError("DEFAULT_ACCOUNT_NUMBER not set in environment variables")
    public_api_client = PublicApiClient(
        ApiKeyAuthConfig(api_secret_key=api_secret_key),
        config=PublicApiClientConfiguration(
            default_account_number=default_account_number,
        ),
    )

    example_accounts(public_api_client)
    example_instruments(public_api_client)
    example_market_data(public_api_client)

    example_preflight_order(public_api_client)
    example_place_order(public_api_client)
