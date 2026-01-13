import datetime
import os
from ibind import IbkrClient, StockQuery, ibind_logs_initialize
from ibind.client.ibkr_utils import OrderRequest
from pprint import pprint
from unittest.mock import patch, MagicMock

ibind_logs_initialize(log_to_file=False)


def example_accounts(client: IbkrClient):
    print("\n#### get_accounts ####")
    accounts = client.portfolio_accounts().data
    client.account_id = accounts[0]["accountId"]
    print(accounts)

    print("\n\n#### get_ledger ####")
    ledger = client.get_ledger().data
    for currency, subledger in ledger.items():
        print(f"\t Ledger currency: {currency}")
        print(f'\t cash balance: {subledger["cashbalance"]}')
        print(f'\t net liquidation value: {subledger["netliquidationvalue"]}')
        print(f'\t stock market value: {subledger["stockmarketvalue"]}')
        print()

    print("\n#### get_positions ####")
    positions = client.positions().data
    for position in positions:
        print(
            f'\t Position {position["ticker"]}: {position["position"]} (${position["mktValue"]})'
        )


def example_query_stocks(client: IbkrClient):
    print("#### get_stocks ####")
    stocks = client.security_stocks_by_symbol("AAPL").data
    print(stocks)

    print("\n#### get_conids ####")
    conids = client.stock_conid_by_symbol("AAPL").data
    print(conids)

    print("\n#### using StockQuery ####")
    conids = client.stock_conid_by_symbol(
        StockQuery("AAPL", contract_conditions={"exchange": "MEXI"}),
        default_filtering=False,
    ).data
    pprint(conids)

    print("\n#### mixed queries ####")
    stock_queries = [
        StockQuery("AAPL", contract_conditions={"exchange": "MEXI"}),
        "HUBS",
        StockQuery("GOOG", name_match="ALPHABET INC - CDR"),
    ]
    conids = client.stock_conid_by_symbol(stock_queries, default_filtering=False).data
    pprint(conids)


def example_query_market_data(client: IbkrClient):
    history = client.marketdata_history_by_symbols(
        "AAPL", period="1d", bar="1d", outside_rth=True
    )
    print("\n\n#### One symbol ####")
    print(f"{history}")

    history = client.marketdata_history_by_symbols(
        ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN"],
        period="1d",
        bar="1d",
        outside_rth=True,
        run_in_parallel=True,
    )
    print("\n\n#### Five symbols parallel ####")
    print(f"{history}")


def example_trading_schedules(client: IbkrClient):
    schedules = client.trading_schedule_by_symbol(
        asset_class="STK",
        symbol="SPY",
    ).data
    print("\n\n#### Trading schedules ####")
    pprint(schedules)


def example_orders(client: IbkrClient, mock: bool = True):
    account_id = os.getenv('IBIND_ACCOUNT_ID', '[YOUR_ACCOUNT_ID]')
    if account_id == '[YOUR_ACCOUNT_ID]':
        raise ValueError("Please set your IBIND_ACCOUNT_ID environment variable.")

    conid = 265598
    side = 'BUY'
    size = 1
    order_type = 'MKT'
    order_tag = f'my_order-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    order_request = OrderRequest(conid=conid, side=side, quantity=size, order_type=order_type, acct_id=account_id, coid=order_tag)


if __name__ == "__main__":
    client = IbkrClient()  # initializes OAuth from environment variables

    example_accounts(client)
    example_query_stocks(client)
    # example_query_market_data(client)
    example_trading_schedules(client)
