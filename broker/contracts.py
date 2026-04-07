from __future__ import annotations

from ibapi.contract import Contract


def create_stock_contract(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
) -> Contract:
    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "STK"
    contract.exchange = exchange
    contract.currency = currency
    return contract
