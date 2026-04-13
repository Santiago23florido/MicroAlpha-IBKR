from __future__ import annotations

from ibapi.contract import Contract


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def create_stock_contract(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
) -> Contract:
    contract = Contract()
    contract.symbol = normalize_symbol(symbol)
    contract.secType = "STK"
    contract.exchange = exchange.upper()
    contract.currency = currency.upper()
    return contract


def create_us_stock_contract(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
) -> Contract:
    return create_stock_contract(symbol=symbol, exchange=exchange, currency=currency)
