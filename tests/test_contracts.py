from __future__ import annotations

from broker.contracts import create_stock_contract, create_us_stock_contract


def test_create_stock_contract_defaults() -> None:
    contract = create_stock_contract("spy")

    assert contract.symbol == "SPY"
    assert contract.secType == "STK"
    assert contract.exchange == "SMART"
    assert contract.currency == "USD"


def test_create_us_stock_contract_accepts_custom_values() -> None:
    contract = create_us_stock_contract("aapl", exchange="ISLAND", currency="USD")

    assert contract.symbol == "AAPL"
    assert contract.exchange == "ISLAND"
    assert contract.currency == "USD"
