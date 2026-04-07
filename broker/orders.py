from __future__ import annotations

from ibapi.order import Order


def create_market_order(action: str, quantity: int) -> Order:
    order = Order()
    order.action = action.upper()
    order.orderType = "MKT"
    order.totalQuantity = quantity
    order.tif = "DAY"
    return order
