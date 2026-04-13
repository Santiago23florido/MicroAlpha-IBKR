from __future__ import annotations

from broker.orders import create_bracket_order, create_limit_order, create_market_order


def test_create_market_order_defaults() -> None:
    order = create_market_order("buy", 2)

    assert order.action == "BUY"
    assert order.orderType == "MKT"
    assert order.totalQuantity == 2
    assert order.transmit is True


def test_create_limit_order_has_limit_price() -> None:
    order = create_limit_order("sell", 3, 512.25)

    assert order.action == "SELL"
    assert order.orderType == "LMT"
    assert order.totalQuantity == 3
    assert order.lmtPrice == 512.25


def test_create_bracket_order_wires_children_safely() -> None:
    orders = create_bracket_order(
        parent_order_id=100,
        action="BUY",
        quantity=1,
        entry_limit_price=100.0,
        take_profit_price=103.0,
        stop_loss_price=98.0,
    )

    parent, take_profit, stop_loss = orders

    assert parent.orderId == 100
    assert parent.orderType == "LMT"
    assert parent.transmit is False

    assert take_profit.orderId == 101
    assert take_profit.parentId == 100
    assert take_profit.orderType == "LMT"
    assert take_profit.transmit is False

    assert stop_loss.orderId == 102
    assert stop_loss.parentId == 100
    assert stop_loss.orderType == "STP"
    assert stop_loss.transmit is True
