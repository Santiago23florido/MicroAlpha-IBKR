from __future__ import annotations

from ibapi.order import Order


def _round_price(price: float, tick_size: float = 0.01) -> float:
    rounded_ticks = round(price / tick_size)
    return round(rounded_ticks * tick_size, 2)


def _build_order_base(
    action: str,
    quantity: int,
    *,
    transmit: bool = True,
) -> Order:
    order = Order()
    order.action = action.upper()
    order.totalQuantity = quantity
    order.tif = "DAY"
    order.transmit = transmit
    order.outsideRth = False
    return order


def create_market_order(action: str, quantity: int, *, transmit: bool = True) -> Order:
    order = _build_order_base(action, quantity, transmit=transmit)
    order.orderType = "MKT"
    return order


def create_limit_order(
    action: str,
    quantity: int,
    limit_price: float,
    *,
    transmit: bool = True,
) -> Order:
    order = _build_order_base(action, quantity, transmit=transmit)
    order.orderType = "LMT"
    order.lmtPrice = float(limit_price)
    return order


def create_stop_order(
    action: str,
    quantity: int,
    stop_price: float,
    *,
    transmit: bool = True,
) -> Order:
    order = _build_order_base(action, quantity, transmit=transmit)
    order.orderType = "STP"
    order.auxPrice = float(stop_price)
    return order


def create_stop_limit_order(
    action: str,
    quantity: int,
    stop_price: float,
    limit_price: float,
    *,
    transmit: bool = True,
) -> Order:
    order = _build_order_base(action, quantity, transmit=transmit)
    order.orderType = "STP LMT"
    order.auxPrice = float(stop_price)
    order.lmtPrice = float(limit_price)
    return order


def calculate_marketable_limit_price(
    *,
    action: str,
    bid: float | None,
    ask: float | None,
    last: float | None = None,
    buffer_bps: float = 2.0,
    tick_size: float = 0.01,
) -> float:
    side = action.upper()
    anchor_price = ask if side == "BUY" else bid
    if anchor_price is None:
        anchor_price = last
    if anchor_price is None or anchor_price <= 0:
        raise ValueError("A valid bid/ask or last price is required for a marketable limit order.")
    buffer_multiplier = 1.0 + (buffer_bps / 10000.0) if side == "BUY" else 1.0 - (buffer_bps / 10000.0)
    raw_price = float(anchor_price) * buffer_multiplier
    if side == "BUY":
        raw_price += tick_size
    else:
        raw_price -= tick_size
    return _round_price(raw_price, tick_size=tick_size)


def create_marketable_limit_order(
    action: str,
    quantity: int,
    *,
    bid: float | None,
    ask: float | None,
    last: float | None = None,
    buffer_bps: float = 2.0,
    tick_size: float = 0.01,
    transmit: bool = True,
) -> Order:
    limit_price = calculate_marketable_limit_price(
        action=action,
        bid=bid,
        ask=ask,
        last=last,
        buffer_bps=buffer_bps,
        tick_size=tick_size,
    )
    return create_limit_order(
        action=action,
        quantity=quantity,
        limit_price=limit_price,
        transmit=transmit,
    )


def create_bracket_order(
    parent_order_id: int,
    action: str,
    quantity: int,
    entry_limit_price: float,
    take_profit_price: float,
    stop_loss_price: float,
) -> list[Order]:
    parent = create_limit_order(
        action=action,
        quantity=quantity,
        limit_price=entry_limit_price,
        transmit=False,
    )
    parent.orderId = parent_order_id

    exit_action = "SELL" if action.upper() == "BUY" else "BUY"

    take_profit = create_limit_order(
        action=exit_action,
        quantity=quantity,
        limit_price=take_profit_price,
        transmit=False,
    )
    take_profit.orderId = parent_order_id + 1
    take_profit.parentId = parent_order_id

    stop_loss = _build_order_base(exit_action, quantity, transmit=True)
    stop_loss.orderId = parent_order_id + 2
    stop_loss.parentId = parent_order_id
    stop_loss.orderType = "STP"
    stop_loss.auxPrice = float(stop_loss_price)

    return [parent, take_profit, stop_loss]
