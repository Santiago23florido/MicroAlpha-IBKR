from __future__ import annotations

from typing import Iterable

from execution.models import FillEvent, Order, OrderStatus


class OrderStateTransitionError(ValueError):
    """Raised when an invalid order transition is attempted."""


class OrderStateMachine:
    _ALLOWED_TRANSITIONS: dict[OrderStatus, tuple[OrderStatus, ...]] = {
        OrderStatus.CREATED: (
            OrderStatus.SUBMITTED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
            OrderStatus.CANCELLED,
        ),
        OrderStatus.SUBMITTED: (
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        ),
        OrderStatus.ACKNOWLEDGED: (
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        ),
        OrderStatus.PARTIALLY_FILLED: (
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        ),
        OrderStatus.REJECTED: (),
        OrderStatus.FILLED: (),
        OrderStatus.CANCELLED: (),
        OrderStatus.EXPIRED: (),
        OrderStatus.FAILED: (),
    }

    def allowed_transitions(self, status: OrderStatus) -> tuple[OrderStatus, ...]:
        return self._ALLOWED_TRANSITIONS[status]

    def ensure_transition(self, current: OrderStatus, target: OrderStatus) -> None:
        if current == target:
            return
        if target not in self.allowed_transitions(current):
            allowed = ", ".join(item.value for item in self.allowed_transitions(current))
            raise OrderStateTransitionError(
                f"Invalid order transition {current.value} -> {target.value}. Allowed: [{allowed}]"
            )

    def transition(self, order: Order, target: OrderStatus, *, updated_at: str) -> Order:
        self.ensure_transition(order.status, target)
        return order.replace(status=target, updated_at=updated_at)

    def apply_fill(self, order: Order, fill: FillEvent) -> Order:
        total_filled = order.filled_quantity + int(fill.quantity)
        if total_filled > order.quantity:
            raise OrderStateTransitionError(
                f"Fill quantity would overfill order {order.order_id}: {total_filled}>{order.quantity}."
            )

        average_fill_price = fill.fill_price
        if order.average_fill_price is not None and order.filled_quantity > 0:
            weighted_notional = (order.average_fill_price * order.filled_quantity) + (fill.fill_price * fill.quantity)
            average_fill_price = weighted_notional / total_filled

        target = OrderStatus.FILLED if total_filled == order.quantity else OrderStatus.PARTIALLY_FILLED
        if order.status == OrderStatus.ACKNOWLEDGED:
            self.ensure_transition(order.status, target)
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            self.ensure_transition(order.status, target)
        else:
            raise OrderStateTransitionError(
                f"Cannot apply fill to order {order.order_id} while in status {order.status.value}."
            )

        return order.replace(
            status=target,
            filled_quantity=total_filled,
            average_fill_price=float(average_fill_price),
            updated_at=fill.filled_at,
        )

    def validate_path(self, statuses: Iterable[OrderStatus]) -> None:
        ordered = list(statuses)
        for current, target in zip(ordered, ordered[1:]):
            self.ensure_transition(current, target)
