from __future__ import annotations

from execution.models import OrderStatus


_ACKNOWLEDGED_STATUSES = {"PendingSubmit", "ApiPending", "PreSubmitted", "Submitted", "PendingCancel"}
_CANCELLED_STATUSES = {"Cancelled", "ApiCancelled"}
_REJECTED_STATUSES = {"Rejected"}
_EXPIRED_STATUSES = {"Expired"}


def map_ibkr_status(
    broker_status: str | None,
    *,
    filled_quantity: float | int | None = None,
    remaining_quantity: float | int | None = None,
    message: str | None = None,
) -> OrderStatus:
    normalized = str(broker_status or "").strip()
    filled = float(filled_quantity or 0.0)
    remaining = float(remaining_quantity or 0.0)
    lower_message = str(message or "").lower()

    if normalized == "Filled":
        return OrderStatus.FILLED
    if normalized in _CANCELLED_STATUSES:
        return OrderStatus.CANCELLED
    if normalized in _REJECTED_STATUSES:
        return OrderStatus.REJECTED
    if normalized in _EXPIRED_STATUSES:
        return OrderStatus.EXPIRED
    if normalized == "Inactive":
        if "reject" in lower_message or "error" in lower_message or "invalid" in lower_message:
            return OrderStatus.REJECTED
        return OrderStatus.FAILED
    if filled > 0.0 and remaining > 0.0:
        return OrderStatus.PARTIALLY_FILLED
    if normalized in _ACKNOWLEDGED_STATUSES or normalized:
        return OrderStatus.ACKNOWLEDGED
    return OrderStatus.SUBMITTED

