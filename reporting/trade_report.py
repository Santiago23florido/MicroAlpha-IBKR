from __future__ import annotations

from typing import Any


def build_trade_report(trades: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(trades)
    by_event: dict[str, int] = {}
    for trade in trades:
        event_type = str(trade.get("event_type", "unknown"))
        by_event[event_type] = by_event.get(event_type, 0) + 1
    return {"total_trades": total, "events": by_event}
