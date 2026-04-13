from __future__ import annotations

from typing import Any


def build_performance_report(trades: list[dict[str, Any]]) -> dict[str, Any]:
    realized_pnl = sum(float(trade.get("realized_pnl") or 0.0) for trade in trades)
    return {
        "realized_pnl": realized_pnl,
        "trade_count": len(trades),
    }
