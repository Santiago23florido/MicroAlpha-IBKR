from __future__ import annotations

from storage.trades import TradeStore


class ExecutionTracker:
    def __init__(self, trade_store: TradeStore) -> None:
        self.trade_store = trade_store

    def recent_execution_events(self, limit: int = 50) -> list[dict[str, object]]:
        return self.trade_store.list_recent_execution_events(limit=limit)

    def recent_trades(self, limit: int = 50) -> list[dict[str, object]]:
        return self.trade_store.list_recent_trades(limit=limit)
