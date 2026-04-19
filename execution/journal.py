from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from config.phase7 import ExecutionLoggingConfig
from execution.models import ExecutionReport, FillEvent, Order, PortfolioSnapshot, PositionState


class ExecutionJournal:
    def __init__(self, config: ExecutionLoggingConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self.journal_dir = Path(config.journal_dir)
        self.state_path = Path(config.state_path)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.orders_path = self.journal_dir / "orders.jsonl"
        self.fills_path = self.journal_dir / "fills.jsonl"
        self.reports_path = self.journal_dir / "reports.jsonl"
        self.positions_path = self.journal_dir / "positions.jsonl"
        self.pnl_path = self.journal_dir / "pnl.jsonl"
        self.backend_events_path = self.journal_dir / "backend_events.jsonl"
        self.reconciliation_path = self.journal_dir / "reconciliation.jsonl"

    def append_order(self, order: Order) -> None:
        self._append_jsonl(self.orders_path, order.to_dict())

    def append_fill(self, fill: FillEvent) -> None:
        self._append_jsonl(self.fills_path, fill.to_dict())

    def append_report(self, report: ExecutionReport) -> None:
        self._append_jsonl(self.reports_path, report.to_dict())

    def append_position(self, position: PositionState, **context: Any) -> None:
        payload = {"position": position.to_dict(), **context}
        self._append_jsonl(self.positions_path, payload)

    def append_pnl(self, portfolio: PortfolioSnapshot, **context: Any) -> None:
        payload = {"portfolio": portfolio.to_dict(), **context}
        self._append_jsonl(self.pnl_path, payload)

    def append_backend_event(self, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(self.backend_events_path, dict(payload))

    def append_reconciliation(self, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(self.reconciliation_path, dict(payload))

    def save_state(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self.state_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def recent_orders(self, limit: int = 5) -> list[dict[str, Any]]:
        return self._read_recent_jsonl(self.orders_path, limit)

    def recent_fills(self, limit: int = 5) -> list[dict[str, Any]]:
        return self._read_recent_jsonl(self.fills_path, limit)

    def recent_reports(self, limit: int = 5) -> list[dict[str, Any]]:
        return self._read_recent_jsonl(self.reports_path, limit)

    def recent_backend_events(self, limit: int = 5) -> list[dict[str, Any]]:
        return self._read_recent_jsonl(self.backend_events_path, limit)

    def recent_reconciliation(self, limit: int = 5) -> list[dict[str, Any]]:
        return self._read_recent_jsonl(self.reconciliation_path, limit)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True, default=str))
            handle.write("\n")

    @staticmethod
    def _read_recent_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
        if limit <= 0 or not path.exists():
            return []
        lines = deque(maxlen=limit)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    lines.append(line)
        return [json.loads(line) for line in reversed(lines)]
