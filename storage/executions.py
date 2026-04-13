from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

CSV_FIELDS = [
    "timestamp_utc",
    "event_type",
    "order_id",
    "parent_order_id",
    "symbol",
    "action",
    "quantity",
    "order_type",
    "limit_price",
    "stop_price",
    "status",
    "filled_quantity",
    "average_fill_price",
    "execution_id",
    "execution_time",
    "message",
]


class ExecutionAuditStore:
    def __init__(self, csv_path: str, logger: logging.Logger) -> None:
        self.csv_path = Path(csv_path)
        self.logger = logger
        self._lock = threading.Lock()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        row["event_type"] = event_type

        for field in CSV_FIELDS:
            if field in {"timestamp_utc", "event_type"}:
                continue
            value = payload.get(field)
            if value is None:
                continue
            row[field] = str(value)

        try:
            with self._lock:
                write_header = not self.csv_path.exists()
                with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
        except OSError as exc:
            self.logger.warning("Failed to append execution audit row: %s", exc)
