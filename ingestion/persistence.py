from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class ParquetMarketDataSink:
    def __init__(
        self,
        root_dir: str | Path,
        logger,
        *,
        batch_size: int,
        flush_interval_seconds: float,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.logger = logger
        self.batch_size = max(int(batch_size), 1)
        self.flush_interval_seconds = max(float(flush_interval_seconds), 1.0)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._last_flush_at = datetime.now(timezone.utc)
        self._flush_count = 0
        self._persisted_records = 0

    @property
    def pending_count(self) -> int:
        return len(self._buffer)

    @property
    def flush_count(self) -> int:
        return self._flush_count

    @property
    def persisted_records(self) -> int:
        return self._persisted_records

    def append(self, record: dict[str, Any]) -> None:
        self._buffer.append(record)

    def extend(self, records: list[dict[str, Any]]) -> None:
        self._buffer.extend(records)

    def flush_if_due(self) -> dict[str, Any] | None:
        if not self._buffer:
            return None
        age_seconds = (datetime.now(timezone.utc) - self._last_flush_at).total_seconds()
        if len(self._buffer) < self.batch_size and age_seconds < self.flush_interval_seconds:
            return None
        return self.flush()

    def flush(self) -> dict[str, Any] | None:
        if not self._buffer:
            return None

        frame = pd.DataFrame(self._buffer)
        written_files: list[str] = []

        for (event_date, symbol), group in frame.groupby(
            [frame["timestamp"].str.slice(0, 10), frame["symbol"]],
            dropna=False,
        ):
            output_dir = self.root_dir / str(event_date) / str(symbol).upper()
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output_path = output_dir / f"collector_{batch_token}_{self._flush_count + 1:05d}.parquet"
            group.to_parquet(output_path, index=False)
            written_files.append(str(output_path))

        persisted = len(self._buffer)
        self._flush_count += 1
        self._persisted_records += persisted
        self._buffer.clear()
        self._last_flush_at = datetime.now(timezone.utc)

        payload = {
            "records": persisted,
            "files": written_files,
            "root_dir": str(self.root_dir),
            "flush_count": self._flush_count,
        }
        self.logger.info(
            "Collector flush completed: records=%s files=%s flush_count=%s",
            persisted,
            len(written_files),
            self._flush_count,
        )
        return payload

    def close(self) -> dict[str, Any] | None:
        return self.flush()
