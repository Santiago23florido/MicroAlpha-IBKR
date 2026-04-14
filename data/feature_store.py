from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

from data.schemas import FeatureSnapshot


class FeatureStore:
    def __init__(
        self,
        db_path: str,
        logger: logging.Logger,
        sequence_length: int = 16,
    ) -> None:
        self.db_path = Path(db_path)
        self.logger = logger
        self.sequence_length = sequence_length
        self._buffers: dict[str, deque[FeatureSnapshot]] = defaultdict(
            lambda: deque(maxlen=self.sequence_length * 8)
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    estimated_cost_bps REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def append(self, snapshot: FeatureSnapshot) -> None:
        self._buffers[snapshot.symbol].append(snapshot)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feature_snapshots (
                    timestamp,
                    symbol,
                    estimated_cost_bps,
                    payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    snapshot.timestamp,
                    snapshot.symbol,
                    snapshot.estimated_cost_bps,
                    json.dumps(snapshot.to_dict(), sort_keys=True),
                ),
            )
            connection.commit()

    def get_recent_sequence(self, symbol: str, limit: int | None = None) -> list[FeatureSnapshot]:
        normalized = symbol.upper()
        buffer = self._buffers.get(normalized)
        if buffer:
            rows = list(buffer)
        else:
            rows = self._load_recent_from_db(normalized, limit or self.sequence_length)
        if limit is None:
            return rows
        return rows[-limit:]

    def list_recent(self, limit: int = 50) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM feature_snapshots
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _load_recent_from_db(self, symbol: str, limit: int) -> list[FeatureSnapshot]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM feature_snapshots
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        restored = []
        for row in reversed(rows):
            payload = json.loads(row[0])
            restored.append(
                FeatureSnapshot(
                    symbol=payload["symbol"],
                    timestamp=payload["timestamp"],
                    feature_values=payload["feature_values"],
                    estimated_cost_bps=payload["estimated_cost_bps"],
                    missing_features=payload.get("missing_features", []),
                    source_mode=payload.get("source_mode", "paper_or_local"),
                )
            )
        return restored
