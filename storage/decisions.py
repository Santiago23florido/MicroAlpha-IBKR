from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from data.schemas import DecisionRecord


class DecisionStore:
    def __init__(self, db_path: str, logger: logging.Logger) -> None:
        self.db_path = Path(db_path)
        self.logger = logger
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    final_action TEXT NOT NULL,
                    direction TEXT,
                    explanation_text TEXT NOT NULL,
                    expected_edge REAL,
                    estimated_cost REAL,
                    risk_passed INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS config_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save_decision(self, decision: DecisionRecord) -> int:
        payload = decision.to_dict()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO decisions (
                    timestamp,
                    symbol,
                    final_action,
                    direction,
                    explanation_text,
                    expected_edge,
                    estimated_cost,
                    risk_passed,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.timestamp,
                    decision.symbol,
                    decision.final_action,
                    decision.direction,
                    decision.explanation_text,
                    decision.expected_edge,
                    decision.estimated_cost,
                    int(decision.risk_passed),
                    json.dumps(payload, sort_keys=True),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def save_config_snapshot(self, payload: Mapping[str, Any]) -> int:
        serialized = json.dumps(dict(payload), sort_keys=True, default=str)
        created_at = str(payload.get("created_at") or payload.get("timestamp") or "")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO config_snapshots (created_at, payload_json)
                VALUES (?, ?)
                """,
                (created_at, serialized),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_latest_decision(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM decisions
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list_recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM decisions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
