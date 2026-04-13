from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from data.schemas import TradeLifecycleEvent


class TradeStore:
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
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    action TEXT,
                    quantity REAL,
                    status TEXT NOT NULL,
                    order_id INTEGER,
                    parent_order_id INTEGER,
                    price REAL,
                    realized_pnl REAL,
                    message TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    symbol TEXT,
                    order_id INTEGER,
                    status TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def append_trade(self, event: TradeLifecycleEvent) -> int:
        payload = event.to_dict()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO trades (
                    timestamp,
                    symbol,
                    event_type,
                    action,
                    quantity,
                    status,
                    order_id,
                    parent_order_id,
                    price,
                    realized_pnl,
                    message,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp,
                    event.symbol,
                    event.event_type,
                    event.action,
                    event.quantity,
                    event.status,
                    event.order_id,
                    event.parent_order_id,
                    event.price,
                    event.realized_pnl,
                    event.message,
                    json.dumps(payload, sort_keys=True),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def append_execution_event(
        self,
        *,
        timestamp: str,
        event_type: str,
        payload: dict[str, Any],
        symbol: str | None = None,
        order_id: int | None = None,
        status: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO execution_events (
                    timestamp,
                    event_type,
                    symbol,
                    order_id,
                    status,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    event_type,
                    symbol,
                    order_id,
                    status,
                    json.dumps(payload, sort_keys=True, default=str),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM trades
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_recent_execution_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM execution_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_daily_trade_count(self, session_date: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM trades
                WHERE substr(timestamp, 1, 10) = ?
                  AND event_type IN ('entry_submitted', 'entry_filled', 'manual_test_order_submitted')
                """,
                (session_date,),
            ).fetchone()
        return int(row[0]) if row else 0

    def get_daily_realized_pnl(self, session_date: str) -> float:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(realized_pnl), 0)
                FROM trades
                WHERE substr(timestamp, 1, 10) = ?
                """,
                (session_date,),
            ).fetchone()
        return float(row[0]) if row else 0.0
