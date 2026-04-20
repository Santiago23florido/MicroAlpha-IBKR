from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from evaluation.io import read_jsonl, write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionTracker:
    def __init__(self, *, session_root: str | Path, registry_path: str | Path, archive_root: str | Path) -> None:
        self.session_root = Path(session_root)
        self.registry_path = Path(registry_path)
        self.archive_root = Path(archive_root)
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_root.mkdir(parents=True, exist_ok=True)

    def start_session(
        self,
        *,
        active_model_name: str,
        active_backend: str,
        symbols: list[str],
    ) -> dict[str, Any]:
        session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        session_dir = self.session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "session_id": session_id,
            "started_at": utc_now_iso(),
            "ended_at": None,
            "active_model_name": active_model_name,
            "active_backend": active_backend,
            "symbols": list(symbols),
            "decisions_count": 0,
            "orders_count": 0,
            "fills_count": 0,
            "pnl": 0.0,
            "alerts_count": 0,
            "reconciliation_status": "PENDING",
            "readiness_status": "PENDING",
            "final_state": "RUNNING",
        }
        self._write_session_metadata(session_id, record)
        self._append_registry_snapshot(record)
        return record

    def update_session(self, session_id: str, **updates: Any) -> dict[str, Any]:
        record = self.load_session(session_id)
        record.update(updates)
        record["session_id"] = session_id
        record["updated_at"] = utc_now_iso()
        self._write_session_metadata(session_id, record)
        self._append_registry_snapshot(record)
        return record

    def finalize_session(self, session_id: str, **updates: Any) -> dict[str, Any]:
        updates.setdefault("ended_at", utc_now_iso())
        return self.update_session(session_id, **updates)

    def load_session(self, session_id: str) -> dict[str, Any]:
        path = self.session_dir(session_id) / "session_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown session_id {session_id!r}. Missing {path}.")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_sessions(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        snapshots = read_jsonl(self.registry_path)
        by_session_id: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            session_id = str(snapshot.get("session_id") or "")
            if not session_id:
                continue
            by_session_id[session_id] = snapshot
        sessions = sorted(by_session_id.values(), key=lambda item: str(item.get("started_at") or ""), reverse=True)
        if limit is not None and limit > 0:
            sessions = sessions[:limit]
        return sessions

    def latest_session(self) -> dict[str, Any] | None:
        sessions = self.list_sessions(limit=1)
        return sessions[0] if sessions else None

    def session_dir(self, session_id: str) -> Path:
        return self.session_root / session_id

    def write_snapshot(self, session_id: str, name: str, payload: dict[str, Any]) -> str:
        return write_json(self.session_dir(session_id) / name, payload)

    def archive_session(self, session_id: str) -> str:
        source = self.session_dir(session_id)
        if not source.exists():
            raise FileNotFoundError(f"Cannot archive session {session_id!r}: missing {source}.")
        date_token = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        destination = self.archive_root / date_token / session_id
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        return str(destination)

    def _write_session_metadata(self, session_id: str, payload: dict[str, Any]) -> None:
        write_json(self.session_dir(session_id) / "session_metadata.json", payload)

    def _append_registry_snapshot(self, payload: dict[str, Any]) -> None:
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True, default=str))
            handle.write("\n")
