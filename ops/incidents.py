from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from evaluation.io import read_jsonl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class IncidentRecord:
    severity: str
    root_component: str
    category: str
    message: str
    session_id: str | None = None
    recovery_attempted: bool = False
    recovery_status: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    incident_id: str = field(default_factory=lambda: f"inc_{uuid4().hex[:12]}")
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "root_component": self.root_component,
            "category": self.category,
            "timestamp": self.timestamp,
            "message": self.message,
            "session_id": self.session_id,
            "recovery_attempted": self.recovery_attempted,
            "recovery_status": self.recovery_status,
            "context": dict(self.context),
        }


class IncidentStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, incident: IncidentRecord | dict[str, Any]) -> dict[str, Any]:
        payload = incident.to_dict() if isinstance(incident, IncidentRecord) else dict(incident)
        payload.setdefault("incident_id", f"inc_{uuid4().hex[:12]}")
        payload.setdefault("timestamp", utc_now_iso())
        payload.setdefault("context", {})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str))
            handle.write("\n")
        return payload

    def emit_many(self, incidents: Iterable[IncidentRecord | dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.emit(incident) for incident in incidents]

    def list_incidents(
        self,
        *,
        session_id: str | None = None,
        severity: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        incidents = read_jsonl(self.path)
        if session_id is not None:
            incidents = [item for item in incidents if str(item.get("session_id") or "") == str(session_id)]
        if severity is not None:
            incidents = [item for item in incidents if str(item.get("severity") or "").lower() == severity.lower()]
        incidents = sorted(incidents, key=lambda item: str(item.get("timestamp") or ""), reverse=True)
        if limit is not None and limit > 0:
            incidents = incidents[:limit]
        return incidents

    def summarize(self, *, session_id: str | None = None) -> dict[str, Any]:
        incidents = self.list_incidents(session_id=session_id)
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        unresolved = 0
        for incident in incidents:
            severity = str(incident.get("severity") or "unknown").lower()
            category = str(incident.get("category") or "unknown").lower()
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
            if str(incident.get("recovery_status") or "").lower() not in {"recovered", "manual_followup_complete"}:
                unresolved += 1
        return {
            "total": len(incidents),
            "unresolved": unresolved,
            "by_severity": by_severity,
            "by_category": by_category,
        }

    def write_session_csv(self, session_id: str, destination: str | Path) -> str:
        rows = list(reversed(self.list_incidents(session_id=session_id)))
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "incident_id",
            "severity",
            "root_component",
            "category",
            "timestamp",
            "message",
            "session_id",
            "recovery_attempted",
            "recovery_status",
            "context",
        ]
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "context": json.dumps(row.get("context", {}), sort_keys=True, default=str),
                    }
                )
        return str(target)


def build_incident(
    *,
    severity: str,
    root_component: str,
    category: str,
    message: str,
    session_id: str | None = None,
    recovery_attempted: bool = False,
    recovery_status: str | None = None,
    context: dict[str, Any] | None = None,
) -> IncidentRecord:
    return IncidentRecord(
        severity=severity,
        root_component=root_component,
        category=category,
        message=message,
        session_id=session_id,
        recovery_attempted=recovery_attempted,
        recovery_status=recovery_status,
        context=dict(context or {}),
    )
