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
class AlertRecord:
    severity: str
    category: str
    message: str
    session_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    recommended_action: str | None = None
    alert_id: str = field(default_factory=lambda: f"alert_{uuid4().hex[:12]}")
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "category": self.category,
            "timestamp": self.timestamp,
            "message": self.message,
            "context": dict(self.context),
            "session_id": self.session_id,
            "recommended_action": self.recommended_action,
        }


class AlertStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, alert: AlertRecord | dict[str, Any]) -> dict[str, Any]:
        payload = alert.to_dict() if isinstance(alert, AlertRecord) else dict(alert)
        payload.setdefault("alert_id", f"alert_{uuid4().hex[:12]}")
        payload.setdefault("timestamp", utc_now_iso())
        payload.setdefault("context", {})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str))
            handle.write("\n")
        return payload

    def emit_many(self, alerts: Iterable[AlertRecord | dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.emit(alert) for alert in alerts]

    def list_alerts(
        self,
        *,
        session_id: str | None = None,
        severity: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        alerts = read_jsonl(self.path)
        if session_id is not None:
            alerts = [item for item in alerts if str(item.get("session_id") or "") == str(session_id)]
        if severity is not None:
            alerts = [item for item in alerts if str(item.get("severity") or "").lower() == severity.lower()]
        alerts = sorted(alerts, key=lambda item: str(item.get("timestamp") or ""), reverse=True)
        if limit is not None and limit > 0:
            alerts = alerts[:limit]
        return alerts

    def summarize(self, *, session_id: str | None = None) -> dict[str, Any]:
        alerts = self.list_alerts(session_id=session_id)
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for alert in alerts:
            severity = str(alert.get("severity") or "unknown").lower()
            category = str(alert.get("category") or "unknown").lower()
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
        return {
            "total": len(alerts),
            "by_severity": by_severity,
            "by_category": by_category,
        }

    def write_session_csv(self, session_id: str, destination: str | Path) -> str:
        rows = list(reversed(self.list_alerts(session_id=session_id)))
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "alert_id",
            "severity",
            "category",
            "timestamp",
            "message",
            "session_id",
            "recommended_action",
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


def build_alert(
    *,
    severity: str,
    category: str,
    message: str,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
    recommended_action: str | None = None,
) -> AlertRecord:
    return AlertRecord(
        severity=severity,
        category=category,
        message=message,
        session_id=session_id,
        context=dict(context or {}),
        recommended_action=recommended_action,
    )
