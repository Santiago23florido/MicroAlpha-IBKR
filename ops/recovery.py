from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import Settings
from config.phase10_11 import load_phase10_11_config
from engine.phase7 import broker_healthcheck, execution_status
from evaluation.io import read_jsonl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecoveryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "recovery_event_id": f"rec_{uuid4().hex[:12]}",
            "timestamp": utc_now_iso(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, sort_keys=True, default=str))
            handle.write("\n")
        return normalized

    def list_events(self, *, session_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        rows = read_jsonl(self.path)
        if session_id is not None:
            rows = [row for row in rows if str(row.get("session_id") or "") == str(session_id)]
        rows = sorted(rows, key=lambda row: str(row.get("timestamp") or ""), reverse=True)
        if limit is not None and limit > 0:
            rows = rows[:limit]
        return rows


def attempt_safe_recovery(
    settings: Settings,
    *,
    category: str,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    store = RecoveryStore(phase10_11.report_paths.recovery_events_path)
    context = dict(context or {})

    attempted = False
    status = "not_attempted"
    message = "No automatic recovery rule was defined for this category."
    details: dict[str, Any] = {}

    if category == "broker_connectivity_failure":
        attempted = True
        try:
            details = broker_healthcheck(settings)
            status = "recovered" if details.get("status") == "ok" else "failed"
            message = (
                "Broker connectivity recovered during conservative recovery attempt."
                if status == "recovered"
                else "Broker connectivity recovery attempt did not restore health."
            )
        except Exception as exc:  # pragma: no cover - depends on broker availability
            status = "failed"
            message = str(exc)
            details = {"status": "error", "message": str(exc)}
    elif category in {"scheduler_failure", "monitor_restart"}:
        attempted = False
        status = "manual_review_required"
        message = "Scheduler and monitor failures require manual review before restart."
    elif category == "duplicate_submission_risk":
        attempted = False
        status = "aborted"
        message = "Automatic resume was aborted to avoid duplicate order submission."

    return store.append(
        {
            "session_id": session_id,
            "category": category,
            "attempted": attempted,
            "status": status,
            "message": message,
            "context": context,
            "details": details,
        }
    )


def safe_restart_assessment(settings: Settings) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    status = execution_status(settings, limit=10)
    open_orders = list(status.get("open_orders", []) or [])
    broker_open_orders = list(status.get("broker_open_orders", []) or [])
    broker_error = status.get("broker_error")
    duplicate_submission_risk = bool(open_orders or broker_open_orders)
    safe_to_resume = not duplicate_submission_risk and not broker_error
    if phase10_11.recovery.abort_on_duplicate_submission_risk and duplicate_submission_risk:
        safe_to_resume = False
    return {
        "status": "ok",
        "safe_to_resume": safe_to_resume,
        "duplicate_submission_risk": duplicate_submission_risk,
        "broker_error": broker_error,
        "open_order_count": len(open_orders),
        "broker_open_order_count": len(broker_open_orders),
    }
