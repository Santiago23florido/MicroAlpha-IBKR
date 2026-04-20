from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Settings
from config.phase10_11 import load_phase10_11_config
from evaluation.io import read_json, write_json
from monitoring.alerts import AlertStore, build_alert
from ops.incidents import IncidentStore, build_incident
from ops.recovery import safe_restart_assessment
from validation.session_tracker import SessionTracker


def postflight_check(settings: Settings, *, session_id: str) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    tracker = SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )
    alert_store = AlertStore(phase10_11.report_paths.alerts_path)
    incident_store = IncidentStore(phase10_11.report_paths.incidents_path)
    session_dir = tracker.session_dir(session_id)

    checks: dict[str, dict[str, Any]] = {}
    alerts: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []

    def record(name: str, passed: bool, message: str, *, severity: str = "critical", context: dict[str, Any] | None = None) -> None:
        checks[name] = {"passed": passed, "message": message, "severity": severity, "context": dict(context or {})}
        if not passed:
            alerts.append(
                build_alert(
                    severity=severity,
                    category="recovery_event" if name == "archive_completed" else "scheduler_failure",
                    message=message,
                    session_id=session_id,
                    context={"check": name, **dict(context or {})},
                    recommended_action="Review postflight outputs and resolve the missing artifact or unsafe restart condition.",
                ).to_dict()
            )
            if severity == "critical":
                incidents.append(
                    build_incident(
                        severity=severity,
                        root_component="postflight",
                        category="reconciliation_failure" if name == "reconciliation_completed" else "scheduler_failure",
                        message=message,
                        session_id=session_id,
                        context={"check": name, **dict(context or {})},
                    ).to_dict()
                )

    required_files = {
        "session_summary.json": session_dir / "session_summary.json",
        "system_health.json": session_dir / "system_health.json",
        "alerts_summary.csv": session_dir / "alerts_summary.csv",
        "reconciliation_summary.json": session_dir / "reconciliation_summary.json",
    }
    reports_created = all(path.exists() for path in required_files.values())
    record(
        "reports_created",
        reports_created,
        "All required session reports were created." if reports_created else "One or more required session reports are missing.",
        context={name: str(path) for name, path in required_files.items()},
    )

    reconciliation_status = None
    reconciliation_path = session_dir / "reconciliation_summary.json"
    if reconciliation_path.exists():
        reconciliation_status = read_json(reconciliation_path).get("status")
    reconciliation_completed = reconciliation_status in {"MATCH", "MISMATCH", "CRITICAL_MISMATCH"}
    record(
        "reconciliation_completed",
        reconciliation_completed,
        "Reconciliation summary is available." if reconciliation_completed else "Reconciliation summary is missing or invalid.",
        context={"reconciliation_status": reconciliation_status},
    )

    alert_summary = alert_store.summarize(session_id=session_id)
    record(
        "alerts_summarized",
        True,
        "Alert summary was generated.",
        severity="info",
        context=alert_summary,
    )

    archive_completed = True
    archived_path = None
    if phase10_11.archival_policy.enabled and phase10_11.archival_policy.archive_completed_sessions:
        archived_path = tracker.archive_session(session_id)
    else:
        archive_completed = False
    record(
        "archive_completed",
        archive_completed,
        "Session artifacts were archived." if archive_completed else "Archival policy is disabled or archive was skipped.",
        severity="warning" if not archive_completed else "info",
        context={"archived_path": archived_path},
    )

    restart_assessment = safe_restart_assessment(settings)
    record(
        "safe_restart_assessment",
        bool(restart_assessment.get("safe_to_resume")),
        "Safe restart assessment passed." if restart_assessment.get("safe_to_resume") else "Safe restart assessment failed.",
        severity="warning",
        context=restart_assessment,
    )

    alert_store.emit_many(alerts)
    incident_store.emit_many(incidents)
    required = set(phase10_11.checks.postflight_required_checks)
    blocking_failures = [name for name, payload in checks.items() if name in required and not payload["passed"]]
    payload = {
        "status": "ok" if not blocking_failures else "error",
        "checks": checks,
        "required_checks": list(phase10_11.checks.postflight_required_checks),
        "blocking_failures": blocking_failures,
        "alerts": alerts,
        "incidents": incidents,
        "archive_path": archived_path,
        "restart_assessment": restart_assessment,
    }
    write_json(session_dir / "postflight_check.json", payload)
    return payload
