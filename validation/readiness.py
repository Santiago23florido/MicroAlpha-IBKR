from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Settings
from config.phase10_11 import load_phase10_11_config
from engine.phase6 import risk_check
from engine.phase7 import show_execution_backend
from evaluation.io import read_json, write_json
from monitoring.alerts import AlertStore
from ops.incidents import IncidentStore
from validation.session_tracker import SessionTracker


def generate_readiness_report(
    settings: Settings,
    *,
    session_id: str | None = None,
    session_summary: dict[str, Any] | None = None,
    monitor_report: dict[str, Any] | None = None,
    reconciliation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    tracker = SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )
    alert_store = AlertStore(phase10_11.report_paths.alerts_path)
    incident_store = IncidentStore(phase10_11.report_paths.incidents_path)
    latest_session = tracker.latest_session()
    session_id = session_id or (latest_session or {}).get("session_id")
    if session_id is None:
        raise FileNotFoundError("No paper validation session is available for readiness evaluation.")
    session_dir = tracker.session_dir(session_id)

    session_summary = session_summary or _read_optional_json(session_dir / "session_summary.json")
    monitor_report = monitor_report or _read_optional_json(session_dir / "system_health.json")
    reconciliation_report = reconciliation_report or _read_optional_json(session_dir / "reconciliation_summary.json")
    sessions = tracker.list_sessions(limit=phase10_11.validation_session_limits.max_sessions_to_compare)
    completed_sessions = [row for row in sessions if str(row.get("final_state") or "").startswith("COMPLETED")]
    failure_count = len([row for row in sessions if str(row.get("final_state") or "").startswith("FAILED") or str(row.get("final_state") or "") == "PRECHECK_FAILED"])
    session_failure_rate = float(failure_count / len(sessions)) if sessions else 0.0

    backend_status = show_execution_backend(settings)
    model_status = risk_check(settings)
    alert_summary = alert_store.summarize(session_id=session_id)
    incident_summary = incident_store.summarize(session_id=session_id)
    reconciliation_summary = dict((reconciliation_report or {}).get("summary", reconciliation_report or {}) or {})
    monitor_snapshot = dict((monitor_report or {}).get("latest_snapshot", monitor_report or {}) or {})
    drift_summary = dict((monitor_snapshot or {}).get("drift_summary", {}) or {})
    execution_summary = dict((monitor_snapshot or {}).get("execution_summary", {}) or {})

    criteria = {
        "broker_connectivity_stable": _criterion(
            bool(((monitor_snapshot.get("broker_status") or {}).get("status") == "ok")),
            "Broker connectivity is stable.",
            "Broker connectivity is not stable.",
        ),
        "no_critical_reconciliation_mismatches": _criterion(
            int(reconciliation_summary.get("critical_order_mismatches", 0) or 0)
            + int(reconciliation_summary.get("critical_fill_mismatches", 0) or 0)
            + int(reconciliation_summary.get("critical_position_mismatches", 0) or 0)
            <= phase10_11.readiness_thresholds.max_allowed_mismatches,
            "No critical reconciliation mismatches were detected.",
            "Critical reconciliation mismatches exceed the allowed threshold.",
        ),
        "alert_rate_within_threshold": _criterion(
            int(alert_summary.get("total", 0) or 0) <= phase10_11.readiness_thresholds.max_allowed_alerts,
            "Alert count is within threshold.",
            "Alert count exceeds the readiness threshold.",
        ),
        "session_failure_rate_acceptable": _criterion(
            session_failure_rate <= phase10_11.readiness_thresholds.max_session_failure_rate,
            "Session failure rate is acceptable.",
            "Session failure rate exceeds the readiness threshold.",
        ),
        "no_live_path_enabled": _criterion(
            str(settings.broker_mode).strip().lower() == "paper"
            and str(backend_status["ibkr_paper_config"]["broker_mode"]).strip().lower() == "paper"
            and str(backend_status["active_execution_backend"]).strip().lower() != "live",
            "No live-trading path is enabled.",
            "A live-trading ambiguity was detected.",
        ),
        "risk_engine_functioning": _criterion(
            model_status.get("status") == "ok",
            "Risk and model checks are loadable.",
            "Risk/model loading checks failed.",
        ),
        "positions_reconciled": _criterion(
            (reconciliation_summary.get("percent_reconciled_positions") or 0.0) == 1.0,
            "Positions are fully reconciled.",
            "Positions are not fully reconciled.",
        ),
        "drawdown_under_threshold": _criterion(
            _to_float(((session_summary or {}).get("performance_summary", {}) or {}).get("max_drawdown")) is None
            or float(((session_summary or {}).get("performance_summary", {}) or {}).get("max_drawdown", 0.0))
            <= phase10_11.readiness_thresholds.max_allowed_drawdown,
            "Drawdown is within threshold.",
            "Drawdown exceeds the readiness threshold.",
        ),
        "drift_under_threshold": _criterion(
            _max_drift(drift_summary) is None or _max_drift(drift_summary) <= phase10_11.readiness_thresholds.max_drift_psi,
            "Drift is within threshold.",
            "Drift exceeds the readiness threshold.",
        ),
        "healthchecks_passing": _criterion(
            (monitor_snapshot.get("status") == "ok"),
            "Monitoring healthchecks are passing.",
            "Monitoring healthchecks reported an error.",
        ),
        "minimum_sessions_completed": _criterion(
            len(completed_sessions) >= phase10_11.readiness_thresholds.min_required_sessions_for_readiness,
            "Minimum completed sessions threshold was met.",
            "Not enough completed paper-validation sessions exist yet.",
        ),
        "critical_alerts_below_threshold": _criterion(
            int((alert_summary.get("by_severity") or {}).get("critical", 0) or 0) <= phase10_11.readiness_thresholds.max_critical_alerts,
            "Critical alerts are below threshold.",
            "Critical alerts exceed the readiness threshold.",
        ),
        "latency_within_threshold": _criterion(
            _to_float(execution_summary.get("avg_submit_to_final_fill_ms")) is None
            or float(execution_summary.get("avg_submit_to_final_fill_ms", 0.0))
            <= phase10_11.readiness_thresholds.max_allowed_latency_ms,
            "Execution latency is within threshold.",
            "Execution latency exceeds the readiness threshold.",
        ),
    }

    critical_failures = [
        name
        for name in (
            "broker_connectivity_stable",
            "no_critical_reconciliation_mismatches",
            "no_live_path_enabled",
            "risk_engine_functioning",
            "positions_reconciled",
            "critical_alerts_below_threshold",
        )
        if not criteria[name]["passed"]
    ]
    review_flags = [
        name
        for name in (
            "alert_rate_within_threshold",
            "session_failure_rate_acceptable",
            "drawdown_under_threshold",
            "drift_under_threshold",
            "minimum_sessions_completed",
            "latency_within_threshold",
        )
        if not criteria[name]["passed"]
    ]

    if critical_failures:
        status = "NOT_READY"
    elif review_flags:
        status = "REVIEW_NEEDED"
    else:
        status = "READY"

    payload = {
        "status": "ok",
        "session_id": session_id,
        "readiness_status": status,
        "criteria": criteria,
        "critical_failures": critical_failures,
        "review_flags": review_flags,
        "alert_summary": alert_summary,
        "incident_summary": incident_summary,
        "session_failure_rate": session_failure_rate,
        "completed_session_count": len(completed_sessions),
        "evaluated_session_count": len(sessions),
    }
    write_json(session_dir / "readiness_report.json", payload)
    return payload


def _criterion(passed: bool, success_message: str, failure_message: str) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "message": success_message if passed else failure_message,
    }


def _max_drift(drift_summary: dict[str, Any]) -> float | None:
    values = [
        _to_float(drift_summary.get("data_drift_max_psi")),
        _to_float(drift_summary.get("prediction_drift_max_psi")),
        _to_float(drift_summary.get("label_drift_max_psi")),
    ]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)
