from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings
from config.phase10_11 import load_phase10_11_config
from engine.phase6 import risk_check
from engine.phase7 import broker_healthcheck, show_execution_backend
from evaluation.io import load_phase7_frame, read_json, write_json
from monitoring.alerts import AlertStore, build_alert
from ops.incidents import IncidentStore, build_incident
from validation.session_tracker import SessionTracker


def monitor_paper_session(
    settings: Settings,
    *,
    session_id: str | None = None,
    summary_path: str | Path | None = None,
    iterations: int = 1,
    persist: bool = True,
) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    active_backend = str(show_execution_backend(settings).get("active_execution_backend") or "").strip().lower()
    if active_backend != "ibkr_paper":
        raise ValueError("monitor-paper-session requires ACTIVE_EXECUTION_BACKEND=ibkr_paper.")
    tracker = SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )
    alert_store = AlertStore(phase10_11.report_paths.alerts_path)
    incident_store = IncidentStore(phase10_11.report_paths.incidents_path)
    iterations = max(1, min(int(iterations), phase10_11.validation_session_limits.max_monitor_iterations))

    snapshots: list[dict[str, Any]] = []
    emitted_alerts: list[dict[str, Any]] = []
    emitted_incidents: list[dict[str, Any]] = []
    for index in range(iterations):
        snapshot = _collect_monitor_snapshot(settings, session_id=session_id, summary_path=summary_path)
        snapshots.append(snapshot)
        alerts = _alerts_from_snapshot(snapshot, session_id=session_id, phase10_11=phase10_11)
        incidents = _incidents_from_alerts(alerts, session_id=session_id)
        emitted_alerts.extend(alert_store.emit_many(alerts))
        emitted_incidents.extend(incident_store.emit_many(incidents))
        if index + 1 < iterations:
            time.sleep(phase10_11.monitoring_intervals.monitoring_sleep_seconds)

    latest = snapshots[-1]
    payload = {
        "status": latest.get("status", "ok"),
        "session_id": session_id,
        "iterations": iterations,
        "latest_snapshot": latest,
        "snapshots": snapshots,
        "alerts_generated": len(emitted_alerts),
        "incidents_generated": len(emitted_incidents),
        "alert_summary": alert_store.summarize(session_id=session_id),
        "incident_summary": incident_store.summarize(session_id=session_id),
    }

    if persist and session_id:
        session_dir = tracker.session_dir(session_id)
        write_json(session_dir / "system_health.json", payload)
        alert_store.write_session_csv(session_id, session_dir / "alerts_summary.csv")
        incident_store.write_session_csv(session_id, session_dir / "incidents_summary.csv")
    elif persist:
        write_json(phase10_11.report_paths.health_report_path, payload)
    return payload


def _collect_monitor_snapshot(
    settings: Settings,
    *,
    session_id: str | None,
    summary_path: str | Path | None,
) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    tracker = SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )
    summary_payload = _resolve_phase7_summary_payload(session_id=session_id, summary_path=summary_path, tracker=tracker)
    parquet_path = summary_payload.get("parquet_path")
    frame = load_phase7_frame(parquet_path) if parquet_path and Path(parquet_path).exists() else None
    latest_timestamp = None if frame is None or frame.empty else frame["timestamp"].dropna().max()
    data_age_seconds = None
    if latest_timestamp is not None:
        now = datetime.now(timezone.utc)
        data_age_seconds = float((now - latest_timestamp.to_pydatetime()).total_seconds())

    backend_status = show_execution_backend(settings)
    model_status = risk_check(settings)
    try:
        broker_status = broker_healthcheck(settings)
    except Exception as exc:  # pragma: no cover - depends on broker availability
        broker_status = {"status": "error", "message": str(exc)}

    phase8_run_report = None
    run_report_path = ((summary_payload.get("phase8_report") or {}).get("run_report_path")) if isinstance(summary_payload.get("phase8_report"), dict) else None
    if run_report_path and Path(run_report_path).exists():
        phase8_run_report = read_json(run_report_path)

    drift_summary = dict((phase8_run_report or {}).get("drift_summary", {}) or {})
    execution_summary = dict((phase8_run_report or {}).get("execution_summary", {}) or {})
    blocked_by_risk_count = int(summary_payload.get("blocked_by_risk_count", 0) or 0)
    row_count = int(summary_payload.get("row_count", 0) or 0)
    risk_block_rate = (blocked_by_risk_count / row_count) if row_count else 0.0

    reconciliation_summary = None
    if session_id:
        reconciliation_path = tracker.session_dir(session_id) / "reconciliation_summary.json"
        if reconciliation_path.exists():
            reconciliation_summary = read_json(reconciliation_path)

    status = "ok"
    if broker_status.get("status") != "ok":
        status = "error"
    elif reconciliation_summary and reconciliation_summary.get("status") == "CRITICAL_MISMATCH":
        status = "error"
    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "backend_status": backend_status,
        "broker_status": broker_status,
        "model_status": model_status,
        "phase7_summary_path": summary_payload.get("summary_path"),
        "data_age_seconds": data_age_seconds,
        "blocked_by_risk_count": blocked_by_risk_count,
        "risk_block_rate": risk_block_rate,
        "execution_summary": execution_summary,
        "drift_summary": drift_summary,
        "reconciliation_summary": reconciliation_summary,
    }


def _resolve_phase7_summary_payload(
    *,
    session_id: str | None,
    summary_path: str | Path | None,
    tracker: SessionTracker,
) -> dict[str, Any]:
    if summary_path is not None:
        payload = read_json(summary_path)
        payload.setdefault("summary_path", str(summary_path))
        return payload
    if session_id:
        session_summary_path = tracker.session_dir(session_id) / "session_summary.json"
        if session_summary_path.exists():
            session_summary = read_json(session_summary_path)
            phase7_summary_path = session_summary.get("phase7_summary_path")
            if phase7_summary_path:
                payload = read_json(phase7_summary_path)
                payload.setdefault("summary_path", str(phase7_summary_path))
                return payload
    raise FileNotFoundError("monitor-paper-session requires a session_id or summary_path that resolves to a Phase 7 summary.")


def _alerts_from_snapshot(
    snapshot: dict[str, Any],
    *,
    session_id: str | None,
    phase10_11,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    broker_status = snapshot.get("broker_status", {})
    if broker_status.get("status") != "ok":
        alerts.append(
            build_alert(
                severity="critical",
                category="connection_issue",
                message="Broker connectivity is degraded during paper monitoring.",
                session_id=session_id,
                context={"broker_status": broker_status},
                recommended_action="Run broker-healthcheck and stop the next session until connectivity is restored.",
            ).to_dict()
        )

    data_age_seconds = snapshot.get("data_age_seconds")
    if data_age_seconds is not None and float(data_age_seconds) > phase10_11.alert_thresholds.max_stale_data_seconds:
        alerts.append(
            build_alert(
                severity="warning",
                category="stale_data",
                message="Observed data age exceeds the configured freshness threshold.",
                session_id=session_id,
                context={"data_age_seconds": data_age_seconds, "threshold": phase10_11.alert_thresholds.max_stale_data_seconds},
                recommended_action="Check the feature pipeline and data import freshness before the next run.",
            ).to_dict()
        )

    execution_summary = dict(snapshot.get("execution_summary", {}) or {})
    if (
        _to_float(execution_summary.get("rejection_rate")) is not None
        and float(execution_summary["rejection_rate"]) > phase10_11.alert_thresholds.max_order_rejection_rate
    ):
        alerts.append(
            build_alert(
                severity="warning",
                category="order_rejection",
                message="Order rejection rate exceeds the configured threshold.",
                session_id=session_id,
                context={"rejection_rate": execution_summary.get("rejection_rate")},
                recommended_action="Review broker rejects and risk-block patterns before continuing.",
            ).to_dict()
        )
    if (
        _to_float(execution_summary.get("avg_submit_to_final_fill_ms")) is not None
        and float(execution_summary["avg_submit_to_final_fill_ms"]) > phase10_11.alert_thresholds.max_submit_to_fill_latency_ms
    ):
        alerts.append(
            build_alert(
                severity="warning",
                category="execution_delay",
                message="Average submit-to-fill latency exceeds the configured threshold.",
                session_id=session_id,
                context={"avg_submit_to_final_fill_ms": execution_summary.get("avg_submit_to_final_fill_ms")},
                recommended_action="Review IBKR Paper latency before trusting automation.",
            ).to_dict()
        )

    if float(snapshot.get("risk_block_rate") or 0.0) > phase10_11.alert_thresholds.max_risk_block_rate:
        alerts.append(
            build_alert(
                severity="warning",
                category="risk_block",
                message="Risk block rate is above the configured threshold.",
                session_id=session_id,
                context={"risk_block_rate": snapshot.get("risk_block_rate")},
                recommended_action="Review session timing, costs, spread conditions, and risk thresholds.",
            ).to_dict()
        )

    drift_summary = dict(snapshot.get("drift_summary", {}) or {})
    drift_values = [
        _to_float(drift_summary.get("data_drift_max_psi")),
        _to_float(drift_summary.get("prediction_drift_max_psi")),
        _to_float(drift_summary.get("label_drift_max_psi")),
    ]
    max_drift = max([value for value in drift_values if value is not None], default=None)
    if max_drift is not None and max_drift > phase10_11.readiness_thresholds.max_drift_psi:
        alerts.append(
            build_alert(
                severity="warning",
                category="drift_warning",
                message="Drift severity exceeds the readiness threshold.",
                session_id=session_id,
                context={"max_drift_psi": max_drift},
                recommended_action="Review drift reports before continuing multi-session validation.",
            ).to_dict()
        )

    reconciliation_summary = dict(snapshot.get("reconciliation_summary", {}) or {})
    if reconciliation_summary.get("status") == "CRITICAL_MISMATCH":
        alerts.append(
            build_alert(
                severity="critical",
                category="reconciliation_mismatch",
                message="Critical reconciliation mismatch detected during monitoring.",
                session_id=session_id,
                context={"reconciliation_summary": reconciliation_summary},
                recommended_action="Stop automation and reconcile broker state manually.",
            ).to_dict()
        )
    return alerts


def _incidents_from_alerts(alerts: list[dict[str, Any]], *, session_id: str | None) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    category_map = {
        "connection_issue": "broker_connectivity_failure",
        "reconciliation_mismatch": "reconciliation_failure",
        "stale_data": "data_pipeline_failure",
        "order_rejection": "order_routing_failure",
        "execution_delay": "order_routing_failure",
        "risk_block": "decision_engine_failure",
        "drift_warning": "scheduler_failure",
    }
    for alert in alerts:
        if str(alert.get("severity") or "").lower() not in {"critical", "warning"}:
            continue
        incidents.append(
            build_incident(
                severity=str(alert.get("severity")),
                root_component="paper_monitor",
                category=category_map.get(str(alert.get("category") or ""), "scheduler_failure"),
                message=str(alert.get("message") or ""),
                session_id=session_id,
                context={"alert_id": alert.get("alert_id"), "alert_context": alert.get("context", {})},
            ).to_dict()
        )
    return incidents


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
