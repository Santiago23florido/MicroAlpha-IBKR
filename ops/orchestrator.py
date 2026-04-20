from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from config import Settings
from config.phase10_11 import load_phase10_11_config
from config.phase12_14 import resolve_runtime_profile
from evaluation.io import write_json
from governance.releases import governance_status, show_active_release
from monitoring.alerts import AlertStore
from monitoring.paper_monitor import monitor_paper_session
from ops.incidents import IncidentStore
from ops.postflight import postflight_check
from ops.preflight import preflight_check
from ops.runbooks import generate_runbooks as _generate_runbooks
from ops.runtime_manager import bootstrap_runtime, runtime_status as runtime_status_command, start_runtime
from ops.scheduler import build_scheduler_plan
from shadow.session import run_shadow_session
from validation.paper_validation import compare_paper_sessions, run_paper_validation_session
from validation.readiness import generate_readiness_report
from validation.session_tracker import SessionTracker


def full_paper_validation_cycle(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    latest_per_symbol: int | None = None,
    decision_log_path: str | None = None,
) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    preflight = preflight_check(settings, symbols=symbols)
    if preflight.get("status") != "ok":
        return {
            "status": "error",
            "stage": "preflight",
            "preflight": preflight,
            "scheduler_plan": build_scheduler_plan(settings),
        }

    session = run_paper_validation_session(
        settings,
        symbols=symbols,
        feature_root=feature_root,
        latest_per_symbol=latest_per_symbol,
        decision_log_path=decision_log_path,
        run_preflight=False,
    )
    session_id = session["session_id"]
    readiness = session.get("readiness_report") or generate_readiness_report(settings, session_id=session_id)
    postflight = postflight_check(settings, session_id=session_id)
    comparison = compare_paper_sessions(settings)
    health = system_health_report(settings, session_id=session_id)
    tracker = SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )
    tracker.write_snapshot(session_id, "scheduler_plan.json", build_scheduler_plan(settings))
    return {
        "status": "ok" if readiness["readiness_status"] != "NOT_READY" and postflight["status"] == "ok" else "error",
        "session_id": session_id,
        "session_dir": session["session_dir"],
        "preflight": preflight,
        "session": session,
        "readiness_report": readiness,
        "postflight": postflight,
        "comparison": comparison,
        "system_health": health,
        "scheduler_plan": build_scheduler_plan(settings),
    }


def system_health_report(settings: Settings, *, session_id: str | None = None) -> dict[str, Any]:
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
    latest_session_payload = None if session_id is None else tracker.load_session(session_id)
    readiness_report = {} if session_id is None else _read_optional_json(tracker.session_dir(session_id) / "readiness_report.json")
    monitor_report = {} if session_id is None else _read_optional_json(tracker.session_dir(session_id) / "system_health.json")
    payload = {
        "status": "ok",
        "latest_session_status": latest_session_payload,
        "broker_connectivity_health": ((monitor_report.get("latest_snapshot") or {}).get("broker_status")) if monitor_report else None,
        "reconciliation_health": _read_optional_json(tracker.session_dir(session_id) / "reconciliation_summary.json") if session_id else {},
        "alert_counts_by_severity": alert_store.summarize(session_id=session_id).get("by_severity", {}),
        "drift_status": ((monitor_report.get("latest_snapshot") or {}).get("drift_summary")) if monitor_report else {},
        "risk_status": ((monitor_report.get("latest_snapshot") or {}).get("model_status")) if monitor_report else {},
        "readiness_trend": readiness_report.get("readiness_status"),
        "outstanding_incidents": incident_store.list_incidents(session_id=session_id, limit=10),
    }
    write_json(phase10_11.report_paths.health_report_path, payload)
    if session_id:
        write_json(tracker.session_dir(session_id) / "system_health_consolidated.json", payload)
    return payload


def generate_runbooks(settings: Settings) -> dict[str, Any]:
    return _generate_runbooks(settings)


def preflight_check_command(settings: Settings, *, symbols: Sequence[str] | None = None) -> dict[str, Any]:
    return preflight_check(settings, symbols=symbols)


def postflight_check_command(settings: Settings, *, session_id: str) -> dict[str, Any]:
    return postflight_check(settings, session_id=session_id)


def full_runtime_cycle(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    latest_per_symbol: int | None = None,
    decision_log_path: str | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    profile = resolve_runtime_profile(settings, profile_name=profile_name)
    bootstrap = bootstrap_runtime(settings, profile_name=profile.name)
    if bootstrap["status"] != "ok":
        return {"status": "error", "stage": "bootstrap", "bootstrap": bootstrap}

    runtime = start_runtime(settings, profile_name=profile.name)
    if runtime["status"] != "ok":
        return {"status": "error", "stage": "start_runtime", "runtime": runtime}

    if profile.shadow_mode_enabled:
        session = run_shadow_session(
            settings,
            symbols=symbols,
            feature_root=feature_root,
            latest_per_symbol=latest_per_symbol,
            decision_log_path=decision_log_path,
        )
        health = system_health_report(settings)
        governance = governance_status(settings)
        return {
            "status": "ok",
            "runtime_profile": profile.name,
            "mode": "shadow",
            "bootstrap": bootstrap,
            "runtime": runtime,
            "session": session,
            "system_health": health,
            "governance": governance,
            "active_release": show_active_release(settings).get("active_release"),
        }

    validation = full_paper_validation_cycle(
        settings,
        symbols=symbols,
        feature_root=feature_root,
        latest_per_symbol=latest_per_symbol,
        decision_log_path=decision_log_path,
    )
    governance = governance_status(settings)
    return {
        "status": validation.get("status", "error"),
        "runtime_profile": profile.name,
        "mode": "paper_validation",
        "bootstrap": bootstrap,
        "runtime": runtime,
        "validation": validation,
        "governance": governance,
        "active_release": show_active_release(settings).get("active_release"),
    }


def governance_status_command(settings: Settings) -> dict[str, Any]:
    return governance_status(settings)


def runtime_status_command_wrapper(settings: Settings) -> dict[str, Any]:
    return runtime_status_command(settings)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    from evaluation.io import read_json

    return read_json(path)
