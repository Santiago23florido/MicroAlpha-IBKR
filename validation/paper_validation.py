from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import Settings
from config.phase10_11 import load_phase10_11_config
from config.phase7 import load_phase7_config
from engine.phase6 import show_active_model
from engine.phase7 import run_paper_session_real, show_execution_backend
from evaluation.io import load_phase7_frame, read_json, write_json
from execution.reconciliation import reconcile_broker_state
from monitoring.alerts import AlertStore, build_alert
from monitoring.paper_monitor import monitor_paper_session
from ops.incidents import IncidentStore, build_incident
from ops.preflight import preflight_check
from validation.readiness import generate_readiness_report
from validation.reconciliation_report import write_reconciliation_reports
from validation.session_tracker import SessionTracker


def run_paper_validation_session(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    latest_per_symbol: int | None = None,
    decision_log_path: str | None = None,
    run_preflight: bool = True,
) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    phase7 = load_phase7_config(settings)
    tracker = SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )
    alert_store = AlertStore(phase10_11.report_paths.alerts_path)
    incident_store = IncidentStore(phase10_11.report_paths.incidents_path)
    requested_symbols = [str(symbol).upper() for symbol in (symbols or settings.supported_symbols)]
    active_model = show_active_model(settings).get("active_model", {})
    backend_status = show_execution_backend(settings)
    session_record = tracker.start_session(
        active_model_name=str(active_model.get("model_name") or "unknown"),
        active_backend=str(backend_status.get("active_execution_backend") or "unknown"),
        symbols=requested_symbols,
    )
    session_id = session_record["session_id"]
    session_dir = tracker.session_dir(session_id)

    _write_session_snapshots(
        settings,
        tracker=tracker,
        session_id=session_id,
        active_model=active_model,
        backend_status=backend_status,
    )

    preflight = None
    if run_preflight:
        preflight = preflight_check(settings, symbols=requested_symbols, session_id=session_id)
        tracker.write_snapshot(session_id, "preflight_check.json", preflight)
        if preflight.get("status") != "ok":
            session_summary = _write_session_summary(
                tracker=tracker,
                session_id=session_id,
                session_record=session_record,
                phase7_summary={},
                final_state="PRECHECK_FAILED",
                alert_store=alert_store,
                reconciliation_status="PENDING",
                readiness_status="NOT_READY",
                initial_cash=phase7.session.initial_cash,
            )
            readiness = generate_readiness_report(
                settings,
                session_id=session_id,
                session_summary=session_summary,
            )
            tracker.finalize_session(
                session_id,
                ended_at=datetime.now(timezone.utc).isoformat(),
                alerts_count=alert_store.summarize(session_id=session_id)["total"],
                readiness_status=readiness["readiness_status"],
                reconciliation_status="PENDING",
                final_state="PRECHECK_FAILED",
            )
            return {
                "status": "error",
                "session_id": session_id,
                "session_dir": str(session_dir),
                "preflight": preflight,
                "readiness_report": readiness,
                "session_summary_path": str(session_dir / "session_summary.json"),
            }

    try:
        phase7_summary = run_paper_session_real(
            settings,
            symbols=requested_symbols,
            feature_root=feature_root,
            latest_per_symbol=latest_per_symbol,
            decision_log_path=decision_log_path,
        )
    except Exception as exc:
        alert_store.emit(
            build_alert(
                severity="critical",
                category="scheduler_failure",
                message=f"Paper validation session failed before completion: {exc}",
                session_id=session_id,
                context={"stage": "run_paper_session_real"},
                recommended_action="Review broker connectivity, active model, and preflight results before retrying.",
            )
        )
        incident_store.emit(
            build_incident(
                severity="critical",
                root_component="paper_validation",
                category="scheduler_failure",
                message=str(exc),
                session_id=session_id,
                context={"stage": "run_paper_session_real"},
            )
        )
        session_summary = _write_session_summary(
            tracker=tracker,
            session_id=session_id,
            session_record=session_record,
            phase7_summary={},
            final_state="FAILED_SESSION",
            alert_store=alert_store,
            reconciliation_status="PENDING",
            readiness_status="NOT_READY",
            initial_cash=phase7.session.initial_cash,
        )
        readiness = generate_readiness_report(
            settings,
            session_id=session_id,
            session_summary=session_summary,
        )
        tracker.finalize_session(
            session_id,
            ended_at=datetime.now(timezone.utc).isoformat(),
            alerts_count=alert_store.summarize(session_id=session_id)["total"],
            readiness_status=readiness["readiness_status"],
            reconciliation_status="PENDING",
            final_state="FAILED_SESSION",
        )
        return {
            "status": "error",
            "session_id": session_id,
            "session_dir": str(session_dir),
            "message": str(exc),
            "readiness_report": readiness,
            "session_summary_path": str(session_dir / "session_summary.json"),
        }

    phase7_frame = load_phase7_frame(phase7_summary["parquet_path"])
    session_summary = _write_session_summary(
        tracker=tracker,
        session_id=session_id,
        session_record=session_record,
        phase7_summary=phase7_summary,
        phase7_frame=phase7_frame,
        final_state="COMPLETED",
        alert_store=alert_store,
        reconciliation_status="PENDING",
        readiness_status="PENDING",
        initial_cash=phase7.session.initial_cash,
    )

    monitor_report = monitor_paper_session(
        settings,
        session_id=session_id,
        summary_path=phase7_summary["summary_path"],
        iterations=1,
        persist=True,
    )
    reconciliation = reconcile_and_report(settings, session_id=session_id)
    readiness = generate_readiness_report(
        settings,
        session_id=session_id,
        session_summary=session_summary,
        monitor_report=monitor_report,
        reconciliation_report=reconciliation,
    )
    session_summary = _write_session_summary(
        tracker=tracker,
        session_id=session_id,
        session_record=session_record,
        phase7_summary=phase7_summary,
        phase7_frame=phase7_frame,
        final_state="COMPLETED",
        alert_store=alert_store,
        reconciliation_status=reconciliation["summary"]["status"],
        readiness_status=readiness["readiness_status"],
        initial_cash=phase7.session.initial_cash,
    )
    tracker.finalize_session(
        session_id,
        ended_at=datetime.now(timezone.utc).isoformat(),
        decisions_count=session_summary["decisions_count"],
        orders_count=session_summary["orders_count"],
        fills_count=session_summary["fills_count"],
        pnl=session_summary["pnl"],
        alerts_count=session_summary["alerts_count"],
        reconciliation_status=session_summary["reconciliation_status"],
        readiness_status=session_summary["readiness_status"],
        final_state=session_summary["final_state"],
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "session_dir": str(session_dir),
        "phase7_summary": phase7_summary,
        "monitor_report": monitor_report,
        "reconciliation_report": reconciliation,
        "readiness_report": readiness,
        "session_summary_path": str(session_dir / "session_summary.json"),
    }


def reconcile_and_report(settings: Settings, *, session_id: str | None = None) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    phase7 = load_phase7_config(settings)
    if str(phase7.execution.active_execution_backend).strip().lower() != "ibkr_paper":
        raise ValueError("reconcile-broker-state requires ACTIVE_EXECUTION_BACKEND=ibkr_paper.")
    if str(phase7.ibkr_paper.broker_mode).strip().lower() != "paper":
        raise ValueError("reconcile-broker-state requires broker_mode=paper.")
    tracker = SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )
    alert_store = AlertStore(phase10_11.report_paths.alerts_path)
    incident_store = IncidentStore(phase10_11.report_paths.incidents_path)
    session_id = session_id or ((tracker.latest_session() or {}).get("session_id"))
    if session_id is None:
        raise FileNotFoundError("No paper validation session is available for reconciliation.")
    session_dir = tracker.session_dir(session_id)

    reconciliation = reconcile_broker_state(settings)
    report_paths = write_reconciliation_reports(
        report_root=phase10_11.report_paths.reconciliation_dir,
        session_dir=session_dir,
        session_id=session_id,
        reconciliation=reconciliation,
    )

    summary = reconciliation["summary"]
    if summary["critical_order_mismatches"] + summary["critical_fill_mismatches"] + summary["critical_position_mismatches"] > 0:
        alert_store.emit(
            build_alert(
                severity="critical",
                category="reconciliation_mismatch",
                message="Critical reconciliation mismatches were detected.",
                session_id=session_id,
                context=summary,
                recommended_action="Stop the next automated session and review reconciliation reports.",
            )
        )
        incident_store.emit(
            build_incident(
                severity="critical",
                root_component="reconciliation",
                category="reconciliation_failure",
                message="Critical reconciliation mismatches were detected.",
                session_id=session_id,
                context=summary,
            )
        )
    return {
        **reconciliation,
        "report_paths": report_paths,
    }


def compare_paper_sessions(settings: Settings) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    tracker = SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )
    sessions = tracker.list_sessions(limit=phase10_11.validation_session_limits.max_sessions_to_compare)
    rows: list[dict[str, Any]] = []
    for session in sessions:
        session_id = session["session_id"]
        session_dir = tracker.session_dir(session_id)
        summary = _read_optional_json(session_dir / "session_summary.json")
        readiness = _read_optional_json(session_dir / "readiness_report.json")
        reconciliation = _read_optional_json(session_dir / "reconciliation_summary.json")
        health = _read_optional_json(session_dir / "system_health.json")
        phase7_summary = _read_optional_json(Path(summary.get("phase7_summary_path", ""))) if summary.get("phase7_summary_path") else {}
        phase8_report = ((phase7_summary.get("phase8_report") or {}).get("run_report_path")) if isinstance(phase7_summary.get("phase8_report"), dict) else None
        phase8_payload = _read_optional_json(Path(phase8_report)) if phase8_report else {}
        execution_summary = dict(phase8_payload.get("execution_summary", {}) or {})
        drift_summary = dict(phase8_payload.get("drift_summary", {}) or {})
        performance_summary = dict(phase8_payload.get("performance_summary", {}) or {})
        rows.append(
            {
                "session_id": session_id,
                "started_at": session.get("started_at"),
                "ended_at": session.get("ended_at"),
                "active_model_name": session.get("active_model_name"),
                "active_backend": session.get("active_backend"),
                "decisions_count": session.get("decisions_count"),
                "orders_count": session.get("orders_count"),
                "fills_count": session.get("fills_count"),
                "pnl": session.get("pnl"),
                "alerts_count": session.get("alerts_count"),
                "reconciliation_status": session.get("reconciliation_status"),
                "readiness_status": session.get("readiness_status"),
                "final_state": session.get("final_state"),
                "max_drawdown": performance_summary.get("max_drawdown"),
                "avg_submit_to_final_fill_ms": execution_summary.get("avg_submit_to_final_fill_ms"),
                "critical_mismatches": (
                    int(reconciliation.get("critical_order_mismatches", 0) or 0)
                    + int(reconciliation.get("critical_fill_mismatches", 0) or 0)
                    + int(reconciliation.get("critical_position_mismatches", 0) or 0)
                ),
                "drift_max_psi": max(
                    [
                        value
                        for value in (
                            _to_float(drift_summary.get("data_drift_max_psi")),
                            _to_float(drift_summary.get("prediction_drift_max_psi")),
                            _to_float(drift_summary.get("label_drift_max_psi")),
                        )
                        if value is not None
                    ],
                    default=None,
                ),
                "health_status": health.get("status"),
                "readiness_overall": readiness.get("readiness_status"),
            }
        )

    frame = pd.DataFrame(rows)
    comparison_dir = Path(phase10_11.report_paths.comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = comparison_dir / "session_leaderboard.csv"
    health_summary_path = comparison_dir / "session_health_summary.csv"
    report_path = comparison_dir / "session_compare_report.json"

    if not frame.empty:
        sort_columns = ["readiness_overall", "critical_mismatches", "alerts_count", "pnl"]
        sortable = frame.copy()
        readiness_rank = {"READY": 0, "REVIEW_NEEDED": 1, "NOT_READY": 2}
        sortable["readiness_rank"] = sortable["readiness_overall"].map(readiness_rank).fillna(3)
        sortable = sortable.sort_values(
            by=["readiness_rank", "critical_mismatches", "alerts_count", "pnl"],
            ascending=[True, True, True, False],
        )
        sortable.drop(columns=["readiness_rank"]).to_csv(leaderboard_path, index=False)
        sortable[
            [
                "session_id",
                "final_state",
                "readiness_overall",
                "health_status",
                "critical_mismatches",
                "alerts_count",
                "avg_submit_to_final_fill_ms",
                "drift_max_psi",
            ]
        ].to_csv(health_summary_path, index=False)
    else:
        frame.to_csv(leaderboard_path, index=False)
        frame.to_csv(health_summary_path, index=False)

    report = {
        "status": "ok",
        "session_count": int(len(frame)),
        "best_pnl_session": None if frame.empty else frame.sort_values("pnl", ascending=False).iloc[0]["session_id"],
        "most_stable_session": None if frame.empty else frame.sort_values(["critical_mismatches", "alerts_count"]).iloc[0]["session_id"],
        "worst_latency_session": None if frame.empty else frame.sort_values("avg_submit_to_final_fill_ms", ascending=False).iloc[0]["session_id"],
        "most_drift_session": None if frame.empty else frame.sort_values("drift_max_psi", ascending=False).iloc[0]["session_id"],
        "leaderboard_path": str(leaderboard_path),
        "health_summary_path": str(health_summary_path),
    }
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def _write_session_summary(
    *,
    tracker: SessionTracker,
    session_id: str,
    session_record: dict[str, Any],
    phase7_summary: dict[str, Any],
    final_state: str,
    alert_store: AlertStore,
    reconciliation_status: str,
    readiness_status: str,
    initial_cash: float,
    phase7_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    phase7_frame = phase7_frame if phase7_frame is not None else pd.DataFrame()
    fills_count = int(pd.to_numeric(phase7_frame.get("fill_count"), errors="coerce").fillna(0).sum()) if not phase7_frame.empty else 0
    pnl = _session_total_pnl(phase7_summary, initial_cash=initial_cash)
    phase8_report_path = ((phase7_summary.get("phase8_report") or {}).get("run_report_path")) if isinstance(phase7_summary.get("phase8_report"), dict) else None
    phase8_report = _read_optional_json(Path(phase8_report_path)) if phase8_report_path else {}
    payload = {
        "session_id": session_id,
        "started_at": session_record.get("started_at"),
        "ended_at": datetime.now(timezone.utc).isoformat() if final_state != "RUNNING" else None,
        "active_model_name": session_record.get("active_model_name"),
        "active_backend": session_record.get("active_backend"),
        "symbols": session_record.get("symbols", []),
        "decisions_count": int(phase7_summary.get("row_count", 0) or 0),
        "orders_count": int(phase7_summary.get("orders_attempted", 0) or 0),
        "fills_count": fills_count,
        "pnl": pnl,
        "alerts_count": int(alert_store.summarize(session_id=session_id)["total"]),
        "reconciliation_status": reconciliation_status,
        "readiness_status": readiness_status,
        "final_state": final_state,
        "phase7_summary_path": phase7_summary.get("summary_path"),
        "phase7_parquet_path": phase7_summary.get("parquet_path"),
        "phase7_phase8_report": phase7_summary.get("phase8_report"),
        "performance_summary": dict(phase8_report.get("performance_summary", {}) or {}),
    }
    write_json(tracker.session_dir(session_id) / "session_summary.json", payload)
    return payload


def _write_session_snapshots(
    settings: Settings,
    *,
    tracker: SessionTracker,
    session_id: str,
    active_model: dict[str, Any],
    backend_status: dict[str, Any],
) -> None:
    phase10_11 = load_phase10_11_config(settings)
    tracker.write_snapshot(session_id, "active_model_snapshot.json", {"active_model": active_model})
    tracker.write_snapshot(session_id, "backend_snapshot.json", backend_status)
    tracker.write_snapshot(session_id, "config_snapshot.json", settings.as_dict())
    tracker.write_snapshot(session_id, "phase10_11_config_snapshot.json", phase10_11.to_dict())


def _session_total_pnl(phase7_summary: dict[str, Any], *, initial_cash: float) -> float:
    portfolio = dict(phase7_summary.get("portfolio_final", {}) or {})
    equity = _to_float(portfolio.get("equity"))
    if equity is None:
        realized = _to_float(portfolio.get("realized_pnl")) or 0.0
        unrealized = _to_float(portfolio.get("unrealized_pnl")) or 0.0
        return float(realized + unrealized)
    return float(equity - initial_cash)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
