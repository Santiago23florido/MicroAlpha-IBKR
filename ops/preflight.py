from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from config import Settings
from config.phase10_11 import load_phase10_11_config
from config.phase6 import load_phase6_config
from config.phase7 import load_phase7_config
from engine.phase6 import risk_check
from engine.phase7 import broker_healthcheck
from monitoring.alerts import AlertStore, build_alert
from ops.incidents import IncidentStore, build_incident
from ops.recovery import attempt_safe_recovery


def preflight_check(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    phase7 = load_phase7_config(settings)
    phase10_11 = load_phase10_11_config(settings)
    alert_store = AlertStore(phase10_11.report_paths.alerts_path)
    incident_store = IncidentStore(phase10_11.report_paths.incidents_path)

    requested_symbols = [str(symbol).upper() for symbol in (symbols or settings.supported_symbols)]
    checks: dict[str, dict[str, Any]] = {}
    alerts: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []

    def record_check(name: str, passed: bool, *, message: str, severity: str = "critical", context: dict[str, Any] | None = None) -> None:
        checks[name] = {
            "passed": bool(passed),
            "message": message,
            "severity": severity,
            "context": dict(context or {}),
        }
        if not passed:
            category = _check_to_alert_category(name)
            alerts.append(
                build_alert(
                    severity=severity,
                    category=category,
                    message=message,
                    session_id=session_id,
                    context={"check": name, **dict(context or {})},
                    recommended_action=_check_recommended_action(name),
                ).to_dict()
            )
            if severity == "critical":
                incidents.append(
                    build_incident(
                        severity=severity,
                        root_component="preflight",
                        category=_check_to_incident_category(name),
                        message=message,
                        session_id=session_id,
                        context={"check": name, **dict(context or {})},
                    ).to_dict()
                )

    backend_ok = str(phase7.execution.active_execution_backend).strip().lower() == "ibkr_paper"
    record_check(
        "execution_backend_real",
        backend_ok,
        message=(
            "Active execution backend is ibkr_paper."
            if backend_ok
            else f"Active execution backend must be ibkr_paper, got {phase7.execution.active_execution_backend!r}."
        ),
        context={"active_execution_backend": phase7.execution.active_execution_backend},
    )

    broker_mode_ok = (
        str(settings.broker_mode).strip().lower() == "paper"
        and str(phase7.ibkr_paper.broker_mode).strip().lower() == "paper"
    )
    record_check(
        "broker_mode_paper",
        broker_mode_ok,
        message="Broker mode is paper." if broker_mode_ok else "Broker mode must remain paper for validation and ops hardening.",
        context={
            "settings_broker_mode": settings.broker_mode,
            "phase7_broker_mode": phase7.ibkr_paper.broker_mode,
        },
    )

    safe_to_trade_ok = bool(settings.safe_to_trade and phase7.ibkr_paper.safe_to_trade)
    record_check(
        "safe_to_trade_enabled",
        safe_to_trade_ok,
        message="safe_to_trade is enabled." if safe_to_trade_ok else "safe_to_trade must be explicitly enabled before paper validation.",
        context={"settings_safe_to_trade": settings.safe_to_trade, "backend_safe_to_trade": phase7.ibkr_paper.safe_to_trade},
    )

    allow_session_ok = bool(settings.trading.allow_session_execution and phase7.ibkr_paper.allow_session_execution)
    record_check(
        "allow_session_execution",
        allow_session_ok,
        message="Session execution is enabled." if allow_session_ok else "ALLOW_SESSION_EXECUTION must be true before paper validation.",
        context={
            "settings_allow_session_execution": settings.trading.allow_session_execution,
            "backend_allow_session_execution": phase7.ibkr_paper.allow_session_execution,
        },
    )

    try:
        model_status = risk_check(settings)
        model_loadable = bool(model_status.get("inference", {}).get("loadable"))
        record_check(
            "active_model_loadable",
            model_loadable,
            message="Active model loaded successfully." if model_loadable else "Active model failed to load during preflight.",
            context={"risk_check_status": model_status.get("status"), "errors": model_status.get("errors", [])},
        )
    except Exception as exc:
        record_check(
            "active_model_loadable",
            False,
            message=f"Active model failed to load during preflight: {exc}",
            context={"exception": str(exc)},
        )

    risk_config_ok = False
    try:
        load_phase6_config(settings)
        risk_config_ok = True
    except Exception as exc:
        record_check("risk_config_present", False, message=f"Phase 6 risk configuration failed to load: {exc}")
    else:
        record_check("risk_config_present", True, message="Phase 6 risk configuration is loadable.")

    feature_pipeline_ok = Path(settings.paths.feature_dir).exists()
    record_check(
        "feature_pipeline_available",
        feature_pipeline_ok,
        message="Feature directory is present." if feature_pipeline_ok else "Feature directory is missing.",
        severity="warning",
        context={"feature_dir": settings.paths.feature_dir},
    )

    supported_symbols = {symbol.upper() for symbol in phase7.ibkr_paper.supported_symbols}
    symbol_list_ok = all(symbol in supported_symbols for symbol in requested_symbols)
    record_check(
        "symbol_list_valid",
        symbol_list_ok,
        message="Requested symbols are allowed." if symbol_list_ok else "Requested symbols are outside the allowed paper-trading universe.",
        context={"requested_symbols": requested_symbols, "supported_symbols": sorted(supported_symbols)},
    )

    writable_paths_ok = _check_paths_writable(
        [
            phase7.logging.journal_dir,
            phase7.logging.report_dir,
            phase10_11.report_paths.session_root,
            phase10_11.report_paths.reconciliation_dir,
            phase10_11.report_paths.alerts_path,
            phase10_11.report_paths.incidents_path,
        ]
    )
    record_check(
        "required_paths_writable",
        writable_paths_ok["ok"],
        message="Required report paths are writable." if writable_paths_ok["ok"] else "One or more required paths are not writable.",
        context={"path_results": writable_paths_ok["paths"]},
    )

    broker_ok = False
    broker_payload: dict[str, Any] | None = None
    recovery_event: dict[str, Any] | None = None
    try:
        broker_payload = broker_healthcheck(settings)
        broker_ok = broker_payload.get("status") == "ok"
    except Exception as exc:  # pragma: no cover - depends on broker availability
        recovery_event = attempt_safe_recovery(
            settings,
            category="broker_connectivity_failure",
            session_id=session_id,
            context={"root_cause": str(exc)},
        )
        broker_payload = {"status": "error", "message": str(exc)}
        broker_ok = recovery_event.get("status") == "recovered"
    record_check(
        "broker_reachable",
        broker_ok,
        message="Broker healthcheck passed." if broker_ok else "Broker healthcheck failed.",
        context={"broker_health": broker_payload, "recovery_event": recovery_event},
    )

    alert_store.emit_many(alerts)
    incident_store.emit_many(incidents)

    required = set(phase10_11.checks.preflight_required_checks)
    blocking_failures = [name for name, payload in checks.items() if name in required and not payload["passed"]]
    status = "ok" if not blocking_failures else "error"
    return {
        "status": status,
        "checks": checks,
        "required_checks": list(phase10_11.checks.preflight_required_checks),
        "blocking_failures": blocking_failures,
        "alerts": alerts,
        "incidents": incidents,
    }


def _check_paths_writable(paths: list[str]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    overall = True
    for path_str in paths:
        path = Path(path_str)
        target = path if path.suffix == "" else path.parent
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".preflight_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            results.append({"path": str(path), "writable": True})
        except Exception as exc:
            overall = False
            results.append({"path": str(path), "writable": False, "error": str(exc)})
    return {"ok": overall, "paths": results}


def _check_to_alert_category(name: str) -> str:
    mapping = {
        "broker_reachable": "connection_issue",
        "required_paths_writable": "data_quality_issue",
        "symbol_list_valid": "data_quality_issue",
        "active_model_loadable": "data_quality_issue",
    }
    return mapping.get(name, "scheduler_failure")


def _check_to_incident_category(name: str) -> str:
    mapping = {
        "broker_reachable": "broker_connectivity_failure",
        "active_model_loadable": "model_load_failure",
        "required_paths_writable": "config_failure",
        "execution_backend_real": "config_failure",
        "broker_mode_paper": "config_failure",
        "allow_session_execution": "config_failure",
        "safe_to_trade_enabled": "config_failure",
    }
    return mapping.get(name, "config_failure")


def _check_recommended_action(name: str) -> str:
    mapping = {
        "broker_reachable": "Verify TWS/IB Gateway paper connectivity and retry broker-healthcheck.",
        "active_model_loadable": "Review the active model artifact and run show-active-model plus risk-check.",
        "required_paths_writable": "Fix filesystem permissions before starting validation.",
        "execution_backend_real": "Switch ACTIVE_EXECUTION_BACKEND to ibkr_paper before validation.",
        "broker_mode_paper": "Ensure every broker mode flag remains paper.",
        "allow_session_execution": "Set ALLOW_SESSION_EXECUTION=true only for paper mode.",
        "safe_to_trade_enabled": "Set SAFE_TO_TRADE=true only after manual review.",
    }
    return mapping.get(name, "Review the failed check and resolve it before continuing.")
