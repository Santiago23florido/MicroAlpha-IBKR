from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import Settings
from config.phase10_11 import load_phase10_11_config
from config.phase12_14 import load_phase12_14_config, resolve_runtime_profile
from config.phase7 import load_phase7_config
from evaluation.io import write_json
from governance.releases import governance_status, show_active_release
from monitoring.alerts import AlertStore, build_alert
from ops.incidents import IncidentStore, build_incident
from ops.preflight import preflight_check


def bootstrap_runtime(settings: Settings, *, profile_name: str | None = None) -> dict[str, Any]:
    config = load_phase12_14_config(settings)
    profile = resolve_runtime_profile(settings, profile_name=profile_name)
    phase10_11 = load_phase10_11_config(settings)
    phase7 = load_phase7_config(settings)
    alerts = AlertStore(phase10_11.report_paths.alerts_path)
    incidents = IncidentStore(phase10_11.report_paths.incidents_path)

    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, message: str, context: dict[str, Any] | None = None) -> None:
        checks[name] = {"passed": passed, "message": message, "context": dict(context or {})}
        if not passed:
            alerts.emit(build_alert(severity="critical", category="runtime_bootstrap", message=message, context={"check": name, **dict(context or {})}))
            incidents.emit(build_incident(severity="critical", root_component="runtime_bootstrap", category="config_failure", message=message, context={"check": name, **dict(context or {})}))

    record("profile_known", profile.name in config.profiles, f"Runtime profile {profile.name} loaded.", {"runtime_profile": profile.name})
    live_backend_forbidden = profile.active_execution_backend not in set(config.runtime_safety_flags.forbid_live_backend_names)
    record("forbid_live_backend", live_backend_forbidden, "Runtime profile does not request a live backend." if live_backend_forbidden else "Runtime profile requests a forbidden live backend.", {"active_execution_backend": profile.active_execution_backend})
    broker_mode_ok = (not config.runtime_safety_flags.require_paper_broker_mode) or profile.broker_mode == "paper"
    record("broker_mode_paper", broker_mode_ok, "Runtime profile uses broker_mode=paper." if broker_mode_ok else "Runtime profile must keep broker_mode=paper.", {"broker_mode": profile.broker_mode})
    backend_match = str(phase7.execution.active_execution_backend).strip().lower() == str(profile.active_execution_backend).strip().lower()
    record(
        "backend_matches_profile",
        backend_match,
        "Active execution backend matches the runtime profile." if backend_match else "ACTIVE_EXECUTION_BACKEND does not match the selected runtime profile.",
        {"configured_backend": phase7.execution.active_execution_backend, "profile_backend": profile.active_execution_backend},
    )
    shadow_coherent = not (profile.shadow_mode_enabled and profile.paper_order_submission_enabled)
    record("shadow_submission_coherent", shadow_coherent, "Shadow mode configuration is coherent." if shadow_coherent else "shadow_mode_enabled cannot coexist with paper_order_submission_enabled.")
    release_status = show_active_release(settings)
    active_release_ok = bool(release_status.get("active_release")) if config.runtime_safety_flags.require_active_release else True
    record("active_release_present", active_release_ok, "Active release is present." if active_release_ok else "Runtime bootstrap requires a valid active release.")

    path_checks = {}
    for path_str in (
        config.deployment_paths.runtime_root,
        config.deployment_paths.runtime_log_dir,
        config.deployment_paths.deployment_snapshot_dir,
        config.deployment_paths.shadow_dir,
        config.deployment_paths.shadow_report_dir,
    ):
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        path_checks[str(path)] = path.exists()
    record("runtime_paths_ready", all(path_checks.values()), "Runtime paths are prepared.", {"paths": path_checks})

    if profile.require_safe_to_trade:
        safe_to_trade_ok = bool(settings.safe_to_trade)
        record("safe_to_trade_enabled", safe_to_trade_ok, "safe_to_trade is enabled." if safe_to_trade_ok else "Runtime profile requires safe_to_trade=true.")

    if profile.require_ibkr_paper_connection:
        preflight = preflight_check(settings, symbols=settings.supported_symbols)
        broker_ok = preflight.get("checks", {}).get("broker_reachable", {}).get("passed", False)
        record("broker_reachable", broker_ok, "Broker is reachable." if broker_ok else "Runtime profile requires IBKR Paper connectivity.", {"preflight_status": preflight.get("status")})

    status = "ok" if all(item["passed"] for item in checks.values()) else "error"
    payload = {
        "status": status,
        "runtime_profile": profile.to_dict(),
        "checks": checks,
        "deployment_paths": config.deployment_paths.to_dict(),
        "active_release": release_status.get("active_release"),
    }
    _append_runtime_event(Path(config.deployment_paths.runtime_service_log_path), {"event": "bootstrap_runtime", "timestamp": _utc_now(), "status": status, "runtime_profile": profile.name, "checks": checks})
    write_json(Path(config.deployment_paths.deployment_snapshot_dir) / f"bootstrap_{profile.name}.json", payload)
    return payload


def start_runtime(settings: Settings, *, profile_name: str | None = None) -> dict[str, Any]:
    config = load_phase12_14_config(settings)
    profile = resolve_runtime_profile(settings, profile_name=profile_name)
    bootstrap = bootstrap_runtime(settings, profile_name=profile.name)
    if bootstrap["status"] != "ok":
        return {"status": "error", "stage": "bootstrap", "bootstrap": bootstrap}
    state = {
        "status": "running",
        "runtime_profile": profile.name,
        "started_at": _utc_now(),
        "shadow_mode_enabled": profile.shadow_mode_enabled,
        "paper_order_submission_enabled": profile.paper_order_submission_enabled,
        "services": _service_map(config, running=True, scheduler_enabled=profile.scheduler_enabled),
        "active_release": show_active_release(settings).get("active_release"),
        "governance": governance_status(settings),
    }
    write_json(config.deployment_paths.runtime_state_path, state)
    write_json(config.deployment_paths.runtime_status_path, state)
    _append_runtime_event(
        Path(config.deployment_paths.deployment_registry_path),
        {
            "event": "runtime_deployment_snapshot",
            "timestamp": _utc_now(),
            "runtime_profile": profile.name,
            "active_release_id": (state.get("active_release") or {}).get("release_id"),
            "shadow_mode_enabled": state["shadow_mode_enabled"],
            "paper_order_submission_enabled": state["paper_order_submission_enabled"],
        },
    )
    _append_runtime_event(Path(config.deployment_paths.runtime_service_log_path), {"event": "start_runtime", "timestamp": _utc_now(), "runtime_profile": profile.name})
    return {"status": "ok", "runtime_state_path": config.deployment_paths.runtime_state_path, "runtime": state}


def stop_runtime(settings: Settings) -> dict[str, Any]:
    config = load_phase12_14_config(settings)
    state = _load_runtime_state(settings)
    state.update(
        {
            "status": "stopped",
            "stopped_at": _utc_now(),
            "services": _service_map(config, running=False, scheduler_enabled=False),
        }
    )
    write_json(config.deployment_paths.runtime_state_path, state)
    write_json(config.deployment_paths.runtime_status_path, state)
    _append_runtime_event(Path(config.deployment_paths.runtime_service_log_path), {"event": "stop_runtime", "timestamp": _utc_now(), "runtime_profile": state.get("runtime_profile")})
    return {"status": "ok", "runtime": state}


def restart_runtime(settings: Settings, *, profile_name: str | None = None) -> dict[str, Any]:
    stopped = stop_runtime(settings)
    started = start_runtime(settings, profile_name=profile_name)
    return {"status": started.get("status"), "stopped": stopped, "started": started}


def service_status(settings: Settings) -> dict[str, Any]:
    config = load_phase12_14_config(settings)
    state = _load_runtime_state(settings)
    return {
        "status": "ok",
        "runtime_state": state,
        "runtime_state_path": config.deployment_paths.runtime_state_path,
        "runtime_service_log_path": config.deployment_paths.runtime_service_log_path,
    }


def runtime_status(settings: Settings) -> dict[str, Any]:
    config = load_phase12_14_config(settings)
    state = _load_runtime_state(settings)
    governance = governance_status(settings)
    latest_readiness = _latest_readiness(Path(load_phase10_11_config(settings).report_paths.session_root))
    payload = {
        "status": "ok",
        "runtime_profile": state.get("runtime_profile", config.runtime_profile),
        "runtime_state": state,
        "active_release": show_active_release(settings).get("active_release"),
        "governance": governance,
        "latest_readiness": latest_readiness,
        "shadow_mode_enabled": state.get("shadow_mode_enabled"),
        "paper_order_submission_enabled": state.get("paper_order_submission_enabled"),
    }
    write_json(config.deployment_paths.runtime_status_path, payload)
    return payload


def _service_map(config, *, running: bool, scheduler_enabled: bool) -> dict[str, dict[str, Any]]:
    status = "running" if running else "stopped"
    scheduler_status = "running" if running and scheduler_enabled else "stopped"
    return {
        config.service_paths.collector_name: {"status": status},
        config.service_paths.runtime_runner_name: {"status": status},
        config.service_paths.monitor_name: {"status": status},
        config.service_paths.reconciliation_name: {"status": status},
        config.service_paths.reporting_name: {"status": status},
        config.service_paths.scheduler_name: {"status": scheduler_status},
    }


def _load_runtime_state(settings: Settings) -> dict[str, Any]:
    path = Path(load_phase12_14_config(settings).deployment_paths.runtime_state_path)
    if not path.exists():
        return {"status": "stopped", "services": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _append_runtime_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")


def _latest_readiness(session_root: Path) -> dict[str, Any] | None:
    readiness_paths = sorted(session_root.glob("session_*/readiness_report.json"))
    if not readiness_paths:
        return None
    return json.loads(readiness_paths[-1].read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
