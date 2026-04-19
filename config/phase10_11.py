from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from config.loader import Settings


PHASE10_11_CONFIG_FILENAME = "phase10_11.yaml"


@dataclass(frozen=True)
class ValidationSessionLimitsConfig:
    max_sessions_to_compare: int
    max_monitor_iterations: int
    max_session_runtime_minutes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_sessions_to_compare": self.max_sessions_to_compare,
            "max_monitor_iterations": self.max_monitor_iterations,
            "max_session_runtime_minutes": self.max_session_runtime_minutes,
        }


@dataclass(frozen=True)
class ReconciliationToleranceConfig:
    quantity_tolerance: float
    average_price_tolerance: float
    pnl_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantity_tolerance": self.quantity_tolerance,
            "average_price_tolerance": self.average_price_tolerance,
            "pnl_tolerance": self.pnl_tolerance,
        }


@dataclass(frozen=True)
class AlertThresholdConfig:
    max_order_rejection_rate: float
    max_partial_fill_rate: float
    max_cancel_rate: float
    max_risk_block_rate: float
    max_alerts_per_session: int
    max_disconnects: int
    max_stale_data_seconds: int
    max_submit_to_fill_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_order_rejection_rate": self.max_order_rejection_rate,
            "max_partial_fill_rate": self.max_partial_fill_rate,
            "max_cancel_rate": self.max_cancel_rate,
            "max_risk_block_rate": self.max_risk_block_rate,
            "max_alerts_per_session": self.max_alerts_per_session,
            "max_disconnects": self.max_disconnects,
            "max_stale_data_seconds": self.max_stale_data_seconds,
            "max_submit_to_fill_latency_ms": self.max_submit_to_fill_latency_ms,
        }


@dataclass(frozen=True)
class ReadinessThresholdConfig:
    max_allowed_mismatches: int
    max_allowed_disconnects: int
    max_allowed_alerts: int
    max_allowed_latency_ms: float
    max_allowed_drawdown: float
    min_required_sessions_for_readiness: int
    max_session_failure_rate: float
    max_critical_alerts: int
    max_drift_psi: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_allowed_mismatches": self.max_allowed_mismatches,
            "max_allowed_disconnects": self.max_allowed_disconnects,
            "max_allowed_alerts": self.max_allowed_alerts,
            "max_allowed_latency_ms": self.max_allowed_latency_ms,
            "max_allowed_drawdown": self.max_allowed_drawdown,
            "min_required_sessions_for_readiness": self.min_required_sessions_for_readiness,
            "max_session_failure_rate": self.max_session_failure_rate,
            "max_critical_alerts": self.max_critical_alerts,
            "max_drift_psi": self.max_drift_psi,
        }


@dataclass(frozen=True)
class MonitoringIntervalConfig:
    broker_healthcheck_seconds: int
    reconciliation_seconds: int
    data_freshness_seconds: int
    monitoring_sleep_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "broker_healthcheck_seconds": self.broker_healthcheck_seconds,
            "reconciliation_seconds": self.reconciliation_seconds,
            "data_freshness_seconds": self.data_freshness_seconds,
            "monitoring_sleep_seconds": self.monitoring_sleep_seconds,
        }


@dataclass(frozen=True)
class SchedulerIntervalConfig:
    preflight_delay_seconds: float
    scheduled_healthcheck_seconds: int
    scheduled_reconciliation_seconds: int
    scheduled_reporting_seconds: int
    default_monitor_iterations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "preflight_delay_seconds": self.preflight_delay_seconds,
            "scheduled_healthcheck_seconds": self.scheduled_healthcheck_seconds,
            "scheduled_reconciliation_seconds": self.scheduled_reconciliation_seconds,
            "scheduled_reporting_seconds": self.scheduled_reporting_seconds,
            "default_monitor_iterations": self.default_monitor_iterations,
        }


@dataclass(frozen=True)
class ReportPathConfig:
    session_root: str
    archive_root: str
    alerts_path: str
    incidents_path: str
    recovery_events_path: str
    registry_path: str
    health_report_path: str
    runbooks_dir: str
    reconciliation_dir: str
    comparison_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_root": self.session_root,
            "archive_root": self.archive_root,
            "alerts_path": self.alerts_path,
            "incidents_path": self.incidents_path,
            "recovery_events_path": self.recovery_events_path,
            "registry_path": self.registry_path,
            "health_report_path": self.health_report_path,
            "runbooks_dir": self.runbooks_dir,
            "reconciliation_dir": self.reconciliation_dir,
            "comparison_dir": self.comparison_dir,
        }


@dataclass(frozen=True)
class ArchivalPolicyConfig:
    enabled: bool
    archive_completed_sessions: bool
    retention_days: int
    session_retention_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "archive_completed_sessions": self.archive_completed_sessions,
            "retention_days": self.retention_days,
            "session_retention_policy": self.session_retention_policy,
        }


@dataclass(frozen=True)
class GateConfig:
    require_ibkr_paper_backend: bool
    block_on_critical_alerts: bool
    block_on_reconciliation_mismatch: bool
    block_on_not_ready: bool
    block_on_broker_mode_ambiguity: bool
    block_on_high_failure_rate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_ibkr_paper_backend": self.require_ibkr_paper_backend,
            "block_on_critical_alerts": self.block_on_critical_alerts,
            "block_on_reconciliation_mismatch": self.block_on_reconciliation_mismatch,
            "block_on_not_ready": self.block_on_not_ready,
            "block_on_broker_mode_ambiguity": self.block_on_broker_mode_ambiguity,
            "block_on_high_failure_rate": self.block_on_high_failure_rate,
        }


@dataclass(frozen=True)
class RecoveryConfig:
    reconnect_attempt_limits: int
    monitor_restart_limit: int
    allow_resume_if_safe: bool
    abort_on_duplicate_submission_risk: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconnect_attempt_limits": self.reconnect_attempt_limits,
            "monitor_restart_limit": self.monitor_restart_limit,
            "allow_resume_if_safe": self.allow_resume_if_safe,
            "abort_on_duplicate_submission_risk": self.abort_on_duplicate_submission_risk,
        }


@dataclass(frozen=True)
class OperationalCheckConfig:
    preflight_required_checks: tuple[str, ...]
    postflight_required_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preflight_required_checks": list(self.preflight_required_checks),
            "postflight_required_checks": list(self.postflight_required_checks),
        }


@dataclass(frozen=True)
class Phase10_11Config:
    validation_session_limits: ValidationSessionLimitsConfig
    reconciliation_tolerances: ReconciliationToleranceConfig
    alert_thresholds: AlertThresholdConfig
    readiness_thresholds: ReadinessThresholdConfig
    monitoring_intervals: MonitoringIntervalConfig
    scheduler_intervals: SchedulerIntervalConfig
    report_paths: ReportPathConfig
    archival_policy: ArchivalPolicyConfig
    gates: GateConfig
    recovery: RecoveryConfig
    checks: OperationalCheckConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_session_limits": self.validation_session_limits.to_dict(),
            "reconciliation_tolerances": self.reconciliation_tolerances.to_dict(),
            "alert_thresholds": self.alert_thresholds.to_dict(),
            "readiness_thresholds": self.readiness_thresholds.to_dict(),
            "monitoring_intervals": self.monitoring_intervals.to_dict(),
            "scheduler_intervals": self.scheduler_intervals.to_dict(),
            "report_paths": self.report_paths.to_dict(),
            "archival_policy": self.archival_policy.to_dict(),
            "gates": self.gates.to_dict(),
            "recovery": self.recovery.to_dict(),
            "checks": self.checks.to_dict(),
        }


def load_phase10_11_config(settings: Settings) -> Phase10_11Config:
    runtime_env = _runtime_env(settings)
    _env_bool_local = lambda name, default: _env_bool(name, default, runtime_env)
    _env_int_local = lambda name, default: _env_int(name, default, runtime_env)
    _env_float_local = lambda name, default: _env_float(name, default, runtime_env)
    payload = _load_yaml(Path(settings.paths.config_dir) / PHASE10_11_CONFIG_FILENAME)
    defaults = payload.get("defaults", {})
    merged = _deep_merge(defaults, payload.get("environments", {}).get(settings.environment, {}))

    session_payload = merged.get("validation_session_limits", {})
    recon_payload = merged.get("reconciliation_tolerances", {})
    alert_payload = merged.get("alert_thresholds", {})
    readiness_payload = merged.get("readiness_thresholds", {})
    monitoring_payload = merged.get("monitoring_intervals", {})
    scheduler_payload = merged.get("scheduler_intervals", {})
    report_payload = merged.get("report_paths", {})
    archival_payload = merged.get("archival_policy", {})
    gate_payload = merged.get("gates", {})
    check_payload = merged.get("checks", {})
    recovery_payload = merged.get("recovery", {})

    return Phase10_11Config(
        validation_session_limits=ValidationSessionLimitsConfig(
            max_sessions_to_compare=_env_int_local(
                "VALIDATION_MAX_SESSIONS_TO_COMPARE",
                int(session_payload.get("max_sessions_to_compare", 25)),
            ),
            max_monitor_iterations=_env_int_local(
                "VALIDATION_MAX_MONITOR_ITERATIONS",
                int(session_payload.get("max_monitor_iterations", 10)),
            ),
            max_session_runtime_minutes=_env_int_local(
                "VALIDATION_MAX_SESSION_RUNTIME_MINUTES",
                int(session_payload.get("max_session_runtime_minutes", 240)),
            ),
        ),
        reconciliation_tolerances=ReconciliationToleranceConfig(
            quantity_tolerance=_env_float_local(
                "RECONCILIATION_QUANTITY_TOLERANCE",
                float(recon_payload.get("quantity_tolerance", 0.0)),
            ),
            average_price_tolerance=_env_float_local(
                "RECONCILIATION_AVG_PRICE_TOLERANCE",
                float(recon_payload.get("average_price_tolerance", 0.05)),
            ),
            pnl_tolerance=_env_float_local(
                "RECONCILIATION_PNL_TOLERANCE",
                float(recon_payload.get("pnl_tolerance", 5.0)),
            ),
        ),
        alert_thresholds=AlertThresholdConfig(
            max_order_rejection_rate=_env_float_local(
                "VALIDATION_MAX_ORDER_REJECTION_RATE",
                float(alert_payload.get("max_order_rejection_rate", 0.25)),
            ),
            max_partial_fill_rate=_env_float_local(
                "VALIDATION_MAX_PARTIAL_FILL_RATE",
                float(alert_payload.get("max_partial_fill_rate", 0.50)),
            ),
            max_cancel_rate=_env_float_local(
                "VALIDATION_MAX_CANCEL_RATE",
                float(alert_payload.get("max_cancel_rate", 0.25)),
            ),
            max_risk_block_rate=_env_float_local(
                "VALIDATION_MAX_RISK_BLOCK_RATE",
                float(alert_payload.get("max_risk_block_rate", 0.80)),
            ),
            max_alerts_per_session=_env_int_local(
                "VALIDATION_MAX_ALERTS_PER_SESSION",
                int(alert_payload.get("max_alerts_per_session", 25)),
            ),
            max_disconnects=_env_int_local(
                "VALIDATION_MAX_ALLOWED_DISCONNECTS",
                int(alert_payload.get("max_disconnects", 2)),
            ),
            max_stale_data_seconds=_env_int_local(
                "VALIDATION_MAX_STALE_DATA_SECONDS",
                int(alert_payload.get("max_stale_data_seconds", 900)),
            ),
            max_submit_to_fill_latency_ms=_env_float_local(
                "VALIDATION_MAX_SUBMIT_TO_FILL_LATENCY_MS",
                float(alert_payload.get("max_submit_to_fill_latency_ms", 15000.0)),
            ),
        ),
        readiness_thresholds=ReadinessThresholdConfig(
            max_allowed_mismatches=_env_int_local(
                "READINESS_MAX_ALLOWED_MISMATCHES",
                int(readiness_payload.get("max_allowed_mismatches", 0)),
            ),
            max_allowed_disconnects=_env_int_local(
                "READINESS_MAX_ALLOWED_DISCONNECTS",
                int(readiness_payload.get("max_allowed_disconnects", 1)),
            ),
            max_allowed_alerts=_env_int_local(
                "READINESS_MAX_ALLOWED_ALERTS",
                int(readiness_payload.get("max_allowed_alerts", 10)),
            ),
            max_allowed_latency_ms=_env_float_local(
                "READINESS_MAX_ALLOWED_LATENCY_MS",
                float(readiness_payload.get("max_allowed_latency_ms", 15000.0)),
            ),
            max_allowed_drawdown=_env_float_local(
                "READINESS_MAX_ALLOWED_DRAWDOWN",
                float(readiness_payload.get("max_allowed_drawdown", 1000.0)),
            ),
            min_required_sessions_for_readiness=_env_int_local(
                "READINESS_MIN_REQUIRED_SESSIONS",
                int(readiness_payload.get("min_required_sessions_for_readiness", 3)),
            ),
            max_session_failure_rate=_env_float_local(
                "READINESS_MAX_SESSION_FAILURE_RATE",
                float(readiness_payload.get("max_session_failure_rate", 0.25)),
            ),
            max_critical_alerts=_env_int_local(
                "READINESS_MAX_CRITICAL_ALERTS",
                int(readiness_payload.get("max_critical_alerts", 0)),
            ),
            max_drift_psi=_env_float_local(
                "READINESS_MAX_DRIFT_PSI",
                float(readiness_payload.get("max_drift_psi", 0.35)),
            ),
        ),
        monitoring_intervals=MonitoringIntervalConfig(
            broker_healthcheck_seconds=_env_int_local(
                "MONITOR_BROKER_HEALTHCHECK_SECONDS",
                int(monitoring_payload.get("broker_healthcheck_seconds", 60)),
            ),
            reconciliation_seconds=_env_int_local(
                "MONITOR_RECONCILIATION_SECONDS",
                int(monitoring_payload.get("reconciliation_seconds", 120)),
            ),
            data_freshness_seconds=_env_int_local(
                "MONITOR_DATA_FRESHNESS_SECONDS",
                int(monitoring_payload.get("data_freshness_seconds", 900)),
            ),
            monitoring_sleep_seconds=_env_float_local(
                "MONITOR_SLEEP_SECONDS",
                float(monitoring_payload.get("monitoring_sleep_seconds", 1.0)),
            ),
        ),
        scheduler_intervals=SchedulerIntervalConfig(
            preflight_delay_seconds=_env_float_local(
                "SCHEDULER_PREFLIGHT_DELAY_SECONDS",
                float(scheduler_payload.get("preflight_delay_seconds", 0.0)),
            ),
            scheduled_healthcheck_seconds=_env_int_local(
                "SCHEDULER_HEALTHCHECK_SECONDS",
                int(scheduler_payload.get("scheduled_healthcheck_seconds", 60)),
            ),
            scheduled_reconciliation_seconds=_env_int_local(
                "SCHEDULER_RECONCILIATION_SECONDS",
                int(scheduler_payload.get("scheduled_reconciliation_seconds", 300)),
            ),
            scheduled_reporting_seconds=_env_int_local(
                "SCHEDULER_REPORTING_SECONDS",
                int(scheduler_payload.get("scheduled_reporting_seconds", 300)),
            ),
            default_monitor_iterations=_env_int_local(
                "SCHEDULER_DEFAULT_MONITOR_ITERATIONS",
                int(scheduler_payload.get("default_monitor_iterations", 1)),
            ),
        ),
        report_paths=ReportPathConfig(
            session_root=_resolve_path(
                settings,
                str(report_payload.get("session_root", runtime_env.get("VALIDATION_SESSION_ROOT", "data/reports/sessions"))),
            ),
            archive_root=_resolve_path(
                settings,
                str(report_payload.get("archive_root", runtime_env.get("VALIDATION_ARCHIVE_ROOT", "data/reports/archive"))),
            ),
            alerts_path=_resolve_path(
                settings,
                str(report_payload.get("alerts_path", runtime_env.get("VALIDATION_ALERTS_PATH", "data/reports/validation/alerts.jsonl"))),
            ),
            incidents_path=_resolve_path(
                settings,
                str(report_payload.get("incidents_path", runtime_env.get("VALIDATION_INCIDENTS_PATH", "data/reports/validation/incidents.jsonl"))),
            ),
            recovery_events_path=_resolve_path(
                settings,
                str(
                    report_payload.get(
                        "recovery_events_path",
                        runtime_env.get("VALIDATION_RECOVERY_EVENTS_PATH", "data/reports/validation/recovery_events.jsonl"),
                    )
                ),
            ),
            registry_path=_resolve_path(
                settings,
                str(report_payload.get("registry_path", runtime_env.get("VALIDATION_REGISTRY_PATH", "data/reports/validation/session_registry.jsonl"))),
            ),
            health_report_path=_resolve_path(
                settings,
                str(
                    report_payload.get(
                        "health_report_path",
                        runtime_env.get("VALIDATION_SYSTEM_HEALTH_PATH", "data/reports/validation/system_health_latest.json"),
                    )
                ),
            ),
            runbooks_dir=_resolve_path(
                settings,
                str(report_payload.get("runbooks_dir", runtime_env.get("VALIDATION_RUNBOOKS_DIR", "docs/runbooks"))),
            ),
            reconciliation_dir=_resolve_path(
                settings,
                str(
                    report_payload.get(
                        "reconciliation_dir",
                        runtime_env.get("VALIDATION_RECONCILIATION_DIR", "data/reports/reconciliation"),
                    )
                ),
            ),
            comparison_dir=_resolve_path(
                settings,
                str(report_payload.get("comparison_dir", runtime_env.get("VALIDATION_COMPARISON_DIR", "data/reports/validation/comparisons"))),
            ),
        ),
        archival_policy=ArchivalPolicyConfig(
            enabled=_env_bool_local(
                "VALIDATION_ARCHIVE_ENABLED",
                bool(archival_payload.get("enabled", True)),
            ),
            archive_completed_sessions=_env_bool_local(
                "VALIDATION_ARCHIVE_COMPLETED_SESSIONS",
                bool(archival_payload.get("archive_completed_sessions", True)),
            ),
            retention_days=_env_int_local(
                "VALIDATION_RETENTION_DAYS",
                int(archival_payload.get("retention_days", 30)),
            ),
            session_retention_policy=str(
                runtime_env.get(
                    "VALIDATION_SESSION_RETENTION_POLICY",
                    archival_payload.get("session_retention_policy", "keep_recent_and_archive"),
                )
            ).strip(),
        ),
        gates=GateConfig(
            require_ibkr_paper_backend=_env_bool_local(
                "GATE_REQUIRE_IBKR_PAPER_BACKEND",
                bool(gate_payload.get("require_ibkr_paper_backend", True)),
            ),
            block_on_critical_alerts=_env_bool_local(
                "GATE_BLOCK_ON_CRITICAL_ALERTS",
                bool(gate_payload.get("block_on_critical_alerts", True)),
            ),
            block_on_reconciliation_mismatch=_env_bool_local(
                "GATE_BLOCK_ON_RECONCILIATION_MISMATCH",
                bool(gate_payload.get("block_on_reconciliation_mismatch", True)),
            ),
            block_on_not_ready=_env_bool_local(
                "GATE_BLOCK_ON_NOT_READY",
                bool(gate_payload.get("block_on_not_ready", True)),
            ),
            block_on_broker_mode_ambiguity=_env_bool_local(
                "GATE_BLOCK_ON_BROKER_MODE_AMBIGUITY",
                bool(gate_payload.get("block_on_broker_mode_ambiguity", True)),
            ),
            block_on_high_failure_rate=_env_bool_local(
                "GATE_BLOCK_ON_HIGH_FAILURE_RATE",
                bool(gate_payload.get("block_on_high_failure_rate", True)),
            ),
        ),
        recovery=RecoveryConfig(
            reconnect_attempt_limits=_env_int_local(
                "RECOVERY_RECONNECT_ATTEMPT_LIMITS",
                int(recovery_payload.get("reconnect_attempt_limits", 2)),
            ),
            monitor_restart_limit=_env_int_local(
                "RECOVERY_MONITOR_RESTART_LIMIT",
                int(recovery_payload.get("monitor_restart_limit", 1)),
            ),
            allow_resume_if_safe=_env_bool_local(
                "RECOVERY_ALLOW_RESUME_IF_SAFE",
                bool(recovery_payload.get("allow_resume_if_safe", False)),
            ),
            abort_on_duplicate_submission_risk=_env_bool_local(
                "RECOVERY_ABORT_ON_DUPLICATE_SUBMISSION_RISK",
                bool(recovery_payload.get("abort_on_duplicate_submission_risk", True)),
            ),
        ),
        checks=OperationalCheckConfig(
            preflight_required_checks=_env_csv(
                "PREFLIGHT_REQUIRED_CHECKS",
                tuple(
                    str(value)
                    for value in check_payload.get(
                        "preflight_required_checks",
                        (
                            "active_model_loadable",
                            "execution_backend_real",
                            "broker_mode_paper",
                            "safe_to_trade_enabled",
                            "allow_session_execution",
                            "required_paths_writable",
                            "symbol_list_valid",
                            "broker_reachable",
                        ),
                    )
                ),
                runtime_env,
            ),
            postflight_required_checks=_env_csv(
                "POSTFLIGHT_REQUIRED_CHECKS",
                tuple(
                    str(value)
                    for value in check_payload.get(
                        "postflight_required_checks",
                        (
                            "reports_created",
                            "reconciliation_completed",
                            "alerts_summarized",
                            "archive_completed",
                        ),
                    )
                ),
                runtime_env,
            ),
        ),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(settings: Settings, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = Path(settings.paths.project_root) / path
    return str(path.resolve())


def _env_bool(name: str, default: bool, env: dict[str, str] | None = None) -> bool:
    raw_value = (env or os.environ).get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw_value!r}.")


def _env_int(name: str, default: int, env: dict[str, str] | None = None) -> int:
    raw_value = (env or os.environ).get(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _env_float(name: str, default: float, env: dict[str, str] | None = None) -> float:
    raw_value = (env or os.environ).get(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _env_csv(name: str, default: tuple[str, ...], env: dict[str, str]) -> tuple[str, ...]:
    raw_value = env.get(name)
    if raw_value is None:
        return tuple(default)
    parts = tuple(part.strip() for part in str(raw_value).split(",") if part.strip())
    return parts or tuple(default)


def _runtime_env(settings: Settings) -> dict[str, str]:
    file_env = {
        key: str(value)
        for key, value in dotenv_values(settings.env_file if settings.env_file else None).items()
        if value is not None
    }
    return {**file_env, **os.environ}
