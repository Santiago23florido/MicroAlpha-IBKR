from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from config.loader import Settings


PHASE12_14_CONFIG_FILENAME = "phase12_14.yaml"
RUNTIME_PROFILE_CONFIG_FILENAME = "runtime_profiles.yaml"


@dataclass(frozen=True)
class RuntimeProfileSpec:
    name: str
    active_execution_backend: str
    broker_mode: str
    logging_level: str
    scheduler_behavior: str
    scheduler_enabled: bool
    model_selection_mode: str
    shadow_mode_enabled: bool
    paper_order_submission_enabled: bool
    require_safe_to_trade: bool
    require_ibkr_paper_connection: bool
    forbid_live_paths: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "active_execution_backend": self.active_execution_backend,
            "broker_mode": self.broker_mode,
            "logging_level": self.logging_level,
            "scheduler_behavior": self.scheduler_behavior,
            "scheduler_enabled": self.scheduler_enabled,
            "model_selection_mode": self.model_selection_mode,
            "shadow_mode_enabled": self.shadow_mode_enabled,
            "paper_order_submission_enabled": self.paper_order_submission_enabled,
            "safety_guards": {
                "require_safe_to_trade": self.require_safe_to_trade,
                "require_ibkr_paper_connection": self.require_ibkr_paper_connection,
                "forbid_live_paths": self.forbid_live_paths,
            },
        }


@dataclass(frozen=True)
class DeploymentPathConfig:
    runtime_root: str
    runtime_state_path: str
    runtime_log_dir: str
    runtime_service_log_path: str
    deployment_snapshot_dir: str
    deployment_registry_path: str
    runtime_status_path: str
    shadow_dir: str
    shadow_intents_path: str
    shadow_report_dir: str
    release_registry_path: str
    active_release_path: str
    release_history_path: str
    promotion_audit_path: str
    rollback_audit_path: str
    governance_report_path: str
    governance_status_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_root": self.runtime_root,
            "runtime_state_path": self.runtime_state_path,
            "runtime_log_dir": self.runtime_log_dir,
            "runtime_service_log_path": self.runtime_service_log_path,
            "deployment_snapshot_dir": self.deployment_snapshot_dir,
            "deployment_registry_path": self.deployment_registry_path,
            "runtime_status_path": self.runtime_status_path,
            "shadow_dir": self.shadow_dir,
            "shadow_intents_path": self.shadow_intents_path,
            "shadow_report_dir": self.shadow_report_dir,
            "release_registry_path": self.release_registry_path,
            "active_release_path": self.active_release_path,
            "release_history_path": self.release_history_path,
            "promotion_audit_path": self.promotion_audit_path,
            "rollback_audit_path": self.rollback_audit_path,
            "governance_report_path": self.governance_report_path,
            "governance_status_path": self.governance_status_path,
        }


@dataclass(frozen=True)
class ServicePathConfig:
    collector_name: str
    runtime_runner_name: str
    monitor_name: str
    reconciliation_name: str
    reporting_name: str
    scheduler_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "collector_name": self.collector_name,
            "runtime_runner_name": self.runtime_runner_name,
            "monitor_name": self.monitor_name,
            "reconciliation_name": self.reconciliation_name,
            "reporting_name": self.reporting_name,
            "scheduler_name": self.scheduler_name,
        }


@dataclass(frozen=True)
class GovernanceThresholdConfig:
    max_open_critical_incidents: int
    max_open_critical_alerts: int
    max_recent_session_failure_rate: float
    min_readiness_status: str
    require_release_validation_summary: bool
    require_release_metrics_summary: bool
    require_release_readiness_summary: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_open_critical_incidents": self.max_open_critical_incidents,
            "max_open_critical_alerts": self.max_open_critical_alerts,
            "max_recent_session_failure_rate": self.max_recent_session_failure_rate,
            "min_readiness_status": self.min_readiness_status,
            "require_release_validation_summary": self.require_release_validation_summary,
            "require_release_metrics_summary": self.require_release_metrics_summary,
            "require_release_readiness_summary": self.require_release_readiness_summary,
        }


@dataclass(frozen=True)
class PromotionCheckConfig:
    required_fields: tuple[str, ...]
    block_on_critical_incidents: bool
    require_paper_readiness: bool
    require_validation_summary: bool
    require_metrics_summary: bool
    allow_review_needed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_fields": list(self.required_fields),
            "block_on_critical_incidents": self.block_on_critical_incidents,
            "require_paper_readiness": self.require_paper_readiness,
            "require_validation_summary": self.require_validation_summary,
            "require_metrics_summary": self.require_metrics_summary,
            "allow_review_needed": self.allow_review_needed,
        }


@dataclass(frozen=True)
class RollbackPolicyConfig:
    require_reason: bool
    keep_failed_target_active: bool
    allow_only_known_releases: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_reason": self.require_reason,
            "keep_failed_target_active": self.keep_failed_target_active,
            "allow_only_known_releases": self.allow_only_known_releases,
        }


@dataclass(frozen=True)
class RuntimeSafetyConfig:
    disable_live_trading: bool
    require_paper_broker_mode: bool
    forbid_live_backend_names: tuple[str, ...]
    require_active_release: bool
    allow_shadow_without_broker: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "disable_live_trading": self.disable_live_trading,
            "require_paper_broker_mode": self.require_paper_broker_mode,
            "forbid_live_backend_names": list(self.forbid_live_backend_names),
            "require_active_release": self.require_active_release,
            "allow_shadow_without_broker": self.allow_shadow_without_broker,
        }


@dataclass(frozen=True)
class Phase12_14Config:
    runtime_profile: str
    profiles: dict[str, RuntimeProfileSpec]
    deployment_paths: DeploymentPathConfig
    service_paths: ServicePathConfig
    governance_thresholds: GovernanceThresholdConfig
    promotion_checks: PromotionCheckConfig
    rollback_policy: RollbackPolicyConfig
    runtime_safety_flags: RuntimeSafetyConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_profile": self.runtime_profile,
            "profiles": {key: value.to_dict() for key, value in self.profiles.items()},
            "deployment_paths": self.deployment_paths.to_dict(),
            "service_paths": self.service_paths.to_dict(),
            "governance_thresholds": self.governance_thresholds.to_dict(),
            "promotion_checks": self.promotion_checks.to_dict(),
            "rollback_policy": self.rollback_policy.to_dict(),
            "runtime_safety_flags": self.runtime_safety_flags.to_dict(),
        }


def load_phase12_14_config(settings: Settings) -> Phase12_14Config:
    runtime_env = _runtime_env(settings)
    profiles_payload = _load_yaml(Path(settings.paths.config_dir) / RUNTIME_PROFILE_CONFIG_FILENAME)
    phase_payload = _load_yaml(Path(settings.paths.config_dir) / PHASE12_14_CONFIG_FILENAME)

    profile_defaults = profiles_payload.get("defaults", {})
    profile_env = profiles_payload.get("environments", {}).get(settings.environment, {})
    merged_profiles_payload = _deep_merge(profile_defaults, profile_env)
    runtime_profile = str(runtime_env.get("RUNTIME_PROFILE", merged_profiles_payload.get("runtime_profile", "development"))).strip()
    profile_specs = {
        str(name): _profile_spec_from_payload(name=str(name), payload=dict(payload))
        for name, payload in (merged_profiles_payload.get("profiles", {}) or {}).items()
    }
    if runtime_profile not in profile_specs:
        raise ValueError(f"Unknown runtime_profile={runtime_profile!r}. Available: {sorted(profile_specs)}")

    defaults = phase_payload.get("defaults", {})
    merged = _deep_merge(defaults, phase_payload.get("environments", {}).get(settings.environment, {}))

    deployment_paths_payload = merged.get("deployment_paths", {})
    service_paths_payload = merged.get("service_paths", {})
    governance_payload = merged.get("governance_thresholds", {})
    promotion_payload = merged.get("promotion_checks", {})
    rollback_payload = merged.get("rollback_policy", {})
    runtime_safety_payload = merged.get("runtime_safety_flags", {})

    return Phase12_14Config(
        runtime_profile=runtime_profile,
        profiles=profile_specs,
        deployment_paths=DeploymentPathConfig(
            runtime_root=_resolve_path(settings, runtime_env.get("RUNTIME_ROOT", str(deployment_paths_payload.get("runtime_root", "data/runtime")))),
            runtime_state_path=_resolve_path(settings, runtime_env.get("RUNTIME_STATE_PATH", str(deployment_paths_payload.get("runtime_state_path", "data/runtime/services/runtime_state.json")))),
            runtime_log_dir=_resolve_path(settings, runtime_env.get("RUNTIME_LOG_DIR", str(deployment_paths_payload.get("runtime_log_dir", "data/runtime/logs")))),
            runtime_service_log_path=_resolve_path(settings, runtime_env.get("RUNTIME_SERVICE_LOG_PATH", str(deployment_paths_payload.get("runtime_service_log_path", "data/runtime/logs/runtime_service_events.jsonl")))),
            deployment_snapshot_dir=_resolve_path(settings, runtime_env.get("DEPLOYMENT_SNAPSHOT_DIR", str(deployment_paths_payload.get("deployment_snapshot_dir", "data/runtime/deployments")))),
            deployment_registry_path=_resolve_path(settings, runtime_env.get("DEPLOYMENT_REGISTRY_PATH", str(deployment_paths_payload.get("deployment_registry_path", "data/runtime/deployments/deployment_registry.jsonl")))),
            runtime_status_path=_resolve_path(settings, runtime_env.get("RUNTIME_STATUS_PATH", str(deployment_paths_payload.get("runtime_status_path", "data/runtime/runtime_status.json")))),
            shadow_dir=_resolve_path(settings, runtime_env.get("SHADOW_DIR", str(deployment_paths_payload.get("shadow_dir", "data/reports/shadow")))),
            shadow_intents_path=_resolve_path(settings, runtime_env.get("SHADOW_INTENTS_PATH", str(deployment_paths_payload.get("shadow_intents_path", "data/reports/shadow/shadow_intents.jsonl")))),
            shadow_report_dir=_resolve_path(settings, runtime_env.get("SHADOW_REPORT_DIR", str(deployment_paths_payload.get("shadow_report_dir", "data/reports/shadow/reports")))),
            release_registry_path=_resolve_path(settings, runtime_env.get("RELEASE_REGISTRY_PATH", str(deployment_paths_payload.get("release_registry_path", "data/models/releases/release_registry.json")))),
            active_release_path=_resolve_path(settings, runtime_env.get("ACTIVE_RELEASE_PATH", str(deployment_paths_payload.get("active_release_path", "data/models/releases/active_release.json")))),
            release_history_path=_resolve_path(settings, runtime_env.get("RELEASE_HISTORY_PATH", str(deployment_paths_payload.get("release_history_path", "data/models/releases/release_history.csv")))),
            promotion_audit_path=_resolve_path(settings, runtime_env.get("PROMOTION_AUDIT_PATH", str(deployment_paths_payload.get("promotion_audit_path", "data/models/releases/promotion_audit.csv")))),
            rollback_audit_path=_resolve_path(settings, runtime_env.get("ROLLBACK_AUDIT_PATH", str(deployment_paths_payload.get("rollback_audit_path", "data/models/releases/rollback_audit.csv")))),
            governance_report_path=_resolve_path(settings, runtime_env.get("GOVERNANCE_REPORT_PATH", str(deployment_paths_payload.get("governance_report_path", "data/models/releases/release_governance_report.json")))),
            governance_status_path=_resolve_path(settings, runtime_env.get("GOVERNANCE_STATUS_PATH", str(deployment_paths_payload.get("governance_status_path", "data/runtime/governance_status.json")))),
        ),
        service_paths=ServicePathConfig(
            collector_name=str(service_paths_payload.get("collector_name", "collector")),
            runtime_runner_name=str(service_paths_payload.get("runtime_runner_name", "runtime_session_runner")),
            monitor_name=str(service_paths_payload.get("monitor_name", "paper_monitor")),
            reconciliation_name=str(service_paths_payload.get("reconciliation_name", "reconciliation")),
            reporting_name=str(service_paths_payload.get("reporting_name", "reporting")),
            scheduler_name=str(service_paths_payload.get("scheduler_name", "scheduler")),
        ),
        governance_thresholds=GovernanceThresholdConfig(
            max_open_critical_incidents=_env_int("GOVERNANCE_MAX_OPEN_CRITICAL_INCIDENTS", int(governance_payload.get("max_open_critical_incidents", 0)), runtime_env),
            max_open_critical_alerts=_env_int("GOVERNANCE_MAX_OPEN_CRITICAL_ALERTS", int(governance_payload.get("max_open_critical_alerts", 2)), runtime_env),
            max_recent_session_failure_rate=_env_float("GOVERNANCE_MAX_RECENT_SESSION_FAILURE_RATE", float(governance_payload.get("max_recent_session_failure_rate", 0.2)), runtime_env),
            min_readiness_status=str(runtime_env.get("GOVERNANCE_MIN_READINESS_STATUS", governance_payload.get("min_readiness_status", "REVIEW_NEEDED"))),
            require_release_validation_summary=_env_bool("GOVERNANCE_REQUIRE_RELEASE_VALIDATION_SUMMARY", bool(governance_payload.get("require_release_validation_summary", True)), runtime_env),
            require_release_metrics_summary=_env_bool("GOVERNANCE_REQUIRE_RELEASE_METRICS_SUMMARY", bool(governance_payload.get("require_release_metrics_summary", True)), runtime_env),
            require_release_readiness_summary=_env_bool("GOVERNANCE_REQUIRE_RELEASE_READINESS_SUMMARY", bool(governance_payload.get("require_release_readiness_summary", False)), runtime_env),
        ),
        promotion_checks=PromotionCheckConfig(
            required_fields=tuple(str(value) for value in promotion_payload.get("required_fields", ("model_name", "run_id", "artifact_dir", "feature_set_name", "target_mode"))),
            block_on_critical_incidents=_env_bool("PROMOTION_BLOCK_ON_CRITICAL_INCIDENTS", bool(promotion_payload.get("block_on_critical_incidents", True)), runtime_env),
            require_paper_readiness=_env_bool("PROMOTION_REQUIRE_PAPER_READINESS", bool(promotion_payload.get("require_paper_readiness", False)), runtime_env),
            require_validation_summary=_env_bool("PROMOTION_REQUIRE_VALIDATION_SUMMARY", bool(promotion_payload.get("require_validation_summary", True)), runtime_env),
            require_metrics_summary=_env_bool("PROMOTION_REQUIRE_METRICS_SUMMARY", bool(promotion_payload.get("require_metrics_summary", True)), runtime_env),
            allow_review_needed=_env_bool("PROMOTION_ALLOW_REVIEW_NEEDED", bool(promotion_payload.get("allow_review_needed", False)), runtime_env),
        ),
        rollback_policy=RollbackPolicyConfig(
            require_reason=_env_bool("ROLLBACK_REQUIRE_REASON", bool(rollback_payload.get("require_reason", True)), runtime_env),
            keep_failed_target_active=_env_bool("ROLLBACK_KEEP_FAILED_TARGET_ACTIVE", bool(rollback_payload.get("keep_failed_target_active", True)), runtime_env),
            allow_only_known_releases=_env_bool("ROLLBACK_ALLOW_ONLY_KNOWN_RELEASES", bool(rollback_payload.get("allow_only_known_releases", True)), runtime_env),
        ),
        runtime_safety_flags=RuntimeSafetyConfig(
            disable_live_trading=_env_bool("RUNTIME_DISABLE_LIVE_TRADING", bool(runtime_safety_payload.get("disable_live_trading", True)), runtime_env),
            require_paper_broker_mode=_env_bool("RUNTIME_REQUIRE_PAPER_BROKER_MODE", bool(runtime_safety_payload.get("require_paper_broker_mode", True)), runtime_env),
            forbid_live_backend_names=tuple(str(value) for value in runtime_safety_payload.get("forbid_live_backend_names", ("ibkr_live", "live", "real_money"))),
            require_active_release=_env_bool("RUNTIME_REQUIRE_ACTIVE_RELEASE", bool(runtime_safety_payload.get("require_active_release", True)), runtime_env),
            allow_shadow_without_broker=_env_bool("RUNTIME_ALLOW_SHADOW_WITHOUT_BROKER", bool(runtime_safety_payload.get("allow_shadow_without_broker", True)), runtime_env),
        ),
    )


def resolve_runtime_profile(settings: Settings, *, profile_name: str | None = None) -> RuntimeProfileSpec:
    config = load_phase12_14_config(settings)
    target = str(profile_name or config.runtime_profile).strip()
    if target not in config.profiles:
        raise ValueError(f"Unknown runtime profile {target!r}. Available: {sorted(config.profiles)}")
    return config.profiles[target]


def _profile_spec_from_payload(*, name: str, payload: dict[str, Any]) -> RuntimeProfileSpec:
    safety = payload.get("safety_guards", {}) or {}
    return RuntimeProfileSpec(
        name=name,
        active_execution_backend=str(payload.get("active_execution_backend", "mock")),
        broker_mode=str(payload.get("broker_mode", "paper")),
        logging_level=str(payload.get("logging_level", "INFO")),
        scheduler_behavior=str(payload.get("scheduler_behavior", "manual")),
        scheduler_enabled=bool(payload.get("scheduler_enabled", False)),
        model_selection_mode=str(payload.get("model_selection_mode", "active_release")),
        shadow_mode_enabled=bool(payload.get("shadow_mode_enabled", False)),
        paper_order_submission_enabled=bool(payload.get("paper_order_submission_enabled", False)),
        require_safe_to_trade=bool(safety.get("require_safe_to_trade", False)),
        require_ibkr_paper_connection=bool(safety.get("require_ibkr_paper_connection", False)),
        forbid_live_paths=bool(safety.get("forbid_live_paths", True)),
    )


def _runtime_env(settings: Settings) -> dict[str, str]:
    env_path = Path(settings.env_file) if settings.env_file else None
    file_env = {
        key: str(value)
        for key, value in dotenv_values(env_path if env_path and env_path.exists() else None).items()
        if value is not None
    }
    return {**file_env, **os.environ}


def _load_yaml(path: Path) -> dict[str, Any]:
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


def _env_bool(name: str, default: bool, env: dict[str, str]) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


def _env_int(name: str, default: int, env: dict[str, str]) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float, env: dict[str, str]) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    return float(raw)
