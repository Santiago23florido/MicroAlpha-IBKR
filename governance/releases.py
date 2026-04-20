from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings
from config.phase10_11 import load_phase10_11_config
from config.phase12_14 import load_phase12_14_config
from config.phase6 import (
    load_active_model_selection,
    required_phase5_artifact_files,
    resolve_phase5_artifact,
    set_active_model_selection,
)
from evaluation.io import read_json, write_json


@dataclass
class ReleaseRegistry:
    active_release_id: str | None
    releases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"active_release_id": self.active_release_id, "releases": list(self.releases)}


def list_model_releases(settings: Settings) -> dict[str, Any]:
    registry = _load_or_bootstrap_registry(settings)
    _persist_release_reports(settings, registry)
    return {
        "status": "ok",
        "active_release_id": registry.active_release_id,
        "release_count": len(registry.releases),
        "releases": sorted(registry.releases, key=lambda item: str(item.get("created_at") or ""), reverse=True),
    }


def show_active_release(settings: Settings) -> dict[str, Any]:
    registry = _load_or_bootstrap_registry(settings)
    active = _get_active_release(registry)
    _persist_release_reports(settings, registry)
    return {
        "status": "ok" if active is not None else "error",
        "active_release": active,
        "active_release_id": registry.active_release_id,
        "active_model": load_active_model_selection(settings).to_dict(),
    }


def promote_model_release(
    settings: Settings,
    *,
    model_name: str | None = None,
    run_id: str | None = None,
    release_id: str | None = None,
    actor: str = "cli",
    reason: str = "manual promotion",
) -> dict[str, Any]:
    registry = _load_or_bootstrap_registry(settings)
    target = _select_release(registry, model_name=model_name, run_id=run_id, release_id=release_id)
    governance = _build_governance_report(settings, registry=registry, target_release=target, operation="promote")
    if governance["governance_decision"] != "APPROVED":
        _write_governance_report(settings, governance)
        return {"status": "error", "message": "Promotion blocked by governance checks.", "governance": governance}

    previous_active = _get_active_release(registry)
    now = _utc_now()
    for release in registry.releases:
        if release["release_id"] == target["release_id"]:
            release["status"] = "active"
            release["promoted_at"] = now
            release["promoted_by"] = actor
        elif previous_active and release["release_id"] == previous_active["release_id"]:
            release["status"] = "approved"
    registry.active_release_id = target["release_id"]
    set_active_model_selection(settings, run_id=target["run_id"])
    audit = {
        "timestamp": now,
        "event_type": "promotion",
        "previous_active_release_id": previous_active["release_id"] if previous_active else None,
        "new_active_release_id": target["release_id"],
        "previous_active_run_id": previous_active["run_id"] if previous_active else None,
        "new_active_run_id": target["run_id"],
        "actor": actor,
        "reason": reason,
        "success": True,
    }
    _write_registry(settings, registry)
    _append_audit_csv(Path(load_phase12_14_config(settings).deployment_paths.promotion_audit_path), audit)
    _persist_release_reports(settings, registry)
    _write_governance_report(settings, governance)
    return {"status": "ok", "active_release": _get_active_release(registry), "audit": audit, "governance": governance}


def rollback_model_release(
    settings: Settings,
    *,
    to: str,
    actor: str = "cli",
    reason: str = "manual rollback",
) -> dict[str, Any]:
    config = load_phase12_14_config(settings)
    if config.rollback_policy.require_reason and not str(reason).strip():
        raise ValueError("Rollback requires a non-empty reason.")
    registry = _load_or_bootstrap_registry(settings)
    current = _get_active_release(registry)
    if current is None:
        raise ValueError("Cannot rollback without an active release.")
    target = _select_release(registry, release_id=to, run_id=to)
    if target["release_id"] == current["release_id"]:
        raise ValueError("Rollback target is already the active release.")

    required_files = required_phase5_artifact_files(resolve_phase5_artifact(settings, run_id=target["run_id"], artifact_dir=target["artifact_dir"]))
    if not all(item["exists"] for item in required_files.values()):
        raise ValueError("Rollback target release is incomplete and cannot be activated.")

    now = _utc_now()
    for release in registry.releases:
        if release["release_id"] == target["release_id"]:
            release["status"] = "active"
            release["promoted_at"] = now
            release["promoted_by"] = actor
        elif release["release_id"] == current["release_id"]:
            release["status"] = "rollback"
    registry.active_release_id = target["release_id"]
    set_active_model_selection(settings, run_id=target["run_id"])
    audit = {
        "timestamp": now,
        "event_type": "rollback",
        "from_release_id": current["release_id"],
        "to_release_id": target["release_id"],
        "from_run_id": current["run_id"],
        "to_run_id": target["run_id"],
        "actor": actor,
        "reason": reason,
        "success": True,
    }
    _write_registry(settings, registry)
    _append_audit_csv(Path(config.deployment_paths.rollback_audit_path), audit)
    _persist_release_reports(settings, registry)
    return {"status": "ok", "active_release": _get_active_release(registry), "audit": audit}


def governance_status(settings: Settings) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    config = load_phase12_14_config(settings)
    registry = _load_or_bootstrap_registry(settings)
    active = _get_active_release(registry)
    incident_count = _count_csv_rows(Path(phase10_11.report_paths.incidents_path), severity="critical")
    alert_count = _count_jsonl_severity(Path(phase10_11.report_paths.alerts_path), severity="critical")
    latest_readiness = _latest_readiness_status(Path(phase10_11.report_paths.session_root))
    payload = {
        "status": "ok",
        "runtime_profile": config.runtime_profile,
        "active_release": active,
        "critical_incident_count": incident_count,
        "critical_alert_count": alert_count,
        "latest_readiness_status": latest_readiness,
        "latest_promotion": _latest_csv_row(Path(config.deployment_paths.promotion_audit_path)),
        "latest_rollback": _latest_csv_row(Path(config.deployment_paths.rollback_audit_path)),
        "promotion_audit_path": config.deployment_paths.promotion_audit_path,
        "rollback_audit_path": config.deployment_paths.rollback_audit_path,
        "governance_report_path": config.deployment_paths.governance_report_path,
    }
    write_json(config.deployment_paths.governance_status_path, payload)
    return payload


def _load_or_bootstrap_registry(settings: Settings) -> ReleaseRegistry:
    config = load_phase12_14_config(settings)
    path = Path(config.deployment_paths.release_registry_path)
    if path.exists():
        payload = read_json(path)
        releases = list(payload.get("releases", []))
        active_release_id = payload.get("active_release_id")
    else:
        releases = []
        active_release_id = None

    discovered = _discover_releases(settings)
    existing_by_id = {release["release_id"]: release for release in releases}
    for release in discovered:
        existing = existing_by_id.get(release["release_id"])
        if existing is None:
            releases.append(release)
        else:
            for key, value in release.items():
                existing.setdefault(key, value)

    selection = load_active_model_selection(settings)
    active_release_id = active_release_id or f"rel_{selection.run_id}"
    for release in releases:
        if release["release_id"] == active_release_id:
            release["status"] = "active"
        elif release.get("status") == "active":
            release["status"] = "approved"
    registry = ReleaseRegistry(active_release_id=active_release_id, releases=releases)
    _write_registry(settings, registry)
    return registry


def _discover_releases(settings: Settings) -> list[dict[str, Any]]:
    from config.phase6 import discover_phase5_artifacts

    releases: list[dict[str, Any]] = []
    for candidate in discover_phase5_artifacts(settings):
        metrics = candidate.get("validation_metrics") or candidate.get("test_metrics") or {}
        releases.append(
            {
                "release_id": f"rel_{candidate['run_id']}",
                "run_id": candidate["run_id"],
                "model_name": candidate["model_name"],
                "model_type": candidate["model_type"],
                "artifact_dir": candidate["artifact_dir"],
                "feature_set_name": candidate["feature_set_name"],
                "target_mode": candidate["target_mode"],
                "training_metadata": candidate.get("training_metadata", {}),
                "metrics_summary": {
                    "validation_metrics": candidate.get("validation_metrics", {}),
                    "test_metrics": candidate.get("test_metrics", {}),
                    "ranking_score": candidate.get("ranking_score"),
                },
                "validation_summary": metrics,
                "readiness_summary": {},
                "created_at": candidate.get("timestamp_utc") or _utc_now(),
                "promoted_at": None,
                "promoted_by": None,
                "status": "candidate",
                "release_notes": f"Imported from Phase 5 run {candidate['run_id']}.",
            }
        )
    return releases


def _build_governance_report(
    settings: Settings,
    *,
    registry: ReleaseRegistry,
    target_release: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    config = load_phase12_14_config(settings)
    phase10_11 = load_phase10_11_config(settings)
    required_files = required_phase5_artifact_files(resolve_phase5_artifact(settings, run_id=target_release["run_id"], artifact_dir=target_release["artifact_dir"]))
    checks = {
        "artifact_complete": all(item["exists"] for item in required_files.values()),
        "required_fields_present": all(bool(target_release.get(field)) for field in config.promotion_checks.required_fields),
        "validation_summary_available": bool(target_release.get("validation_summary")),
        "metrics_summary_available": bool(target_release.get("metrics_summary")),
        "broker_mode_safe": str(settings.broker_mode).strip().lower() == "paper",
        "no_critical_incidents_open": _count_csv_rows(Path(phase10_11.report_paths.incidents_path), severity="critical") <= config.governance_thresholds.max_open_critical_incidents,
        "no_critical_alerts_open": _count_jsonl_severity(Path(phase10_11.report_paths.alerts_path), severity="critical") <= config.governance_thresholds.max_open_critical_alerts,
    }
    if config.promotion_checks.require_paper_readiness:
        checks["readiness_summary_available"] = bool(target_release.get("readiness_summary"))
    failed = [name for name, passed in checks.items() if not passed]
    if any(name in failed for name in ("artifact_complete", "required_fields_present", "broker_mode_safe")):
        decision = "BLOCKED"
    elif failed:
        decision = "REVIEW_NEEDED"
    else:
        decision = "APPROVED"
    governance = {
        "status": "ok",
        "operation": operation,
        "target_release_id": target_release["release_id"],
        "target_run_id": target_release["run_id"],
        "checks": checks,
        "failed_checks": failed,
        "governance_decision": decision,
        "evaluated_at": _utc_now(),
        "active_release_id": registry.active_release_id,
    }
    return governance


def _write_governance_report(settings: Settings, payload: dict[str, Any]) -> None:
    write_json(load_phase12_14_config(settings).deployment_paths.governance_report_path, payload)


def _persist_release_reports(settings: Settings, registry: ReleaseRegistry) -> None:
    config = load_phase12_14_config(settings)
    active_release = _get_active_release(registry)
    write_json(config.deployment_paths.active_release_path, {"active_release": active_release, "active_release_id": registry.active_release_id})
    _write_registry(settings, registry)
    _write_release_history_csv(Path(config.deployment_paths.release_history_path), registry.releases)


def _write_registry(settings: Settings, registry: ReleaseRegistry) -> None:
    write_json(load_phase12_14_config(settings).deployment_paths.release_registry_path, registry.to_dict())


def _write_release_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "release_id",
        "run_id",
        "model_name",
        "model_type",
        "feature_set_name",
        "target_mode",
        "status",
        "created_at",
        "promoted_at",
        "promoted_by",
        "artifact_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item.get("created_at") or "")):
            writer.writerow({field: row.get(field) for field in fieldnames})


def _append_audit_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(payload)


def _get_active_release(registry: ReleaseRegistry) -> dict[str, Any] | None:
    for release in registry.releases:
        if release["release_id"] == registry.active_release_id:
            return release
    return None


def _select_release(
    registry: ReleaseRegistry,
    *,
    model_name: str | None = None,
    run_id: str | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    releases = registry.releases
    if release_id:
        for release in releases:
            if release["release_id"] == release_id:
                return release
    if run_id:
        for release in releases:
            if release["run_id"] == run_id or release["release_id"] == run_id:
                return release
    if model_name:
        matches = [release for release in releases if release["model_name"] == model_name]
        if matches:
            return sorted(matches, key=lambda item: str(item.get("created_at") or ""), reverse=True)[0]
    raise ValueError("No matching release found.")


def _count_jsonl_severity(path: Path, *, severity: str) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if str(payload.get("severity") or "").lower() == severity.lower():
            count += 1
    return count


def _count_csv_rows(path: Path, *, severity: str) -> int:
    if not path.exists():
        return 0
    return _count_jsonl_severity(path, severity=severity)


def _latest_readiness_status(session_root: Path) -> str | None:
    if not session_root.exists():
        return None
    readiness_paths = sorted(session_root.glob("session_*/readiness_report.json"))
    if not readiness_paths:
        return None
    payload = read_json(readiness_paths[-1])
    return payload.get("readiness_status")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_csv_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None
