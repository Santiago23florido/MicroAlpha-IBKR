from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import Settings
from monitoring.logging import setup_logger

from deployment.drive_sync import (
    discover_sync_candidates,
    resolve_sync_categories,
    validate_synced_file,
)


def cleanup_local_artifacts(
    settings: Settings,
    *,
    categories: Sequence[str] | None = None,
    dry_run: bool | None = None,
    logger=None,
) -> dict[str, Any]:
    cleanup_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.cleanup_local",
    )
    dry_run_enabled = settings.sync.dry_run if dry_run is None else dry_run
    resolved_categories = resolve_sync_categories(settings, categories)
    threshold_hours = max(settings.sync.delete_min_age_hours, max(settings.sync.retention_days_local, 0) * 24.0)

    candidates = discover_sync_candidates(settings, categories=resolved_categories)
    cleanup_logger.info(
        "Starting local cleanup: dry_run=%s categories=%s threshold_hours=%s candidates=%s",
        dry_run_enabled,
        list(resolved_categories),
        threshold_hours,
        len(candidates),
    )

    deleted_files = 0
    deleted_bytes = 0
    planned_files = 0
    skipped_files = 0
    results: list[dict[str, Any]] = []

    for candidate in candidates:
        validation = validate_synced_file(candidate.source_path, candidate.destination_path)
        item = {
            "category": candidate.category,
            "source": candidate.source_path,
            "destination": candidate.destination_path,
            "size_bytes": candidate.size_bytes,
            "age_hours": round(candidate.age_hours, 3),
            "validation": validation,
        }

        if not validation["valid"]:
            item["status"] = "skipped_not_validated"
            skipped_files += 1
            results.append(item)
            continue

        if candidate.age_hours < threshold_hours:
            item["status"] = "skipped_too_young"
            skipped_files += 1
            results.append(item)
            continue

        if dry_run_enabled:
            item["status"] = "planned"
            planned_files += 1
            deleted_bytes += candidate.size_bytes
            results.append(item)
            continue

        try:
            Path(candidate.source_path).unlink(missing_ok=False)
            item["status"] = "deleted"
            deleted_files += 1
            deleted_bytes += candidate.size_bytes
        except OSError as exc:
            item["status"] = "delete_failed"
            item["error"] = str(exc)
            skipped_files += 1
        results.append(item)

    payload = {
        "status": "ok",
        "dry_run": dry_run_enabled,
        "categories": list(resolved_categories),
        "threshold_hours": threshold_hours,
        "detected_files": len(candidates),
        "deleted_files": deleted_files,
        "planned_files": planned_files,
        "skipped_files": skipped_files,
        "reclaimed_bytes": deleted_bytes,
        "results": results,
    }
    cleanup_logger.info(
        "Local cleanup complete: deleted=%s planned=%s skipped=%s reclaimed_bytes=%s dry_run=%s",
        deleted_files,
        planned_files,
        skipped_files,
        deleted_bytes,
        dry_run_enabled,
    )
    return _write_cleanup_report(settings, payload)


def _write_cleanup_report(settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    report_dir = Path(settings.paths.sync_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"cleanup_local_{token}.json"
    report_payload = {
        "created_at_utc": created_at,
        "operation": "cleanup_local",
        **payload,
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    result = dict(payload)
    result["report_path"] = str(report_path)
    return result
