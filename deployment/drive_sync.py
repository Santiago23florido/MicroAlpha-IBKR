from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import Settings
from monitoring.logging import setup_logger

from deployment.sqlite_backup import create_sqlite_backup


SYNC_CATEGORIES = ("raw", "features", "sqlite", "logs")


@dataclass(frozen=True)
class SyncCandidate:
    category: str
    source_path: str
    destination_path: str
    size_bytes: int
    age_hours: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sync_to_drive(
    settings: Settings,
    *,
    categories: Sequence[str] | None = None,
    dry_run: bool | None = None,
    delete_after_sync: bool | None = None,
    validate_checksum: bool | None = None,
    include_sqlite_backup: bool = True,
    logger=None,
) -> dict[str, Any]:
    sync_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.drive_sync",
    )
    dry_run_enabled = settings.sync.dry_run if dry_run is None else dry_run
    delete_enabled = settings.sync.delete_after_sync if delete_after_sync is None else delete_after_sync
    checksum_enabled = settings.sync.validate_checksum if validate_checksum is None else validate_checksum

    if not settings.deployment.sync_enabled:
        return _write_report(
            settings,
            "sync_drive",
            {
                "status": "disabled",
                "dry_run": dry_run_enabled,
                "message": "Sync is disabled by configuration.",
            },
        )

    drive_status = _drive_root_status(settings)
    if not drive_status["available"]:
        return _write_report(
            settings,
            "sync_drive",
            {
                "status": "error",
                "dry_run": dry_run_enabled,
                "message": "Google Drive root is not available.",
                "drive": drive_status,
            },
        )

    backup_result: dict[str, Any] | None = None
    resolved_categories = resolve_sync_categories(settings, categories)
    if include_sqlite_backup and "sqlite" in resolved_categories:
        backup_result = create_sqlite_backup(
            settings,
            dry_run=dry_run_enabled,
            logger=sync_logger,
        )

    candidates = discover_sync_candidates(settings, categories=resolved_categories)
    sync_logger.info(
        "Starting drive sync: dry_run=%s categories=%s candidates=%s drive=%s",
        dry_run_enabled,
        list(resolved_categories),
        len(candidates),
        drive_status["base_dir"],
    )

    results: list[dict[str, Any]] = []
    copied_count = 0
    skipped_count = 0
    failed_count = 0
    deleted_count = 0
    planned_delete_count = 0

    for candidate in candidates:
        source = Path(candidate.source_path)
        destination = Path(candidate.destination_path)

        current_validation = validate_synced_file(
            source,
            destination,
            validate_checksum=checksum_enabled,
        )
        destination_current = (
            current_validation["valid"]
            and destination.exists()
            and destination.stat().st_mtime >= source.stat().st_mtime
        )

        item: dict[str, Any] = {
            "category": candidate.category,
            "source": candidate.source_path,
            "destination": candidate.destination_path,
            "size_bytes": candidate.size_bytes,
            "age_hours": round(candidate.age_hours, 3),
            "dry_run": dry_run_enabled,
        }

        if dry_run_enabled:
            item["status"] = "already_synced" if destination_current else "planned"
            item["validation"] = current_validation
            if delete_enabled and _meets_min_delete_age(candidate.age_hours, settings.sync.delete_min_age_hours) and current_validation["valid"]:
                item["delete_status"] = "planned"
                planned_delete_count += 1
            results.append(item)
            continue

        if destination_current:
            item["status"] = "already_synced"
            item["validation"] = current_validation
            skipped_count += 1
        else:
            try:
                _copy_with_tempfile(source, destination)
                validation = validate_synced_file(
                    source,
                    destination,
                    validate_checksum=checksum_enabled,
                )
                item["status"] = "copied" if validation["valid"] else "validation_failed"
                item["validation"] = validation
                if validation["valid"]:
                    copied_count += 1
                else:
                    failed_count += 1
            except Exception as exc:
                item["status"] = "copy_failed"
                item["error"] = str(exc)
                failed_count += 1
                sync_logger.error(
                    "Drive sync copy failed: source=%s destination=%s error=%s",
                    source,
                    destination,
                    exc,
                )

        if delete_enabled and item.get("validation", {}).get("valid") and _meets_min_delete_age(candidate.age_hours, settings.sync.delete_min_age_hours):
            try:
                source.unlink(missing_ok=False)
                item["delete_status"] = "deleted"
                deleted_count += 1
            except OSError as exc:
                item["delete_status"] = "delete_failed"
                item["delete_error"] = str(exc)
                failed_count += 1
                sync_logger.error("Post-sync delete failed: source=%s error=%s", source, exc)
        elif delete_enabled and item.get("validation", {}).get("valid"):
            item["delete_status"] = "too_young"

        results.append(item)

    summary = {
        "status": "ok" if failed_count == 0 else "partial_failure",
        "dry_run": dry_run_enabled,
        "delete_after_sync": delete_enabled,
        "validate_checksum": checksum_enabled,
        "drive": drive_status,
        "categories": list(resolved_categories),
        "backup": backup_result,
        "detected_files": len(candidates),
        "copied_files": copied_count,
        "skipped_files": skipped_count,
        "failed_files": failed_count,
        "deleted_files": deleted_count,
        "planned_delete_files": planned_delete_count,
        "results": results,
    }
    sync_logger.info(
        "Drive sync complete: detected=%s copied=%s skipped=%s failed=%s deleted=%s dry_run=%s",
        len(candidates),
        copied_count,
        skipped_count,
        failed_count,
        deleted_count,
        dry_run_enabled,
    )
    return _write_report(settings, "sync_drive", summary)


def build_sync_status(
    settings: Settings,
    *,
    categories: Sequence[str] | None = None,
    validate_checksum: bool | None = None,
    logger=None,
) -> dict[str, Any]:
    sync_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.sync_status",
    )
    checksum_enabled = settings.sync.validate_checksum if validate_checksum is None else validate_checksum
    resolved_categories = resolve_sync_categories(settings, categories)
    drive_status = _drive_root_status(settings)
    candidates = discover_sync_candidates(settings, categories=resolved_categories) if drive_status["configured"] else []

    pending_count = 0
    synced_count = 0
    invalid_count = 0
    pending_bytes = 0
    deletable_bytes = 0
    per_category: dict[str, dict[str, int]] = {}
    retention_threshold_hours = _cleanup_age_threshold_hours(settings)

    for category in resolved_categories:
        per_category[category] = {"count": 0, "pending": 0, "synced": 0, "invalid": 0}

    for candidate in candidates:
        per_category[candidate.category]["count"] += 1
        validation = validate_synced_file(
            Path(candidate.source_path),
            Path(candidate.destination_path),
            validate_checksum=checksum_enabled,
        )
        if validation["valid"]:
            synced_count += 1
            per_category[candidate.category]["synced"] += 1
            if _meets_min_delete_age(candidate.age_hours, retention_threshold_hours):
                deletable_bytes += candidate.size_bytes
        elif Path(candidate.destination_path).exists():
            invalid_count += 1
            per_category[candidate.category]["invalid"] += 1
            pending_bytes += candidate.size_bytes
        else:
            pending_count += 1
            per_category[candidate.category]["pending"] += 1
            pending_bytes += candidate.size_bytes

    latest_report = read_latest_sync_report(settings)
    summary = {
        "status": "ok" if drive_status["available"] else "warning",
        "sync_enabled": settings.deployment.sync_enabled,
        "dry_run_default": settings.sync.dry_run,
        "drive": drive_status,
        "categories": list(resolved_categories),
        "pending_files": pending_count,
        "synced_files": synced_count,
        "invalid_destination_files": invalid_count,
        "pending_bytes": pending_bytes,
        "estimated_deletable_bytes": deletable_bytes,
        "retention_threshold_hours": retention_threshold_hours,
        "per_category": per_category,
        "latest_report": latest_report,
    }
    sync_logger.info(
        "Sync status generated: pending=%s synced=%s invalid=%s drive_available=%s",
        pending_count,
        synced_count,
        invalid_count,
        drive_status["available"],
    )
    return summary


def discover_sync_candidates(
    settings: Settings,
    *,
    categories: Sequence[str] | None = None,
) -> list[SyncCandidate]:
    drive_base = get_drive_base_dir(settings)
    resolved_categories = resolve_sync_categories(settings, categories)
    discovered: list[SyncCandidate] = []

    for category in resolved_categories:
        if category == "raw":
            source_root = Path(settings.paths.market_raw_dir)
            destination_root = drive_base / "raw" / "market"
            patterns = ("*.parquet",)
        elif category == "features":
            source_root = Path(settings.paths.feature_dir)
            destination_root = drive_base / "features"
            patterns = ("*.parquet",)
        elif category == "sqlite":
            source_root = Path(settings.paths.sqlite_backup_dir)
            destination_root = drive_base / "meta" / "sqlite"
            patterns = ("*.sqlite", "*.db", "*.sqlite3")
        elif category == "logs":
            source_root = Path(settings.paths.log_dir)
            destination_root = drive_base / "logs"
            patterns = ("*",)
        else:  # pragma: no cover
            continue

        if not source_root.exists():
            continue

        for source_path in _iter_category_files(source_root, patterns):
            relative = source_path.relative_to(source_root)
            destination = destination_root / relative
            stat = source_path.stat()
            discovered.append(
                SyncCandidate(
                    category=category,
                    source_path=str(source_path),
                    destination_path=str(destination),
                    size_bytes=stat.st_size,
                    age_hours=_age_hours(stat.st_mtime),
                )
            )

    return sorted(discovered, key=lambda item: (item.category, item.source_path))


def resolve_sync_categories(settings: Settings, categories: Sequence[str] | None = None) -> tuple[str, ...]:
    if categories:
        requested = tuple(category.lower() for category in categories)
    else:
        requested = tuple(
            category
            for category, enabled in (
                ("raw", settings.sync.raw_enabled),
                ("features", settings.sync.features_enabled),
                ("sqlite", settings.sync.sqlite_enabled),
                ("logs", settings.sync.logs_enabled),
            )
            if enabled
        )
    invalid = sorted(set(requested) - set(SYNC_CATEGORIES))
    if invalid:
        raise ValueError(f"Unsupported sync categories: {', '.join(invalid)}")
    return requested


def validate_synced_file(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    validate_checksum: bool = False,
) -> dict[str, Any]:
    source = Path(source_path)
    destination = Path(destination_path)
    result: dict[str, Any] = {
        "source_exists": source.exists(),
        "destination_exists": destination.exists(),
        "source_size_bytes": source.stat().st_size if source.exists() else None,
        "destination_size_bytes": destination.stat().st_size if destination.exists() else None,
        "non_empty": False,
        "size_match": False,
        "checksum_match": None,
        "valid": False,
    }
    if not source.exists() or not destination.exists():
        return result

    result["non_empty"] = bool(destination.stat().st_size > 0)
    result["size_match"] = bool(source.stat().st_size == destination.stat().st_size)

    if validate_checksum and result["size_match"]:
        result["checksum_match"] = _sha256(source) == _sha256(destination)
    elif validate_checksum:
        result["checksum_match"] = False

    result["valid"] = bool(
        result["destination_exists"]
        and result["non_empty"]
        and result["size_match"]
        and (result["checksum_match"] in {None, True})
    )
    return result


def read_latest_sync_report(settings: Settings) -> dict[str, Any] | None:
    report_dir = Path(settings.paths.sync_report_dir)
    if not report_dir.exists():
        return None
    report_files = sorted(report_dir.glob("*.json"))
    if not report_files:
        return None
    latest = report_files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"path": str(latest), "status": "invalid_json"}
    return {
        "path": str(latest),
        "created_at_utc": payload.get("created_at_utc"),
        "operation": payload.get("operation"),
        "status": payload.get("status"),
    }


def get_drive_base_dir(settings: Settings) -> Path:
    if not settings.sync.google_drive_root:
        return Path(settings.paths.data_root) / ".unconfigured_drive_root"
    return Path(settings.sync.google_drive_root).expanduser().resolve() / settings.sync.drive_subdirectory


def _drive_root_status(settings: Settings) -> dict[str, Any]:
    root = Path(settings.sync.google_drive_root).expanduser().resolve() if settings.sync.google_drive_root else None
    base_dir = get_drive_base_dir(settings)
    configured = bool(settings.sync.google_drive_root)
    available = bool(configured and root and root.exists() and root.is_dir())
    writable = False
    if available:
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            writable = os.access(base_dir, os.W_OK)
        except OSError:
            writable = False
    return {
        "configured": configured,
        "root": str(root) if root else None,
        "base_dir": str(base_dir),
        "available": available,
        "writable": writable,
    }


def _copy_with_tempfile(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(f"{destination.suffix}.part")
    if temp_path.exists():
        temp_path.unlink()
    shutil.copy2(source, temp_path)
    temp_path.replace(destination)


def _iter_category_files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def _age_hours(mtime: float) -> float:
    return max((datetime.now(timezone.utc).timestamp() - mtime) / 3600.0, 0.0)


def _meets_min_delete_age(age_hours: float, threshold_hours: float) -> bool:
    return age_hours >= max(threshold_hours, 0.0)


def _cleanup_age_threshold_hours(settings: Settings) -> float:
    retention_hours = max(settings.sync.retention_days_local, 0) * 24.0
    return max(settings.sync.delete_min_age_hours, retention_hours)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(settings: Settings, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    report_dir = Path(settings.paths.sync_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"{operation}_{token}.json"
    report_payload = {
        "created_at_utc": created_at,
        "operation": operation,
        **payload,
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    result = dict(payload)
    result["report_path"] = str(report_path)
    return result
