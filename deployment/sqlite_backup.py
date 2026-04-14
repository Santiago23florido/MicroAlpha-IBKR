from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings
from monitoring.logging import setup_logger


def create_sqlite_backup(
    settings: Settings,
    *,
    source_path: str | Path | None = None,
    destination_dir: str | Path | None = None,
    filename: str | None = None,
    dry_run: bool | None = None,
    logger=None,
) -> dict[str, Any]:
    backup_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.sqlite_backup",
    )
    source = Path(source_path or settings.sync.sqlite_source_path).resolve()
    target_dir = Path(destination_dir or settings.paths.sqlite_backup_dir).resolve()
    target_filename = _build_backup_filename(filename or settings.sync.sqlite_backup_filename)
    target_path = target_dir / target_filename
    dry_run_enabled = settings.sync.dry_run if dry_run is None else dry_run

    if not source.exists():
        return {
            "status": "missing_source",
            "dry_run": dry_run_enabled,
            "source_path": str(source),
            "backup_path": str(target_path),
            "message": "SQLite source file does not exist.",
        }

    if dry_run_enabled:
        backup_logger.info(
            "Dry-run SQLite backup: source=%s planned_backup=%s",
            source,
            target_path,
        )
        return {
            "status": "planned",
            "dry_run": True,
            "source_path": str(source),
            "backup_path": str(target_path),
            "source_size_bytes": source.stat().st_size,
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    if temp_path.exists():
        temp_path.unlink()

    backup_logger.info("Creating SQLite backup: source=%s target=%s", source, target_path)
    with sqlite3.connect(source) as source_connection, sqlite3.connect(temp_path) as destination_connection:
        source_connection.backup(destination_connection)
    temp_path.replace(target_path)

    return {
        "status": "created",
        "dry_run": False,
        "source_path": str(source),
        "backup_path": str(target_path),
        "source_size_bytes": source.stat().st_size,
        "backup_size_bytes": target_path.stat().st_size,
    }


def _build_backup_filename(base_filename: str) -> str:
    stem = Path(base_filename).stem
    suffix = Path(base_filename).suffix or ".sqlite"
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}_{token}{suffix}"
