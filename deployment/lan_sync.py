from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from config import Settings
from monitoring.logging import setup_logger


SUPPORTED_TRANSFER_CATEGORIES = ("raw", "meta", "logs")


@dataclass(frozen=True)
class TransferCandidate:
    category: str
    source_path: Path
    destination_path: Path
    relative_path: Path
    size_bytes: int
    modified_at_utc: str


def pull_from_pc2(
    settings: Settings,
    *,
    network_root: str | Path | None = None,
    destination_root: str | Path | None = None,
    categories: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool | None = None,
    overwrite_policy: str | None = None,
    validate_parquet: bool | None = None,
    logger=None,
) -> dict[str, Any]:
    sync_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.lan_sync",
    )
    started_at = datetime.now(timezone.utc)
    remote_root = _resolve_remote_root(settings, network_root)
    import_paths = _resolve_import_paths(settings, destination_root)
    dry_run_enabled = settings.lan_sync.dry_run if dry_run is None else dry_run
    overwrite_mode = (overwrite_policy or settings.lan_sync.overwrite_policy).strip().lower()
    validate_parquet_enabled = settings.lan_sync.validate_parquet if validate_parquet is None else validate_parquet

    if overwrite_mode not in {"if_newer", "always", "never"}:
        raise ValueError("LAN overwrite policy must be one of: if_newer, always, never.")

    if not remote_root.exists():
        raise FileNotFoundError(
            f"PC2 network root {remote_root} is not reachable. Mount or share the remote folder and set PC2_NETWORK_ROOT."
        )

    resolved_categories = resolve_lan_categories(settings, categories)
    resolved_symbols = _resolve_symbols(settings, symbols)
    candidates = discover_pc2_candidates(
        settings,
        remote_root=remote_root,
        categories=resolved_categories,
        symbols=resolved_symbols,
        start_date=start_date,
        end_date=end_date,
    )
    sync_logger.info(
        "Starting LAN pull from PC2: remote_root=%s destination=%s dry_run=%s categories=%s candidates=%s",
        remote_root,
        import_paths["import_root"],
        dry_run_enabled,
        list(resolved_categories),
        len(candidates),
    )

    results: list[dict[str, Any]] = []
    copied_files = 0
    skipped_files = 0
    failed_files = 0
    validated_files = 0

    for candidate in candidates:
        transfer_needed, reason = _should_copy_candidate(candidate, overwrite_mode)

        item = {
            "category": candidate.category,
            "source_path": str(candidate.source_path),
            "destination_path": str(candidate.destination_path),
            "relative_path": str(candidate.relative_path),
            "size_bytes": candidate.size_bytes,
            "modified_at_utc": candidate.modified_at_utc,
            "dry_run": dry_run_enabled,
        }

        if not transfer_needed:
            item["status"] = "skipped"
            item["reason"] = reason
            skipped_files += 1
            results.append(item)
            _append_transfer_event(settings, item)
            continue

        if dry_run_enabled:
            item["status"] = "planned"
            item["reason"] = reason
            results.append(item)
            _append_transfer_event(settings, item)
            continue

        try:
            candidate.destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate.source_path, candidate.destination_path)
            validation = validate_transferred_file(
                candidate.source_path,
                candidate.destination_path,
                validate_parquet=validate_parquet_enabled,
            )
            item["validation"] = validation
            if validation["valid"]:
                item["status"] = "copied"
                copied_files += 1
                validated_files += 1
            else:
                item["status"] = "validation_failed"
                failed_files += 1
            results.append(item)
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
            failed_files += 1
            results.append(item)
            sync_logger.error(
                "LAN transfer failed: source=%s destination=%s error=%s",
                candidate.source_path,
                candidate.destination_path,
                exc,
            )

        _append_transfer_event(settings, item)

    report = {
        "status": "ok" if failed_files == 0 else "error",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "remote_root": str(remote_root),
        "destination_root": str(import_paths["import_root"]),
        "categories": list(resolved_categories),
        "symbols": list(resolved_symbols),
        "filters": {"start_date": start_date, "end_date": end_date},
        "dry_run": dry_run_enabled,
        "overwrite_policy": overwrite_mode,
        "validate_parquet": validate_parquet_enabled,
        "detected_files": len(candidates),
        "copied_files": copied_files,
        "skipped_files": skipped_files,
        "validated_files": validated_files,
        "failed_files": failed_files,
        "results": results,
        "duration_seconds": round((datetime.now(timezone.utc) - started_at).total_seconds(), 3),
    }
    report["report_path"] = _write_pull_report(settings, report)
    sync_logger.info(
        "LAN pull complete: detected=%s copied=%s skipped=%s failed=%s dry_run=%s report=%s",
        report["detected_files"],
        copied_files,
        skipped_files,
        failed_files,
        dry_run_enabled,
        report["report_path"],
    )
    return report


def discover_pc2_candidates(
    settings: Settings,
    *,
    remote_root: Path,
    categories: Sequence[str],
    symbols: Sequence[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[TransferCandidate]:
    candidates: list[TransferCandidate] = []
    for category in categories:
        if category == "raw":
            source_root = remote_root / settings.lan_sync.source_market_subdir
            destination_root = Path(settings.paths.import_market_dir)
            for source_path in _iter_raw_files(source_root, symbols=symbols, start_date=start_date, end_date=end_date):
                relative_path = source_path.relative_to(source_root)
                candidates.append(_build_candidate(category, source_path, destination_root / relative_path, relative_path))
        elif category == "meta":
            source_root = remote_root / settings.lan_sync.source_meta_subdir
            destination_root = Path(settings.paths.import_meta_dir)
            for source_path in _iter_all_files(source_root):
                relative_path = source_path.relative_to(source_root)
                candidates.append(_build_candidate(category, source_path, destination_root / relative_path, relative_path))
        elif category == "logs":
            source_root = remote_root / settings.lan_sync.source_log_subdir
            destination_root = Path(settings.paths.import_log_dir)
            for source_path in _iter_all_files(source_root):
                relative_path = source_path.relative_to(source_root)
                candidates.append(_build_candidate(category, source_path, destination_root / relative_path, relative_path))
    return sorted(candidates, key=lambda item: (item.category, str(item.relative_path)))


def resolve_lan_categories(settings: Settings, categories: Sequence[str] | None = None) -> tuple[str, ...]:
    if categories:
        resolved = tuple(category.strip().lower() for category in categories if category and category.strip())
    else:
        resolved = tuple(
            category
            for category, enabled in (
                ("raw", settings.lan_sync.include_raw),
                ("meta", settings.lan_sync.include_meta),
                ("logs", settings.lan_sync.include_logs),
            )
            if enabled
        )
    invalid = [category for category in resolved if category not in SUPPORTED_TRANSFER_CATEGORIES]
    if invalid:
        raise ValueError(f"Unsupported LAN transfer categories: {', '.join(invalid)}")
    return resolved


def validate_transferred_file(
    source_path: Path,
    destination_path: Path,
    *,
    validate_parquet: bool = True,
) -> dict[str, Any]:
    validation = {
        "valid": False,
        "source_exists": source_path.exists(),
        "destination_exists": destination_path.exists(),
        "source_size_bytes": source_path.stat().st_size if source_path.exists() else 0,
        "destination_size_bytes": destination_path.stat().st_size if destination_path.exists() else 0,
        "parquet_readable": None,
        "error": None,
    }
    if not validation["source_exists"] or not validation["destination_exists"]:
        validation["error"] = "missing_source_or_destination"
        return validation
    if validation["source_size_bytes"] <= 0 or validation["destination_size_bytes"] <= 0:
        validation["error"] = "empty_file"
        return validation
    if validation["source_size_bytes"] != validation["destination_size_bytes"]:
        validation["error"] = "size_mismatch"
        return validation

    if validate_parquet and destination_path.suffix.lower() == ".parquet":
        try:
            pd.read_parquet(destination_path).head(1)
            validation["parquet_readable"] = True
        except Exception as exc:
            validation["parquet_readable"] = False
            validation["error"] = f"parquet_validation_failed: {exc}"
            return validation

    validation["valid"] = True
    return validation


def _resolve_remote_root(settings: Settings, network_root: str | Path | None) -> Path:
    resolved = network_root or settings.lan_sync.pc2_network_root
    if not resolved:
        raise ValueError(
            "PC2 network root is not configured. Set PC2_NETWORK_ROOT in .env or pass a network root override."
        )
    return Path(str(resolved)).expanduser()


def _resolve_import_paths(settings: Settings, destination_root: str | Path | None) -> dict[str, Path]:
    if destination_root is None:
        import_root = Path(settings.paths.import_root)
    else:
        import_root = Path(destination_root)
    return {
        "import_root": import_root,
        "import_market_dir": import_root / "raw" / "market",
        "import_meta_dir": import_root / "meta",
        "import_log_dir": import_root / "logs",
    }


def _resolve_symbols(settings: Settings, symbols: Sequence[str] | None) -> tuple[str, ...]:
    if symbols:
        return tuple(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())
    if settings.lan_sync.allowed_symbols:
        return tuple(symbol.upper() for symbol in settings.lan_sync.allowed_symbols)
    return settings.supported_symbols


def _iter_raw_files(
    source_root: Path,
    *,
    symbols: Sequence[str],
    start_date: str | None,
    end_date: str | None,
) -> Iterable[Path]:
    if not source_root.exists():
        return []
    allowed_symbols = {symbol.upper() for symbol in symbols}
    start = _coerce_session_date(start_date)
    end = _coerce_session_date(end_date)
    files: list[Path] = []
    for date_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        session_date = _coerce_session_date(date_dir.name)
        if session_date is None:
            continue
        if start and session_date < start:
            continue
        if end and session_date > end:
            continue
        for path in sorted(date_dir.rglob("*.parquet")):
            inferred_symbol = _infer_symbol_from_raw_path(path)
            if inferred_symbol and inferred_symbol not in allowed_symbols:
                continue
            files.append(path)
    return files


def _iter_all_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _infer_symbol_from_raw_path(path: Path) -> str | None:
    if path.parent.name.upper() and path.parent.name != path.parent.parent.name and _coerce_session_date(path.parent.name) is None:
        return path.parent.name.upper()
    if path.suffix.lower() == ".parquet":
        return path.stem.upper()
    return None


def _build_candidate(category: str, source_path: Path, destination_path: Path, relative_path: Path) -> TransferCandidate:
    stat = source_path.stat()
    return TransferCandidate(
        category=category,
        source_path=source_path,
        destination_path=destination_path,
        relative_path=relative_path,
        size_bytes=stat.st_size,
        modified_at_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    )


def _should_copy_candidate(candidate: TransferCandidate, overwrite_mode: str) -> tuple[bool, str]:
    destination = candidate.destination_path
    if not destination.exists():
        return True, "destination_missing"
    if overwrite_mode == "never":
        return False, "destination_exists"
    source_stat = candidate.source_path.stat()
    destination_stat = destination.stat()
    if overwrite_mode == "always":
        return True, "overwrite_policy_always"
    source_is_newer = source_stat.st_mtime > destination_stat.st_mtime + 1e-6
    size_changed = source_stat.st_size != destination_stat.st_size
    if source_is_newer or size_changed:
        return True, "source_changed"
    return False, "unchanged"


def _append_transfer_event(settings: Settings, event: dict[str, Any]) -> None:
    event_payload = dict(event)
    event_payload["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    log_path = Path(settings.paths.transfer_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_payload, sort_keys=True))
        handle.write("\n")


def _write_pull_report(settings: Settings, report: dict[str, Any]) -> str:
    report_dir = Path(settings.paths.transfer_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"pull_from_pc2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(report_path)


def _coerce_session_date(value: str | None):
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None
