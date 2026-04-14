from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from config import Settings


SYNC_PATHS = (
    "raw_dir",
    "processed_dir",
    "feature_dir",
    "model_dir",
    "report_dir",
    "log_dir",
)


def sync_data_artifacts(
    settings: Settings,
    *,
    destination_root: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    destination = Path(destination_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, Any]] = []

    for attribute in SYNC_PATHS:
        source = Path(getattr(settings.paths, attribute))
        target = destination / source.name
        item = {
            "source": str(source),
            "destination": str(target),
            "exists": source.exists(),
        }
        plan.append(item)
        if execute and source.exists():
            _copy_tree(source, target)

    return {
        "status": "synced" if execute else "planned",
        "destination_root": str(destination),
        "items": plan,
        "sync_enabled": settings.deployment.sync_enabled,
    }


def _copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
