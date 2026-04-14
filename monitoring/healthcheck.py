from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings


def build_healthcheck_report(
    settings: Settings,
    *,
    broker_client: Any | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": settings.environment,
        "machine_role": settings.deployment.machine_role,
        "role": settings.deployment.role,
        "broker_mode": settings.broker_mode,
        "flags": {
            "collector_enabled": settings.collector_enabled,
            "training_enabled": settings.training_enabled,
            "backtest_enabled": settings.backtest_enabled,
            "safe_to_trade": settings.safe_to_trade,
            "dry_run": settings.dry_run,
        },
        "paths": {
            "data_root": _path_status(settings.paths.data_root),
            "raw_dir": _path_status(settings.paths.raw_dir),
            "processed_dir": _path_status(settings.paths.processed_dir),
            "feature_dir": _path_status(settings.paths.feature_dir),
            "model_dir": _path_status(settings.paths.model_dir),
            "log_dir": _path_status(settings.paths.log_dir),
            "report_dir": _path_status(settings.paths.report_dir),
            "runtime_db_parent": _path_status(Path(settings.runtime_db_path).parent),
            "model_registry_parent": _path_status(Path(settings.models.registry_path).parent),
        },
        "broker": {
            "configured": {
                "host": settings.ib_host,
                "port": settings.ib_port,
                "client_id": settings.ib_client_id,
                "ui_client_id": settings.ib_ui_client_id,
                "symbol": settings.ib_symbol,
            },
            "connected": None,
        },
    }

    if broker_client is None:
        report["broker"]["status"] = "not_checked"
        return report

    try:
        broker_client.connect()
        report["broker"]["connected"] = broker_client.is_connected()
        report["broker"]["status"] = "ok" if broker_client.is_connected() else "disconnected"
    except Exception as exc:
        report["broker"]["connected"] = False
        report["broker"]["status"] = "error"
        report["broker"]["error"] = str(exc)
    finally:
        with suppress(Exception):
            broker_client.disconnect()

    return report


def _path_status(value: str | Path) -> dict[str, Any]:
    path = Path(value)
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
    }
