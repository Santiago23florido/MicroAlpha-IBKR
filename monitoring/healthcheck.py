from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
import tempfile
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
            "market_raw_dir": _path_status(settings.paths.market_raw_dir),
            "import_root": _path_status(settings.paths.import_root),
            "import_market_dir": _path_status(settings.paths.import_market_dir),
            "import_meta_dir": _path_status(settings.paths.import_meta_dir),
            "import_log_dir": _path_status(settings.paths.import_log_dir),
            "processed_dir": _path_status(settings.paths.processed_dir),
            "feature_dir": _path_status(settings.paths.feature_dir),
            "model_dir": _path_status(settings.paths.model_dir),
            "log_dir": _path_status(settings.paths.log_dir),
            "report_dir": _path_status(settings.paths.report_dir),
            "runtime_db_parent": _path_status(Path(settings.runtime_db_path).parent),
            "model_registry_parent": _path_status(Path(settings.models.registry_path).parent),
            "transfer_log_parent": _path_status(Path(settings.paths.transfer_log_path).parent),
            "transfer_report_dir": _path_status(settings.paths.transfer_report_dir),
        },
        "broker": {
            "configured": {
                "host": settings.ib_host,
                "port": settings.ib_port,
                "client_id": settings.ib_client_id,
                "ui_client_id": settings.ib_ui_client_id,
                "collector_client_id": settings.ib_collector_client_id,
                "symbol": settings.ib_symbol,
            },
            "connected": None,
        },
        "collector": {
            "enabled": settings.collector_enabled,
            "mode": settings.collector.mode,
            "symbols": list(settings.supported_symbols),
            "poll_interval_seconds": settings.collector.poll_interval_seconds,
            "flush_interval_seconds": settings.collector.flush_interval_seconds,
            "batch_size": settings.collector.batch_size,
            "reconnect_delay_seconds": settings.collector.reconnect_delay_seconds,
            "max_reconnect_attempts": settings.collector.max_reconnect_attempts,
            "output_root": str(settings.paths.market_raw_dir),
            "output_root_writable": _path_is_writable(settings.paths.market_raw_dir),
        },
        "lan_sync": {
            "network_root": _path_status(settings.lan_sync.pc2_network_root) if settings.lan_sync.pc2_network_root else None,
            "source_market_subdir": settings.lan_sync.source_market_subdir,
            "source_meta_subdir": settings.lan_sync.source_meta_subdir,
            "source_log_subdir": settings.lan_sync.source_log_subdir,
            "categories": {
                "raw": settings.lan_sync.include_raw,
                "meta": settings.lan_sync.include_meta,
                "logs": settings.lan_sync.include_logs,
            },
            "overwrite_policy": settings.lan_sync.overwrite_policy,
            "validate_parquet": settings.lan_sync.validate_parquet,
            "dry_run": settings.lan_sync.dry_run,
            "allowed_symbols": list(settings.lan_sync.allowed_symbols or settings.supported_symbols),
            "transfer_log_path": str(settings.paths.transfer_log_path),
            "transfer_report_dir": _path_status(settings.paths.transfer_report_dir),
        },
    }
    report["collector"]["ready"] = bool(
        report["collector"]["symbols"]
        and report["collector"]["output_root_writable"]
    )
    report["lan_sync"]["ready"] = bool(
        settings.lan_sync.pc2_network_root
        and report["paths"]["import_market_dir"]["is_dir"]
        and report["paths"]["transfer_report_dir"]["is_dir"]
    )

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


def _path_is_writable(value: str | Path) -> bool:
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".microalpha_health_", delete=True):
            return True
    except OSError:
        return False
