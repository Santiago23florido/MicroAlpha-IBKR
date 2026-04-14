from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from app import build_parser
from config import load_settings
from deployment.drive_sync import build_sync_status, sync_to_drive
from deployment.retention import cleanup_local_artifacts
from deployment.sqlite_backup import create_sqlite_backup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def build_settings(tmp_path: Path):
    data_root = tmp_path / "data"
    drive_root = tmp_path / "google_drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_ENV=deploy",
                f"DATA_ROOT={data_root}",
                f"GOOGLE_DRIVE_ROOT={drive_root}",
                "GOOGLE_DRIVE_SUBDIR=microalpha",
                "SYNC_ENABLED=true",
                "SYNC_RAW_ENABLED=true",
                "SYNC_FEATURES_ENABLED=true",
                "SYNC_SQLITE_ENABLED=true",
                "SYNC_LOGS_ENABLED=true",
                "SYNC_DRY_RUN=true",
                "SYNC_VALIDATE_CHECKSUM=false",
                "DELETE_AFTER_SYNC=false",
                "DELETE_MIN_AGE_HOURS=1",
                "RETENTION_DAYS_LOCAL=0",
                "SQLITE_BACKUP_FILENAME=collector.sqlite",
                f"SQLITE_SOURCE_PATH={data_root / 'processed' / 'runtime' / 'collector.sqlite'}",
                f"LOG_FILE={data_root / 'logs' / 'microalpha.log'}",
                f"RUNTIME_DB_PATH={data_root / 'processed' / 'runtime' / 'collector.sqlite'}",
                "SUPPORTED_SYMBOLS=SPY,QQQ",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file=env_file, config_dir=CONFIG_DIR, environment="deploy")


def write_parquet(path: Path, symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-14T13:30:00+00:00",
                "symbol": symbol,
                "last_price": 500.0,
                "bid": 499.99,
                "ask": 500.01,
                "spread": 0.02,
                "bid_size": 100,
                "ask_size": 99,
                "last_size": 10,
                "volume": 1000,
                "event_type": "snapshot",
                "source": "ib_snapshot",
                "session_window": "opening_range",
                "is_market_open": True,
                "exchange_time": "2026-04-14T09:30:00-04:00",
                "collected_at": "2026-04-14T13:30:00+00:00",
            }
        ]
    )
    frame.to_parquet(path, index=False)


def write_sqlite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS sample (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO sample (value) VALUES ('ok')")
        connection.commit()


def age_file(path: Path, *, hours: float) -> None:
    aged_timestamp = path.stat().st_mtime - (hours * 3600.0)
    path.touch()
    import os

    os.utime(path, (aged_timestamp, aged_timestamp))


def test_parser_exposes_phase4_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "sync-drive" in help_text
    assert "cleanup-local" in help_text
    assert "backup-sqlite" in help_text
    assert "sync-status" in help_text


def test_create_sqlite_backup_uses_local_snapshot_dir(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    source_path = Path(settings.sync.sqlite_source_path)
    write_sqlite(source_path)

    result = create_sqlite_backup(settings, dry_run=False)

    backup_path = Path(result["backup_path"])
    assert result["status"] == "created"
    assert backup_path.exists()
    assert backup_path.is_file()
    assert str(backup_path).startswith(str(Path(settings.paths.sqlite_backup_dir)))
    assert not str(backup_path).startswith(str(Path(settings.sync.google_drive_root)))


def test_sync_to_drive_copies_and_validates_categories(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_file = Path(settings.paths.market_raw_dir) / "2026-04-14" / "SPY" / "collector_00001.parquet"
    feature_file = Path(settings.paths.feature_dir) / "2026-04-14" / "SPY.parquet"
    log_file = Path(settings.paths.log_dir) / "microalpha.log"
    sqlite_source = Path(settings.sync.sqlite_source_path)
    write_parquet(raw_file, "SPY")
    write_parquet(feature_file, "SPY")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("sync me\n", encoding="utf-8")
    write_sqlite(sqlite_source)

    result = sync_to_drive(settings, dry_run=False, delete_after_sync=False)

    assert result["status"] == "ok"
    assert result["copied_files"] >= 3
    assert (Path(settings.sync.google_drive_root) / "microalpha" / "raw" / "market" / "2026-04-14" / "SPY" / "collector_00001.parquet").exists()
    assert (Path(settings.sync.google_drive_root) / "microalpha" / "features" / "2026-04-14" / "SPY.parquet").exists()
    assert (Path(settings.sync.google_drive_root) / "microalpha" / "logs" / "microalpha.log").exists()
    assert any(item["category"] == "sqlite" and item["validation"]["valid"] for item in result["results"])


def test_sync_to_drive_dry_run_keeps_local_files(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_file = Path(settings.paths.market_raw_dir) / "2026-04-14" / "SPY" / "collector_00002.parquet"
    write_parquet(raw_file, "SPY")
    age_file(raw_file, hours=2)

    result = sync_to_drive(
        settings,
        categories=["raw"],
        dry_run=True,
        delete_after_sync=True,
        include_sqlite_backup=False,
    )

    assert result["dry_run"] is True
    assert raw_file.exists()
    assert not (Path(settings.sync.google_drive_root) / "microalpha" / "raw" / "market" / "2026-04-14" / "SPY" / "collector_00002.parquet").exists()
    assert result["planned_delete_files"] == 0 or result["results"][0].get("delete_status") == "planned"


def test_cleanup_local_artifacts_deletes_only_validated_old_files(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    synced_file = Path(settings.paths.market_raw_dir) / "2026-04-14" / "SPY" / "collector_synced.parquet"
    pending_file = Path(settings.paths.market_raw_dir) / "2026-04-14" / "QQQ" / "collector_pending.parquet"
    write_parquet(synced_file, "SPY")
    age_file(synced_file, hours=2)

    sync_to_drive(
        settings,
        categories=["raw"],
        dry_run=False,
        delete_after_sync=False,
        include_sqlite_backup=False,
    )

    write_parquet(pending_file, "QQQ")
    age_file(pending_file, hours=2)

    result = cleanup_local_artifacts(settings, categories=["raw"], dry_run=False)

    assert result["deleted_files"] == 1
    assert not synced_file.exists()
    assert pending_file.exists()


def test_build_sync_status_reports_pending_and_latest_report(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    synced_file = Path(settings.paths.market_raw_dir) / "2026-04-14" / "SPY" / "collector_00003.parquet"
    pending_feature = Path(settings.paths.feature_dir) / "2026-04-14" / "QQQ.parquet"
    write_parquet(synced_file, "SPY")
    write_parquet(pending_feature, "QQQ")

    sync_to_drive(
        settings,
        categories=["raw"],
        dry_run=False,
        delete_after_sync=False,
        include_sqlite_backup=False,
    )

    status = build_sync_status(settings)

    assert status["drive"]["available"] is True
    assert status["synced_files"] >= 1
    assert status["pending_files"] >= 1
    assert status["latest_report"] is not None
