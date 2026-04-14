from __future__ import annotations

from pathlib import Path

import pandas as pd

from app import build_parser, run_dev_sync_and_build
from config import load_settings
from deployment.lan_sync import pull_from_pc2
from monitoring.data_quality import validate_imports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def build_settings(tmp_path: Path):
    data_root = tmp_path / "pc1_data"
    import_root = tmp_path / "imports" / "from_pc2"
    network_root = tmp_path / "pc2_share"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_ENV=development",
                f"DATA_ROOT={data_root}",
                f"MARKET_RAW_DIR={data_root / 'raw' / 'market'}",
                f"IMPORT_ROOT={import_root}",
                f"IMPORT_MARKET_DIR={import_root / 'raw' / 'market'}",
                f"IMPORT_META_DIR={import_root / 'meta'}",
                f"IMPORT_LOG_DIR={import_root / 'logs'}",
                f"TRANSFER_LOG_PATH={import_root / 'transfer_log.jsonl'}",
                f"TRANSFER_REPORT_DIR={data_root / 'reports' / 'lan_sync'}",
                f"PC2_NETWORK_ROOT={network_root}",
                f"LOG_FILE={data_root / 'logs' / 'microalpha.log'}",
                f"EXECUTION_LOG_FILE={data_root / 'reports' / 'executions.csv'}",
                f"RUNTIME_DB_PATH={data_root / 'processed' / 'runtime' / 'microalpha.db'}",
                f"MODEL_ARTIFACTS_DIR={data_root / 'models' / 'artifacts'}",
                f"MODEL_REGISTRY_PATH={data_root / 'models' / 'artifacts' / 'registry.json'}",
                "SUPPORTED_SYMBOLS=SPY,QQQ",
                "LAN_INCLUDE_RAW=true",
                "LAN_INCLUDE_META=true",
                "LAN_INCLUDE_LOGS=false",
                "LAN_DRY_RUN=false",
                "LAN_OVERWRITE_POLICY=if_newer",
                "LAN_VALIDATE_PARQUET=true",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file=env_file, config_dir=CONFIG_DIR, environment="development")


def write_pc2_market_partition(root: Path, session_date: str, symbol: str) -> Path:
    start = pd.Timestamp(f"{session_date} 13:30:00+00:00")
    rows = []
    for minute in range(8):
        timestamp = start + pd.Timedelta(minutes=minute)
        last_price = 600.0 + minute * 0.1 + (5 if symbol == "QQQ" else 0)
        bid = last_price - 0.02
        ask = last_price + 0.02
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "last_price": last_price,
                "bid": bid,
                "ask": ask,
                "spread": ask - bid,
                "bid_size": 100 + minute,
                "ask_size": 101 + minute,
                "last_size": 10 + minute,
                "volume": 1000 + minute * 10,
                "event_type": "snapshot",
                "source": "ib_snapshot",
                "session_window": "opening_range" if minute < 5 else "primary",
                "is_market_open": True,
                "exchange_time": timestamp.tz_convert("America/New_York").isoformat(),
                "collected_at": timestamp.isoformat(),
            }
        )
    frame = pd.DataFrame(rows)
    target = root / "data" / "raw" / "market" / session_date / symbol / "collector_00001.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return target


def test_parser_exposes_lan_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    assert "pull-from-pc2" in help_text
    assert "validate-imports" in help_text
    assert "dev-sync-and-build" in help_text
    assert "build-features" in help_text


def test_pull_from_pc2_copies_and_tracks_files(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    source_file = write_pc2_market_partition(Path(settings.lan_sync.pc2_network_root), "2026-04-14", "SPY")
    meta_file = Path(settings.lan_sync.pc2_network_root) / "data" / "meta" / "collector_state.json"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text('{"status":"ok"}\n', encoding="utf-8")

    result = pull_from_pc2(settings, categories=["raw", "meta"], symbols=["SPY"])

    copied_file = Path(settings.paths.import_market_dir) / "2026-04-14" / "SPY" / "collector_00001.parquet"
    assert result["status"] == "ok"
    assert result["copied_files"] == 2
    assert copied_file.exists()
    assert Path(settings.paths.transfer_log_path).exists()
    assert any(item["destination_path"] == str(copied_file) for item in result["results"])
    assert source_file.exists()


def test_validate_imports_reports_quality_and_partition_summary(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    write_pc2_market_partition(Path(settings.lan_sync.pc2_network_root), "2026-04-14", "SPY")
    pull_from_pc2(settings, categories=["raw"], symbols=["SPY"])

    result = validate_imports(settings, input_root=settings.paths.import_market_dir, symbols=["SPY"])

    assert result["status"] == "ok"
    assert result["file_count"] == 1
    assert result["rows"] == 8
    assert result["quality"]["row_count"] == 8
    assert result["summary_by_partition"][0]["symbol"] == "SPY"
    assert Path(result["report_path"]).exists()


def test_dev_sync_and_build_creates_feature_files_from_imports(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    write_pc2_market_partition(Path(settings.lan_sync.pc2_network_root), "2026-04-14", "SPY")
    write_pc2_market_partition(Path(settings.lan_sync.pc2_network_root), "2026-04-15", "SPY")

    result = run_dev_sync_and_build(settings, symbols=["SPY"])

    assert result["status"] == "ok"
    assert result["pull"]["copied_files"] == 2
    assert result["validation"]["status"] == "ok"
    assert result["features"]["feature_rows"] == 16
    assert len(result["features"]["written_files"]) == 2
    assert Path(result["features"]["written_files"][0]).exists()
