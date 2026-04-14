from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from broker.ib_client import IBClientError
from config import load_settings
from data.schemas import MarketSnapshot
from engine.market_clock import MarketClock
from ingestion.collector import MarketDataCollector, collect_market_data
from ingestion.market_data import normalize_market_snapshot
from ingestion.persistence import ParquetMarketDataSink
from monitoring.healthcheck import build_healthcheck_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def build_settings(tmp_path: Path, extra_lines: list[str] | None = None):
    data_root = tmp_path / "data"
    env_file = tmp_path / ".env"
    lines = [
        f"DATA_ROOT={data_root}",
        f"LOG_FILE={data_root / 'logs' / 'collector.log'}",
        f"EXECUTION_LOG_FILE={data_root / 'reports' / 'executions.csv'}",
        f"RUNTIME_DB_PATH={data_root / 'processed' / 'runtime' / 'microalpha.db'}",
        f"MODEL_ARTIFACTS_DIR={data_root / 'models' / 'artifacts'}",
        f"MODEL_REGISTRY_PATH={data_root / 'models' / 'artifacts' / 'registry.json'}",
        "SUPPORTED_SYMBOLS=SPY,QQQ",
        "COLLECTOR_BATCH_SIZE=1",
        "COLLECTOR_FLUSH_INTERVAL_SECONDS=1",
        "COLLECTOR_RECONNECT_DELAY_SECONDS=0",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    env_file.write_text("\n".join(lines), encoding="utf-8")
    return load_settings(env_file=env_file, config_dir=CONFIG_DIR, environment="deploy")


class FakeCollectorClient:
    def __init__(self, snapshots: dict[str, MarketSnapshot], *, fail_once: bool = False) -> None:
        self.snapshots = snapshots
        self.fail_once = fail_once
        self.logger = logging.getLogger("test.collector")
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.fetch_calls = 0
        self.client_id = 201

    def connect(self) -> bool:
        self.connect_calls += 1
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def get_server_time(self) -> dict[str, str | int]:
        return {"epoch": 1_776_108_394, "iso_utc": "2026-04-13T19:26:34+00:00"}

    def fetch_market_snapshot(self, symbol: str) -> MarketSnapshot:
        self.fetch_calls += 1
        if self.fail_once:
            self.fail_once = False
            self.connected = False
            raise IBClientError("simulated connection loss")
        return self.snapshots[symbol.upper()]


def test_normalize_market_snapshot_produces_consistent_record(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    clock = MarketClock(settings.session)
    snapshot = MarketSnapshot(
        symbol="SPY",
        timestamp="2026-04-13T19:26:35+00:00",
        source="ib_snapshot",
        bid=500.0,
        ask=500.2,
        last=500.1,
        volume=1000,
        bid_size=10,
        ask_size=12,
    )

    record = normalize_market_snapshot(snapshot, clock).to_dict()

    assert record["symbol"] == "SPY"
    assert record["spread"] == 0.2
    assert record["event_type"] == "snapshot"
    assert record["source"] == "ib_snapshot"
    assert record["session_window"] in {"primary", "secondary", "between_windows", "opening_range"}


def test_parquet_market_data_sink_writes_partitioned_files(tmp_path: Path) -> None:
    sink = ParquetMarketDataSink(tmp_path / "market", logging.getLogger("test.sink"), batch_size=1, flush_interval_seconds=1)
    sink.append(
        {
            "timestamp": "2026-04-13T19:26:35+00:00",
            "symbol": "SPY",
            "last_price": 500.1,
            "bid": 500.0,
            "ask": 500.2,
            "spread": 0.2,
            "bid_size": 10,
            "ask_size": 12,
            "last_size": None,
            "volume": 1000.0,
            "event_type": "snapshot",
            "source": "ib_snapshot",
            "session_window": "primary",
            "is_market_open": True,
            "exchange_time": "2026-04-13T15:26:35-04:00",
            "collected_at": "2026-04-13T19:26:35+00:00",
        }
    )

    result = sink.flush()

    assert result is not None
    file_path = Path(result["files"][0])
    assert file_path.exists()
    frame = pd.read_parquet(file_path)
    assert list(frame.columns)[0] == "timestamp"
    assert frame.iloc[0]["symbol"] == "SPY"


def test_market_data_collector_runs_once_and_persists_records(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    snapshots = {
        "SPY": MarketSnapshot(symbol="SPY", timestamp="2026-04-13T19:26:35+00:00", source="ib_snapshot", bid=500.0, ask=500.2, last=500.1),
        "QQQ": MarketSnapshot(symbol="QQQ", timestamp="2026-04-13T19:26:36+00:00", source="ib_snapshot", bid=400.0, ask=400.1, last=400.05),
    }
    client = FakeCollectorClient(snapshots)

    result = collect_market_data(settings, client=client, once=True, output_root=tmp_path / "market")

    assert result["status"] == "ok"
    assert result["cycles"] == 1
    assert result["records_collected"] == 2
    assert result["records_persisted"] == 2
    assert client.connect_calls == 1


def test_market_data_collector_reconnects_after_ib_error(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, ["COLLECTOR_MAX_RECONNECT_ATTEMPTS=2"])
    snapshots = {
        "SPY": MarketSnapshot(symbol="SPY", timestamp="2026-04-13T19:26:35+00:00", source="ib_snapshot", bid=500.0, ask=500.2, last=500.1),
        "QQQ": MarketSnapshot(symbol="QQQ", timestamp="2026-04-13T19:26:36+00:00", source="ib_snapshot", bid=400.0, ask=400.1, last=400.05),
    }
    client = FakeCollectorClient(snapshots, fail_once=True)
    sink = ParquetMarketDataSink(tmp_path / "market", logging.getLogger("test.collector.reconnect"), batch_size=10, flush_interval_seconds=30)
    collector = MarketDataCollector(settings, client, sink)

    result = collector.run(once=True)

    assert result["status"] == "ok"
    assert result["reconnect_attempts"] == 1
    assert result["records_collected"] == 2
    assert client.connect_calls >= 2


def test_healthcheck_reports_collector_readiness(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)

    report = build_healthcheck_report(settings)

    assert report["collector"]["ready"] is True
    assert report["collector"]["output_root_writable"] is True
    assert report["broker"]["configured"]["collector_client_id"] == settings.ib_collector_client_id
