from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from config import load_ibkr_historical_config, load_settings
from ingestion.ibkr_historical_backfill import (
    HistoricalBackfillServices,
    export_training_csv_from_backfill,
    ibkr_backfill,
    ibkr_backfill_status,
    ibkr_head_timestamp,
    prepare_ibkr_training_data,
)
from ingestion.ibkr_rate_limiter import IBKRHistoricalRateLimiter
from ingestion.ibkr_resume_store import IBKRBackfillResumeStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeHistoricalIBClient:
    def __init__(self) -> None:
        self.connected = False
        self.head_calls = 0
        self.bar_calls = 0

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def get_head_timestamp(self, *, symbol: str, exchange: str, currency: str, what_to_show: str, use_rth: bool) -> dict[str, str | bool]:
        self.head_calls += 1
        return {
            "symbol": symbol,
            "what_to_show": what_to_show,
            "use_rth": use_rth,
            "head_timestamp": "2025-01-01T14:30:00+00:00",
            "raw_head_timestamp": "20250101-14:30:00",
        }

    def get_historical_bars(
        self,
        *,
        symbol: str,
        exchange: str,
        currency: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
        end_datetime: str,
    ) -> list[dict[str, object]]:
        self.bar_calls += 1
        end = datetime.strptime(end_datetime, "%Y%m%d-%H:%M:%S").replace(tzinfo=timezone.utc)
        rows = []
        for minute in range(3, 0, -1):
            ts = end - timedelta(minutes=minute)
            price = 590.0 + (self.bar_calls * 0.1) + (3 - minute) * 0.05
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": price - 0.03,
                    "high": price + 0.06,
                    "low": price - 0.06,
                    "close": price,
                    "volume": 1000 + self.bar_calls,
                    "bar_count": 12,
                    "average": price - 0.01,
                }
            )
        return rows


def build_settings(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for source in (PROJECT_ROOT / "config").glob("*.yaml"):
        shutil.copy(source, config_dir / source.name)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_ENV=development",
                "SUPPORTED_SYMBOLS=SPY",
                "IBKR_HISTORICAL_ENABLED=true",
                "IBKR_BACKFILL_CHUNK_DAYS_1M=1000",
                "IBKR_BACKFILL_OUTPUT_ROOT=data/raw/ibkr_backfill",
                "IBKR_BACKFILL_STATE_ROOT=data/processed/ibkr_backfill_state",
                "IBKR_BACKFILL_EXPORT_ROOT=data/training/ibkr",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file=env_file, config_dir=config_dir, environment="development")


def build_services(settings, fake_client: FakeHistoricalIBClient) -> HistoricalBackfillServices:
    config = load_ibkr_historical_config(settings)
    return HistoricalBackfillServices(
        settings=settings,
        config=config,
        client=fake_client,
        rate_limiter=IBKRHistoricalRateLimiter(
            max_requests_per_10_min=config.max_requests_per_10_min,
            max_same_contract_requests_per_2_sec=config.max_same_contract_requests_per_2_sec,
            dedupe_window_seconds=0,
        ),
        resume_store=IBKRBackfillResumeStore(config.state_root, config.output_root),
    )


def test_ibkr_head_timestamp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    fake_client = FakeHistoricalIBClient()
    monkeypatch.setattr(
        "ingestion.ibkr_historical_backfill._build_services",
        lambda settings, config: build_services(settings, fake_client),
    )

    payload = ibkr_head_timestamp(settings, symbol="SPY", what_to_show="TRADES", use_rth=True)

    assert payload["status"] == "ok"
    assert payload["provider"] == "ibkr"
    assert payload["head_timestamp"] == "2025-01-01T14:30:00+00:00"


def test_prepare_ibkr_training_data_runs_backfill_and_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    fake_client = FakeHistoricalIBClient()
    monkeypatch.setattr(
        "ingestion.ibkr_historical_backfill._build_services",
        lambda settings, config: build_services(settings, fake_client),
    )
    output_path = tmp_path / "training" / "SPY_1m_training.csv"

    payload = prepare_ibkr_training_data(
        settings,
        symbol="SPY",
        what_to_show="TRADES",
        bar_size="1 min",
        use_rth=True,
        output_path=output_path,
    )

    exported = pd.read_csv(output_path)
    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert payload["status"] == "ok"
    assert payload["export"]["row_count"] > 0
    assert output_path.exists()
    assert output_path.with_suffix(".parquet").exists()
    assert {"timestamp", "symbol", "open", "high", "low", "close", "last", "bid", "ask", "bid_size", "ask_size"}.issubset(exported.columns)
    assert manifest["provider"] == "ibkr"
    assert manifest["source"] == "ibkr_historical_backfill"
    assert "bid" in manifest["synthetic_columns"]


def test_ibkr_backfill_resume_and_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    fake_client = FakeHistoricalIBClient()
    monkeypatch.setattr(
        "ingestion.ibkr_historical_backfill._build_services",
        lambda settings, config: build_services(settings, fake_client),
    )

    first = ibkr_backfill(
        settings,
        symbol="SPY",
        what_to_show="TRADES",
        bar_size="1 min",
        use_rth=True,
        resume=False,
    )
    second = ibkr_backfill(
        settings,
        symbol="SPY",
        what_to_show="TRADES",
        bar_size="1 min",
        use_rth=True,
        resume=True,
    )
    status = ibkr_backfill_status(settings, symbol="SPY", what_to_show="TRADES", bar_size="1 min")

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert status["state"]["status"] == "completed"
    assert Path(status["raw_output_path"]).exists()
    assert status["state"]["chunk_count"] >= 1


def test_export_training_csv_requires_existing_backfill(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    with pytest.raises(ValueError, match="No backfilled data available"):
        export_training_csv_from_backfill(
            settings,
            symbol="SPY",
            what_to_show="TRADES",
            bar_size="1 min",
            output_path=tmp_path / "missing.csv",
        )
