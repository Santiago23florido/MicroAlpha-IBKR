from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from config import load_settings
from data_sources.polygon_download import fetch_training_data, prepare_training_data
from data_sources.polygon_normalizer import CANONICAL_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_polygon_test_settings(tmp_path: Path, *, include_api_key: bool = True):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for source in (PROJECT_ROOT / "config").glob("*.yaml"):
        shutil.copy(source, config_dir / source.name)

    env_lines = [
        "APP_ENV=development",
        "SUPPORTED_SYMBOLS=SPY",
        "DRY_RUN=true",
        "TRAINING_DATA_OUTPUT_ROOT=data/training/polygon",
        "TRAINING_DATA_WRITE_PARQUET=true",
        "TRAINING_DATA_WRITE_MANIFEST=true",
        "POLYGON_SYNTHETIC_SPREAD_BPS=2.0",
        "POLYGON_DEFAULT_DEPTH_SIZE=100",
    ]
    if include_api_key:
        env_lines.append("POLYGON_API_KEY=test_polygon_key")

    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(env_lines), encoding="utf-8")
    return load_settings(env_file=env_file, config_dir=config_dir, environment="development")


def test_prepare_training_data_from_manual_polygon_csv(tmp_path: Path) -> None:
    settings = build_polygon_test_settings(tmp_path)
    raw_path = tmp_path / "raw_polygon.csv"
    output_path = tmp_path / "exports" / "SPY_manual_training.csv"
    pd.DataFrame(
        [
            {"t": 1735738200000, "o": 588.10, "h": 588.50, "l": 587.90, "c": 588.25, "v": 105000},
            {"t": 1735738260000, "o": 588.25, "h": 588.60, "l": 588.10, "c": 588.40, "v": 98000},
        ]
    ).to_csv(raw_path, index=False)

    payload = prepare_training_data(
        settings,
        provider="polygon",
        input_path=raw_path,
        symbol="SPY",
        interval="1m",
        output_path=output_path,
    )

    exported = pd.read_csv(output_path)
    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert payload["status"] == "ok"
    assert payload["mode"] == "manual_csv"
    assert output_path.exists()
    assert output_path.with_suffix(".parquet").exists()
    assert set(CANONICAL_COLUMNS).issubset(exported.columns)
    assert exported["symbol"].tolist() == ["SPY", "SPY"]
    assert exported["last"].tolist() == exported["close"].tolist()
    assert exported["synthetic_bid_ask_flag"].all()
    assert exported["synthetic_depth_flag"].all()
    assert "bid" in manifest["synthetic_columns"]
    assert "ask" in manifest["synthetic_columns"]
    assert "last" in manifest["synthetic_columns"]
    assert manifest["provider"] == "polygon"
    assert manifest["source_mode"] == "manual_csv"


def test_prepare_training_data_from_polygon_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = build_polygon_test_settings(tmp_path, include_api_key=True)
    output_path = tmp_path / "exports" / "SPY_1m_training.csv"

    def fake_fetch_aggregates(self, *, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        assert symbol == "SPY"
        assert start_date == "2025-01-01"
        assert end_date == "2025-01-03"
        assert interval == "1m"
        return pd.DataFrame(
            [
                {"t": 1735738200000, "o": 588.10, "h": 588.50, "l": 587.90, "c": 588.25, "v": 105000},
                {"t": 1735738260000, "o": 588.25, "h": 588.60, "l": 588.10, "c": 588.40, "v": 98000},
            ]
        )

    monkeypatch.setattr("data_sources.polygon_client.PolygonClient.fetch_aggregates", fake_fetch_aggregates)

    payload = prepare_training_data(
        settings,
        provider="polygon",
        symbol="SPY",
        start_date="2025-01-01",
        end_date="2025-01-03",
        interval="1m",
        output_path=output_path,
    )

    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert payload["status"] == "ok"
    assert payload["mode"] == "api"
    assert payload["symbol"] == "SPY"
    assert payload["row_count"] == 2
    assert output_path.exists()
    assert output_path.with_suffix(".parquet").exists()
    assert manifest["source_mode"] == "api"
    assert manifest["synthetic_bid_ask_flag"] is True
    assert manifest["synthetic_depth_flag"] is True


def test_fetch_training_data_requires_polygon_api_key(tmp_path: Path) -> None:
    settings = build_polygon_test_settings(tmp_path, include_api_key=False)

    with pytest.raises(ValueError, match="POLYGON_API_KEY"):
        fetch_training_data(
            settings,
            provider="polygon",
            symbol="SPY",
            start_date="2025-01-01",
            end_date="2025-01-03",
            interval="1m",
            output_path=tmp_path / "exports" / "SPY_1m_training.csv",
        )
