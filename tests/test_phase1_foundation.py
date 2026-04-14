from __future__ import annotations

from pathlib import Path

import pandas as pd

from app import build_parser
from backtest.runner import run_backtest_stub
from config import load_settings
from ingestion.collector import persist_collection_payload
from monitoring.healthcheck import build_healthcheck_report
from monitoring.sync import sync_data_artifacts
from scripts._launcher import _inject_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def test_load_settings_supports_environment_overrides(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "IB_CLIENT_ID=7",
                "IB_UI_CLIENT_ID=107",
                "SUPPORTED_SYMBOLS=SPY,QQQ",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file, config_dir=CONFIG_DIR, environment="deploy")

    assert settings.environment == "deploy"
    assert settings.deployment.role == "deploy"
    assert settings.collector_enabled is True
    assert settings.training_enabled is False
    assert settings.ib_client_id == 7
    assert settings.ib_ui_client_id == 107
    assert settings.supported_symbols == ("SPY", "QQQ")
    assert settings.paths.model_artifacts_dir.endswith("data/models/artifacts")


def test_phase1_parser_exposes_required_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    for command in ["collect", "train", "backtest", "run-session", "dashboard", "healthcheck"]:
        assert command in help_text


def test_script_launcher_preserves_global_options() -> None:
    argv = _inject_command("collect", ["--environment", "deploy", "--symbol", "SPY"])

    assert argv == ["--environment", "deploy", "collect", "--symbol", "SPY"]


def test_persist_collection_payload_writes_snapshot_and_bars(tmp_path: Path) -> None:
    payload = {
        "symbol": "SPY",
        "timestamp": "2026-04-13T19:26:30+00:00",
        "source": "historical_bar_fallback",
        "last": 500.5,
    }
    bars = pd.DataFrame(
        [
            {"timestamp": "2026-04-13T19:25:00+00:00", "open": 500.0, "high": 501.0, "low": 499.5, "close": 500.5, "volume": 123},
        ]
    )

    result = persist_collection_payload(tmp_path / "collector" / "spy", payload, bars)

    assert Path(result["snapshot_path"]).exists()
    assert Path(result["bars_path"]).exists()


def test_backtest_stub_reports_dataset_shape() -> None:
    settings = load_settings(env_file=PROJECT_ROOT / ".env.example", config_dir=CONFIG_DIR, environment="development")

    report = run_backtest_stub(settings)

    assert report["status"] == "placeholder"
    assert report["rows"] > 0
    assert "timestamp" in report["columns"]


def test_healthcheck_and_sync_plan_are_built_without_broker(tmp_path: Path) -> None:
    settings = load_settings(env_file=PROJECT_ROOT / ".env.example", config_dir=CONFIG_DIR, environment="development")

    healthcheck = build_healthcheck_report(settings)
    sync_plan = sync_data_artifacts(settings, destination_root=tmp_path, execute=False)

    assert healthcheck["broker"]["status"] == "not_checked"
    assert sync_plan["status"] == "planned"
    assert len(sync_plan["items"]) >= 4
