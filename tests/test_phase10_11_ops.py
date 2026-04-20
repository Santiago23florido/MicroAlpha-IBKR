from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from config import load_settings
from config.phase10_11 import load_phase10_11_config
from engine import phase7 as phase7_engine
from execution import IBKRPaperExecutionBackend
from execution import reconciliation as reconciliation_module
from ops.orchestrator import full_paper_validation_cycle, generate_runbooks, system_health_report
from ops.preflight import preflight_check
from tests.test_phase6_operations import PROJECT_ROOT, create_mock_feature_store, create_mock_phase5_artifact
from tests.test_phase9_ibkr_paper import FakeIBClient
from validation.paper_validation import compare_paper_sessions, reconcile_and_report, run_paper_validation_session
from validation.session_tracker import SessionTracker


def build_phase10_11_test_settings(tmp_path: Path, *, extra_env: list[str] | None = None):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "settings.yaml",
        "risk.yaml",
        "symbols.yaml",
        "deployment.yaml",
        "feature_sets.yaml",
        "modeling.yaml",
        "phase6.yaml",
        "phase7.yaml",
        "phase8.yaml",
        "phase10_11.yaml",
    ):
        shutil.copy(PROJECT_ROOT / "config" / name, config_dir / name)
    env_lines = [
        "APP_ENV=development",
        "SUPPORTED_SYMBOLS=SPY",
        "DRY_RUN=false",
        "SAFE_TO_TRADE=true",
        "ALLOW_SESSION_EXECUTION=true",
        "ACTIVE_EXECUTION_BACKEND=ibkr_paper",
        "BROKER_MODE=paper",
        "IBKR_PAPER_CLIENT_ID=911",
    ]
    env_lines.extend(extra_env or [])
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(env_lines), encoding="utf-8")
    return load_settings(env_file=env_file, config_dir=config_dir, environment="development")


class MatchingFakeIBClient(FakeIBClient):
    def get_positions(self) -> list[dict[str, object]]:
        return [{"account": "DU123456", "symbol": "SPY", "position": 1.0, "avgCost": 100.02}]


def patch_fake_paper_backend(monkeypatch: pytest.MonkeyPatch, settings) -> None:
    def fake_build_execution_backend(config, **kwargs):
        return IBKRPaperExecutionBackend(config, settings=settings, client=MatchingFakeIBClient())

    monkeypatch.setattr(phase7_engine, "build_execution_backend", fake_build_execution_backend)
    monkeypatch.setattr(reconciliation_module, "build_execution_backend", fake_build_execution_backend)


def tracker_for(settings) -> SessionTracker:
    phase10_11 = load_phase10_11_config(settings)
    return SessionTracker(
        session_root=phase10_11.report_paths.session_root,
        registry_path=phase10_11.report_paths.registry_path,
        archive_root=phase10_11.report_paths.archive_root,
    )


def test_preflight_detects_invalid_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = build_phase10_11_test_settings(tmp_path, extra_env=["ACTIVE_EXECUTION_BACKEND=mock"])
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)
    monkeypatch.setattr("ops.preflight.broker_healthcheck", lambda settings: {"status": "ok"})

    result = preflight_check(settings, symbols=["SPY"])

    assert result["status"] == "error"
    assert "execution_backend_real" in result["blocking_failures"]


def test_run_paper_validation_session_creates_session_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = build_phase10_11_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)
    patch_fake_paper_backend(monkeypatch, settings)

    result = run_paper_validation_session(settings, symbols=["SPY"], latest_per_symbol=1)

    tracker = tracker_for(settings)
    session = tracker.load_session(result["session_id"])
    session_dir = tracker.session_dir(result["session_id"])

    assert result["status"] == "ok"
    assert session["final_state"] == "COMPLETED"
    assert (session_dir / "session_summary.json").exists()
    assert (session_dir / "readiness_report.json").exists()
    assert (session_dir / "reconciliation_summary.json").exists()
    assert (session_dir / "system_health.json").exists()


def test_full_paper_validation_cycle_generates_postflight_and_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = build_phase10_11_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)
    patch_fake_paper_backend(monkeypatch, settings)

    result = full_paper_validation_cycle(settings, symbols=["SPY"], latest_per_symbol=1)
    runbooks = generate_runbooks(settings)

    tracker = tracker_for(settings)
    session_dir = tracker.session_dir(result["session_id"])

    assert result["status"] in {"ok", "error"}
    assert result["preflight"]["status"] == "ok"
    assert result["postflight"]["checks"]["reports_created"]["passed"] is True
    assert result["postflight"]["archive_path"] is not None
    assert (session_dir / "postflight_check.json").exists()
    assert Path(runbooks["files"]["preflight_checklist.md"]).exists()


def test_compare_sessions_and_system_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = build_phase10_11_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)
    patch_fake_paper_backend(monkeypatch, settings)

    first = run_paper_validation_session(settings, symbols=["SPY"], latest_per_symbol=1)
    second = run_paper_validation_session(settings, symbols=["SPY"], latest_per_symbol=1)
    reconciliation = reconcile_and_report(settings, session_id=second["session_id"])
    comparison = compare_paper_sessions(settings)
    health = system_health_report(settings, session_id=second["session_id"])

    assert reconciliation["status"] == "ok"
    assert comparison["status"] == "ok"
    assert comparison["session_count"] >= 2
    assert Path(comparison["leaderboard_path"]).exists()
    assert health["status"] == "ok"
    assert health["latest_session_status"]["session_id"] == second["session_id"]
