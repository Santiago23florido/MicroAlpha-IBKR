from __future__ import annotations

import json
import shutil
from pathlib import Path

from config import load_settings, set_active_model_selection
from config.phase12_14 import load_phase12_14_config
from governance.releases import (
    governance_status,
    list_model_releases,
    promote_model_release,
    rollback_model_release,
    show_active_release,
)
from ops.orchestrator import full_runtime_cycle
from ops.runtime_manager import bootstrap_runtime, runtime_status, service_status, start_runtime, stop_runtime
from shadow.session import run_shadow_session
from tests.test_phase6_operations import PROJECT_ROOT, create_mock_feature_store, create_mock_phase5_artifact


def build_phase12_14_test_settings(tmp_path: Path, *, extra_env: list[str] | None = None):
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
        "phase12_14.yaml",
        "runtime_profiles.yaml",
    ):
        shutil.copy(PROJECT_ROOT / "config" / name, config_dir / name)
    env_lines = [
        "APP_ENV=development",
        "SUPPORTED_SYMBOLS=SPY",
        "BROKER_MODE=paper",
        "ACTIVE_EXECUTION_BACKEND=ibkr_paper",
        "SAFE_TO_TRADE=false",
        "ALLOW_SESSION_EXECUTION=false",
        "RUNTIME_PROFILE=shadow",
    ]
    env_lines.extend(extra_env or [])
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(env_lines), encoding="utf-8")
    return load_settings(env_file=env_file, config_dir=config_dir, environment="development")


def test_runtime_bootstrap_and_service_lifecycle_shadow(tmp_path: Path) -> None:
    settings = build_phase12_14_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)

    bootstrap = bootstrap_runtime(settings, profile_name="shadow")
    started = start_runtime(settings, profile_name="shadow")
    status = service_status(settings)
    runtime = runtime_status(settings)
    stopped = stop_runtime(settings)

    assert bootstrap["status"] == "ok"
    assert started["status"] == "ok"
    assert status["runtime_state"]["status"] == "running"
    assert runtime["runtime_profile"] == "shadow"
    assert runtime["shadow_mode_enabled"] is True
    assert stopped["runtime"]["status"] == "stopped"


def test_run_shadow_session_writes_intents_and_comparison_reports(tmp_path: Path) -> None:
    settings = build_phase12_14_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)

    result = run_shadow_session(settings, symbols=["SPY"], latest_per_symbol=1)
    config = load_phase12_14_config(settings)
    intents_path = Path(config.deployment_paths.shadow_intents_path)

    assert result["status"] == "ok"
    assert result["shadow_mode_enabled"] is True
    assert result["paper_order_submission_enabled"] is False
    assert Path(result["summary_path"]).exists()
    assert Path(result["comparison"]["shadow_alignment_summary_path"]).exists()
    assert intents_path.exists()


def test_release_lifecycle_promote_and_rollback(tmp_path: Path) -> None:
    settings = build_phase12_14_test_settings(tmp_path)
    create_mock_feature_store(settings)
    first = create_mock_phase5_artifact(settings, run_id="run_test_logistic_hybrid_intraday_classification_binary_0001")
    second = create_mock_phase5_artifact(settings, run_id="run_test_logistic_hybrid_intraday_classification_binary_0002")
    set_active_model_selection(settings, run_id="run_test_logistic_hybrid_intraday_classification_binary_0001")

    releases = list_model_releases(settings)
    release_ids = {item["run_id"]: item["release_id"] for item in releases["releases"]}
    promotion = promote_model_release(
        settings,
        release_id=release_ids["run_test_logistic_hybrid_intraday_classification_binary_0002"],
        actor="pytest",
        reason="promote newer candidate",
    )
    active_after_promotion = show_active_release(settings)
    rollback = rollback_model_release(
        settings,
        to=release_ids["run_test_logistic_hybrid_intraday_classification_binary_0001"],
        actor="pytest",
        reason="rollback to previous release",
    )
    governance = governance_status(settings)

    assert Path(first).exists()
    assert Path(second).exists()
    assert promotion["status"] == "ok"
    assert active_after_promotion["active_release"]["run_id"] == "run_test_logistic_hybrid_intraday_classification_binary_0002"
    assert rollback["status"] == "ok"
    assert rollback["active_release"]["run_id"] == "run_test_logistic_hybrid_intraday_classification_binary_0001"
    assert governance["status"] == "ok"
    assert Path(load_phase12_14_config(settings).deployment_paths.promotion_audit_path).exists()
    assert Path(load_phase12_14_config(settings).deployment_paths.rollback_audit_path).exists()


def test_full_runtime_cycle_shadow_mode(tmp_path: Path) -> None:
    settings = build_phase12_14_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)

    result = full_runtime_cycle(settings, symbols=["SPY"], latest_per_symbol=1, profile_name="shadow")

    assert result["status"] == "ok"
    assert result["mode"] == "shadow"
    assert result["session"]["status"] == "ok"
    assert result["active_release"]["run_id"].startswith("run_test_logistic")
