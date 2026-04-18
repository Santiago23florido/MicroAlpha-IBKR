from __future__ import annotations

from pathlib import Path

from config.phase6 import load_active_model_selection
from config.phase7 import load_phase7_config
from engine.phase7 import execution_status, run_paper_session, run_paper_sim_offline, show_execution_backend
from execution import ExecutionJournal, OrderManager, PositionManager, build_execution_backend
from tests.test_phase6_operations import (
    build_phase6_test_settings,
    create_mock_feature_store,
    create_mock_phase5_artifact,
)


def test_show_execution_backend_uses_mock_backend(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)

    payload = show_execution_backend(settings)

    assert payload["status"] == "ok"
    assert payload["active_execution_backend"] == "mock"
    assert payload["backend"]["name"] == "mock"
    assert payload["backend"]["paper_mode"] is True


def test_order_manager_generates_mock_fills_and_updates_portfolio(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    phase7 = load_phase7_config(settings)
    selection = load_active_model_selection(settings)
    backend = build_execution_backend(phase7)
    journal = ExecutionJournal(phase7.logging)
    positions = PositionManager(initial_cash=phase7.session.initial_cash)
    manager = OrderManager(
        phase7,
        backend=backend,
        journal=journal,
        position_manager=positions,
    )

    result = manager.process_decision(
        {
            "symbol": "SPY",
            "action": "LONG",
            "size_suggestion": 2,
            "timestamp": "2026-04-17T13:40:00+00:00",
        },
        model_trace=selection_to_model_trace(selection),
        decision_id="dec_test_0001",
        market_data={"price_proxy": 100.0, "spread_bps": 1.5},
    )

    assert result.accepted is True
    assert result.order.status.value == "FILLED"
    assert result.order.filled_quantity == 2
    assert len(result.fills) >= 1
    assert result.portfolio is not None
    assert result.portfolio.open_position_count == 1
    assert result.portfolio.positions["SPY"].quantity == 2
    assert journal.recent_orders(5)
    assert journal.recent_fills(5)


def test_phase7_offline_runner_writes_reports_and_state(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)

    summary = run_paper_sim_offline(settings, symbols=["SPY"])
    status = execution_status(settings)

    assert summary["status"] == "ok"
    assert summary["backend_name"] == "mock"
    assert Path(summary["summary_path"]).exists()
    assert Path(summary["parquet_path"]).exists()
    assert Path(summary["csv_path"]).exists()
    assert summary["portfolio_final"]["open_position_count"] >= 0
    assert status["status"] == "ok"
    assert status["active_execution_backend"] == "mock"
    assert Path(status["paths"]["state_path"]).exists()


def test_phase7_session_runner_reuses_pipeline_without_real_broker(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)

    summary = run_paper_session(settings, symbols=["SPY"], latest_per_symbol=2)

    assert summary["status"] == "ok"
    assert summary["backend_name"] == "mock"
    assert summary["paper_mode"] is True
    assert summary["row_count"] == 2


def selection_to_model_trace(selection):
    from execution import ModelTrace

    return ModelTrace(
        model_name=selection.model_name,
        model_type=selection.model_type,
        run_id=selection.run_id,
        feature_set_name=selection.feature_set_name,
        target_mode=selection.target_mode,
        artifact_dir=selection.artifact_dir,
        selection_reason=selection.selection_reason,
        source_leaderboard=selection.source_leaderboard,
        updated_at_utc=selection.updated_at_utc,
    )
