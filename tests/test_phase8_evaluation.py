from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.phase8 import load_phase8_config
from engine.phase7 import run_paper_session, run_paper_sim_offline
from evaluation.compare_runs import compare_runs
from evaluation.performance import evaluate_performance
from evaluation.signal_analysis import analyze_signal_quality
from monitoring.drift import population_stability_index
from reporting.report_bundle import detect_drift_report, evaluate_performance_report, generate_report
from tests.test_phase6_operations import (
    build_phase6_test_settings,
    create_mock_feature_store,
    create_mock_phase5_artifact,
)


def test_performance_engine_computes_trade_metrics() -> None:
    decision_frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-17T13:40:00+00:00",
                "portfolio_before": {"equity": 100000.0},
                "portfolio_after": {"equity": 99999.9},
                "execution_status": "FILLED",
            },
            {
                "timestamp": "2026-04-17T13:45:00+00:00",
                "portfolio_before": {"equity": 99999.9},
                "portfolio_after": {"equity": 100000.8},
                "execution_status": "FILLED",
            },
        ]
    )
    fills = [
        {
            "fill_id": "fill_entry",
            "order_id": "ord_entry",
            "symbol": "SPY",
            "action": "BUY",
            "quantity": 1,
            "fill_price": 100.0,
            "commission": 0.1,
            "filled_at": "2026-04-17T13:40:00+00:00",
            "source_decision_id": "dec_entry",
            "source_model_name": "model_a",
        },
        {
            "fill_id": "fill_exit",
            "order_id": "ord_exit",
            "symbol": "SPY",
            "action": "SELL",
            "quantity": 1,
            "fill_price": 101.0,
            "commission": 0.1,
            "filled_at": "2026-04-17T13:45:00+00:00",
            "source_decision_id": "dec_exit",
            "source_model_name": "model_a",
        },
    ]

    result = evaluate_performance(
        decision_frame,
        fills=fills,
        final_portfolio={"realized_pnl": 0.8, "unrealized_pnl": 0.0, "positions": {}},
    )

    assert result["summary"]["closed_trade_count"] == 1
    assert abs(result["summary"]["net_total_pnl"] - 0.8) < 1e-9
    assert result["summary"]["win_rate"] == 1.0
    assert result["summary"]["session_total_pnl"] == 0.8


def test_signal_analysis_detects_monotonic_signal_quality() -> None:
    frame = pd.DataFrame(
        {
            "score": [float(index) for index in range(20)],
            "probability": [index / 20.0 for index in range(20)],
            "future_net_return_bps": [float(index) - 5.0 for index in range(20)],
            "action": ["LONG"] * 20,
        }
    )

    result = analyze_signal_quality(frame)

    assert result["summary"]["score_monotonicity_ratio"] is not None
    assert result["summary"]["score_monotonicity_ratio"] >= 0.8
    assert "score_deciles" in result["tables"]


def test_population_stability_index_detects_shift() -> None:
    psi = population_stability_index([0.0] * 100 + [1.0] * 100, [2.0] * 200)
    assert psi is not None
    assert psi > 0.1


def test_phase8_reports_auto_generate_and_compare_runs(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)
    phase8 = load_phase8_config(settings)

    offline = run_paper_sim_offline(settings, symbols=["SPY"])
    session = run_paper_session(settings, symbols=["SPY"], latest_per_symbol=2)

    assert offline["status"] == "ok"
    assert session["status"] == "ok"
    assert "phase8_report" in offline
    assert Path(offline["phase8_report"]["run_report_path"]).exists()

    generated = generate_report(settings)
    performance = evaluate_performance_report(settings)
    drift = detect_drift_report(settings)
    comparison = compare_runs(phase8.report_paths.report_dir, output_dir=phase8.report_paths.compare_runs_dir)

    assert generated["status"] == "ok"
    assert performance["status"] == "ok"
    assert drift["status"] == "ok"
    assert comparison["status"] == "ok"
    assert Path(generated["run_report_path"]).exists()
    assert Path(comparison["csv_path"]).exists()
