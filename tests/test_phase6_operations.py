from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import load_settings
from config.phase6 import load_active_model_selection, load_phase6_config
from engine.phase6 import risk_check, run_decisions_offline, run_session, show_active_model
from models.inference import OperationalInferenceEngine
from risk.risk_engine import OperationalRiskEngine, OperationalRiskState
from strategy.decision_engine import DecisionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_phase6_test_settings(tmp_path: Path):
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
    ):
        shutil.copy(PROJECT_ROOT / "config" / name, config_dir / name)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_ENV=development",
                "SUPPORTED_SYMBOLS=SPY",
                "DRY_RUN=true",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file=env_file, config_dir=config_dir, environment="development")


def create_mock_phase5_artifact(settings, *, run_id: str = "run_test_logistic_hybrid_intraday_classification_binary_0001") -> Path:
    artifact_dir = Path(settings.paths.model_dir) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = ["signal_strength", "spread_bps", "estimated_cost_bps"]
    train_x = pd.DataFrame(
        {
            "signal_strength": [0.0, 0.2, 0.8, 1.0, 1.2, 1.4],
            "spread_bps": [2.0, 2.2, 1.5, 1.4, 1.3, 1.2],
            "estimated_cost_bps": [1.0, 1.0, 0.9, 0.8, 0.8, 0.7],
        }
    )
    train_y = [0, 0, 1, 1, 1, 1]
    preprocessor = Pipeline([("scaler", StandardScaler())])
    model = LogisticRegression(max_iter=200)
    model.fit(preprocessor.fit_transform(train_x), train_y)

    joblib.dump(model, artifact_dir / "model.joblib")
    joblib.dump(preprocessor, artifact_dir / "preprocessing.joblib")
    (artifact_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (artifact_dir / "target_config.json").write_text(
        json.dumps(
            {
                "name": "classification_binary",
                "description": "Binary future return target.",
                "task_type": "classification",
                "horizon_bars": 1,
                "threshold_bps": 2.5,
                "negative_threshold_bps": None,
                "bin_edges_bps": [],
                "class_labels": [],
                "quantiles": [],
                "cost_adjustment_bps": 0.0,
                "cost_adjustment_multiplier": 0.0,
                "positive_label": 1,
                "negative_label": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    training_metadata = {
        "run_id": run_id,
        "created_at_utc": "2026-04-17T08:00:00+00:00",
        "feature_set_name": "hybrid_intraday",
        "target_mode": "classification_binary",
        "model_name": "logistic_regression",
        "artifact_files": {
            "artifact_path": str((artifact_dir / "model.joblib").resolve()),
            "preprocessing_path": str((artifact_dir / "preprocessing.joblib").resolve()),
            "feature_columns_path": str((artifact_dir / "feature_columns.json").resolve()),
            "target_config_path": str((artifact_dir / "target_config.json").resolve()),
        },
        "dataset": {
            "feature_columns": feature_columns,
        },
    }
    (artifact_dir / "training_metadata.json").write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")
    (artifact_dir / "leaderboard_row.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "artifact_dir": str(artifact_dir.resolve()),
                "model_name": "logistic_regression",
                "model_type": "logistic_regression",
                "feature_set": "hybrid_intraday",
                "target_mode": "classification_binary",
                "symbols": ["SPY"],
                "ranking_score": 0.61,
                "timestamp_utc": "2026-04-17T08:00:00+00:00",
                "validation_metrics": {"f1_macro": 0.61, "roc_auc": 0.82},
                "test_metrics": {"f1_macro": 0.58, "roc_auc": 0.79},
                "hyperparameters": {"C": 1.0, "max_iter": 200},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_dir


def create_mock_feature_store(settings) -> None:
    root = Path(settings.paths.feature_dir) / "hybrid_intraday" / "2026-04-17"
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-04-17T13:40:00+00:00",
                "exchange_timestamp": "2026-04-17T13:40:00+00:00",
                "session_date": "2026-04-17",
                "session_time": "09:40:00",
                "symbol": "SPY",
                "signal_strength": 1.25,
                "spread_bps": 1.5,
                "estimated_cost_bps": 0.8,
                "price_proxy": 100.0,
            },
            {
                "timestamp": "2026-04-17T13:41:00+00:00",
                "exchange_timestamp": "2026-04-17T13:41:00+00:00",
                "session_date": "2026-04-17",
                "session_time": "09:41:00",
                "symbol": "SPY",
                "signal_strength": 0.15,
                "spread_bps": 1.8,
                "estimated_cost_bps": 0.9,
                "price_proxy": 100.1,
            },
        ]
    )
    frame.to_parquet(root / "SPY.parquet", index=False)

    label_root = Path(settings.paths.processed_dir) / "labels" / "hybrid_intraday" / "classification_binary" / "2026-04-17"
    label_root.mkdir(parents=True, exist_ok=True)
    label_frame = frame.copy()
    label_frame["future_return_bps"] = [6.0, -1.0]
    label_frame["future_net_return_bps"] = [5.2, -1.9]
    label_frame["target_cost_adjustment_bps"] = [0.8, 0.9]
    label_frame["target_classification_binary"] = [1, 0]
    label_frame.to_parquet(label_root / "SPY.parquet", index=False)


def test_active_model_selection_and_show_status(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)

    selection = load_active_model_selection(settings)
    status = show_active_model(settings)

    assert selection.model_name == "logistic_regression"
    assert selection.target_mode == "classification_binary"
    assert status["artifact_status"]["ready"] is True


def test_operational_inference_and_decision_pipeline(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)

    inference = OperationalInferenceEngine.from_active_model(settings)
    phase6 = load_phase6_config(settings)
    decision_engine = DecisionEngine(phase6.decision, phase6.sizing)
    row = pd.Series(
        {
            "timestamp": "2026-04-17T13:40:00+00:00",
            "exchange_timestamp": "2026-04-17T13:40:00+00:00",
            "session_time": "09:40:00",
            "session_date": "2026-04-17",
            "symbol": "SPY",
            "signal_strength": 1.30,
            "spread_bps": 1.2,
            "estimated_cost_bps": 0.7,
            "price_proxy": 100.0,
        }
    )

    prediction = inference.predict_row(row).to_dict()
    decision = decision_engine.decide(row, prediction).to_dict()

    assert prediction["valid"] is True
    assert prediction["model_name"] == "logistic_regression"
    assert decision["target_mode"] == "classification_binary"
    assert decision["action"] in {"LONG", "NO_TRADE"}
    assert "reasons" in decision


def test_risk_engine_blocks_after_cooldown_trigger(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    phase6 = load_phase6_config(settings)
    engine = OperationalRiskEngine(phase6.risk)
    state = OperationalRiskState(session_date="2026-04-17")

    decision = {
        "action": "LONG",
        "symbol": "SPY",
        "timestamp": "2026-04-17T13:40:00+00:00",
        "size_suggestion": 1,
        "reasons": [],
    }
    prediction = {"valid": True}
    row = pd.Series(
        {
            "timestamp": "2026-04-17T13:40:00+00:00",
            "session_date": "2026-04-17",
            "symbol": "SPY",
            "spread_bps": 1.0,
            "estimated_cost_bps": 1.0,
        }
    )
    evaluation = engine.evaluate(decision, row, prediction, state)
    assert evaluation.blocked_by_risk is False
    state = engine.record_post_decision(state, decision, realized_net_return_bps=-5.0)

    later_row = row.copy()
    later_row["timestamp"] = "2026-04-17T13:45:00+00:00"
    blocked = engine.evaluate(decision, later_row, prediction, state)
    assert blocked.blocked_by_risk is True
    assert "cooldown_ok" in blocked.checks


def test_offline_and_session_runners_write_outputs(tmp_path: Path) -> None:
    settings = build_phase6_test_settings(tmp_path)
    create_mock_phase5_artifact(settings)
    create_mock_feature_store(settings)

    offline = run_decisions_offline(settings, symbols=["SPY"])
    session = run_session(settings, symbols=["SPY"])
    check = risk_check(settings)

    assert offline["status"] == "ok"
    assert Path(offline["parquet_path"]).exists()
    assert Path(offline["summary_path"]).exists()
    assert session["status"] == "ok"
    assert session["orders_sent"] is False
    assert check["artifact_status"]["ready"] is True
