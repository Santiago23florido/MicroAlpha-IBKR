from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import PathSettings, load_settings
from data.feature_loader import infer_feature_columns
from labels.labeling import build_labels
from models.experiments import compare_model_variants, train_baseline_variant
from models.registry import ModelRegistry


def test_infer_feature_columns_excludes_targets_and_future_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-04-01T13:30:00Z")],
            "symbol": ["SPY"],
            "sma_short": [1.0],
            "future_return_bps": [5.0],
            "target_classification_binary": [1],
            "target_cost_adjustment_bps": [0.2],
        }
    )
    columns = infer_feature_columns(frame)
    assert "sma_short" in columns
    assert "future_return_bps" not in columns
    assert "target_classification_binary" not in columns
    assert "target_cost_adjustment_bps" not in columns


def test_phase5_labeling_and_variant_runner(tmp_path: Path) -> None:
    settings = _build_temp_settings(tmp_path)
    _write_feature_store(settings.paths.feature_dir)

    label_result = build_labels(
        settings,
        feature_set_name="hybrid_intraday",
        target_mode="classification_binary",
        symbols=["SPY"],
    )
    assert label_result["labeled_row_count"] > 0
    assert Path(label_result["report_path"]).exists()

    train_result = train_baseline_variant(
        settings,
        feature_set_name="hybrid_intraday",
        target_mode="classification_binary",
        model_name="logistic_regression",
        symbols=["SPY"],
        hyperparameters={"C": 0.5, "max_iter": 200, "class_weight": None},
    )
    assert Path(train_result["artifact_path"]).exists()
    assert Path(train_result["preprocessing_path"]).exists()

    comparison = compare_model_variants(
        settings,
        feature_sets=["hybrid_intraday"],
        target_modes=["quantile_regression"],
        models=["quantile_gradient_boosting"],
        symbols=["SPY"],
        max_combinations=1,
    )
    assert comparison["attempted_combinations"] == 1
    assert Path(comparison["leaderboard_paths"]["json"]).exists()
    assert Path(comparison["leaderboard_paths"]["csv"]).exists()
    assert Path(comparison["leaderboard_paths"]["parquet"]).exists()

    registry = ModelRegistry(settings.models.registry_path)
    assert registry.list_phase5_runs()


def _build_temp_settings(tmp_path: Path):
    base_settings = load_settings(".env", config_dir="config")
    tmp_root = tmp_path / "workspace"
    paths = replace(
        base_settings.paths,
        data_root=str((tmp_root / "data").resolve()),
        feature_dir=str((tmp_root / "data/features").resolve()),
        processed_dir=str((tmp_root / "data/processed").resolve()),
        model_dir=str((tmp_root / "data/models").resolve()),
        model_artifacts_dir=str((tmp_root / "data/models/artifacts").resolve()),
        report_dir=str((tmp_root / "data/reports").resolve()),
        log_dir=str((tmp_root / "data/logs").resolve()),
    )
    models = replace(
        base_settings.models,
        artifacts_dir=str((tmp_root / "data/models/artifacts").resolve()),
        registry_path=str((tmp_root / "data/models/artifacts/registry.json").resolve()),
    )
    storage = replace(
        base_settings.storage,
        log_file=str((tmp_root / "data/logs/microalpha.log").resolve()),
        execution_log_file=str((tmp_root / "data/reports/executions.csv").resolve()),
        runtime_db_path=str((tmp_root / "runtime/microalpha.db").resolve()),
    )
    for path in [
        Path(paths.feature_dir),
        Path(paths.processed_dir),
        Path(paths.model_dir),
        Path(paths.model_artifacts_dir),
        Path(paths.report_dir),
        Path(paths.log_dir),
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return replace(base_settings, paths=paths, models=models, storage=storage)


def _write_feature_store(feature_root: str) -> None:
    root = Path(feature_root) / "hybrid_intraday"
    rows_per_day = 50
    base_timestamp = pd.Timestamp("2026-04-06T13:30:00Z")
    for day_index, session_date in enumerate(["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10"]):
        day_dir = root / session_date
        day_dir.mkdir(parents=True, exist_ok=True)
        timestamps = [base_timestamp + pd.Timedelta(days=day_index, minutes=index) for index in range(rows_per_day)]
        price = pd.Series(100 + day_index + (pd.Series(range(rows_per_day)) * 0.05))
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "exchange_timestamp": timestamps,
                "session_date": [session_date] * rows_per_day,
                "session_time": [ts.time().isoformat() for ts in timestamps],
                "symbol": ["SPY"] * rows_per_day,
                "event_type": ["snapshot"] * rows_per_day,
                "source": ["simulated"] * rows_per_day,
                "session_window": ["regular"] * rows_per_day,
                "is_market_open": [True] * rows_per_day,
                "last_price": price,
                "mid_price": price + 0.01,
                "price_proxy": price + 0.01,
                "high_price_proxy": price + 0.03,
                "low_price_proxy": price - 0.03,
                "spread_bps": 2.0 + (pd.Series(range(rows_per_day)) % 5) * 0.1,
                "imbalance": ((pd.Series(range(rows_per_day)) % 7) - 3) / 10.0,
                "relative_volume": 1.0 + (pd.Series(range(rows_per_day)) % 10) / 10.0,
                "breakout_distance_bps": (pd.Series(range(rows_per_day)) - 25) / 2.0,
                "return_1_bps": pd.Series(range(rows_per_day)).diff().fillna(0.0),
                "estimated_cost_bps": 0.6,
                "orb_width_bps": 12.0,
                "vwap_approx": price + 0.02,
                "spread_proxy_bps": 1.0,
                "slippage_proxy_bps": 0.4,
            }
        )
        frame.to_parquet(day_dir / "SPY.parquet", index=False)
