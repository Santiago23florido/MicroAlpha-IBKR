from __future__ import annotations

from pathlib import Path

import pandas as pd

from app import build_parser
from config import load_settings
from data.cleaning import clean_market_data
from data.loader import load_market_data
from features.feature_pipeline import (
    FEATURE_COLUMNS,
    build_feature_frame,
    inspect_feature_dependencies_for_build,
    list_available_feature_sets,
    run_feature_build_pipeline,
)
from features.validation import validate_feature_store
from labels.dataset_builder import build_training_dataset
from monitoring.data_quality import assess_market_data_quality


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def build_settings(tmp_path: Path):
    data_root = tmp_path / "data"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"DATA_ROOT={data_root}",
                f"MARKET_RAW_DIR={data_root / 'raw' / 'market'}",
                f"LOG_FILE={data_root / 'logs' / 'pipeline.log'}",
                f"EXECUTION_LOG_FILE={data_root / 'reports' / 'executions.csv'}",
                f"RUNTIME_DB_PATH={data_root / 'processed' / 'runtime' / 'microalpha.db'}",
                f"MODEL_ARTIFACTS_DIR={data_root / 'models' / 'artifacts'}",
                f"MODEL_REGISTRY_PATH={data_root / 'models' / 'artifacts' / 'registry.json'}",
                "SUPPORTED_SYMBOLS=SPY,QQQ",
                "FEATURE_GAP_THRESHOLD_SECONDS=90",
                "FEATURE_ROLLING_SHORT_WINDOW=3",
                "FEATURE_ROLLING_MEDIUM_WINDOW=5",
                "FEATURE_ROLLING_LONG_WINDOW=8",
                "FEATURE_VWAP_WINDOW=4",
                "FEATURE_VOLUME_WINDOW=4",
                "FEATURE_SET=hybrid_intraday",
                "FEATURE_VALIDATION_MAX_NAN_RATIO=0.5",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file=env_file, config_dir=CONFIG_DIR, environment="development")


def write_market_partition(root: Path, session_date: str, symbol: str, *, nested: bool = True) -> Path:
    start = pd.Timestamp(f"{session_date} 13:30:00+00:00")
    rows = []
    for minute in range(12):
        timestamp = start + pd.Timedelta(minutes=minute)
        last_price = 500.0 + minute * 0.05 + (10 if symbol == "QQQ" else 0)
        bid = last_price - 0.02
        ask = last_price + 0.02
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "last_price": last_price,
                "bid": bid,
                "ask": ask,
                "spread": ask - bid,
                "bid_size": 100 + minute,
                "ask_size": 98 + minute,
                "last_size": 10 + minute,
                "volume": 1000 + minute * 10,
                "event_type": "snapshot",
                "source": "ib_snapshot",
                "session_window": "opening_range" if minute < 5 else "primary",
                "is_market_open": True,
                "exchange_time": (timestamp.tz_convert("America/New_York")).isoformat(),
                "collected_at": timestamp.isoformat(),
            }
        )
    frame = pd.DataFrame(rows)
    date_dir = root / session_date
    if nested:
        target_dir = date_dir / symbol
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "collector_00001.parquet"
    else:
        date_dir.mkdir(parents=True, exist_ok=True)
        target_path = date_dir / f"{symbol}.parquet"
    frame.to_parquet(target_path, index=False)
    return target_path


def test_parser_exposes_build_features_command() -> None:
    parser = build_parser()
    assert "build-features" in parser.format_help()
    assert "list-feature-sets" in parser.format_help()
    assert "inspect-feature-dependencies" in parser.format_help()
    assert "validate-features" in parser.format_help()


def test_load_market_data_supports_multiple_days_and_layouts(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_root = Path(settings.paths.market_raw_dir)
    write_market_partition(raw_root, "2026-04-14", "SPY", nested=True)
    write_market_partition(raw_root, "2026-04-15", "QQQ", nested=False)

    frame = load_market_data(
        settings,
        symbols=["SPY", "QQQ"],
        start_date="2026-04-14",
        end_date="2026-04-15",
    )

    assert len(frame) == 24
    assert set(frame["symbol"].unique()) == {"SPY", "QQQ"}
    assert "source_file" in frame.columns


def test_data_quality_detects_duplicates_bad_quotes_and_gaps(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_root = Path(settings.paths.market_raw_dir)
    write_market_partition(raw_root, "2026-04-14", "SPY", nested=True)
    frame = load_market_data(settings, symbols=["SPY"])
    broken = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    broken.loc[0, "bid"] = broken.loc[0, "ask"] + 1.0
    broken.loc[1, "timestamp"] = pd.Timestamp("2026-04-14 22:30:00+00:00")
    broken.loc[2, "timestamp"] = pd.Timestamp("2026-04-14 13:50:00+00:00")

    report = assess_market_data_quality(broken, settings)

    assert report.duplicate_count > 0
    assert report.bid_gt_ask_count > 0
    assert report.outside_regular_hours_count > 0
    assert report.large_gap_count > 0


def test_clean_build_and_store_features_end_to_end(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_root = Path(settings.paths.market_raw_dir)
    feature_root = tmp_path / "features_output"
    write_market_partition(raw_root, "2026-04-14", "SPY", nested=True)
    write_market_partition(raw_root, "2026-04-14", "QQQ", nested=True)

    summary = run_feature_build_pipeline(
        settings,
        symbols=["SPY", "QQQ"],
        input_root=raw_root,
        output_root=feature_root,
    )

    assert summary["input_rows"] == 24
    assert summary["cleaned_rows"] == 24
    assert summary["feature_rows"] == 24
    assert len(summary["written_files"]) == 2
    assert summary["feature_set_name"] == "hybrid_intraday"
    assert summary["manifest"]["feature_set_name"] == "hybrid_intraday"
    written_path = Path(summary["written_files"][0])
    assert written_path.exists()
    frame = pd.read_parquet(written_path)
    for column in FEATURE_COLUMNS:
        assert column in frame.columns
    assert Path(summary["report_path"]).exists()


def test_list_feature_sets_and_inspect_dependencies(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_root = Path(settings.paths.market_raw_dir)
    write_market_partition(raw_root, "2026-04-14", "SPY", nested=True)

    listing = list_available_feature_sets(settings)
    assert listing["default_feature_set"] == "hybrid_intraday"
    assert any(item["name"] == "technical_basic" for item in listing["feature_sets"])

    report = inspect_feature_dependencies_for_build(
        settings,
        symbols=["SPY"],
        input_root=raw_root,
        feature_set_name="hybrid_intraday",
    )
    assert report["feature_set"]["name"] == "hybrid_intraday"
    assert len(report["compatible_indicators"]) > 0
    assert "spread_bps" in report["planned_feature_columns"]


def test_validate_feature_store_reports_feature_quality(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_root = Path(settings.paths.market_raw_dir)
    feature_root = tmp_path / "features_output"
    write_market_partition(raw_root, "2026-04-14", "SPY", nested=True)

    run_feature_build_pipeline(
        settings,
        symbols=["SPY"],
        input_root=raw_root,
        output_root=feature_root,
        feature_set_name="hybrid_intraday",
    )

    report = validate_feature_store(settings, feature_root=feature_root, symbols=["SPY"])
    assert report["status"] in {"ok", "warning"}
    assert report["quality"]["feature_count"] > 0
    assert Path(report["report_path"]).exists()


def test_dataset_builder_creates_temporal_split(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_root = Path(settings.paths.market_raw_dir)
    write_market_partition(raw_root, "2026-04-14", "SPY", nested=True)

    raw_frame = load_market_data(settings, symbols=["SPY"])
    cleaned_frame = clean_market_data(raw_frame, settings)
    feature_frame = build_feature_frame(cleaned_frame, settings)
    dataset = build_training_dataset(
        feature_frame,
        settings,
        feature_columns=[
            "spread_bps",
            "relative_volume",
            "seconds_since_open",
            "orb_width_bps",
            "estimated_cost_bps",
        ],
    )

    assert dataset.x_train.shape[1] == len(dataset.feature_columns)
    assert len(dataset.train_frame) > 0
    assert len(dataset.test_frame) > 0
    assert dataset.train_frame["timestamp"].max() <= dataset.test_frame["timestamp"].min()
