from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_settings
from features.feature_pipeline import inspect_feature_dependencies_for_build


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
                "SUPPORTED_SYMBOLS=SPY",
                "FEATURE_SET=hybrid_intraday",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file=env_file, config_dir=CONFIG_DIR, environment="development")


def write_market_partition_without_depth_or_volume(root: Path, session_date: str, symbol: str) -> None:
    start = pd.Timestamp(f"{session_date} 13:30:00+00:00")
    frame = pd.DataFrame(
        [
            {
                "timestamp": (start + pd.Timedelta(minutes=minute)).isoformat(),
                "symbol": symbol,
                "last_price": 500.0 + minute * 0.05,
                "bid": 499.98 + minute * 0.05,
                "ask": 500.02 + minute * 0.05,
                "event_type": "snapshot",
                "source": "ib_snapshot",
                "collected_at": (start + pd.Timedelta(minutes=minute)).isoformat(),
            }
            for minute in range(12)
        ]
    )
    target = root / session_date / symbol / "collector_00001.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)


def test_dependency_inspection_omits_incompatible_indicators(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    raw_root = Path(settings.paths.market_raw_dir)
    write_market_partition_without_depth_or_volume(raw_root, "2026-04-14", "SPY")

    report = inspect_feature_dependencies_for_build(
        settings,
        symbols=["SPY"],
        input_root=raw_root,
        feature_set_name="full_experimental",
    )

    omitted = {item["name"]: item for item in report["omitted_indicators"]}
    assert "relative_volume" in omitted
    assert "microprice_proxy" in omitted
    assert "rolling_volume_mean" in omitted
    assert "Missing usable dependency columns" in omitted["relative_volume"]["reason"]
