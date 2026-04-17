from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from config import Settings
from data.loader import list_market_data_files


FEATURE_METADATA_COLUMNS = {
    "timestamp",
    "exchange_timestamp",
    "exchange_time",
    "collected_at",
    "symbol",
    "session_date",
    "session_time",
    "source_file",
    "event_type",
    "source",
    "session_window",
    "is_market_open",
}

NON_FEATURE_PREFIXES = ("target_", "future_")
NON_FEATURE_COLUMNS = {
    "target_cost_adjustment_bps",
}


def resolve_feature_root(
    settings: Settings,
    *,
    feature_set_name: str | None = None,
    feature_root: str | Path | None = None,
) -> Path:
    base_root = Path(feature_root or settings.paths.feature_dir)
    if feature_set_name:
        candidate = base_root / feature_set_name
        if candidate.exists():
            return candidate
        if feature_root is None and _looks_like_partition_root(base_root):
            return base_root
        if feature_root is None:
            return candidate
    return base_root


def load_feature_data(
    settings: Settings,
    *,
    feature_set_name: str | None = None,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_root: str | Path | None = None,
) -> pd.DataFrame:
    root_dir = resolve_feature_root(settings, feature_set_name=feature_set_name, feature_root=feature_root)
    files = list_market_data_files(
        root_dir,
        symbols=symbols or settings.supported_symbols,
        start_date=start_date,
        end_date=end_date,
    )
    if not files:
        raise FileNotFoundError(
            f"No feature parquet files found under {root_dir}. Run build-features first or pass --feature-root."
        )

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    if "exchange_timestamp" in combined.columns:
        combined["exchange_timestamp"] = pd.to_datetime(combined["exchange_timestamp"], utc=True, errors="coerce")
    if "collected_at" in combined.columns:
        combined["collected_at"] = pd.to_datetime(combined["collected_at"], utc=True, errors="coerce")
    if "symbol" in combined.columns:
        combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.sort_values(["symbol", "timestamp", "source_file"]).reset_index(drop=True)
    return combined


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in FEATURE_METADATA_COLUMNS
        and column not in NON_FEATURE_COLUMNS
        and not any(column.startswith(prefix) for prefix in NON_FEATURE_PREFIXES)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _looks_like_partition_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(child.is_dir() and child.name[:4].isdigit() for child in path.iterdir())
