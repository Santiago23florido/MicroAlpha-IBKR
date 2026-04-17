from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import Settings
from data.feature_loader import FEATURE_METADATA_COLUMNS, infer_feature_columns
from data.loader import list_market_data_files
from features.validation import assess_feature_quality
from models.config import DatasetConfig, TargetConfig, load_modeling_config, resolve_target_config


@dataclass(frozen=True)
class TemporalSplit:
    name: str
    frame: pd.DataFrame
    dates: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return int(len(self.frame))


@dataclass(frozen=True)
class ModelingDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    dropped_feature_columns: dict[str, str]
    target_column: str
    target_config: dict[str, Any]
    split_config: dict[str, Any]
    train: TemporalSplit
    validation: TemporalSplit
    test: TemporalSplit
    metadata_columns: list[str] = field(default_factory=list)

    @property
    def x_train(self) -> np.ndarray:
        return self.train.frame[self.feature_columns].to_numpy(dtype=float)

    @property
    def y_train(self) -> np.ndarray:
        return self.train.frame[self.target_column].to_numpy()

    @property
    def x_validation(self) -> np.ndarray:
        return self.validation.frame[self.feature_columns].to_numpy(dtype=float)

    @property
    def y_validation(self) -> np.ndarray:
        return self.validation.frame[self.target_column].to_numpy()

    @property
    def x_test(self) -> np.ndarray:
        return self.test.frame[self.feature_columns].to_numpy(dtype=float)

    @property
    def y_test(self) -> np.ndarray:
        return self.test.frame[self.target_column].to_numpy()

    @property
    def train_frame(self) -> pd.DataFrame:
        return self.train.frame

    @property
    def validation_frame(self) -> pd.DataFrame:
        return self.validation.frame

    @property
    def test_frame(self) -> pd.DataFrame:
        return self.test.frame

    def to_metadata(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "target_config": self.target_config,
            "feature_columns": list(self.feature_columns),
            "dropped_feature_columns": dict(self.dropped_feature_columns),
            "metadata_columns": list(self.metadata_columns),
            "split_config": dict(self.split_config),
            "rows": {
                "total": int(len(self.frame)),
                "train": self.train.row_count,
                "validation": self.validation.row_count,
                "test": self.test.row_count,
            },
            "date_ranges": {
                "train": list(self.train.dates),
                "validation": list(self.validation.dates),
                "test": list(self.test.dates),
            },
            "symbols": sorted(self.frame["symbol"].dropna().astype(str).unique().tolist()) if "symbol" in self.frame.columns else [],
        }


def build_training_dataset(
    feature_frame: pd.DataFrame,
    settings: Settings,
    *,
    feature_columns: Sequence[str] | None = None,
    target_column: str = "target_future_return_bps_placeholder",
) -> ModelingDataset:
    frame = feature_frame.copy()
    if target_column not in frame.columns:
        horizon = settings.feature_pipeline.label_horizon_rows
        price_column = "mid_price" if "mid_price" in frame.columns else "last_price"
        group_columns = ["symbol", "session_date"] if "session_date" in frame.columns else ["symbol"]
        frame[target_column] = (frame.groupby(group_columns)[price_column].shift(-horizon) / frame[price_column] - 1.0) * 10000.0
    dataset_config = load_modeling_config(settings).dataset
    target_config = TargetConfig(
        name=target_column,
        description="Compatibility dataset target.",
        task_type="regression",
        horizon_bars=settings.feature_pipeline.label_horizon_rows,
    )
    relaxed_config = replace(
        dataset_config,
        min_rows=min(dataset_config.min_rows, 8),
        min_train_rows=min(dataset_config.min_train_rows, 4),
        min_validation_rows=min(dataset_config.min_validation_rows, 1),
        min_test_rows=min(dataset_config.min_test_rows, 1),
        min_unique_dates=1,
        strict_feature_validation=False,
    )
    try:
        return build_modeling_dataset_from_frame(
            frame,
            settings,
            feature_columns=feature_columns,
            target_column=target_column,
            target_config=target_config,
            dataset_config=relaxed_config,
        )
    except ValueError as exc:
        if "Temporal split requires at least" not in str(exc) and "Temporal split produced an empty" not in str(exc):
            raise
        return _build_rowwise_compatibility_dataset(
            frame,
            settings,
            feature_columns=feature_columns,
            target_column=target_column,
            target_config=target_config,
        )


def build_modeling_dataset(
    settings: Settings,
    *,
    feature_set_name: str,
    target_mode: str,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    label_root: str | Path | None = None,
    feature_columns: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
) -> ModelingDataset:
    labeled_frame = load_labeled_data(
        settings,
        feature_set_name=feature_set_name,
        target_mode=target_mode,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        label_root=label_root,
    )
    target_config = resolve_target_config(settings, target_mode)
    target_column = f"target_{target_mode}"
    dataset_config = load_modeling_config(settings).dataset
    return build_modeling_dataset_from_frame(
        labeled_frame,
        settings,
        feature_columns=feature_columns,
        exclude_columns=exclude_columns,
        target_column=target_column,
        target_config=target_config,
        dataset_config=dataset_config,
    )


def build_modeling_dataset_from_frame(
    frame: pd.DataFrame,
    settings: Settings,
    *,
    feature_columns: Sequence[str] | None,
    exclude_columns: Sequence[str] | None = None,
    target_column: str,
    target_config: TargetConfig,
    dataset_config: DatasetConfig,
) -> ModelingDataset:
    if frame.empty:
        raise ValueError("Dataset frame is empty.")
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is missing from the dataset frame.")

    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    if "session_date" not in working.columns:
        working["session_date"] = working["timestamp"].dt.strftime("%Y-%m-%d")
    working = working.sort_values(["session_date", "timestamp", "symbol"]).reset_index(drop=True)
    working = working.replace([np.inf, -np.inf], np.nan)

    requested_features = list(feature_columns or infer_feature_columns(working))
    excluded = set(exclude_columns or [])
    requested_features = [column for column in requested_features if column not in excluded]
    requested_features = [column for column in requested_features if column in working.columns]
    if not requested_features:
        raise ValueError("No feature columns are available after selection.")

    if dataset_config.dropna_target:
        working = working.dropna(subset=[target_column]).reset_index(drop=True)
    if target_config.task_type in {"classification", "ordinal", "distribution_bins"}:
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        if dataset_config.dropna_target:
            working = working.dropna(subset=[target_column]).reset_index(drop=True)
        working[target_column] = working[target_column].astype(int)
    else:
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")

    candidate_columns = list(dict.fromkeys([*requested_features, target_column]))
    feature_quality = assess_feature_quality(working[candidate_columns].copy(), settings)

    dropped_feature_columns: dict[str, str] = {}
    for column in requested_features:
        if column in feature_quality.empty_features:
            dropped_feature_columns[column] = "empty_feature"
            continue
        nan_ratio = feature_quality.excessive_nan_features.get(column, 0.0)
        if nan_ratio > settings.feature_pipeline.validation_max_nan_ratio:
            dropped_feature_columns[column] = f"excessive_nan_ratio:{nan_ratio:.6f}"
            continue
        if column in feature_quality.constant_features:
            dropped_feature_columns[column] = "constant_feature"

    valid_feature_columns = [column for column in requested_features if column not in dropped_feature_columns]
    if not valid_feature_columns:
        raise ValueError("No valid feature columns remain after validation filters.")

    working = working.dropna(subset=[*valid_feature_columns, target_column]).reset_index(drop=True)
    if len(working) < dataset_config.min_rows:
        raise ValueError(
            f"Dataset has {len(working)} rows after filtering, below the minimum required {dataset_config.min_rows}."
        )

    split_frames = temporal_train_validation_test_split(working, dataset_config)
    if len(split_frames["train"]) < dataset_config.min_train_rows:
        raise ValueError("Temporal train split is too small.")
    if len(split_frames["validation"]) < dataset_config.min_validation_rows:
        raise ValueError("Temporal validation split is too small.")
    if len(split_frames["test"]) < dataset_config.min_test_rows:
        raise ValueError("Temporal test split is too small.")

    metadata_columns = [
        column
        for column in working.columns
        if column not in valid_feature_columns and column != target_column and column in FEATURE_METADATA_COLUMNS | {
            "future_price_proxy",
            "future_return_bps",
            "future_net_return_bps",
            "target_cost_adjustment_bps",
            "source_file",
        }
    ]

    return ModelingDataset(
        frame=working,
        feature_columns=valid_feature_columns,
        dropped_feature_columns=dropped_feature_columns,
        target_column=target_column,
        target_config=target_config.to_dict(),
        split_config={
            "train_ratio": dataset_config.train_ratio,
            "validation_ratio": dataset_config.validation_ratio,
            "test_ratio": dataset_config.test_ratio,
            "min_rows": dataset_config.min_rows,
            "min_train_rows": dataset_config.min_train_rows,
            "min_validation_rows": dataset_config.min_validation_rows,
            "min_test_rows": dataset_config.min_test_rows,
            "min_unique_dates": dataset_config.min_unique_dates,
            "dropna_target": dataset_config.dropna_target,
            "strict_feature_validation": dataset_config.strict_feature_validation,
        },
        train=TemporalSplit(
            "train",
            split_frames["train"],
            tuple(sorted(split_frames["train"]["session_date"].unique().tolist())),
        ),
        validation=TemporalSplit(
            "validation",
            split_frames["validation"],
            tuple(sorted(split_frames["validation"]["session_date"].unique().tolist())),
        ),
        test=TemporalSplit(
            "test",
            split_frames["test"],
            tuple(sorted(split_frames["test"]["session_date"].unique().tolist())),
        ),
        metadata_columns=metadata_columns,
    )


def temporal_train_validation_test_split(frame: pd.DataFrame, dataset_config: DatasetConfig) -> dict[str, pd.DataFrame]:
    unique_dates = sorted(frame["session_date"].dropna().astype(str).unique().tolist())
    if len(unique_dates) < dataset_config.min_unique_dates:
        raise ValueError(
            f"Temporal split requires at least {dataset_config.min_unique_dates} unique session dates, found {len(unique_dates)}."
        )

    total_dates = len(unique_dates)
    train_cut = max(int(total_dates * dataset_config.train_ratio), 1)
    validation_cut = max(int(total_dates * (dataset_config.train_ratio + dataset_config.validation_ratio)), train_cut + 1)
    validation_cut = min(validation_cut, total_dates - 1)
    train_dates = unique_dates[:train_cut]
    validation_dates = unique_dates[train_cut:validation_cut]
    test_dates = unique_dates[validation_cut:]

    if not validation_dates or not test_dates:
        raise ValueError("Temporal split produced an empty validation or test date range.")

    return {
        "train": frame[frame["session_date"].isin(train_dates)].reset_index(drop=True),
        "validation": frame[frame["session_date"].isin(validation_dates)].reset_index(drop=True),
        "test": frame[frame["session_date"].isin(test_dates)].reset_index(drop=True),
    }


def temporal_train_test_split(frame: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    if "session_date" not in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        working["session_date"] = working["timestamp"].dt.strftime("%Y-%m-%d")
    unique_dates = sorted(working["session_date"].dropna().astype(str).unique().tolist())
    if len(unique_dates) < 2:
        raise ValueError("Temporal train/test split requires at least two unique session dates.")
    train_cut = max(int(len(unique_dates) * train_ratio), 1)
    train_cut = min(train_cut, len(unique_dates) - 1)
    train_dates = unique_dates[:train_cut]
    test_dates = unique_dates[train_cut:]
    return (
        working[working["session_date"].isin(train_dates)].reset_index(drop=True),
        working[working["session_date"].isin(test_dates)].reset_index(drop=True),
    )


def load_labeled_data(
    settings: Settings,
    *,
    feature_set_name: str,
    target_mode: str,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    label_root: str | Path | None = None,
) -> pd.DataFrame:
    base_root = Path(label_root or Path(settings.paths.processed_dir) / "labels")
    root_dir = base_root / feature_set_name / target_mode
    files = list_market_data_files(
        root_dir,
        symbols=symbols or settings.supported_symbols,
        start_date=start_date,
        end_date=end_date,
    )
    if not files:
        raise FileNotFoundError(
            f"No labeled parquet files found under {root_dir}. Run build-labels first or pass --label-root."
        )

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    if "symbol" in combined.columns:
        combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.sort_values(["symbol", "timestamp", "source_file"]).reset_index(drop=True)
    return combined


def _build_rowwise_compatibility_dataset(
    frame: pd.DataFrame,
    settings: Settings,
    *,
    feature_columns: Sequence[str] | None,
    target_column: str,
    target_config: TargetConfig,
) -> ModelingDataset:
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    if "session_date" not in working.columns:
        working["session_date"] = working["timestamp"].dt.strftime("%Y-%m-%d")
    working = working.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    selected_features = list(feature_columns or infer_feature_columns(working))
    selected_features = [column for column in selected_features if column in working.columns]
    working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=[*selected_features, target_column]).reset_index(drop=True)
    if len(working) < 3:
        raise ValueError("Compatibility dataset requires at least three rows after filtering.")

    total_rows = len(working)
    train_end = max(int(total_rows * 0.6), 1)
    validation_end = max(int(total_rows * 0.8), train_end + 1)
    validation_end = min(validation_end, total_rows - 1)
    train_frame = working.iloc[:train_end].reset_index(drop=True)
    validation_frame = working.iloc[train_end:validation_end].reset_index(drop=True)
    test_frame = working.iloc[validation_end:].reset_index(drop=True)
    if validation_frame.empty:
        validation_frame = test_frame.copy()
    return ModelingDataset(
        frame=working,
        feature_columns=selected_features,
        dropped_feature_columns={},
        target_column=target_column,
        target_config=target_config.to_dict(),
        split_config={"mode": "rowwise_compatibility"},
        train=TemporalSplit(
            "train",
            train_frame,
            tuple(sorted(train_frame["session_date"].dropna().astype(str).unique().tolist())),
        ),
        validation=TemporalSplit(
            "validation",
            validation_frame,
            tuple(sorted(validation_frame["session_date"].dropna().astype(str).unique().tolist())),
        ),
        test=TemporalSplit(
            "test",
            test_frame,
            tuple(sorted(test_frame["session_date"].dropna().astype(str).unique().tolist())),
        ),
        metadata_columns=[column for column in FEATURE_METADATA_COLUMNS if column in working.columns],
    )
