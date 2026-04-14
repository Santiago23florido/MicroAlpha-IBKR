from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import Settings
from features.feature_pipeline import FEATURE_COLUMNS


@dataclass(frozen=True)
class TrainingDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def build_training_dataset(
    feature_frame: pd.DataFrame,
    settings: Settings,
    *,
    feature_columns: Sequence[str] | None = None,
    target_column: str = "target_future_return_bps_placeholder",
) -> TrainingDataset:
    if feature_frame.empty:
        raise ValueError("Feature frame is empty. Build features before generating a training dataset.")

    columns = [column for column in (feature_columns or FEATURE_COLUMNS) if column in feature_frame.columns]
    if not columns:
        raise ValueError("No requested feature columns are available in the feature frame.")

    frame = feature_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    horizon = settings.feature_pipeline.label_horizon_rows
    group_columns = ["symbol", "session_date"] if "session_date" in frame.columns else ["symbol"]
    frame[target_column] = (
        frame.groupby(group_columns)["mid_price"].shift(-horizon) / frame["mid_price"] - 1.0
    ) * 10000.0
    frame = frame.dropna(subset=columns + [target_column]).reset_index(drop=True)

    split = temporal_train_test_split(frame, train_ratio=settings.feature_pipeline.train_split_ratio)
    x_train = split["train"][columns].to_numpy(dtype=float)
    y_train = split["train"][target_column].to_numpy(dtype=float)
    x_test = split["test"][columns].to_numpy(dtype=float)
    y_test = split["test"][target_column].to_numpy(dtype=float)

    return TrainingDataset(
        frame=frame,
        feature_columns=list(columns),
        target_column=target_column,
        train_frame=split["train"],
        test_frame=split["test"],
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def temporal_train_test_split(frame: pd.DataFrame, *, train_ratio: float = 0.8) -> dict[str, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    ordered = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    split_index = max(int(len(ordered) * train_ratio), 1)
    split_index = min(split_index, len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    return {
        "train": ordered.iloc[:split_index].reset_index(drop=True),
        "test": ordered.iloc[split_index:].reset_index(drop=True),
    }
