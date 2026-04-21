from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LOBSequenceBatch:
    features: np.ndarray
    target_class: np.ndarray
    target_return: np.ndarray
    session_dates: np.ndarray
    target_timestamps: np.ndarray


class LOBSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, batch: LOBSequenceBatch) -> None:
        self.features = torch.tensor(batch.features, dtype=torch.float32)
        self.target_class = torch.tensor(batch.target_class, dtype=torch.long)
        self.target_return = torch.tensor(batch.target_return, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.target_class[index], self.target_return[index]


class DeepLOBReferenceLike(nn.Module):
    def __init__(self, feature_dim: int, sequence_length: int, channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), padding=(0, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, channels, kernel_size=(1, 2), padding=(0, 1)),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((sequence_length, 1)),
        )
        self.temporal = nn.LSTM(channels, channels, batch_first=True)
        self.cls_head = nn.Linear(channels, 3)
        self.reg_head = nn.Linear(channels, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = features.unsqueeze(1)
        spatial = self.spatial(tensor).squeeze(-1).transpose(1, 2)
        temporal, _ = self.temporal(spatial)
        final_hidden = temporal[:, -1, :]
        return self.cls_head(final_hidden), self.reg_head(final_hidden).squeeze(-1)


def is_lob_dataset(frame: pd.DataFrame) -> bool:
    if "dataset_type" in frame.columns:
        try:
            return str(frame["dataset_type"].iloc[0]) == "ibkr_lob_depth"
        except IndexError:
            return False
    return {"bid_px_1", "ask_px_1", "bid_sz_1", "ask_sz_1"}.issubset(frame.columns)


def lob_feature_columns(depth_levels: int) -> list[str]:
    return (
        [f"bid_px_{index}" for index in range(1, depth_levels + 1)]
        + [f"ask_px_{index}" for index in range(1, depth_levels + 1)]
        + [f"bid_sz_{index}" for index in range(1, depth_levels + 1)]
        + [f"ask_sz_{index}" for index in range(1, depth_levels + 1)]
    )


def build_lob_sequence_batch(
    frame: pd.DataFrame,
    *,
    depth_levels: int,
    sequence_length: int,
) -> LOBSequenceBatch:
    feature_columns = lob_feature_columns(depth_levels)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"LOB dataset is missing required depth columns: {missing}")

    normalized = frame.copy()
    normalized["event_ts_utc"] = pd.to_datetime(normalized["event_ts_utc"], utc=True)
    if "session_date" not in normalized.columns:
        normalized["session_date"] = normalized["event_ts_utc"].dt.date.astype(str)
    if "target_class" not in normalized.columns or "future_return_bps" not in normalized.columns:
        raise ValueError("LOB dataset must include target_class and future_return_bps.")

    sequences: list[np.ndarray] = []
    classes: list[int] = []
    returns: list[float] = []
    session_dates: list[str] = []
    target_timestamps: list[str] = []
    target_class_map = {-1: 0, 0: 1, 1: 2}
    for session_date, session_frame in normalized.groupby("session_date", sort=True):
        session_rows = session_frame.reset_index(drop=True)
        raw_values = session_rows[feature_columns].to_numpy(dtype=np.float32)
        mid_prices = (
            (session_rows["bid_px_1"].to_numpy(dtype=np.float32) + session_rows["ask_px_1"].to_numpy(dtype=np.float32))
            / 2.0
        )
        for end_index in range(sequence_length - 1, len(session_rows)):
            start_index = end_index - sequence_length + 1
            window = raw_values[start_index : end_index + 1].copy()
            last_mid = float(max(mid_prices[end_index], 1e-9))
            price_count = depth_levels * 2
            price_window = window[:, :price_count]
            size_window = window[:, price_count:]
            window[:, :price_count] = ((price_window / last_mid) - 1.0) * 10000.0
            window[:, price_count:] = np.log1p(np.clip(size_window, a_min=0.0, a_max=None))
            sequences.append(window)
            classes.append(target_class_map[int(session_rows.iloc[end_index]["target_class"])])
            returns.append(float(session_rows.iloc[end_index]["future_return_bps"]))
            session_dates.append(str(session_date))
            target_timestamps.append(session_rows.iloc[end_index]["event_ts_utc"].isoformat())
    if not sequences:
        raise ValueError(
            f"LOB dataset does not have enough rows to build sequences of length {sequence_length}."
        )
    return LOBSequenceBatch(
        features=np.asarray(sequences, dtype=np.float32),
        target_class=np.asarray(classes, dtype=np.int64),
        target_return=np.asarray(returns, dtype=np.float32),
        session_dates=np.asarray(session_dates),
        target_timestamps=np.asarray(target_timestamps),
    )


def evaluate_lob_predictions(
    *,
    actual_classes: list[int],
    predicted_classes: list[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual_classes, predicted_classes)),
        "macro_f1": float(f1_score(actual_classes, predicted_classes, average="macro")),
    }
