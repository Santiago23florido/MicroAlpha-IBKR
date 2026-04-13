from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import Settings
from data.historical_loader import load_historical_dataset
from features.preprocessing import prepare_training_dataframe
from models.registry import ModelRegistry


@dataclass(frozen=True)
class SequenceBatch:
    features: np.ndarray
    target_class: np.ndarray
    target_return: np.ndarray


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, batch: SequenceBatch) -> None:
        self.features = torch.tensor(batch.features, dtype=torch.float32)
        self.target_class = torch.tensor(batch.target_class, dtype=torch.long)
        self.target_return = torch.tensor(batch.target_return, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.target_class[index], self.target_return[index]


class DeepLOBLite(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.cls_head = nn.Linear(hidden_dim, 3)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_input = features.transpose(1, 2)
        conv_output = self.conv(conv_input).transpose(1, 2)
        lstm_output, _ = self.lstm(conv_output)
        final_hidden = lstm_output[:, -1, :]
        return self.cls_head(final_hidden), self.reg_head(final_hidden).squeeze(-1)


def train_deep_model(
    settings: Settings,
    *,
    data_path: str | None = None,
    model_name: str = "deep_lob_lite",
    epochs: int = 6,
    set_active: bool = True,
) -> dict[str, object]:
    raw_frame = load_historical_dataset(settings, data_path)
    prepared = prepare_training_dataframe(raw_frame, settings)
    frame = prepared.frame.reset_index(drop=True)
    sequence_length = settings.models.sequence_length
    feature_columns = prepared.feature_columns
    batch = _build_sequence_batch(frame, feature_columns, sequence_length)
    split_index = max(int(len(batch.features) * 0.8), 1)

    train_dataset = SequenceDataset(
        SequenceBatch(
            features=batch.features[:split_index],
            target_class=batch.target_class[:split_index],
            target_return=batch.target_return[:split_index],
        )
    )
    valid_dataset = SequenceDataset(
        SequenceBatch(
            features=batch.features[split_index:],
            target_class=batch.target_class[split_index:],
            target_return=batch.target_return[split_index:],
        )
    )
    if len(valid_dataset) == 0:
        valid_dataset = train_dataset

    model = DeepLOBLite(feature_dim=len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    for _ in range(epochs):
        model.train()
        for features, target_class, target_return in train_loader:
            optimizer.zero_grad()
            logits, predicted_return = model(features)
            loss = cls_loss_fn(logits, target_class) + 0.1 * reg_loss_fn(predicted_return, target_return)
            loss.backward()
            optimizer.step()

    metrics = _evaluate_model(model, valid_loader)
    artifact_id = f"deep-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    artifact_dir = Path(settings.models.artifacts_dir) / "deep"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{artifact_id}.pt"
    metadata_path = artifact_dir / f"{artifact_id}.metadata.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "sequence_length": sequence_length,
            "model_name": model_name,
            "class_index": {-1: 0, 0: 1, 1: 2},
        },
        artifact_path,
    )

    metadata = {
        "artifact_id": artifact_id,
        "model_name": model_name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_source": data_path or "data/sample/spy_microstructure_sample.csv",
        "feature_set": feature_columns,
        "target_definition": {
            "horizon_minutes": settings.models.target_horizon_minutes,
            "classification_threshold_bps": prepared.class_threshold_bps,
            "sequence_length": sequence_length,
        },
        "metrics": metrics,
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    registry = ModelRegistry(settings.models.registry_path)
    record = registry.register_model(
        "deep",
        {
            "artifact_id": artifact_id,
            "model_name": model_name,
            "artifact_path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "metrics": metrics,
            "feature_set": feature_columns,
            "target_definition": metadata["target_definition"],
        },
        set_active=set_active,
    )
    return {"record": record, "metadata": metadata}


def _build_sequence_batch(
    frame,
    feature_columns: list[str],
    sequence_length: int,
) -> SequenceBatch:
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    target_class_map = {-1: 0, 0: 1, 1: 2}
    sequence_rows = []
    class_rows = []
    return_rows = []
    for end_index in range(sequence_length - 1, len(frame)):
        start_index = end_index - sequence_length + 1
        sequence_rows.append(features[start_index : end_index + 1])
        class_rows.append(target_class_map[int(frame.iloc[end_index]["target_class"])])
        return_rows.append(float(frame.iloc[end_index]["future_return_bps"]))
    return SequenceBatch(
        features=np.asarray(sequence_rows, dtype=np.float32),
        target_class=np.asarray(class_rows, dtype=np.int64),
        target_return=np.asarray(return_rows, dtype=np.float32),
    )


def _evaluate_model(model: nn.Module, valid_loader: DataLoader) -> dict[str, float]:
    model.eval()
    actual_classes = []
    predicted_classes = []
    actual_returns = []
    predicted_returns = []
    with torch.no_grad():
        for features, target_class, target_return in valid_loader:
            logits, predicted_return = model(features)
            predicted_class = torch.argmax(logits, dim=1)
            actual_classes.extend(target_class.cpu().numpy().tolist())
            predicted_classes.extend(predicted_class.cpu().numpy().tolist())
            actual_returns.extend(target_return.cpu().numpy().tolist())
            predicted_returns.extend(predicted_return.cpu().numpy().tolist())
    return {
        "accuracy": float(accuracy_score(actual_classes, predicted_classes)),
        "return_mae_bps": float(mean_absolute_error(actual_returns, predicted_returns)),
    }
