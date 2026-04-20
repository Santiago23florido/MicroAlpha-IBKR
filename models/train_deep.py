from __future__ import annotations

import json
import logging
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

logger = logging.getLogger("microalpha.training.deep")


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
    device = _resolve_training_device()
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

    model = DeepLOBLite(feature_dim=len(feature_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    total_batches = len(train_loader)
    batch_log_interval = max(total_batches // 10, 1)
    logger.info(
        "Deep training device | device=%s",
        _describe_device(device),
    )

    for epoch_index in range(epochs):
        model.train()
        cumulative_loss = 0.0
        logger.info(
            "Deep training progress | epoch %s/%s | batches=%s",
            epoch_index + 1,
            epochs,
            total_batches,
        )
        for batch_index, (features, target_class, target_return) in enumerate(train_loader, start=1):
            features = features.to(device, non_blocking=device.type == "cuda")
            target_class = target_class.to(device, non_blocking=device.type == "cuda")
            target_return = target_return.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad()
            logits, predicted_return = model(features)
            loss = cls_loss_fn(logits, target_class) + 0.1 * reg_loss_fn(predicted_return, target_return)
            loss.backward()
            optimizer.step()
            cumulative_loss += float(loss.item())
            if batch_index == total_batches or batch_index % batch_log_interval == 0:
                logger.info(
                    "Deep training progress | epoch %s/%s | batch %s/%s | loss=%.6f",
                    epoch_index + 1,
                    epochs,
                    batch_index,
                    total_batches,
                    float(loss.item()),
                )

        epoch_metrics = _evaluate_model(model, valid_loader, device)
        logger.info(
            "Deep training epoch complete | epoch %s/%s | avg_loss=%.6f | accuracy=%.4f | return_mae_bps=%.4f",
            epoch_index + 1,
            epochs,
            cumulative_loss / max(total_batches, 1),
            epoch_metrics["accuracy"],
            epoch_metrics["return_mae_bps"],
        )

    metrics = _evaluate_model(model, valid_loader, device)
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


def _evaluate_model(model: nn.Module, valid_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    actual_classes = []
    predicted_classes = []
    actual_returns = []
    predicted_returns = []
    with torch.no_grad():
        for features, target_class, target_return in valid_loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            target_class = target_class.to(device, non_blocking=device.type == "cuda")
            target_return = target_return.to(device, non_blocking=device.type == "cuda")
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


def _resolve_training_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def _describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})"
