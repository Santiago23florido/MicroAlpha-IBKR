from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import Settings
from data.historical_loader import load_historical_dataset
from data.lob_dataset import load_lob_capture_frame
from data.lob_labels import attach_lob_mid_price_labels
from features.preprocessing import prepare_training_dataframe
from models.deep_lob import (
    DeepLOBReferenceLike,
    LOBSequenceBatch,
    LOBSequenceDataset,
    build_lob_sequence_batch,
    evaluate_lob_predictions,
    is_lob_dataset,
    lob_feature_columns,
)
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
    model_name: str = "deep_lob_reference_like",
    epochs: int = 6,
    set_active: bool = True,
) -> dict[str, object]:
    device = _resolve_training_device()
    raw_frame = load_historical_dataset(settings, data_path)
    if is_lob_dataset(raw_frame):
        return _train_lob_deep_model(
            settings,
            raw_frame=raw_frame,
            data_path=data_path,
            model_name=model_name,
            epochs=epochs,
            set_active=set_active,
            device=device,
        )
    return _train_bar_deep_model(
        settings,
        raw_frame=raw_frame,
        data_path=data_path,
        model_name=model_name or "deep_lob_lite",
        epochs=epochs,
        set_active=set_active,
        device=device,
    )


def evaluate_deep_daily(
    settings: Settings,
    *,
    symbol: str,
    from_date: str,
    epochs: int = 2,
) -> dict[str, object]:
    raw_frame, source_files = load_lob_capture_frame(
        settings,
        symbol=symbol.upper(),
        from_date=from_date,
        to_date=None,
    )
    if raw_frame.empty:
        raise ValueError(f"No LOB capture data available for {symbol.upper()} from {from_date}.")

    labeled = attach_lob_mid_price_labels(
        raw_frame,
        horizon_events=settings.models.lob_horizon_events,
        stationary_threshold_bps=settings.models.lob_stationary_threshold_bps,
    )
    labeled["dataset_type"] = "ibkr_lob_depth"
    batch = build_lob_sequence_batch(
        labeled,
        depth_levels=settings.models.lob_depth_levels,
        sequence_length=settings.models.lob_sequence_length,
    )
    unique_dates = sorted(pd.unique(batch.session_dates).tolist())
    if len(unique_dates) < 2:
        raise ValueError("Walk-forward daily evaluation requires at least two captured session dates.")

    device = _resolve_training_device()
    rows: list[dict[str, object]] = []
    for eval_date in unique_dates[1:]:
        train_mask = batch.session_dates < eval_date
        test_mask = batch.session_dates == eval_date
        if not np.any(train_mask) or not np.any(test_mask):
            continue
        train_batch = _select_lob_batch(batch, train_mask)
        valid_batch = _select_lob_batch(batch, test_mask)
        model = _fit_lob_model(
            train_batch,
            feature_dim=train_batch.features.shape[2],
            sequence_length=settings.models.lob_sequence_length,
            train_batch_size=settings.models.lob_train_batch_size,
            epochs=epochs,
            device=device,
        )
        metrics = _evaluate_lob_batch(model, valid_batch, device)
        rows.append(
            {
                "session_date": eval_date,
                "row_count": int(len(valid_batch.target_class)),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "return_mae_bps": metrics["return_mae_bps"],
            }
        )
    if not rows:
        raise ValueError("Walk-forward daily evaluation could not build any train/test splits.")

    report_root = Path(settings.lob_capture.report_root) / symbol.upper()
    report_root.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = report_root / f"deep_daily_eval_{token}.json"
    csv_path = report_root / f"deep_daily_eval_{token}.csv"
    payload = {
        "status": "ok",
        "symbol": symbol.upper(),
        "provider": "ibkr",
        "dataset_type": "ibkr_lob_depth",
        "from_date": from_date,
        "epochs": epochs,
        "sequence_length": settings.models.lob_sequence_length,
        "depth_levels": settings.models.lob_depth_levels,
        "horizon_events": settings.models.lob_horizon_events,
        "source_files": source_files,
        "daily_metrics": rows,
        "mean_accuracy": float(pd.DataFrame(rows)["accuracy"].mean()),
        "mean_macro_f1": float(pd.DataFrame(rows)["macro_f1"].mean()),
        "mean_return_mae_bps": float(pd.DataFrame(rows)["return_mae_bps"].mean()),
        "report_path": str(report_path),
        "csv_path": str(csv_path),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return payload


def _train_lob_deep_model(
    settings: Settings,
    *,
    raw_frame: pd.DataFrame,
    data_path: str | None,
    model_name: str,
    epochs: int,
    set_active: bool,
    device: torch.device,
) -> dict[str, object]:
    depth_levels = _infer_depth_levels(raw_frame, settings.models.lob_depth_levels)
    sequence_length = settings.models.lob_sequence_length
    batch = build_lob_sequence_batch(
        raw_frame,
        depth_levels=depth_levels,
        sequence_length=sequence_length,
    )
    train_batch, valid_batch = _split_lob_batch(batch)
    model = _fit_lob_model(
        train_batch,
        feature_dim=train_batch.features.shape[2],
        sequence_length=sequence_length,
        train_batch_size=settings.models.lob_train_batch_size,
        epochs=epochs,
        device=device,
    )
    metrics = _evaluate_lob_batch(model, valid_batch, device)
    daily_metrics = _evaluate_lob_daily_metrics(model, valid_batch, device)
    artifact_id = f"deep-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    artifact_dir = Path(settings.models.artifacts_dir) / "deep"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{artifact_id}.pt"
    metadata_path = artifact_dir / f"{artifact_id}.metadata.json"
    feature_columns = lob_feature_columns(depth_levels)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "sequence_length": sequence_length,
            "depth_levels": depth_levels,
            "model_name": model_name,
            "model_family": "deep_lob_reference_like",
            "dataset_type": "ibkr_lob_depth",
            "horizon_events": int(raw_frame.get("horizon_events", pd.Series([settings.models.lob_horizon_events])).iloc[0]),
            "normalization": "price_rel_to_last_mid_bps_and_log1p_sizes",
            "class_index": {-1: 0, 0: 1, 1: 2},
        },
        artifact_path,
    )
    metadata = {
        "artifact_id": artifact_id,
        "model_name": model_name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_source": data_path or "lob_capture",
        "dataset_type": "ibkr_lob_depth",
        "feature_set": feature_columns,
        "target_definition": {
            "horizon_events": int(raw_frame.get("horizon_events", pd.Series([settings.models.lob_horizon_events])).iloc[0]),
            "stationary_threshold_bps": float(
                raw_frame.get(
                    "stationary_threshold_bps",
                    pd.Series([settings.models.lob_stationary_threshold_bps]),
                ).iloc[0]
            ),
            "sequence_length": sequence_length,
            "depth_levels": depth_levels,
        },
        "metrics": metrics,
        "daily_metrics": daily_metrics,
        "normalization": "price_rel_to_last_mid_bps_and_log1p_sizes",
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
            "dataset_type": "ibkr_lob_depth",
            "target_definition": metadata["target_definition"],
            "model_family": "deep_lob_reference_like",
        },
        set_active=set_active,
    )
    return {"record": record, "metadata": metadata}


def _train_bar_deep_model(
    settings: Settings,
    *,
    raw_frame: pd.DataFrame,
    data_path: str | None,
    model_name: str,
    epochs: int,
    set_active: bool,
    device: torch.device,
) -> dict[str, object]:
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
    class_weights = _compute_class_weights(batch.target_class[:split_index], device)
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
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
        "Deep training device | device=%s | profile=%s | target_horizon_minutes=%s | class_threshold_bps=%.4f",
        _describe_device(device),
        prepared.training_profile,
        prepared.target_horizon_minutes,
        prepared.class_threshold_bps,
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

        epoch_metrics = _evaluate_bar_model(model, valid_loader, device)
        logger.info(
            "Deep training epoch complete | epoch %s/%s | avg_loss=%.6f | accuracy=%.4f | return_mae_bps=%.4f",
            epoch_index + 1,
            epochs,
            cumulative_loss / max(total_batches, 1),
            epoch_metrics["accuracy"],
            epoch_metrics["return_mae_bps"],
        )

    metrics = _evaluate_bar_model(model, valid_loader, device)
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
            "model_family": "deep_lob_lite",
            "dataset_type": prepared.training_profile,
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
            "horizon_minutes": prepared.target_horizon_minutes,
            "training_profile": prepared.training_profile,
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
            "dataset_type": prepared.training_profile,
            "target_definition": metadata["target_definition"],
            "model_family": "deep_lob_lite",
        },
        set_active=set_active,
    )
    return {"record": record, "metadata": metadata}


def _build_sequence_batch(
    frame: pd.DataFrame,
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


def _split_lob_batch(batch: LOBSequenceBatch) -> tuple[LOBSequenceBatch, LOBSequenceBatch]:
    unique_dates = sorted(pd.unique(batch.session_dates).tolist())
    if len(unique_dates) > 1:
        split_index = max(len(unique_dates) - max(int(round(len(unique_dates) * 0.2)), 1), 1)
        train_dates = set(unique_dates[:split_index])
        valid_dates = set(unique_dates[split_index:])
        train_mask = np.isin(batch.session_dates, list(train_dates))
        valid_mask = np.isin(batch.session_dates, list(valid_dates))
    else:
        split_index = max(int(len(batch.features) * 0.8), 1)
        train_mask = np.arange(len(batch.features)) < split_index
        valid_mask = ~train_mask
    train_batch = _select_lob_batch(batch, train_mask)
    valid_batch = _select_lob_batch(batch, valid_mask if np.any(valid_mask) else train_mask)
    return train_batch, valid_batch


def _select_lob_batch(batch: LOBSequenceBatch, mask: np.ndarray) -> LOBSequenceBatch:
    return LOBSequenceBatch(
        features=batch.features[mask],
        target_class=batch.target_class[mask],
        target_return=batch.target_return[mask],
        session_dates=batch.session_dates[mask],
        target_timestamps=batch.target_timestamps[mask],
    )


def _fit_lob_model(
    batch: LOBSequenceBatch,
    *,
    feature_dim: int,
    sequence_length: int,
    train_batch_size: int,
    epochs: int,
    device: torch.device,
) -> DeepLOBReferenceLike:
    model = DeepLOBReferenceLike(feature_dim=feature_dim, sequence_length=sequence_length).to(device)
    dataset = LOBSequenceDataset(batch)
    loader = DataLoader(
        dataset,
        batch_size=max(train_batch_size, 1),
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_weights = _compute_class_weights(batch.target_class, device)
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    reg_loss_fn = nn.MSELoss()
    batch_log_interval = max(len(loader) // 10, 1)
    logger.info(
        "Deep training device | device=%s | dataset_type=ibkr_lob_depth | sequence_length=%s | depth_levels=%s | horizon_events=%s",
        _describe_device(device),
        sequence_length,
        feature_dim // 4,
        "paper-style",
    )
    for epoch_index in range(epochs):
        model.train()
        logger.info(
            "Deep training progress | epoch %s/%s | batches=%s",
            epoch_index + 1,
            epochs,
            len(loader),
        )
        for batch_index, (features, target_class, target_return) in enumerate(loader, start=1):
            features = features.to(device, non_blocking=device.type == "cuda")
            target_class = target_class.to(device, non_blocking=device.type == "cuda")
            target_return = target_return.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad()
            logits, predicted_return = model(features)
            loss = cls_loss_fn(logits, target_class) + 0.1 * reg_loss_fn(predicted_return, target_return)
            loss.backward()
            optimizer.step()
            if batch_index == len(loader) or batch_index % batch_log_interval == 0:
                logger.info(
                    "Deep training progress | epoch %s/%s | batch %s/%s | loss=%.6f",
                    epoch_index + 1,
                    epochs,
                    batch_index,
                    len(loader),
                    float(loss.item()),
                )
    return model


def _evaluate_lob_batch(
    model: nn.Module,
    batch: LOBSequenceBatch,
    device: torch.device,
) -> dict[str, float]:
    dataset = LOBSequenceDataset(batch)
    loader = DataLoader(
        dataset,
        batch_size=max(32, 1),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    actual_classes: list[int] = []
    predicted_classes: list[int] = []
    actual_returns: list[float] = []
    predicted_returns: list[float] = []
    with torch.no_grad():
        for features, target_class, target_return in loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            target_class = target_class.to(device, non_blocking=device.type == "cuda")
            target_return = target_return.to(device, non_blocking=device.type == "cuda")
            logits, predicted_return = model(features)
            predicted_class = torch.argmax(logits, dim=1)
            actual_classes.extend(target_class.cpu().numpy().tolist())
            predicted_classes.extend(predicted_class.cpu().numpy().tolist())
            actual_returns.extend(target_return.cpu().numpy().tolist())
            predicted_returns.extend(predicted_return.cpu().numpy().tolist())
    metrics = evaluate_lob_predictions(
        actual_classes=actual_classes,
        predicted_classes=predicted_classes,
    )
    metrics["return_mae_bps"] = float(mean_absolute_error(actual_returns, predicted_returns))
    return metrics


def _evaluate_lob_daily_metrics(
    model: nn.Module,
    batch: LOBSequenceBatch,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset = LOBSequenceDataset(batch)
    loader = DataLoader(
        dataset,
        batch_size=max(32, 1),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    predicted_classes: list[int] = []
    predicted_returns: list[float] = []
    with torch.no_grad():
        for features, _target_class, _target_return in loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            logits, predicted_return = model(features)
            predicted_classes.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            predicted_returns.extend(predicted_return.cpu().numpy().tolist())
    frame = pd.DataFrame(
        {
            "session_date": batch.session_dates,
            "target_class": batch.target_class,
            "predicted_class": predicted_classes,
            "target_return": batch.target_return,
            "predicted_return": predicted_returns,
        }
    )
    rows: list[dict[str, object]] = []
    for session_date, group in frame.groupby("session_date", sort=True):
        metrics = evaluate_lob_predictions(
            actual_classes=group["target_class"].tolist(),
            predicted_classes=group["predicted_class"].tolist(),
        )
        rows.append(
            {
                "session_date": str(session_date),
                "row_count": int(len(group)),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "return_mae_bps": float(
                    mean_absolute_error(group["target_return"], group["predicted_return"])
                ),
            }
        )
    return rows


def _evaluate_bar_model(model: nn.Module, valid_loader: DataLoader, device: torch.device) -> dict[str, float]:
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
        "macro_f1": float(f1_score(actual_classes, predicted_classes, average="macro")),
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


def _compute_class_weights(target_class: np.ndarray, device: torch.device) -> torch.Tensor:
    values, counts = np.unique(target_class, return_counts=True)
    weight_map = {
        int(value): float(len(target_class)) / float(len(values) * count)
        for value, count in zip(values, counts, strict=False)
    }
    ordered = [weight_map.get(index, 1.0) for index in range(3)]
    return torch.tensor(ordered, dtype=torch.float32, device=device)


def _infer_depth_levels(frame: pd.DataFrame, default_levels: int) -> int:
    candidates = []
    for column in frame.columns:
        if column.startswith("bid_px_"):
            with suppress(ValueError):
                candidates.append(int(column.split("_")[-1]))
    return max(candidates) if candidates else default_levels
