from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from config import Settings
from data.schemas import FeatureSnapshot, ModelPrediction
from features.preprocessing import build_feature_vector
from models.registry import ModelRegistry
from models.train_deep import DeepLOBLite


class InferenceEngine:
    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self.settings = settings
        self.registry = registry

    def predict_baseline(self, feature_snapshot: FeatureSnapshot) -> ModelPrediction:
        record = self.registry.get_active_model("baseline")
        if record is None:
            return _rejected_prediction("baseline", "No active baseline model is configured.")
        artifact = _load_joblib_artifact(record["artifact_path"])
        feature_columns = artifact["feature_columns"]
        vector = build_feature_vector(feature_snapshot.feature_values, feature_columns).reshape(1, -1)
        classifier = artifact["classifier"]
        regressor = artifact["regressor"]
        probabilities = classifier.predict_proba(vector)[0]
        class_labels = list(map(int, classifier.named_steps["classifier"].classes_))
        probability_map = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
        predicted_return = float(regressor.predict(vector)[0])
        return _build_prediction(
            record=record,
            probability_map=probability_map,
            predicted_return=predicted_return,
        )

    def predict_deep(self, sequence: list[FeatureSnapshot]) -> ModelPrediction:
        record = self.registry.get_active_model("deep")
        if record is None:
            return _rejected_prediction("deep", "No active deep model is configured.")
        payload = _load_torch_artifact(record["artifact_path"])
        feature_columns = payload["feature_columns"]
        sequence_length = int(payload["sequence_length"])
        if len(sequence) < sequence_length:
            return _rejected_prediction(
                "deep",
                f"Deep model requires {sequence_length} feature rows but only {len(sequence)} are available.",
            )
        rows = sequence[-sequence_length:]
        tensor = np.asarray(
            [build_feature_vector(snapshot.feature_values, feature_columns) for snapshot in rows],
            dtype=np.float32,
        )
        model = DeepLOBLite(feature_dim=len(feature_columns))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            logits, predicted_return = model(torch.tensor(tensor[None, :, :], dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probability_map = {-1: float(probabilities[0]), 0: float(probabilities[1]), 1: float(probabilities[2])}
        return _build_prediction(
            record=record,
            probability_map=probability_map,
            predicted_return=float(predicted_return.item()),
        )


def _build_prediction(
    *,
    record: dict[str, Any],
    probability_map: dict[int, float],
    predicted_return: float,
) -> ModelPrediction:
    prob_up = probability_map.get(1, 0.0)
    prob_down = probability_map.get(-1, 0.0)
    prob_flat = probability_map.get(0, 0.0)
    direction = "long" if prob_up > max(prob_down, prob_flat) else "short" if prob_down > max(prob_up, prob_flat) else None
    confidence = max(prob_up, prob_down, prob_flat)
    return ModelPrediction(
        model_name=record["model_name"],
        model_type=record["model_type"],
        artifact_id=record["artifact_id"],
        probability_up=prob_up,
        probability_down=prob_down,
        probability_flat=prob_flat,
        directional_probability=max(prob_up, prob_down),
        predicted_return_bps=predicted_return,
        confidence=confidence,
        direction=direction,
        eligible=True,
        metadata={"metrics": record.get("metrics", {})},
    )


def _rejected_prediction(model_type: str, reason: str) -> ModelPrediction:
    return ModelPrediction(
        model_name=f"{model_type}_unavailable",
        model_type=model_type,
        artifact_id=None,
        probability_up=0.0,
        probability_down=0.0,
        probability_flat=1.0,
        directional_probability=0.0,
        predicted_return_bps=0.0,
        confidence=0.0,
        direction=None,
        eligible=False,
        reasons=[reason],
    )


@lru_cache(maxsize=8)
def _load_joblib_artifact(path: str) -> dict[str, Any]:
    return joblib.load(path)


@lru_cache(maxsize=8)
def _load_torch_artifact(path: str) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    return payload
