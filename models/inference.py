from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from config import Settings
from config.phase6 import load_active_model_selection, resolve_phase5_artifact
from data.schemas import FeatureSnapshot, ModelPrediction
from features.preprocessing import build_feature_vector
from models.config import TargetConfig
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


@dataclass(frozen=True)
class OperationalPrediction:
    run_id: str
    model_name: str
    model_type: str
    target_mode: str
    feature_set_name: str
    score: float | None = None
    probability: float | None = None
    predicted_return_bps: float | None = None
    predicted_quantiles: dict[str, float] = field(default_factory=dict)
    class_label: int | float | str | None = None
    action_bias: str | None = None
    valid: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperationalInferenceEngine:
    def __init__(self, settings: Settings, artifact_record: dict[str, Any]) -> None:
        self.settings = settings
        self.artifact_record = artifact_record
        self._model = _load_phase5_joblib(artifact_record["artifact_path"])
        self._preprocessor = _load_phase5_joblib(artifact_record["preprocessing_path"])
        self._feature_columns = list(_load_json_file(artifact_record["feature_columns_path"]))
        self._target_config = _target_config_from_dict(_load_json_file(artifact_record["target_config_path"]))
        self._training_metadata = _load_json_file(artifact_record["training_metadata_path"])

    @classmethod
    def from_active_model(cls, settings: Settings) -> "OperationalInferenceEngine":
        selection = load_active_model_selection(settings)
        artifact = resolve_phase5_artifact(
            settings,
            run_id=selection.run_id,
            artifact_dir=selection.artifact_dir,
        )
        return cls(settings, artifact)

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def target_config(self) -> TargetConfig:
        return self._target_config

    @property
    def training_metadata(self) -> dict[str, Any]:
        return dict(self._training_metadata)

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.artifact_record["run_id"],
            "model_name": self.artifact_record["model_name"],
            "model_type": self.artifact_record["model_type"],
            "feature_set_name": self.artifact_record["feature_set_name"],
            "target_mode": self.artifact_record["target_mode"],
            "artifact_dir": self.artifact_record["artifact_dir"],
            "feature_column_count": len(self._feature_columns),
            "feature_columns": list(self._feature_columns),
            "target_config": self._target_config.to_dict(),
        }

    def predict_row(self, row: pd.Series | dict[str, Any]) -> OperationalPrediction:
        frame = pd.DataFrame([dict(row)])
        return self.predict_frame(frame)[0]

    def predict_frame(self, frame: pd.DataFrame) -> list[OperationalPrediction]:
        if frame.empty:
            return []
        missing_columns = [column for column in self._feature_columns if column not in frame.columns]
        if missing_columns:
            return [
                _invalid_operational_prediction(
                    self.artifact_record,
                    reasons=[f"Missing required feature columns: {missing_columns}"],
                )
                for _ in range(len(frame))
            ]

        model_input = frame.loc[:, self._feature_columns].replace([np.inf, -np.inf], np.nan)
        transformed = self._preprocessor.transform(model_input)
        raw_prediction = self._model.predict(transformed)
        probabilities = None
        classes = None
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(transformed)
            classes = list(getattr(self._model, "classes_", []))
        quantiles: dict[float, np.ndarray] = {}
        if hasattr(self._model, "predict_quantiles"):
            quantiles = self._model.predict_quantiles(transformed)

        predictions: list[OperationalPrediction] = []
        for index in range(len(frame)):
            predictions.append(
                _normalize_operational_prediction(
                    artifact_record=self.artifact_record,
                    target_config=self._target_config,
                    raw_prediction=raw_prediction[index],
                    probability_row=None if probabilities is None else probabilities[index],
                    classes=classes,
                    quantile_row={
                        float(key): float(values[index]) for key, values in quantiles.items()
                    }
                    if quantiles
                    else {},
                )
            )
        return predictions


def _normalize_operational_prediction(
    *,
    artifact_record: dict[str, Any],
    target_config: TargetConfig,
    raw_prediction: Any,
    probability_row: np.ndarray | None,
    classes: list[Any] | None,
    quantile_row: dict[float, float],
) -> OperationalPrediction:
    task_type = target_config.task_type
    try:
        if task_type == "classification":
            return _normalize_binary_classification_prediction(
                artifact_record,
                target_config,
                raw_prediction,
                probability_row,
                classes,
            )
        if task_type in {"ordinal", "distribution_bins"}:
            return _normalize_binned_prediction(
                artifact_record,
                target_config,
                raw_prediction,
                probability_row,
                classes,
            )
        if task_type == "quantile_regression":
            return _normalize_quantile_prediction(
                artifact_record,
                target_config,
                raw_prediction,
                quantile_row,
            )
        return _normalize_regression_prediction(
            artifact_record,
            target_config,
            raw_prediction,
        )
    except Exception as exc:
        return _invalid_operational_prediction(
            artifact_record,
            reasons=[f"Failed to normalize model output: {exc}"],
        )


def _normalize_binary_classification_prediction(
    artifact_record: dict[str, Any],
    target_config: TargetConfig,
    raw_prediction: Any,
    probability_row: np.ndarray | None,
    classes: list[Any] | None,
) -> OperationalPrediction:
    positive_label = int(target_config.positive_label)
    predicted_label = int(raw_prediction)
    probability_map = _probability_map(probability_row, classes)
    probability = probability_map.get(positive_label)
    if probability is None:
        probability = 1.0 if predicted_label == positive_label else 0.0
    score = float((probability * 2.0) - 1.0)
    threshold_bps = float(target_config.threshold_bps or 0.0)
    predicted_return_bps = max(score, 0.0) * max(threshold_bps, 1.0)
    return _validated_operational_prediction(
        OperationalPrediction(
            run_id=artifact_record["run_id"],
            model_name=artifact_record["model_name"],
            model_type=artifact_record["model_type"],
            target_mode=artifact_record["target_mode"],
            feature_set_name=artifact_record["feature_set_name"],
            score=score,
            probability=float(probability),
            predicted_return_bps=float(predicted_return_bps),
            predicted_quantiles={},
            class_label=predicted_label,
            action_bias="LONG" if predicted_label == positive_label else "NO_TRADE",
            metadata={
                "probability_map": {str(key): float(value) for key, value in probability_map.items()},
                "positive_label": positive_label,
                "negative_label": int(target_config.negative_label),
                "threshold_bps_proxy": threshold_bps,
            },
        )
    )


def _normalize_binned_prediction(
    artifact_record: dict[str, Any],
    target_config: TargetConfig,
    raw_prediction: Any,
    probability_row: np.ndarray | None,
    classes: list[Any] | None,
) -> OperationalPrediction:
    probability_map = _probability_map(probability_row, classes)
    bin_midpoints = _bin_midpoint_map(target_config)
    predicted_label = int(raw_prediction)
    probability = None
    if probability_map:
        probability = float(max(probability_map.values()))
        expected_return_bps = float(
            sum(probability_map.get(label, 0.0) * midpoint for label, midpoint in bin_midpoints.items())
        )
    else:
        probability = 1.0
        expected_return_bps = float(bin_midpoints.get(predicted_label, float(predicted_label)))
    return _validated_operational_prediction(
        OperationalPrediction(
            run_id=artifact_record["run_id"],
            model_name=artifact_record["model_name"],
            model_type=artifact_record["model_type"],
            target_mode=artifact_record["target_mode"],
            feature_set_name=artifact_record["feature_set_name"],
            score=float(expected_return_bps),
            probability=float(probability),
            predicted_return_bps=float(expected_return_bps),
            class_label=predicted_label,
            action_bias="LONG" if expected_return_bps > 0 else "SHORT" if expected_return_bps < 0 else "NO_TRADE",
            metadata={
                "probability_map": {str(key): float(value) for key, value in probability_map.items()},
                "bin_midpoints_bps": {str(key): float(value) for key, value in bin_midpoints.items()},
            },
        )
    )


def _normalize_regression_prediction(
    artifact_record: dict[str, Any],
    target_config: TargetConfig,
    raw_prediction: Any,
) -> OperationalPrediction:
    predicted_return_bps = float(raw_prediction)
    return _validated_operational_prediction(
        OperationalPrediction(
            run_id=artifact_record["run_id"],
            model_name=artifact_record["model_name"],
            model_type=artifact_record["model_type"],
            target_mode=artifact_record["target_mode"],
            feature_set_name=artifact_record["feature_set_name"],
            score=predicted_return_bps,
            probability=None,
            predicted_return_bps=predicted_return_bps,
            class_label=None,
            action_bias="LONG" if predicted_return_bps > 0 else "SHORT" if predicted_return_bps < 0 else "NO_TRADE",
            metadata={"task_type": target_config.task_type},
        )
    )


def _normalize_quantile_prediction(
    artifact_record: dict[str, Any],
    target_config: TargetConfig,
    raw_prediction: Any,
    quantile_row: dict[float, float],
) -> OperationalPrediction:
    predicted_return_bps = float(raw_prediction)
    quantiles_payload = {str(key): float(value) for key, value in quantile_row.items()}
    sorted_quantiles = sorted(quantile_row.items())
    interval_width = None
    if len(sorted_quantiles) >= 2:
        interval_width = float(sorted_quantiles[-1][1] - sorted_quantiles[0][1])
    return _validated_operational_prediction(
        OperationalPrediction(
            run_id=artifact_record["run_id"],
            model_name=artifact_record["model_name"],
            model_type=artifact_record["model_type"],
            target_mode=artifact_record["target_mode"],
            feature_set_name=artifact_record["feature_set_name"],
            score=predicted_return_bps,
            probability=None,
            predicted_return_bps=predicted_return_bps,
            predicted_quantiles=quantiles_payload,
            class_label=None,
            action_bias="LONG" if predicted_return_bps > 0 else "SHORT" if predicted_return_bps < 0 else "NO_TRADE",
            metadata={
                "interval_width_bps": interval_width,
                "quantiles": list(target_config.quantiles),
            },
        )
    )


def _validated_operational_prediction(prediction: OperationalPrediction) -> OperationalPrediction:
    numeric_candidates = [
        prediction.score,
        prediction.probability,
        prediction.predicted_return_bps,
        *prediction.predicted_quantiles.values(),
    ]
    invalid_reasons = list(prediction.reasons)
    for value in numeric_candidates:
        if value is None:
            continue
        if not np.isfinite(float(value)):
            invalid_reasons.append("Prediction output contains non-finite values.")
            break
    if invalid_reasons:
        return OperationalPrediction(
            **{
                **prediction.to_dict(),
                "valid": False,
                "reasons": invalid_reasons,
            }
        )
    return prediction


def _invalid_operational_prediction(
    artifact_record: dict[str, Any],
    *,
    reasons: list[str],
) -> OperationalPrediction:
    return OperationalPrediction(
        run_id=str(artifact_record.get("run_id")),
        model_name=str(artifact_record.get("model_name")),
        model_type=str(artifact_record.get("model_type")),
        target_mode=str(artifact_record.get("target_mode")),
        feature_set_name=str(artifact_record.get("feature_set_name")),
        valid=False,
        reasons=reasons,
        metadata={},
    )


def _probability_map(probability_row: np.ndarray | None, classes: list[Any] | None) -> dict[int, float]:
    if probability_row is None or classes is None:
        return {}
    return {
        int(label): float(probability)
        for label, probability in zip(classes, probability_row)
    }


def _bin_midpoint_map(target_config: TargetConfig) -> dict[int, float]:
    edges = list(target_config.bin_edges_bps)
    labels = list(target_config.class_labels)
    if len(edges) < 2 or not labels:
        return {int(label): float(label) for label in labels}
    midpoints = [(left + right) / 2.0 for left, right in zip(edges[:-1], edges[1:])]
    return {int(label): float(midpoints[index]) for index, label in enumerate(labels[: len(midpoints)])}


def _target_config_from_dict(payload: dict[str, Any]) -> TargetConfig:
    return TargetConfig(
        name=str(payload["name"]),
        description=str(payload.get("description", "")),
        task_type=str(payload["task_type"]),
        horizon_bars=payload.get("horizon_bars"),
        horizon_minutes=payload.get("horizon_minutes"),
        threshold_bps=payload.get("threshold_bps"),
        negative_threshold_bps=payload.get("negative_threshold_bps"),
        bin_edges_bps=tuple(payload.get("bin_edges_bps", [])),
        class_labels=tuple(payload.get("class_labels", [])),
        quantiles=tuple(payload.get("quantiles", [])),
        cost_adjustment_bps=float(payload.get("cost_adjustment_bps", 0.0)),
        cost_adjustment_multiplier=float(payload.get("cost_adjustment_multiplier", 0.0)),
        positive_label=int(payload.get("positive_label", 1)),
        negative_label=int(payload.get("negative_label", 0)),
    )


@lru_cache(maxsize=32)
def _load_phase5_joblib(path: str) -> Any:
    return joblib.load(path)


@lru_cache(maxsize=32)
def _load_json_file(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
