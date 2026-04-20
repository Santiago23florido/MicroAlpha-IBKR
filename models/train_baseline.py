from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Settings
from data.historical_loader import load_historical_dataset
from features.preprocessing import prepare_training_dataframe
from models.registry import ModelRegistry


def train_baseline_model(
    settings: Settings,
    *,
    data_path: str | None = None,
    model_name: str = "baseline_logreg",
    set_active: bool = True,
) -> dict[str, object]:
    raw_frame = load_historical_dataset(settings, data_path)
    prepared = prepare_training_dataframe(raw_frame, settings)
    frame = prepared.frame
    split_index = max(int(len(frame) * 0.8), 1)
    train_frame = frame.iloc[:split_index]
    valid_frame = frame.iloc[split_index:] if split_index < len(frame) else frame.iloc[-1:]

    feature_columns = prepared.feature_columns
    X_train = train_frame[feature_columns]
    X_valid = valid_frame[feature_columns]
    y_train_class = train_frame["target_class"]
    y_valid_class = valid_frame["target_class"]
    y_train_return = train_frame["future_return_bps"]
    y_valid_return = valid_frame["future_return_bps"]

    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    classifier.fit(X_train, y_train_class)
    regressor.fit(X_train, y_train_return)

    class_predictions = classifier.predict(X_valid)
    return_predictions = regressor.predict(X_valid)
    metrics = {
        "accuracy": float(accuracy_score(y_valid_class, class_predictions)),
        "macro_f1": float(f1_score(y_valid_class, class_predictions, average="macro")),
        "return_mae_bps": float(mean_absolute_error(y_valid_return, return_predictions)),
    }

    artifact_id = f"baseline-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    artifact_dir = Path(settings.models.artifacts_dir) / "baseline"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{artifact_id}.joblib"
    metadata_path = artifact_dir / f"{artifact_id}.metadata.json"
    payload = {
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": feature_columns,
        "class_labels": list(map(int, classifier.named_steps["classifier"].classes_)),
        "target_horizon_minutes": settings.models.target_horizon_minutes,
        "model_name": model_name,
        "class_threshold_bps": prepared.class_threshold_bps,
    }
    joblib.dump(payload, artifact_path)

    metadata = {
        "artifact_id": artifact_id,
        "model_name": model_name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_source": data_path or "data/sample/spy_microstructure_sample.csv",
        "feature_set": feature_columns,
        "target_definition": {
            "horizon_minutes": settings.models.target_horizon_minutes,
            "classification_threshold_bps": prepared.class_threshold_bps,
        },
        "metrics": metrics,
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    registry = ModelRegistry(settings.models.registry_path)
    record = registry.register_model(
        "baseline",
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
