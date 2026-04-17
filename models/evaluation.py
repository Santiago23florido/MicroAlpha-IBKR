from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from labels.dataset_builder import ModelingDataset, TemporalSplit
from models.config import TargetConfig
from models.factory import QuantileGradientBoostingEnsemble


def evaluate_split(
    model: Any,
    preprocessor: Any,
    dataset: ModelingDataset,
    split: TemporalSplit,
    target_config: TargetConfig,
) -> dict[str, Any]:
    frame = split.frame
    X = preprocessor.transform(frame[dataset.feature_columns])
    y_true = frame[dataset.target_column].to_numpy()
    predictions = _predict_outputs(model, X)

    if target_config.task_type in {"classification", "ordinal", "distribution_bins"}:
        metrics = _evaluate_classification(frame, y_true, predictions)
    elif target_config.task_type == "quantile_regression":
        metrics = _evaluate_quantile_regression(frame, y_true.astype(float), predictions, target_config)
    else:
        metrics = _evaluate_regression(frame, y_true.astype(float), predictions)

    metrics["rows"] = int(len(frame))
    return metrics


def _predict_outputs(model: Any, X: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {"prediction": model.predict(X)}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        payload["probabilities"] = probabilities
        payload["classes"] = list(getattr(model, "classes_", []))
    if isinstance(model, QuantileGradientBoostingEnsemble):
        payload["quantiles"] = model.predict_quantiles(X)
    return payload


def _evaluate_classification(frame: pd.DataFrame, y_true: np.ndarray, predictions: dict[str, Any]) -> dict[str, Any]:
    y_pred = np.asarray(predictions["prediction"])
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_labels": labels,
        "actual_distribution": _distribution(y_true),
        "predicted_distribution": _distribution(y_pred),
    }

    probabilities = predictions.get("probabilities")
    classes = predictions.get("classes") or labels
    if probabilities is not None and len(classes) == 2 and len(np.unique(y_true)) > 1:
        positive_class = classes[-1]
        positive_index = list(classes).index(positive_class)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, positive_index]))
        except ValueError:
            pass

    score = _classification_score(predictions)
    metrics["economics"] = _economic_metrics(frame, score=score)
    return metrics


def _evaluate_regression(frame: pd.DataFrame, y_true: np.ndarray, predictions: dict[str, Any]) -> dict[str, Any]:
    y_pred = np.asarray(predictions["prediction"], dtype=float)
    metrics = {
        "mae_bps": float(mean_absolute_error(y_true, y_pred)),
        "rmse_bps": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }
    metrics["economics"] = _economic_metrics(frame, score=y_pred)
    return metrics


def _evaluate_quantile_regression(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, Any],
    target_config: TargetConfig,
) -> dict[str, Any]:
    point_prediction = np.asarray(predictions["prediction"], dtype=float)
    quantiles = {float(key): np.asarray(value, dtype=float) for key, value in predictions.get("quantiles", {}).items()}
    metrics: dict[str, Any] = {
        "mae_bps": float(mean_absolute_error(y_true, point_prediction)),
        "rmse_bps": float(np.sqrt(mean_squared_error(y_true, point_prediction))),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(point_prediction))),
        "pinball_loss": {},
    }
    for quantile in target_config.quantiles:
        estimate = quantiles.get(float(quantile))
        if estimate is None:
            continue
        metrics["pinball_loss"][str(quantile)] = float(mean_pinball_loss(y_true, estimate, alpha=float(quantile)))

    lower = quantiles.get(min(target_config.quantiles, default=0.1))
    upper = quantiles.get(max(target_config.quantiles, default=0.9))
    if lower is not None and upper is not None:
        interval_coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        interval_width = float(np.mean(upper - lower))
        metrics["interval_coverage"] = interval_coverage
        metrics["mean_interval_width_bps"] = interval_width

    metrics["economics"] = _economic_metrics(frame, score=point_prediction)
    return metrics


def _classification_score(predictions: dict[str, Any]) -> np.ndarray:
    probabilities = predictions.get("probabilities")
    classes = predictions.get("classes")
    if probabilities is None or classes is None:
        return np.asarray(predictions["prediction"], dtype=float)

    ordered_classes = np.asarray(classes, dtype=float)
    return probabilities @ ordered_classes


def _economic_metrics(frame: pd.DataFrame, *, score: np.ndarray) -> dict[str, Any]:
    returns = _resolve_return_series(frame)
    if returns.isna().all():
        return {
            "top_decile_count": 0,
            "top_decile_mean_future_return_bps": 0.0,
            "top_decile_mean_net_return_bps": 0.0,
            "bottom_decile_mean_future_return_bps": 0.0,
            "score_spread_bps": 0.0,
            "top_signal_hit_rate": 0.0,
            "score_buckets": [],
        }

    score_series = pd.Series(np.asarray(score, dtype=float), index=frame.index)
    valid = score_series.notna() & returns.notna()
    if not valid.any():
        return {
            "top_decile_count": 0,
            "top_decile_mean_future_return_bps": 0.0,
            "top_decile_mean_net_return_bps": 0.0,
            "bottom_decile_mean_future_return_bps": 0.0,
            "score_spread_bps": 0.0,
            "top_signal_hit_rate": 0.0,
            "score_buckets": [],
        }

    working = pd.DataFrame(
        {
            "score": score_series[valid],
            "future_return_bps": returns[valid],
            "future_net_return_bps": _resolve_net_return_series(frame)[valid],
        }
    ).sort_values("score")
    top_count = max(int(np.ceil(len(working) * 0.1)), 1)
    bottom = working.head(top_count)
    top = working.tail(top_count)
    buckets = _score_buckets(working)
    return {
        "top_decile_count": int(top_count),
        "top_decile_mean_future_return_bps": float(top["future_return_bps"].mean()),
        "top_decile_mean_net_return_bps": float(top["future_net_return_bps"].mean()),
        "bottom_decile_mean_future_return_bps": float(bottom["future_return_bps"].mean()),
        "score_spread_bps": float(top["future_return_bps"].mean() - bottom["future_return_bps"].mean()),
        "top_signal_hit_rate": float((top["future_net_return_bps"] > 0).mean()),
        "score_buckets": buckets,
    }


def _score_buckets(working: pd.DataFrame) -> list[dict[str, Any]]:
    if len(working) < 5 or working["score"].nunique() < 3:
        return []
    bucket_frame = working.copy()
    bucket_frame["bucket"] = pd.qcut(bucket_frame["score"], q=5, duplicates="drop")
    summary = (
        bucket_frame.groupby("bucket", observed=False)
        .agg(
            count=("score", "size"),
            mean_score=("score", "mean"),
            mean_future_return_bps=("future_return_bps", "mean"),
            mean_future_net_return_bps=("future_net_return_bps", "mean"),
        )
        .reset_index()
    )
    return [
        {
            "bucket": str(row["bucket"]),
            "count": int(row["count"]),
            "mean_score": float(row["mean_score"]),
            "mean_future_return_bps": float(row["mean_future_return_bps"]),
            "mean_future_net_return_bps": float(row["mean_future_net_return_bps"]),
        }
        for _, row in summary.iterrows()
    ]


def _resolve_return_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("future_net_return_bps", "future_return_bps"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _resolve_net_return_series(frame: pd.DataFrame) -> pd.Series:
    if "future_net_return_bps" in frame.columns:
        return pd.to_numeric(frame["future_net_return_bps"], errors="coerce")
    return _resolve_return_series(frame)


def _distribution(values: np.ndarray) -> dict[str, int]:
    series = pd.Series(values)
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}
