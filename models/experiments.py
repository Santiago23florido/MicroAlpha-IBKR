from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import joblib
import pandas as pd

from config import Settings
from features.feature_pipeline import run_feature_build_pipeline
from labels.dataset_builder import ModelingDataset, build_modeling_dataset
from labels.labeling import build_labels
from models.config import (
    ModelingConfig,
    TargetConfig,
    build_parameter_grid,
    load_modeling_config,
    resolve_experiment_profile,
    resolve_target_config,
)
from models.evaluation import evaluate_split
from models.factory import create_model_components, is_model_supported_for_target
from models.registry import ModelRegistry
from monitoring.logging import setup_logger


@dataclass(frozen=True)
class TrainedRun:
    run_id: str
    artifact_dir: str
    artifact_path: str
    preprocessing_path: str
    metrics: dict[str, Any]
    leaderboard_row: dict[str, Any]
    training_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "artifact_dir": self.artifact_dir,
            "artifact_path": self.artifact_path,
            "preprocessing_path": self.preprocessing_path,
            "metrics": self.metrics,
            "leaderboard_row": self.leaderboard_row,
            "training_metadata": self.training_metadata,
        }


def train_baseline_variant(
    settings: Settings,
    *,
    feature_set_name: str,
    target_mode: str,
    model_name: str,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    label_root: str | Path | None = None,
    feature_columns: Sequence[str] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    logger=None,
) -> dict[str, Any]:
    train_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.phase5.train_baseline",
    )
    target_config = resolve_target_config(settings, target_mode)
    compatible, reason = is_model_supported_for_target(model_name, target_config)
    if not compatible:
        raise ValueError(f"Incompatible combination model={model_name!r} target_mode={target_mode!r}: {reason}")

    dataset = build_modeling_dataset(
        settings,
        feature_set_name=feature_set_name,
        target_mode=target_mode,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        label_root=label_root,
        feature_columns=feature_columns,
    )
    params = dict(hyperparameters or {})
    preprocessor, model, model_info = create_model_components(model_name, target_config, params)

    X_train = preprocessor.fit_transform(dataset.train.frame[dataset.feature_columns])
    y_train = dataset.y_train
    model.fit(X_train, y_train)

    validation_metrics = evaluate_split(model, preprocessor, dataset, dataset.validation, target_config)
    test_metrics = evaluate_split(model, preprocessor, dataset, dataset.test, target_config)
    metrics = {"validation": validation_metrics, "test": test_metrics}

    run_id = _build_run_id(model_name, feature_set_name, target_mode)
    artifact_dir = Path(settings.paths.model_dir) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model.joblib"
    preprocessing_path = artifact_dir / "preprocessing.joblib"
    metrics_path = artifact_dir / "metrics.json"
    feature_columns_path = artifact_dir / "feature_columns.json"
    config_snapshot_path = artifact_dir / "config_snapshot.json"
    training_metadata_path = artifact_dir / "training_metadata.json"
    target_config_path = artifact_dir / "target_config.json"

    joblib.dump(model, artifact_path)
    joblib.dump(preprocessor, preprocessing_path)
    feature_columns_path.write_text(json.dumps(dataset.feature_columns, indent=2, default=str), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
    config_snapshot_path.write_text(json.dumps(settings.as_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    target_config_path.write_text(json.dumps(target_config.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")

    training_metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set_name": feature_set_name,
        "target_mode": target_mode,
        "target_config": target_config.to_dict(),
        "model_name": model_name,
        "model_info": model_info,
        "hyperparameters": params,
        "symbols": list(symbols or settings.supported_symbols),
        "start_date": start_date,
        "end_date": end_date,
        "dataset": dataset.to_metadata(),
        "artifact_files": {
            "artifact_path": str(artifact_path),
            "preprocessing_path": str(preprocessing_path),
            "feature_columns_path": str(feature_columns_path),
            "metrics_path": str(metrics_path),
            "config_snapshot_path": str(config_snapshot_path),
            "training_metadata_path": str(training_metadata_path),
            "target_config_path": str(target_config_path),
        },
    }
    training_metadata_path.write_text(json.dumps(training_metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")

    leaderboard_row = build_leaderboard_row(
        run_id=run_id,
        feature_set_name=feature_set_name,
        target_mode=target_mode,
        model_name=model_name,
        hyperparameters=params,
        dataset=dataset,
        metrics=metrics,
        artifact_dir=artifact_dir,
    )
    leaderboard_row_path = artifact_dir / "leaderboard_row.json"
    leaderboard_row_path.write_text(json.dumps(leaderboard_row, indent=2, sort_keys=True, default=str), encoding="utf-8")

    registry = ModelRegistry(settings.models.registry_path)
    registry_record = registry.register_phase5_run(
        {
            **leaderboard_row,
            "artifact_path": str(artifact_path),
            "preprocessing_path": str(preprocessing_path),
            "feature_columns": list(dataset.feature_columns),
            "target_config_path": str(target_config_path),
            "training_metadata_path": str(training_metadata_path),
            "metrics": metrics,
            "hyperparameters": params,
            "symbols": list(symbols or settings.supported_symbols),
        }
    )
    train_logger.info(
        "Trained phase5 run: run_id=%s model=%s feature_set=%s target_mode=%s validation_score=%s artifact_dir=%s",
        run_id,
        model_name,
        feature_set_name,
        target_mode,
        leaderboard_row["ranking_score"],
        artifact_dir,
    )
    return TrainedRun(
        run_id=run_id,
        artifact_dir=str(artifact_dir),
        artifact_path=str(artifact_path),
        preprocessing_path=str(preprocessing_path),
        metrics=metrics,
        leaderboard_row=registry_record,
        training_metadata=training_metadata,
    ).to_dict()


def evaluate_baseline_variant(
    settings: Settings,
    *,
    run_id: str | None = None,
    artifact_dir: str | Path | None = None,
    logger=None,
) -> dict[str, Any]:
    eval_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.phase5.evaluate_baseline",
    )
    registry = ModelRegistry(settings.models.registry_path)
    record = _resolve_run_record(settings, registry, run_id=run_id, artifact_dir=artifact_dir)
    target_metadata = json.loads(Path(record["target_config_path"]).read_text(encoding="utf-8"))
    target_config = TargetConfig(
        name=target_metadata["name"],
        description=target_metadata["description"],
        task_type=target_metadata["task_type"],
        horizon_bars=target_metadata.get("horizon_bars"),
        horizon_minutes=target_metadata.get("horizon_minutes"),
        threshold_bps=target_metadata.get("threshold_bps"),
        negative_threshold_bps=target_metadata.get("negative_threshold_bps"),
        bin_edges_bps=tuple(target_metadata.get("bin_edges_bps", [])),
        class_labels=tuple(target_metadata.get("class_labels", [])),
        quantiles=tuple(target_metadata.get("quantiles", [])),
        cost_adjustment_bps=float(target_metadata.get("cost_adjustment_bps", 0.0)),
        cost_adjustment_multiplier=float(target_metadata.get("cost_adjustment_multiplier", 0.0)),
        positive_label=int(target_metadata.get("positive_label", 1)),
        negative_label=int(target_metadata.get("negative_label", 0)),
    )
    training_metadata = json.loads(Path(record["training_metadata_path"]).read_text(encoding="utf-8"))
    dataset = build_modeling_dataset(
        settings,
        feature_set_name=record["feature_set"],
        target_mode=record["target_mode"],
        symbols=training_metadata.get("symbols"),
        start_date=training_metadata.get("start_date"),
        end_date=training_metadata.get("end_date"),
        feature_columns=training_metadata["dataset"]["feature_columns"],
    )

    model = joblib.load(record["artifact_path"])
    preprocessor = joblib.load(record["preprocessing_path"])
    metrics = {
        "validation": evaluate_split(model, preprocessor, dataset, dataset.validation, target_config),
        "test": evaluate_split(model, preprocessor, dataset, dataset.test, target_config),
    }
    evaluation_path = Path(record["artifact_dir"]) / "evaluation.json"
    payload = {
        "run_id": record["run_id"],
        "model_name": record["model_name"],
        "feature_set_name": record["feature_set"],
        "target_mode": record["target_mode"],
        "metrics": metrics,
        "evaluation_path": str(evaluation_path),
    }
    evaluation_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    registry.update_phase5_run(record["run_id"], {"metrics": metrics, "evaluation_path": str(evaluation_path)})
    eval_logger.info(
        "Evaluated phase5 run: run_id=%s validation_ranking=%s evaluation=%s",
        record["run_id"],
        _ranking_score(target_config, metrics["validation"]),
        evaluation_path,
    )
    return payload


def compare_model_variants(
    settings: Settings,
    *,
    feature_sets: Sequence[str] | None = None,
    target_modes: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    profile_name: str = "default",
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_combinations: int | None = None,
    logger=None,
) -> dict[str, Any]:
    compare_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.phase5.compare_model_variants",
    )
    modeling_config = load_modeling_config(settings)
    profile = resolve_experiment_profile(settings, profile_name)
    selected_feature_sets = tuple(feature_sets or profile.feature_sets)
    selected_target_modes = tuple(target_modes or profile.target_modes)
    requested_models = {model for model in (models or [])}
    hard_limit = max_combinations or profile.max_combinations

    compare_logger.info(
        "Starting phase5 comparison profile=%s feature_sets=%s target_modes=%s models=%s limit=%s",
        profile_name,
        list(selected_feature_sets),
        list(selected_target_modes),
        list(models or []),
        hard_limit,
    )

    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    attempted = 0
    for feature_set_name in selected_feature_sets:
        for target_mode in selected_target_modes:
            build_labels(
                settings,
                feature_set_name=feature_set_name,
                target_mode=target_mode,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                logger=compare_logger,
            )
            target_config = resolve_target_config(settings, target_mode)
            model_candidates = tuple(requested_models) if requested_models else profile.model_map.get(target_mode, ())
            for model_name in model_candidates:
                compatible, reason = is_model_supported_for_target(model_name, target_config)
                if not compatible:
                    skipped.append(
                        {
                            "feature_set": feature_set_name,
                            "target_mode": target_mode,
                            "model_name": model_name,
                            "reason": reason,
                        }
                    )
                    continue
                for hyperparameters in build_parameter_grid(modeling_config, model_name):
                    if attempted >= hard_limit:
                        break
                    attempted += 1
                    runs.append(
                        train_baseline_variant(
                            settings,
                            feature_set_name=feature_set_name,
                            target_mode=target_mode,
                            model_name=model_name,
                            symbols=symbols,
                            start_date=start_date,
                            end_date=end_date,
                            hyperparameters=hyperparameters,
                            logger=compare_logger,
                        )
                    )
                if attempted >= hard_limit:
                    break
            if attempted >= hard_limit:
                break
        if attempted >= hard_limit:
            break

    leaderboard = [run["leaderboard_row"] for run in runs]
    leaderboard = sorted(leaderboard, key=lambda row: row["ranking_score"], reverse=True)
    leaderboard_paths = _persist_leaderboard(settings, leaderboard, profile_name=profile_name)
    return {
        "status": "ok",
        "profile_name": profile_name,
        "runs": runs,
        "leaderboard": leaderboard,
        "leaderboard_paths": leaderboard_paths,
        "skipped": skipped,
        "attempted_combinations": attempted,
    }


def run_phase5_experiments(
    settings: Settings,
    *,
    feature_sets: Sequence[str] | None = None,
    target_modes: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    profile_name: str = "default",
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    build_features_first: bool = True,
    logger=None,
) -> dict[str, Any]:
    run_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.phase5.run_phase5_experiments",
    )
    selected_feature_sets = tuple(feature_sets or resolve_experiment_profile(settings, profile_name).feature_sets)
    feature_results: list[dict[str, Any]] = []
    if build_features_first:
        for feature_set_name in selected_feature_sets:
            feature_results.append(
                run_feature_build_pipeline(
                    settings,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    feature_set_name=feature_set_name,
                    logger=run_logger,
                )
            )
    comparison = compare_model_variants(
        settings,
        feature_sets=selected_feature_sets,
        target_modes=target_modes,
        models=models,
        profile_name=profile_name,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        logger=run_logger,
    )
    return {
        "status": "ok",
        "profile_name": profile_name,
        "feature_builds": feature_results,
        "comparison": comparison,
    }


def build_leaderboard_row(
    *,
    run_id: str,
    feature_set_name: str,
    target_mode: str,
    model_name: str,
    hyperparameters: dict[str, Any],
    dataset: ModelingDataset,
    metrics: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    target_config = TargetConfig(
        name=dataset.target_config["name"],
        description=dataset.target_config["description"],
        task_type=dataset.target_config["task_type"],
        horizon_bars=dataset.target_config.get("horizon_bars"),
        horizon_minutes=dataset.target_config.get("horizon_minutes"),
        threshold_bps=dataset.target_config.get("threshold_bps"),
        negative_threshold_bps=dataset.target_config.get("negative_threshold_bps"),
        bin_edges_bps=tuple(dataset.target_config.get("bin_edges_bps", [])),
        class_labels=tuple(dataset.target_config.get("class_labels", [])),
        quantiles=tuple(dataset.target_config.get("quantiles", [])),
        cost_adjustment_bps=float(dataset.target_config.get("cost_adjustment_bps", 0.0)),
        cost_adjustment_multiplier=float(dataset.target_config.get("cost_adjustment_multiplier", 0.0)),
        positive_label=int(dataset.target_config.get("positive_label", 1)),
        negative_label=int(dataset.target_config.get("negative_label", 0)),
    )
    validation_metrics = metrics["validation"]
    test_metrics = metrics["test"]
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_type": model_name,
        "target_mode": target_mode,
        "feature_set": feature_set_name,
        "hyperparameters": hyperparameters,
        "split_config": dataset.split_config,
        "symbols": sorted(dataset.frame["symbol"].dropna().astype(str).unique().tolist()),
        "train_range": [dataset.train.dates[0], dataset.train.dates[-1]],
        "validation_range": [dataset.validation.dates[0], dataset.validation.dates[-1]],
        "test_range": [dataset.test.dates[0], dataset.test.dates[-1]],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "artifact_dir": str(artifact_dir),
        "ranking_score": _ranking_score(target_config, validation_metrics),
    }


def _persist_leaderboard(settings: Settings, leaderboard: list[dict[str, Any]], *, profile_name: str) -> dict[str, str]:
    report_dir = Path(settings.paths.report_dir) / "phase5"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"leaderboard_{profile_name}_{timestamp_token}.json"
    csv_path = report_dir / f"leaderboard_{profile_name}_{timestamp_token}.csv"
    parquet_path = report_dir / f"leaderboard_{profile_name}_{timestamp_token}.parquet"
    json_path.write_text(json.dumps(leaderboard, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if leaderboard:
        frame = pd.json_normalize(leaderboard, sep=".")
        frame.to_csv(csv_path, index=False)
        frame.to_parquet(parquet_path, index=False)
    else:
        pd.DataFrame().to_csv(csv_path, index=False)
        pd.DataFrame().to_parquet(parquet_path, index=False)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "parquet": str(parquet_path),
    }


def _build_run_id(model_name: str, feature_set_name: str, target_mode: str) -> str:
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"run_{timestamp_token}_{model_name}_{feature_set_name}_{target_mode}_{uuid4().hex[:8]}"


def _ranking_score(target_config: TargetConfig, validation_metrics: dict[str, Any]) -> float:
    if target_config.task_type in {"classification", "ordinal", "distribution_bins"}:
        return float(validation_metrics.get("f1_macro", validation_metrics.get("accuracy", 0.0)))
    if target_config.task_type == "quantile_regression":
        pinball = validation_metrics.get("pinball_loss", {})
        if pinball:
            return -float(sum(pinball.values()) / max(len(pinball), 1))
        return -float(validation_metrics.get("rmse_bps", 0.0))
    return -float(validation_metrics.get("rmse_bps", 0.0))


def _resolve_run_record(
    settings: Settings,
    registry: ModelRegistry,
    *,
    run_id: str | None,
    artifact_dir: str | Path | None,
) -> dict[str, Any]:
    if run_id:
        record = registry.get_phase5_run(run_id)
        if record is None:
            raise ValueError(f"No phase5 run found with id {run_id}.")
        return record
    if artifact_dir:
        artifact_path = Path(artifact_dir).resolve()
        for record in registry.list_phase5_runs():
            if Path(record["artifact_dir"]).resolve() == artifact_path:
                return record
        raise ValueError(f"No phase5 run found for artifact dir {artifact_path}.")
    runs = sorted(
        registry.list_phase5_runs(),
        key=lambda row: row.get("timestamp_utc", row.get("registered_at", "")),
        reverse=True,
    )
    if not runs:
        raise ValueError("No phase5 runs are registered yet.")
    return runs[0]
