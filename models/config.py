from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from config import Settings


@dataclass(frozen=True)
class DatasetConfig:
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    min_rows: int
    min_train_rows: int
    min_validation_rows: int
    min_test_rows: int
    min_unique_dates: int
    dropna_target: bool
    strict_feature_validation: bool


@dataclass(frozen=True)
class TargetConfig:
    name: str
    description: str
    task_type: str
    horizon_bars: int | None = None
    horizon_minutes: int | None = None
    threshold_bps: float | None = None
    negative_threshold_bps: float | None = None
    bin_edges_bps: tuple[float, ...] = ()
    class_labels: tuple[int, ...] = ()
    quantiles: tuple[float, ...] = ()
    cost_adjustment_bps: float = 0.0
    cost_adjustment_multiplier: float = 0.0
    positive_label: int = 1
    negative_label: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "horizon_bars": self.horizon_bars,
            "horizon_minutes": self.horizon_minutes,
            "threshold_bps": self.threshold_bps,
            "negative_threshold_bps": self.negative_threshold_bps,
            "bin_edges_bps": list(self.bin_edges_bps),
            "class_labels": list(self.class_labels),
            "quantiles": list(self.quantiles),
            "cost_adjustment_bps": self.cost_adjustment_bps,
            "cost_adjustment_multiplier": self.cost_adjustment_multiplier,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
        }


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    description: str
    feature_sets: tuple[str, ...]
    target_modes: tuple[str, ...]
    model_map: dict[str, tuple[str, ...]]
    max_combinations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "feature_sets": list(self.feature_sets),
            "target_modes": list(self.target_modes),
            "model_map": {key: list(value) for key, value in self.model_map.items()},
            "max_combinations": self.max_combinations,
        }


@dataclass(frozen=True)
class ModelingConfig:
    dataset: DatasetConfig
    targets: dict[str, TargetConfig]
    model_grids: dict[str, dict[str, list[Any]]]
    experiment_profiles: dict[str, ExperimentProfile]


def load_modeling_config(settings: Settings) -> ModelingConfig:
    config_path = Path(settings.paths.config_dir) / "modeling.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults = payload.get("defaults", {})
    dataset_payload = defaults.get("dataset", {})
    targets_payload = defaults.get("targets", {})
    grids_payload = defaults.get("model_grids", {})
    profiles_payload = defaults.get("experiment_profiles", {})

    dataset = DatasetConfig(
        train_ratio=float(os.getenv("MODEL_TRAIN_RATIO", dataset_payload.get("train_ratio", 0.7))),
        validation_ratio=float(os.getenv("MODEL_VALIDATION_RATIO", dataset_payload.get("validation_ratio", 0.15))),
        test_ratio=float(os.getenv("MODEL_TEST_RATIO", dataset_payload.get("test_ratio", 0.15))),
        min_rows=int(os.getenv("MODEL_MIN_ROWS", dataset_payload.get("min_rows", 120))),
        min_train_rows=int(os.getenv("MODEL_MIN_TRAIN_ROWS", dataset_payload.get("min_train_rows", 60))),
        min_validation_rows=int(os.getenv("MODEL_MIN_VALIDATION_ROWS", dataset_payload.get("min_validation_rows", 20))),
        min_test_rows=int(os.getenv("MODEL_MIN_TEST_ROWS", dataset_payload.get("min_test_rows", 20))),
        min_unique_dates=int(os.getenv("MODEL_MIN_UNIQUE_DATES", dataset_payload.get("min_unique_dates", 3))),
        dropna_target=_env_bool("MODEL_DROPNA_TARGET", bool(dataset_payload.get("dropna_target", True))),
        strict_feature_validation=_env_bool(
            "MODEL_STRICT_FEATURE_VALIDATION",
            bool(dataset_payload.get("strict_feature_validation", True)),
        ),
    )

    targets = {
        name: TargetConfig(
            name=name,
            description=str(item.get("description", "")),
            task_type=str(item.get("task_type")),
            horizon_bars=_optional_int(item.get("horizon_bars")),
            horizon_minutes=_optional_int(item.get("horizon_minutes")),
            threshold_bps=_optional_float(item.get("threshold_bps")),
            negative_threshold_bps=_optional_float(item.get("negative_threshold_bps")),
            bin_edges_bps=tuple(float(value) for value in item.get("bin_edges_bps", [])),
            class_labels=tuple(int(value) for value in item.get("class_labels", [])),
            quantiles=tuple(float(value) for value in item.get("quantiles", [])),
            cost_adjustment_bps=float(item.get("cost_adjustment_bps", 0.0)),
            cost_adjustment_multiplier=float(item.get("cost_adjustment_multiplier", 0.0)),
            positive_label=int(item.get("positive_label", 1)),
            negative_label=int(item.get("negative_label", 0)),
        )
        for name, item in targets_payload.items()
    }

    model_grids = {
        str(name): {str(param): list(values) for param, values in grid.items()}
        for name, grid in grids_payload.items()
    }
    experiment_profiles = {
        name: ExperimentProfile(
            name=name,
            description=str(item.get("description", "")),
            feature_sets=tuple(str(value) for value in item.get("feature_sets", [])),
            target_modes=tuple(str(value) for value in item.get("target_modes", [])),
            model_map={str(key): tuple(str(value) for value in values) for key, values in item.get("model_map", {}).items()},
            max_combinations=int(item.get("max_combinations", 24)),
        )
        for name, item in profiles_payload.items()
    }
    return ModelingConfig(
        dataset=dataset,
        targets=targets,
        model_grids=model_grids,
        experiment_profiles=experiment_profiles,
    )


def resolve_target_config(settings: Settings, target_mode: str) -> TargetConfig:
    config = load_modeling_config(settings)
    try:
        return config.targets[target_mode]
    except KeyError as exc:
        available = ", ".join(sorted(config.targets))
        raise KeyError(f"Unknown target mode {target_mode!r}. Available target modes: {available}") from exc


def resolve_experiment_profile(settings: Settings, profile_name: str) -> ExperimentProfile:
    config = load_modeling_config(settings)
    try:
        return config.experiment_profiles[profile_name]
    except KeyError as exc:
        available = ", ".join(sorted(config.experiment_profiles))
        raise KeyError(f"Unknown experiment profile {profile_name!r}. Available profiles: {available}") from exc


def build_parameter_grid(modeling_config: ModelingConfig, model_name: str) -> list[dict[str, Any]]:
    grid = modeling_config.model_grids.get(model_name, {})
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values_product = product(*(grid[key] for key in keys))
    return [dict(zip(keys, values)) for values in values_product]


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
