from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from config import Settings
from features.definitions import IndicatorDefinition, IndicatorDependency
from features.indicators import (
    build_intraday_indicator_definitions,
    build_microstructure_indicator_definitions,
    build_momentum_indicator_definitions,
    build_trend_indicator_definitions,
    build_volatility_indicator_definitions,
    build_volume_flow_indicator_definitions,
)


@dataclass(frozen=True)
class FeatureSetDefinition:
    name: str
    description: str
    families: tuple[str, ...]
    indicators: tuple[str, ...]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    minimum_columns: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "families": list(self.families),
            "indicators": list(self.indicators),
            "params": self.params,
            "minimum_columns": list(self.minimum_columns),
        }


@dataclass(frozen=True)
class IndicatorResolution:
    name: str
    family: str
    output_columns: tuple[str, ...]
    status: str
    resolved_dependencies: dict[str, str] = field(default_factory=dict)
    missing_dependencies: dict[str, list[str]] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "output_columns": list(self.output_columns),
            "status": self.status,
            "resolved_dependencies": dict(self.resolved_dependencies),
            "missing_dependencies": {key: list(value) for key, value in self.missing_dependencies.items()},
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FeatureExecutionPlan:
    feature_set: FeatureSetDefinition
    compatible_indicators: tuple[IndicatorResolution, ...]
    omitted_indicators: tuple[IndicatorResolution, ...]
    input_columns: tuple[str, ...]
    non_null_columns: tuple[str, ...]
    planned_feature_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set.to_dict(),
            "compatible_indicators": [item.to_dict() for item in self.compatible_indicators],
            "omitted_indicators": [item.to_dict() for item in self.omitted_indicators],
            "input_columns": list(self.input_columns),
            "non_null_columns": list(self.non_null_columns),
            "planned_feature_columns": list(self.planned_feature_columns),
        }


def build_indicator_registry() -> dict[str, IndicatorDefinition]:
    definitions = (
        build_trend_indicator_definitions()
        + build_momentum_indicator_definitions()
        + build_volatility_indicator_definitions()
        + build_volume_flow_indicator_definitions()
        + build_microstructure_indicator_definitions()
        + build_intraday_indicator_definitions()
    )
    registry: dict[str, IndicatorDefinition] = {}
    for definition in definitions:
        if definition.name in registry:
            raise ValueError(f"Duplicate indicator definition detected for {definition.name!r}.")
        registry[definition.name] = definition
    return registry


@lru_cache(maxsize=8)
def _load_feature_sets_yaml(config_dir: str) -> dict[str, Any]:
    config_path = Path(config_dir) / "feature_sets.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Feature set configuration not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping content in {config_path}.")
    return payload


def load_feature_sets(settings: Settings) -> dict[str, FeatureSetDefinition]:
    payload = _load_feature_sets_yaml(settings.paths.config_dir)
    feature_sets_payload = payload.get("feature_sets", {})
    feature_sets: dict[str, FeatureSetDefinition] = {}
    for name, item in feature_sets_payload.items():
        feature_sets[name] = FeatureSetDefinition(
            name=name,
            description=str(item.get("description", "")),
            families=tuple(str(value) for value in item.get("families", [])),
            indicators=tuple(str(value) for value in item.get("indicators", [])),
            params={str(key): dict(value) for key, value in item.get("params", {}).items()},
            minimum_columns=tuple(str(value) for value in item.get("minimum_columns", [])),
        )
    return feature_sets


def default_feature_set_name(settings: Settings) -> str:
    payload = _load_feature_sets_yaml(settings.paths.config_dir)
    configured_default = str(payload.get("defaults", {}).get("default_feature_set", "hybrid_intraday"))
    return settings.feature_pipeline.default_feature_set or configured_default


def list_feature_sets(settings: Settings) -> dict[str, Any]:
    feature_sets = load_feature_sets(settings)
    return {
        "default_feature_set": default_feature_set_name(settings),
        "feature_sets": [definition.to_dict() for definition in feature_sets.values()],
    }


def resolve_feature_set(settings: Settings, feature_set_name: str | None = None) -> FeatureSetDefinition:
    feature_sets = load_feature_sets(settings)
    resolved_name = feature_set_name or default_feature_set_name(settings)
    if resolved_name not in feature_sets:
        available = ", ".join(sorted(feature_sets))
        raise KeyError(f"Unknown feature set {resolved_name!r}. Available feature sets: {available}")
    return feature_sets[resolved_name]


def inspect_feature_dependencies(
    frame: pd.DataFrame,
    settings: Settings,
    *,
    feature_set_name: str | None = None,
) -> FeatureExecutionPlan:
    registry = build_indicator_registry()
    feature_set = resolve_feature_set(settings, feature_set_name)
    input_columns = tuple(str(column) for column in frame.columns)
    non_null_columns = tuple(
        str(column)
        for column in frame.columns
        if pd.Series(frame[column]).notna().any()
    )
    selected_names = _expand_feature_set(feature_set, registry)

    compatible: list[IndicatorResolution] = []
    omitted: list[IndicatorResolution] = []
    planned_columns: list[str] = []

    for indicator_name in selected_names:
        definition = registry[indicator_name]
        resolution = _resolve_indicator(definition, frame)
        if resolution.status == "compatible":
            compatible.append(resolution)
            planned_columns.extend(definition.output_columns)
        else:
            omitted.append(resolution)

    return FeatureExecutionPlan(
        feature_set=feature_set,
        compatible_indicators=tuple(compatible),
        omitted_indicators=tuple(omitted),
        input_columns=input_columns,
        non_null_columns=non_null_columns,
        planned_feature_columns=tuple(dict.fromkeys(planned_columns)),
    )


def _expand_feature_set(
    feature_set: FeatureSetDefinition,
    registry: Mapping[str, IndicatorDefinition],
) -> tuple[str, ...]:
    selected: list[str] = []
    if feature_set.families:
        for name, definition in registry.items():
            if definition.family in set(feature_set.families):
                selected.append(name)
    if feature_set.indicators:
        for indicator_name in feature_set.indicators:
            if indicator_name not in registry:
                raise KeyError(f"Feature set {feature_set.name!r} references unknown indicator {indicator_name!r}.")
            selected.append(indicator_name)
    return tuple(dict.fromkeys(selected))


def _resolve_indicator(definition: IndicatorDefinition, frame: pd.DataFrame) -> IndicatorResolution:
    resolved_dependencies: dict[str, str] = {}
    missing_dependencies: dict[str, list[str]] = {}
    for dependency in definition.required_inputs:
        resolved = _resolve_dependency(frame, dependency)
        if resolved is None:
            missing_dependencies[dependency.label] = list(dependency.any_of)
        else:
            resolved_dependencies[dependency.label] = resolved

    if missing_dependencies:
        missing_parts = ", ".join(
            f"{label}: any of {columns}" for label, columns in missing_dependencies.items()
        )
        return IndicatorResolution(
            name=definition.name,
            family=definition.family,
            output_columns=definition.output_columns,
            status="omitted",
            resolved_dependencies=resolved_dependencies,
            missing_dependencies=missing_dependencies,
            reason=f"Missing usable dependency columns ({missing_parts}).",
        )

    return IndicatorResolution(
        name=definition.name,
        family=definition.family,
        output_columns=definition.output_columns,
        status="compatible",
        resolved_dependencies=resolved_dependencies,
        missing_dependencies={},
        reason=None,
    )


def _resolve_dependency(frame: pd.DataFrame, dependency: IndicatorDependency) -> str | None:
    for column in dependency.any_of:
        if column in frame.columns and pd.Series(frame[column]).notna().any():
            return column
    return None
