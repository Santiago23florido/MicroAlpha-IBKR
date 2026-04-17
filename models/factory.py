from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.config import TargetConfig


PreprocessorFactory = Callable[[], Pipeline]
EstimatorFactory = Callable[[dict[str, Any], TargetConfig], Any]


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    description: str
    supported_task_types: tuple[str, ...]
    preprocessor_factory: PreprocessorFactory
    estimator_factory: EstimatorFactory
    optional_dependency: str | None = None

    @property
    def available(self) -> bool:
        if self.optional_dependency is None:
            return True
        return importlib.util.find_spec(self.optional_dependency) is not None


class QuantileGradientBoostingEnsemble:
    def __init__(self, quantiles: list[float], base_params: dict[str, Any]) -> None:
        self.quantiles = list(quantiles)
        self.base_params = dict(base_params)
        self.models: dict[float, GradientBoostingRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileGradientBoostingEnsemble":
        self.models = {}
        for quantile in self.quantiles:
            params = {
                **self.base_params,
                "loss": "quantile",
                "alpha": float(quantile),
            }
            model = GradientBoostingRegressor(**params)
            model.fit(X, y)
            self.models[float(quantile)] = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        median_key = min(self.quantiles, key=lambda value: abs(value - 0.5))
        return self.models[float(median_key)].predict(X)

    def predict_quantiles(self, X: np.ndarray) -> dict[float, np.ndarray]:
        return {quantile: model.predict(X) for quantile, model in self.models.items()}


def list_model_definitions() -> dict[str, ModelDefinition]:
    return {
        "logistic_regression": ModelDefinition(
            name="logistic_regression",
            description="Multiclass logistic regression baseline with scaling.",
            supported_task_types=("classification", "ordinal", "distribution_bins"),
            preprocessor_factory=_scaling_preprocessor,
            estimator_factory=lambda params, _: LogisticRegression(**params),
        ),
        "random_forest_classifier": ModelDefinition(
            name="random_forest_classifier",
            description="Random forest classifier for binary or multiclass targets.",
            supported_task_types=("classification", "ordinal", "distribution_bins"),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=lambda params, _: RandomForestClassifier(**params),
        ),
        "hist_gradient_boosting_classifier": ModelDefinition(
            name="hist_gradient_boosting_classifier",
            description="Histogram gradient boosting classifier for tabular features.",
            supported_task_types=("classification", "ordinal", "distribution_bins"),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=lambda params, _: HistGradientBoostingClassifier(**params),
        ),
        "ridge_regression": ModelDefinition(
            name="ridge_regression",
            description="Scaled ridge regression for point-return prediction.",
            supported_task_types=("regression",),
            preprocessor_factory=_scaling_preprocessor,
            estimator_factory=lambda params, _: Ridge(**params),
        ),
        "random_forest_regressor": ModelDefinition(
            name="random_forest_regressor",
            description="Random forest regressor for non-linear point forecasts.",
            supported_task_types=("regression",),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=lambda params, _: RandomForestRegressor(**params),
        ),
        "hist_gradient_boosting_regressor": ModelDefinition(
            name="hist_gradient_boosting_regressor",
            description="Histogram gradient boosting regressor for point targets.",
            supported_task_types=("regression",),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=lambda params, _: HistGradientBoostingRegressor(**params),
        ),
        "quantile_gradient_boosting": ModelDefinition(
            name="quantile_gradient_boosting",
            description="Multiple quantile gradient boosting regressors for q10/q50/q90 style forecasts.",
            supported_task_types=("quantile_regression",),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=lambda params, target: QuantileGradientBoostingEnsemble(
                quantiles=list(target.quantiles or (0.1, 0.5, 0.9)),
                base_params=params,
            ),
        ),
        "xgboost_classifier": ModelDefinition(
            name="xgboost_classifier",
            description="Optional XGBoost classifier if xgboost is installed.",
            supported_task_types=("classification", "ordinal", "distribution_bins"),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=_build_xgboost_classifier,
            optional_dependency="xgboost",
        ),
        "xgboost_regressor": ModelDefinition(
            name="xgboost_regressor",
            description="Optional XGBoost regressor if xgboost is installed.",
            supported_task_types=("regression",),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=_build_xgboost_regressor,
            optional_dependency="xgboost",
        ),
        "lightgbm_classifier": ModelDefinition(
            name="lightgbm_classifier",
            description="Optional LightGBM classifier if lightgbm is installed.",
            supported_task_types=("classification", "ordinal", "distribution_bins"),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=_build_lightgbm_classifier,
            optional_dependency="lightgbm",
        ),
        "lightgbm_regressor": ModelDefinition(
            name="lightgbm_regressor",
            description="Optional LightGBM regressor if lightgbm is installed.",
            supported_task_types=("regression",),
            preprocessor_factory=_tree_preprocessor,
            estimator_factory=_build_lightgbm_regressor,
            optional_dependency="lightgbm",
        ),
    }


def list_supported_models() -> list[dict[str, Any]]:
    return [
        {
            "name": definition.name,
            "description": definition.description,
            "supported_task_types": list(definition.supported_task_types),
            "available": definition.available,
            "optional_dependency": definition.optional_dependency,
        }
        for definition in list_model_definitions().values()
    ]


def create_model_components(
    model_name: str,
    target_config: TargetConfig,
    params: dict[str, Any],
) -> tuple[Pipeline, Any, dict[str, Any]]:
    definitions = list_model_definitions()
    if model_name not in definitions:
        available = ", ".join(sorted(definitions))
        raise KeyError(f"Unknown model {model_name!r}. Available models: {available}")

    definition = definitions[model_name]
    if not definition.available:
        raise ValueError(
            f"Model {model_name!r} is unavailable because dependency {definition.optional_dependency!r} is not installed."
        )
    if target_config.task_type not in definition.supported_task_types:
        raise ValueError(
            f"Model {model_name!r} does not support target task type {target_config.task_type!r}."
        )

    preprocessor = definition.preprocessor_factory()
    estimator = definition.estimator_factory(dict(params), target_config)
    return preprocessor, estimator, {
        "model_name": definition.name,
        "description": definition.description,
        "task_types": list(definition.supported_task_types),
        "params": dict(params),
    }


def is_model_supported_for_target(model_name: str, target_config: TargetConfig) -> tuple[bool, str | None]:
    definitions = list_model_definitions()
    definition = definitions.get(model_name)
    if definition is None:
        return False, f"Unknown model {model_name!r}."
    if not definition.available:
        return False, f"Dependency {definition.optional_dependency!r} is not installed."
    if target_config.task_type not in definition.supported_task_types:
        return False, f"Model does not support task type {target_config.task_type!r}."
    return True, None


def _scaling_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _tree_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])


def _build_xgboost_classifier(params: dict[str, Any], _: TargetConfig) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(**params)


def _build_xgboost_regressor(params: dict[str, Any], _: TargetConfig) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**params)


def _build_lightgbm_classifier(params: dict[str, Any], _: TargetConfig) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params)


def _build_lightgbm_regressor(params: dict[str, Any], _: TargetConfig) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params)
