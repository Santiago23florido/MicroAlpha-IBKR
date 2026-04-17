from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pandas as pd

from config import Settings


IndicatorCalculator = Callable[[pd.DataFrame, Settings, Mapping[str, Any]], pd.DataFrame]


@dataclass(frozen=True)
class IndicatorDependency:
    label: str
    any_of: tuple[str, ...]


@dataclass(frozen=True)
class IndicatorDefinition:
    name: str
    family: str
    description: str
    required_inputs: tuple[IndicatorDependency, ...]
    output_columns: tuple[str, ...]
    output_type: str
    default_params: dict[str, Any] = field(default_factory=dict)
    calculator: IndicatorCalculator | None = None
