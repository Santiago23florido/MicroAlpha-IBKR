from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    timestamp: str
    source: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last_size: float | None = None
    exchange_time: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class ORBState:
    symbol: str
    timestamp: str
    exchange_time: str
    range_start: str
    range_end: str
    range_high: float | None
    range_low: float | None
    range_mid: float | None
    range_width: float | None
    range_complete: bool
    breakout_direction: str | None
    breakout_price: float | None
    breakout_distance: float | None
    session_window: str
    trading_allowed: bool
    flatten_required: bool
    time_to_close_minutes: float
    candidate_reason: str
    no_trade_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class FeatureSnapshot:
    symbol: str
    timestamp: str
    feature_values: dict[str, float | None]
    estimated_cost_bps: float
    missing_features: list[str] = field(default_factory=list)
    source_mode: str = "paper_or_local"

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class ModelPrediction:
    model_name: str
    model_type: str
    artifact_id: str | None
    probability_up: float
    probability_down: float
    probability_flat: float
    directional_probability: float
    predicted_return_bps: float
    confidence: float
    direction: str | None
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class RiskCheckResult:
    passed: bool
    checks: dict[str, bool]
    failures: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class DecisionRecord:
    symbol: str
    timestamp: str
    orb_state: dict[str, Any]
    direction: str | None
    candidate_trigger_reason: str
    baseline_model_output: dict[str, Any]
    deep_model_output: dict[str, Any]
    selected_final_score: float | None
    expected_edge: float | None
    estimated_cost: float | None
    risk_checks: dict[str, bool]
    risk_passed: bool
    final_action: str
    explanation_text: str
    structured_explanation_data: dict[str, Any]
    feature_values: dict[str, float | None] = field(default_factory=dict)
    threshold_checks: dict[str, bool] = field(default_factory=dict)
    market_status: dict[str, Any] = field(default_factory=dict)
    execution_allowed: bool = False
    model_confirmation_passed: bool = False
    cost_check_passed: bool = False
    orb_condition_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class TradeLifecycleEvent:
    timestamp: str
    symbol: str
    event_type: str
    action: str | None
    quantity: float | None
    status: str
    order_id: int | None = None
    parent_order_id: int | None = None
    price: float | None = None
    realized_pnl: float | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)
