from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


class OrderAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ModelTrace:
    model_name: str
    model_type: str
    run_id: str
    feature_set_name: str
    target_mode: str
    artifact_dir: str
    selection_reason: str | None = None
    source_leaderboard: str | None = None
    updated_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType
    source_model_name: str
    source_decision_id: str
    limit_price: float | None = None
    stop_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class Order:
    order_id: str
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType
    status: OrderStatus
    created_at: str
    updated_at: str
    source_model_name: str
    source_decision_id: str
    limit_price: float | None = None
    stop_price: float | None = None
    filled_quantity: int = 0
    average_fill_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> int:
        return max(int(self.quantity) - int(self.filled_quantity), 0)

    def replace(self, **updates: Any) -> "Order":
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        payload = _serialize_value(self)
        payload["remaining_quantity"] = self.remaining_quantity
        return payload


@dataclass(frozen=True)
class FillEvent:
    fill_id: str
    order_id: str
    symbol: str
    action: OrderAction
    quantity: int
    fill_price: float
    commission: float
    filled_at: str
    backend_name: str
    source_model_name: str
    source_decision_id: str
    slippage_bps: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class ExecutionReport:
    report_id: str
    order_id: str
    symbol: str
    status: OrderStatus
    backend_name: str
    created_at: str
    source_model_name: str
    source_decision_id: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class PositionState:
    symbol: str
    quantity: int = 0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commissions: float = 0.0
    trade_count: int = 0
    last_price: float | None = None
    market_value: float = 0.0
    notional_exposure: float = 0.0
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class PortfolioSnapshot:
    positions: dict[str, PositionState] = field(default_factory=dict)
    cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commissions: float = 0.0
    trade_count: int = 0
    open_position_count: int = 0
    equity: float = 0.0
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class FillApplicationResult:
    position: PositionState
    portfolio: PortfolioSnapshot
    realized_pnl_delta: float
    realized_return_bps: float | None

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)


@dataclass(frozen=True)
class OrderProcessingResult:
    accepted: bool
    order: Order
    reports: list[ExecutionReport] = field(default_factory=list)
    fills: list[FillEvent] = field(default_factory=list)
    portfolio: PortfolioSnapshot | None = None
    realized_pnl_delta: float = 0.0
    realized_return_bps: float | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)
