from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from config.phase7 import Phase7Config
from execution.models import FillEvent, Order, OrderStatus


@dataclass(frozen=True)
class BackendSubmissionResult:
    accepted: bool
    backend_name: str
    fills: list[FillEvent] = field(default_factory=list)
    rejection_reason: str | None = None
    terminal_status: OrderStatus | None = None
    broker_order_id: int | None = None
    broker_perm_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "backend_name": self.backend_name,
            "fills": [fill.to_dict() for fill in self.fills],
            "rejection_reason": self.rejection_reason,
            "terminal_status": None if self.terminal_status is None else self.terminal_status.value,
            "broker_order_id": self.broker_order_id,
            "broker_perm_id": self.broker_perm_id,
            "metadata": dict(self.metadata),
        }


class BaseExecutionBackend(ABC):
    name: str

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: Order, market_data: Mapping[str, Any] | None = None) -> BackendSubmissionResult:
        raise NotImplementedError

    def cancel_order(self, order: Order) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support cancel_order().")

    def get_order_status(self, order: Order) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support get_order_status().")

    def get_open_orders(self) -> list[dict[str, Any]]:
        return []

    def get_positions(self) -> list[dict[str, Any]]:
        return []

    def get_recent_executions(self) -> list[dict[str, Any]]:
        return []

    def healthcheck(self) -> dict[str, Any]:
        return self.describe()

def build_execution_backend(
    config: Phase7Config,
    *,
    settings=None,
    logger=None,
    client=None,
    backend_name: str | None = None,
) -> BaseExecutionBackend:
    backend_name = (backend_name or config.execution.active_execution_backend).strip().lower()
    if backend_name == "mock":
        from execution.paper_broker_mock import MockExecutionBackend

        return MockExecutionBackend(config)
    if backend_name in {"ibkr_paper", "ibkr-paper"}:
        from execution.ibkr_paper_backend import IBKRPaperExecutionBackend

        if settings is None:
            raise ValueError("IBKR paper backend requires Settings to resolve broker connectivity and safety guards.")
        return IBKRPaperExecutionBackend(config, settings=settings, logger=logger, client=client)
    raise ValueError(
        f"Unsupported execution backend {config.execution.active_execution_backend!r}. "
        "Supported values: mock, ibkr_paper."
    )
