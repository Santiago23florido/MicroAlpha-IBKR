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
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "backend_name": self.backend_name,
            "fills": [fill.to_dict() for fill in self.fills],
            "rejection_reason": self.rejection_reason,
            "terminal_status": None if self.terminal_status is None else self.terminal_status.value,
            "metadata": dict(self.metadata),
        }


class BaseExecutionBackend(ABC):
    name: str

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: Order, market_data: Mapping[str, Any] | None = None) -> BackendSubmissionResult:
        raise NotImplementedError


class FutureIBKRPaperBackend(BaseExecutionBackend):
    name = "ibkr_paper"

    def __init__(self, config: Phase7Config) -> None:
        self.config = config

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": False,
            "paper_mode": self.config.execution.paper_mode,
            "message": "Placeholder backend. Phase 7 uses the mock execution backend until IBKR paper routing is connected.",
        }

    def submit_order(self, order: Order, market_data: Mapping[str, Any] | None = None) -> BackendSubmissionResult:
        raise NotImplementedError(
            "IBKR paper backend is not implemented yet. Set ACTIVE_EXECUTION_BACKEND=mock for Phase 7."
        )


def build_execution_backend(config: Phase7Config) -> BaseExecutionBackend:
    backend_name = config.execution.active_execution_backend.strip().lower()
    if backend_name == "mock":
        from execution.paper_broker_mock import MockExecutionBackend

        return MockExecutionBackend(config)
    if backend_name in {"ibkr_paper", "ibkr-paper"}:
        return FutureIBKRPaperBackend(config)
    raise ValueError(
        f"Unsupported execution backend {config.execution.active_execution_backend!r}. "
        "Supported values: mock, ibkr_paper."
    )
