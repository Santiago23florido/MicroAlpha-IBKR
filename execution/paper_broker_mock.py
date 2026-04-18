from __future__ import annotations

from typing import Any, Mapping

from config.phase7 import Phase7Config
from execution.backend import BackendSubmissionResult, BaseExecutionBackend
from execution.fill_simulator import FillSimulator
from execution.models import Order


class MockExecutionBackend(BaseExecutionBackend):
    name = "mock"

    def __init__(self, config: Phase7Config) -> None:
        self.config = config
        self.fill_simulator = FillSimulator(config.execution)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": True,
            "paper_mode": self.config.execution.paper_mode,
            "default_order_type": self.config.execution.default_order_type,
            "allow_partial_fills": self.config.execution.allow_partial_fills,
            "fill_delay_ms": self.config.execution.fill_delay_ms,
            "slippage_bps": self.config.execution.slippage_bps,
            "commission_per_trade": self.config.execution.commission_per_trade,
            "commission_per_share": self.config.execution.commission_per_share,
            "reject_probability": self.config.execution.reject_probability,
            "spread_aware_fills": self.config.execution.spread_aware_fills,
            "simulate_immediate_fills": self.config.execution.simulate_immediate_fills,
        }

    def submit_order(self, order: Order, market_data: Mapping[str, Any] | None = None) -> BackendSubmissionResult:
        simulation = self.fill_simulator.simulate_fill(order, market_data)
        if not simulation.accepted:
            return BackendSubmissionResult(
                accepted=False,
                backend_name=self.name,
                rejection_reason=simulation.rejection_reason or "mock_execution_rejected",
                metadata=dict(simulation.metadata or {}),
            )
        return BackendSubmissionResult(
            accepted=True,
            backend_name=self.name,
            fills=list(simulation.fills),
            metadata=dict(simulation.metadata or {}),
        )
