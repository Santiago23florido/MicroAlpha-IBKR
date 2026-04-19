from __future__ import annotations

from typing import Any, Mapping
from uuid import uuid4

from config.phase7 import Phase7Config
from execution.backend import BaseExecutionBackend
from execution.journal import ExecutionJournal
from execution.models import (
    ExecutionReport,
    ModelTrace,
    Order,
    OrderAction,
    OrderProcessingResult,
    OrderRequest,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    utc_now_iso,
)
from execution.order_state_machine import OrderStateMachine
from execution.position_manager import PositionConsistencyError, PositionManager


class OrderValidationError(ValueError):
    """Raised when a decision or order request is not executable."""


class OrderManager:
    def __init__(
        self,
        config: Phase7Config,
        *,
        backend: BaseExecutionBackend,
        journal: ExecutionJournal,
        position_manager: PositionManager,
        state_machine: OrderStateMachine | None = None,
    ) -> None:
        if backend is None:
            raise ValueError("OrderManager requires an execution backend.")
        self.config = config
        self.backend = backend
        self.journal = journal
        self.position_manager = position_manager
        self.state_machine = state_machine or OrderStateMachine()
        self.orders: dict[str, Order] = {}

    def restore_orders(self, payloads: list[Mapping[str, Any]] | None) -> None:
        self.orders = {}
        for payload in payloads or []:
            order = Order(
                order_id=str(payload.get("order_id")),
                symbol=str(payload.get("symbol", "")).upper(),
                action=OrderAction(str(payload.get("action")).upper()),
                quantity=int(payload.get("quantity", 0) or 0),
                order_type=OrderType(str(payload.get("order_type", "MARKET")).upper()),
                status=OrderStatus(str(payload.get("status", "CREATED")).upper()),
                created_at=str(payload.get("created_at")),
                updated_at=str(payload.get("updated_at")),
                source_model_name=str(payload.get("source_model_name")),
                source_decision_id=str(payload.get("source_decision_id")),
                backend_name=payload.get("backend_name"),
                broker_order_id=_coerce_optional_int(payload.get("broker_order_id")),
                broker_perm_id=_coerce_optional_int(payload.get("broker_perm_id")),
                limit_price=_coerce_optional_float(payload.get("limit_price")),
                stop_price=_coerce_optional_float(payload.get("stop_price")),
                filled_quantity=int(payload.get("filled_quantity", 0) or 0),
                average_fill_price=_coerce_optional_float(payload.get("average_fill_price")),
                metadata=dict(payload.get("metadata", {}) or {}),
            )
            self.orders[order.order_id] = order

    def process_decision(
        self,
        decision: Mapping[str, Any],
        *,
        model_trace: ModelTrace,
        decision_id: str,
        market_data: Mapping[str, Any] | None = None,
    ) -> OrderProcessingResult:
        request = self._build_order_request(decision, model_trace=model_trace, decision_id=decision_id)
        return self.submit_order_request(request, market_data=market_data)

    def submit_order_request(
        self,
        request: OrderRequest,
        *,
        market_data: Mapping[str, Any] | None = None,
    ) -> OrderProcessingResult:
        created_at = utc_now_iso()
        order = Order(
            order_id=f"ord_{uuid4().hex[:12]}",
            symbol=request.symbol.upper(),
            action=request.action,
            quantity=int(request.quantity),
            order_type=request.order_type,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            status=OrderStatus.CREATED,
            created_at=created_at,
            updated_at=created_at,
            source_model_name=request.source_model_name,
            source_decision_id=request.source_decision_id,
            backend_name=self.backend.name,
            metadata=dict(request.metadata),
        )
        self.orders[order.order_id] = order
        self.journal.append_order(order)

        validation_errors = self._validate_order(order)
        reports: list[ExecutionReport] = []
        fills = []
        portfolio = self.position_manager.snapshot()

        if validation_errors:
            target_status = OrderStatus.REJECTED if self.config.execution.reject_invalid_orders else OrderStatus.FAILED
            order = self._transition(
                order,
                target_status,
                reports,
                message="; ".join(validation_errors),
                metadata={"validation_errors": validation_errors},
            )
            return OrderProcessingResult(
                accepted=False,
                order=order,
                reports=reports,
                fills=fills,
                portfolio=portfolio,
                errors=validation_errors,
            )

        order = self._transition(order, OrderStatus.SUBMITTED, reports, message="Order submitted to execution backend.")
        backend_result = self.backend.submit_order(order, market_data)
        order = self._merge_backend_identity(order, backend_result)
        self.journal.append_backend_event(
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "source_decision_id": order.source_decision_id,
                "backend_name": backend_result.backend_name,
                "payload": backend_result.to_dict(),
            }
        )
        self.journal.append_reconciliation(
            {
                "stage": "submission_result",
                "order_id": order.order_id,
                "symbol": order.symbol,
                "source_decision_id": order.source_decision_id,
                "source_model_name": order.source_model_name,
                "backend_name": backend_result.backend_name,
                "broker_order_id": order.broker_order_id,
                "broker_perm_id": order.broker_perm_id,
                "metadata": dict(backend_result.metadata),
            }
        )

        if not backend_result.accepted:
            order = self._transition(
                order,
                backend_result.terminal_status or OrderStatus.REJECTED,
                reports,
                message=backend_result.rejection_reason or "Execution backend rejected order.",
                metadata=dict(backend_result.metadata),
            )
            return OrderProcessingResult(
                accepted=False,
                order=order,
                reports=reports,
                fills=fills,
                portfolio=portfolio,
                errors=[backend_result.rejection_reason or "backend_rejected_order"],
            )

        order = self._transition(
            order,
            OrderStatus.ACKNOWLEDGED,
            reports,
            message="Execution backend acknowledged order.",
            metadata=dict(backend_result.metadata),
            backend_name=backend_result.backend_name,
        )

        realized_pnl_delta = 0.0
        realized_return_bps: float | None = None
        try:
            for fill in backend_result.fills:
                self._validate_fill(order, fill)
                order = self.state_machine.apply_fill(order, fill)
                self.orders[order.order_id] = order
                self.journal.append_fill(fill)
                self.journal.append_order(order)
                self.journal.append_reconciliation(
                    {
                        "stage": "fill",
                        "order_id": order.order_id,
                        "broker_order_id": fill.broker_order_id,
                        "execution_id": fill.execution_id,
                        "fill_id": fill.fill_id,
                        "symbol": fill.symbol,
                        "source_decision_id": fill.source_decision_id,
                        "source_model_name": fill.source_model_name,
                        "backend_name": fill.backend_name,
                        "quantity": fill.quantity,
                        "fill_price": fill.fill_price,
                        "filled_at": fill.filled_at,
                        "metadata": dict(fill.metadata),
                    }
                )
                fill_result = self.position_manager.apply_fill(fill)
                realized_pnl_delta += fill_result.realized_pnl_delta
                if fill_result.realized_return_bps is not None:
                    realized_return_bps = fill_result.realized_return_bps
                portfolio = fill_result.portfolio
                self.journal.append_position(
                    fill_result.position,
                    order_id=order.order_id,
                    fill_id=fill.fill_id,
                    source_decision_id=order.source_decision_id,
                )
                self.journal.append_pnl(
                    portfolio,
                    order_id=order.order_id,
                    fill_id=fill.fill_id,
                    source_decision_id=order.source_decision_id,
                    realized_pnl_delta=fill_result.realized_pnl_delta,
                )
                reports.append(
                    self._build_report(
                        order,
                        order.status,
                        backend_name=backend_result.backend_name,
                        message=f"Processed fill {fill.fill_id}.",
                        metadata={"fill_id": fill.fill_id, "fill_quantity": fill.quantity},
                    )
                )
                self.journal.append_report(reports[-1])
        except PositionConsistencyError as exc:
            order = self._transition(
                order,
                OrderStatus.FAILED,
                reports,
                message=str(exc),
                backend_name=backend_result.backend_name,
            )
            return OrderProcessingResult(
                accepted=False,
                order=order,
                reports=reports,
                fills=list(backend_result.fills),
                portfolio=portfolio,
                realized_pnl_delta=float(realized_pnl_delta),
                realized_return_bps=realized_return_bps,
                errors=[str(exc)],
            )

        if backend_result.terminal_status and order.status != backend_result.terminal_status:
            order = self._transition(
                order,
                backend_result.terminal_status,
                reports,
                message="Backend closed the order lifecycle.",
                metadata=dict(backend_result.metadata),
                backend_name=backend_result.backend_name,
            )

        self.orders[order.order_id] = order
        return OrderProcessingResult(
            accepted=True,
            order=order,
            reports=reports,
            fills=list(backend_result.fills),
            portfolio=portfolio,
            realized_pnl_delta=float(realized_pnl_delta),
            realized_return_bps=realized_return_bps,
        )

    def open_orders(self) -> list[Order]:
        terminal = {
            OrderStatus.REJECTED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        }
        return [order for order in self.orders.values() if order.status not in terminal]

    def snapshot_orders(self) -> list[dict[str, Any]]:
        return [order.to_dict() for order in sorted(self.orders.values(), key=lambda item: item.created_at)]

    def _build_order_request(
        self,
        decision: Mapping[str, Any],
        *,
        model_trace: ModelTrace,
        decision_id: str,
    ) -> OrderRequest:
        symbol = str(decision.get("symbol") or "").strip().upper()
        if not symbol:
            raise OrderValidationError("Decision is missing symbol.")

        desired_quantity = int(decision.get("size_suggestion") or self.config.execution.default_position_size)
        if desired_quantity <= 0:
            raise OrderValidationError(f"Decision produced invalid quantity {desired_quantity}.")

        raw_action = str(decision.get("action") or "").strip().upper()
        if raw_action == "NO_TRADE":
            raise OrderValidationError("NO_TRADE decisions are not executable.")

        current_quantity = self.position_manager.current_quantity(symbol)
        action, quantity = self._resolve_order_action(raw_action, current_quantity, desired_quantity)
        order_type = self._resolve_order_type(self.config.execution.default_order_type)

        metadata = {
            "decision_action": raw_action,
            "decision_snapshot": dict(decision),
            "decision_generated_at_utc": decision.get("decision_generated_at_utc"),
            "model_trace": model_trace.to_dict(),
            "run_id": model_trace.run_id,
            "feature_set_name": model_trace.feature_set_name,
            "target_mode": model_trace.target_mode,
            "artifact_dir": model_trace.artifact_dir,
        }
        return OrderRequest(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=order_type,
            source_model_name=model_trace.model_name,
            source_decision_id=decision_id,
            metadata=metadata,
        )

    def _validate_order(self, order: Order) -> list[str]:
        errors: list[str] = []
        if not order.symbol:
            errors.append("missing_symbol")
        if order.quantity <= 0:
            errors.append("invalid_quantity")
        if not order.source_model_name or not order.source_decision_id:
            errors.append("missing_traceability_metadata")

        model_trace = dict(order.metadata.get("model_trace", {}) or {})
        for field in ("run_id", "feature_set_name", "target_mode", "artifact_dir"):
            if not model_trace.get(field):
                errors.append(f"missing_model_trace_{field}")

        decision_snapshot = dict(order.metadata.get("decision_snapshot", {}) or {})
        if self.backend.name == "ibkr_paper":
            if self.config.ibkr_paper.broker_mode.strip().lower() != "paper":
                errors.append(f"broker_mode_must_be_paper:{self.config.ibkr_paper.broker_mode}")
            if not self.config.execution.paper_mode:
                errors.append("paper_mode_disabled")
            if not self.config.ibkr_paper.safe_to_trade:
                errors.append("safe_to_trade_disabled")
            if not self.config.ibkr_paper.allow_session_execution:
                errors.append("allow_session_execution_disabled")
            if self.config.ibkr_paper.supported_symbols and order.symbol not in set(self.config.ibkr_paper.supported_symbols):
                errors.append(f"symbol_not_supported:{order.symbol}")
            if not decision_snapshot:
                errors.append("missing_decision_snapshot")
            if decision_snapshot.get("blocked_by_risk"):
                errors.append("decision_blocked_by_risk")
            risk_checks = dict(decision_snapshot.get("risk_checks", {}) or {})
            if not risk_checks:
                errors.append("missing_risk_checks")

        current_quantity = self.position_manager.current_quantity(order.symbol)
        projected_quantity = current_quantity
        if order.action == OrderAction.BUY and current_quantity >= 0:
            projected_quantity = current_quantity + order.quantity
        elif order.action == OrderAction.SHORT and current_quantity <= 0:
            projected_quantity = current_quantity - order.quantity

        if abs(projected_quantity) > self.config.execution.max_position_size:
            errors.append(
                f"projected_position_size_exceeds_limit:{abs(projected_quantity)}>{self.config.execution.max_position_size}"
            )
        return errors

    def _validate_fill(self, order: Order, fill) -> None:
        if fill.order_id != order.order_id:
            raise PositionConsistencyError(
                f"Fill {fill.fill_id} references order {fill.order_id}, expected {order.order_id}."
            )
        if fill.symbol.upper() != order.symbol.upper():
            raise PositionConsistencyError(
                f"Fill {fill.fill_id} symbol {fill.symbol} does not match order symbol {order.symbol}."
            )
        if int(fill.quantity) <= 0:
            raise PositionConsistencyError(f"Fill {fill.fill_id} has invalid quantity {fill.quantity}.")

    def _transition(
        self,
        order: Order,
        target: OrderStatus,
        reports: list[ExecutionReport],
        *,
        message: str,
        metadata: Mapping[str, Any] | None = None,
        backend_name: str | None = None,
    ) -> Order:
        merged_metadata = {**dict(order.metadata), **dict(metadata or {})}
        updated = self.state_machine.transition(
            order.replace(
                metadata=merged_metadata,
                backend_name=backend_name or order.backend_name or self.backend.name,
                broker_order_id=_coerce_optional_int(merged_metadata.get("broker_order_id")) or order.broker_order_id,
                broker_perm_id=_coerce_optional_int(merged_metadata.get("broker_perm_id")) or order.broker_perm_id,
            ),
            target,
            updated_at=utc_now_iso(),
        )
        self.orders[updated.order_id] = updated
        self.journal.append_order(updated)
        report = self._build_report(
            updated,
            target,
            backend_name=backend_name,
            message=message,
            metadata=dict(metadata or {}),
        )
        reports.append(report)
        self.journal.append_report(report)
        return updated

    def _build_report(
        self,
        order: Order,
        status: OrderStatus,
        *,
        message: str,
        metadata: Mapping[str, Any] | None = None,
        backend_name: str | None = None,
    ) -> ExecutionReport:
        return ExecutionReport(
            report_id=f"rpt_{uuid4().hex[:12]}",
            order_id=order.order_id,
            symbol=order.symbol,
            status=status,
            backend_name=backend_name or self.backend.name,
            created_at=utc_now_iso(),
            source_model_name=order.source_model_name,
            source_decision_id=order.source_decision_id,
            broker_order_id=order.broker_order_id,
            message=message,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _resolve_order_type(raw_value: str) -> OrderType:
        try:
            return OrderType(str(raw_value).upper())
        except ValueError as exc:
            raise OrderValidationError(f"Unsupported order type {raw_value!r}.") from exc

    @staticmethod
    def _resolve_order_action(decision_action: str, current_quantity: int, desired_quantity: int) -> tuple[OrderAction, int]:
        if decision_action in {"BUY", "LONG"}:
            if current_quantity < 0:
                return OrderAction.COVER, min(abs(current_quantity), desired_quantity)
            return OrderAction.BUY, desired_quantity
        if decision_action in {"SELL", "EXIT_LONG", "CLOSE_LONG"}:
            if current_quantity <= 0:
                raise OrderValidationError("Cannot SELL without a current long position.")
            return OrderAction.SELL, min(abs(current_quantity), desired_quantity)
        if decision_action in {"SHORT"}:
            if current_quantity > 0:
                return OrderAction.SELL, min(abs(current_quantity), desired_quantity)
            return OrderAction.SHORT, desired_quantity
        if decision_action in {"COVER", "EXIT_SHORT", "CLOSE_SHORT"}:
            if current_quantity >= 0:
                raise OrderValidationError("Cannot COVER without a current short position.")
            return OrderAction.COVER, min(abs(current_quantity), desired_quantity)
        raise OrderValidationError(f"Unsupported decision action {decision_action!r}.")

    @staticmethod
    def _merge_backend_identity(order: Order, backend_result) -> Order:
        metadata = {**dict(order.metadata), **dict(backend_result.metadata or {})}
        merged = order.replace(
            metadata=metadata,
            backend_name=backend_result.backend_name or order.backend_name,
            broker_order_id=_coerce_optional_int(
                metadata.get("broker_order_id", backend_result.broker_order_id)
            )
            or order.broker_order_id,
            broker_perm_id=_coerce_optional_int(
                metadata.get("broker_perm_id", backend_result.broker_perm_id)
            )
            or order.broker_perm_id,
        )
        return merged


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
