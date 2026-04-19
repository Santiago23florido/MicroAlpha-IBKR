from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from broker.ib_client import IBClient, IBClientError
from config import Settings
from config.phase7 import Phase7Config
from execution.backend import BackendSubmissionResult, BaseExecutionBackend
from execution.ibkr_state_mapper import map_ibkr_status
from execution.models import FillEvent, Order, OrderAction, OrderStatus, OrderType, utc_now_iso
from monitoring.logging import setup_logger


class IBKRPaperExecutionBackend(BaseExecutionBackend):
    name = "ibkr_paper"

    def __init__(
        self,
        config: Phase7Config,
        *,
        settings: Settings,
        logger: logging.Logger | None = None,
        client: IBClient | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.logger = logger or setup_logger(
            settings.log_level,
            settings.log_file,
            logger_name="microalpha.execution.ibkr_paper",
        )
        self.client = client or IBClient(
            host=config.ibkr_paper.host,
            port=config.ibkr_paper.port,
            client_id=config.ibkr_paper.client_id,
            logger=self.logger,
            request_timeout=config.ibkr_paper.request_timeout_seconds,
            order_follow_up_seconds=config.ibkr_paper.order_follow_up_seconds,
            account_summary_group=settings.account_summary_group,
            exchange=config.ibkr_paper.exchange,
            currency=config.ibkr_paper.currency,
        )
        self._last_error: str | None = None
        self._last_healthcheck: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": True,
            "paper_mode": self.config.execution.paper_mode,
            "broker_mode": self.config.ibkr_paper.broker_mode,
            "connected": self.client.is_connected(),
            "host": self.config.ibkr_paper.host,
            "port": self.config.ibkr_paper.port,
            "client_id": self.config.ibkr_paper.client_id,
            "exchange": self.config.ibkr_paper.exchange,
            "currency": self.config.ibkr_paper.currency,
            "reconnect_attempts": self.config.ibkr_paper.reconnect_attempts,
            "reconnect_delay_seconds": self.config.ibkr_paper.reconnect_delay_seconds,
            "healthcheck_timeout_seconds": self.config.ibkr_paper.healthcheck_timeout_seconds,
            "safe_to_trade": self.config.ibkr_paper.safe_to_trade,
            "allow_session_execution": self.config.ibkr_paper.allow_session_execution,
            "supported_symbols": list(self.config.ibkr_paper.supported_symbols),
            "last_error": self._last_error,
            "last_healthcheck": self._last_healthcheck,
        }

    def connect(self) -> None:
        self._ensure_paper_only()
        if self.client.is_connected():
            return

        attempts = max(int(self.config.ibkr_paper.reconnect_attempts), 1)
        delay_seconds = max(float(self.config.ibkr_paper.reconnect_delay_seconds), 0.0)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self.client.connect(timeout=self.config.ibkr_paper.healthcheck_timeout_seconds)
                self._last_error = None
                return
            except Exception as exc:  # pragma: no cover - exercised through stub tests
                last_error = exc
                self._last_error = str(exc)
                if attempt >= attempts:
                    break
                self.logger.warning(
                    "IBKR paper connection attempt %s/%s failed: %s. Retrying in %.2fs.",
                    attempt,
                    attempts,
                    exc,
                    delay_seconds,
                )
                time.sleep(delay_seconds)

        assert last_error is not None
        raise IBClientError(f"Unable to connect to IBKR Paper after {attempts} attempt(s): {last_error}")

    def disconnect(self) -> None:
        self.client.disconnect()

    def healthcheck(self) -> dict[str, Any]:
        self._ensure_paper_only()
        self.connect()
        started = time.monotonic()
        server_time = self.client.get_server_time()
        account_summary = self.client.get_account_summary()
        open_orders = self.client.get_open_orders()
        positions = self.client.get_positions()
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
        payload = {
            "status": "ok",
            "backend_name": self.name,
            "connected": self.client.is_connected(),
            "broker_mode": self.config.ibkr_paper.broker_mode,
            "paper_mode": self.config.execution.paper_mode,
            "host": self.config.ibkr_paper.host,
            "port": self.config.ibkr_paper.port,
            "client_id": self.config.ibkr_paper.client_id,
            "safe_to_trade": self.config.ibkr_paper.safe_to_trade,
            "allow_session_execution": self.config.ibkr_paper.allow_session_execution,
            "server_time": server_time,
            "account_summary_row_count": len(account_summary),
            "account_summary_tags": sorted({row.get("tag") for row in account_summary if row.get("tag")}),
            "open_order_count": len(open_orders),
            "position_count": len(positions),
            "latency_ms": elapsed_ms,
        }
        self._last_healthcheck = payload
        return payload

    def submit_order(self, order: Order, market_data: Mapping[str, Any] | None = None) -> BackendSubmissionResult:
        self._ensure_submission_allowed(order)
        self.connect()

        submitted_at_utc = utc_now_iso()
        decision_generated_at = order.metadata.get("decision_generated_at_utc")
        broker_payload = self._submit_to_ibkr(order, market_data)
        broker_status = str(broker_payload.get("status") or "")
        broker_message = broker_payload.get("message")
        internal_status = map_ibkr_status(
            broker_status,
            filled_quantity=broker_payload.get("filled_quantity"),
            remaining_quantity=broker_payload.get("remaining_quantity"),
            message=broker_message,
        )

        fills = self._build_fills(order, broker_payload, market_data)
        fill_ratio = 0.0 if order.quantity <= 0 else min(sum(fill.quantity for fill in fills) / float(order.quantity), 1.0)
        latency_payload = self._build_latency_payload(
            decision_generated_at_utc=decision_generated_at,
            submitted_at_utc=submitted_at_utc,
            broker_payload=broker_payload,
            fills=fills,
        )
        discrepancy_payload = self._build_decision_vs_execution_payload(
            order=order,
            fills=fills,
            market_data=market_data,
            latency_payload=latency_payload,
            broker_payload=broker_payload,
        )
        metadata = {
            "broker_order_id": broker_payload.get("order_id"),
            "broker_perm_id": broker_payload.get("broker_perm_id"),
            "broker_status": broker_status,
            "broker_snapshot": broker_payload,
            "status_history": list(broker_payload.get("status_history", []) or []),
            "latency_ms": latency_payload,
            "decision_vs_execution": discrepancy_payload,
            "fill_ratio": fill_ratio,
        }
        terminal_status = internal_status if internal_status in {
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        } else None

        if internal_status == OrderStatus.FILLED and not fills:
            terminal_status = OrderStatus.FILLED
        elif internal_status == OrderStatus.FILLED:
            terminal_status = None

        accepted = internal_status not in {OrderStatus.REJECTED, OrderStatus.FAILED}
        if internal_status == OrderStatus.CANCELLED and not fills:
            accepted = True

        return BackendSubmissionResult(
            accepted=accepted,
            backend_name=self.name,
            fills=fills,
            rejection_reason=None if accepted else (broker_message or "ibkr_paper_rejected_order"),
            terminal_status=terminal_status,
            broker_order_id=_coerce_optional_int(broker_payload.get("order_id")),
            broker_perm_id=_coerce_optional_int(broker_payload.get("broker_perm_id")),
            metadata=metadata,
        )

    def cancel_order(self, order: Order) -> dict[str, Any]:
        self.connect()
        broker_order_id = self._resolve_broker_order_id(order)
        snapshot = self.client.cancel_order(broker_order_id)
        return {
            "status": "ok",
            "backend_name": self.name,
            "broker_order_id": broker_order_id,
            "broker_status": snapshot.get("status"),
            "internal_status": map_ibkr_status(snapshot.get("status"), message=snapshot.get("message")).value,
            "payload": snapshot,
        }

    def get_order_status(self, order: Order) -> dict[str, Any]:
        self.connect()
        broker_order_id = self._resolve_broker_order_id(order)
        snapshot = self.client.get_order_status(broker_order_id)
        return {
            "status": "ok",
            "backend_name": self.name,
            "broker_order_id": broker_order_id,
            "broker_status": snapshot.get("status"),
            "internal_status": map_ibkr_status(snapshot.get("status"), message=snapshot.get("message")).value,
            "payload": snapshot,
        }

    def get_open_orders(self) -> list[dict[str, Any]]:
        self.connect()
        return self.client.get_open_orders()

    def get_positions(self) -> list[dict[str, Any]]:
        self.connect()
        return self.client.get_positions()

    def get_recent_executions(self) -> list[dict[str, Any]]:
        self.connect()
        return self.client.request_recent_executions()

    def _submit_to_ibkr(self, order: Order, market_data: Mapping[str, Any] | None) -> dict[str, Any]:
        action = _to_ibkr_action(order.action)
        if order.order_type == OrderType.MARKET:
            return self.client.submit_market_order(
                symbol=order.symbol,
                action=action,
                quantity=order.quantity,
                exchange=self.config.ibkr_paper.exchange,
                currency=self.config.ibkr_paper.currency,
            )
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError(f"Order {order.order_id} requires limit_price for LIMIT order type.")
            return self.client.submit_limit_order(
                symbol=order.symbol,
                action=action,
                quantity=order.quantity,
                limit_price=float(order.limit_price),
                exchange=self.config.ibkr_paper.exchange,
                currency=self.config.ibkr_paper.currency,
            )
        if order.order_type == OrderType.STOP:
            if order.stop_price is None:
                raise ValueError(f"Order {order.order_id} requires stop_price for STOP order type.")
            return self.client.submit_stop_order(
                symbol=order.symbol,
                action=action,
                quantity=order.quantity,
                stop_price=float(order.stop_price),
                exchange=self.config.ibkr_paper.exchange,
                currency=self.config.ibkr_paper.currency,
            )
        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                raise ValueError(f"Order {order.order_id} requires stop_price and limit_price for STOP_LIMIT order type.")
            return self.client.submit_stop_limit_order(
                symbol=order.symbol,
                action=action,
                quantity=order.quantity,
                stop_price=float(order.stop_price),
                limit_price=float(order.limit_price),
                exchange=self.config.ibkr_paper.exchange,
                currency=self.config.ibkr_paper.currency,
            )
        raise ValueError(f"Unsupported order type {order.order_type.value!r} for IBKR paper backend.")

    def _build_fills(
        self,
        order: Order,
        broker_payload: Mapping[str, Any],
        market_data: Mapping[str, Any] | None,
    ) -> list[FillEvent]:
        fills: list[FillEvent] = []
        executions = list(broker_payload.get("executions", []) or [])
        broker_order_id = _coerce_optional_int(broker_payload.get("order_id"))
        broker_perm_id = _coerce_optional_int(broker_payload.get("broker_perm_id"))
        fixed_commission_consumed = False
        for index, execution in enumerate(executions):
            quantity = int(round(float(execution.get("shares", 0.0) or 0.0)))
            if quantity <= 0:
                continue
            fill_price = float(execution.get("price", 0.0) or 0.0)
            if fill_price <= 0:
                continue
            commission_report = dict(execution.get("commission_report") or {})
            estimated_commission = self._estimate_commission(
                quantity=quantity,
                commission_report=commission_report,
                fixed_commission_consumed=fixed_commission_consumed,
            )
            fixed_commission_consumed = True
            filled_at = (
                execution.get("timestamp_utc")
                or execution.get("execution_time")
                or broker_payload.get("final_at_utc")
                or broker_payload.get("updated_at_utc")
                or utc_now_iso()
            )
            fills.append(
                FillEvent(
                    fill_id=f"fill_{uuid4().hex[:12]}",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    action=order.action,
                    quantity=quantity,
                    fill_price=fill_price,
                    commission=estimated_commission,
                    filled_at=str(filled_at),
                    backend_name=self.name,
                    source_model_name=order.source_model_name,
                    source_decision_id=order.source_decision_id,
                    execution_id=execution.get("execution_id"),
                    broker_order_id=broker_order_id,
                    broker_perm_id=broker_perm_id,
                    slippage_bps=_slippage_bps(order.action, fill_price, market_data),
                    metadata={
                        "fill_index": index,
                        "exchange": execution.get("exchange"),
                        "execution_time": execution.get("execution_time"),
                        "reference_price": _reference_price(order.action, market_data),
                        "commission_report": commission_report,
                    },
                )
            )
        return fills

    def _build_latency_payload(
        self,
        *,
        decision_generated_at_utc: str | None,
        submitted_at_utc: str,
        broker_payload: Mapping[str, Any],
        fills: list[FillEvent],
    ) -> dict[str, Any]:
        acknowledged_at = broker_payload.get("acknowledged_at_utc") or broker_payload.get("updated_at_utc")
        first_fill_at = fills[0].filled_at if fills else None
        final_fill_at = fills[-1].filled_at if fills else broker_payload.get("final_at_utc")
        payload = {
            "decision_generated_at_utc": decision_generated_at_utc,
            "submitted_at_utc": submitted_at_utc,
            "acknowledged_at_utc": acknowledged_at,
            "first_fill_at_utc": first_fill_at,
            "final_fill_at_utc": final_fill_at,
            "decision_to_submit_ms": _latency_ms(decision_generated_at_utc, submitted_at_utc),
            "submit_to_ack_ms": _latency_ms(submitted_at_utc, acknowledged_at),
            "submit_to_first_fill_ms": _latency_ms(submitted_at_utc, first_fill_at),
            "submit_to_final_fill_ms": _latency_ms(submitted_at_utc, final_fill_at),
        }
        breaches: list[str] = []
        if payload["decision_to_submit_ms"] is not None and payload["decision_to_submit_ms"] > self.config.ibkr_paper.max_decision_to_submit_ms:
            breaches.append("decision_to_submit_latency_exceeded")
        if payload["submit_to_ack_ms"] is not None and payload["submit_to_ack_ms"] > self.config.ibkr_paper.max_submit_to_ack_ms:
            breaches.append("submit_to_ack_latency_exceeded")
        if payload["submit_to_final_fill_ms"] is not None and payload["submit_to_final_fill_ms"] > self.config.ibkr_paper.max_submit_to_fill_ms:
            breaches.append("submit_to_fill_latency_exceeded")
        payload["threshold_breaches"] = breaches
        return payload

    def _build_decision_vs_execution_payload(
        self,
        *,
        order: Order,
        fills: list[FillEvent],
        market_data: Mapping[str, Any] | None,
        latency_payload: Mapping[str, Any],
        broker_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        decision_snapshot = dict(order.metadata.get("decision_snapshot", {}) or {})
        expected_size = _coerce_optional_int(decision_snapshot.get("size_suggestion")) or order.quantity
        filled_size = sum(fill.quantity for fill in fills)
        discrepancy_flags: list[str] = []
        if str(decision_snapshot.get("action")) not in {order.action.value, order.metadata.get("decision_action")}:
            discrepancy_flags.append("decision_action_differs_from_order_action")
        if expected_size != order.quantity:
            discrepancy_flags.append("expected_size_differs_from_submitted_size")
        if filled_size != order.quantity:
            discrepancy_flags.append("submitted_size_differs_from_filled_size")
        if broker_payload.get("status") in {"Rejected", "Inactive"}:
            discrepancy_flags.append("broker_rejected_or_inactive")
        threshold_breaches = list(latency_payload.get("threshold_breaches", []) or [])
        discrepancy_flags.extend(threshold_breaches)
        weighted_slippage = _weighted_average([fill.slippage_bps for fill in fills], [fill.quantity for fill in fills])
        return {
            "expected_action": decision_snapshot.get("action"),
            "actual_order_action": order.action.value,
            "expected_size": expected_size,
            "submitted_size": order.quantity,
            "filled_size": filled_size,
            "expected_cost_bps": decision_snapshot.get("expected_cost_bps"),
            "average_slippage_bps": weighted_slippage,
            "reference_price": _reference_price(order.action, market_data),
            "broker_status": broker_payload.get("status"),
            "discrepancy_flags": discrepancy_flags,
        }

    def _estimate_commission(
        self,
        *,
        quantity: int,
        commission_report: Mapping[str, Any],
        fixed_commission_consumed: bool,
    ) -> float:
        explicit = commission_report.get("commission")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass
        commission = float(quantity) * float(self.config.execution.commission_per_share)
        if not fixed_commission_consumed:
            commission += float(self.config.execution.commission_per_trade)
        return float(commission)

    def _resolve_broker_order_id(self, order: Order) -> int:
        broker_order_id = order.broker_order_id or _coerce_optional_int(order.metadata.get("broker_order_id"))
        if broker_order_id is None:
            raise ValueError(f"Order {order.order_id} does not have a mapped broker_order_id.")
        return int(broker_order_id)

    def _ensure_submission_allowed(self, order: Order) -> None:
        self._ensure_paper_only()
        if not self.config.ibkr_paper.safe_to_trade:
            raise ValueError("safe_to_trade is false. Refusing to send orders to IBKR Paper.")
        if not self.config.ibkr_paper.allow_session_execution:
            raise ValueError("allow_session_execution is false. Refusing to send orders to IBKR Paper.")
        if self.config.ibkr_paper.supported_symbols and order.symbol not in set(self.config.ibkr_paper.supported_symbols):
            raise ValueError(f"Symbol {order.symbol} is not enabled in SUPPORTED_SYMBOLS.")

    def _ensure_paper_only(self) -> None:
        broker_mode = self.config.ibkr_paper.broker_mode.strip().lower()
        if broker_mode != "paper":
            raise ValueError(
                f"IBKR paper backend refuses to run when broker_mode={self.config.ibkr_paper.broker_mode!r}. "
                "Only paper trading is allowed."
            )
        if not self.config.execution.paper_mode:
            raise ValueError("PAPER_MODE is false. Refusing to connect the IBKR paper backend.")


def _to_ibkr_action(action: OrderAction) -> str:
    if action in {OrderAction.BUY, OrderAction.COVER}:
        return "BUY"
    return "SELL"


def _reference_price(action: OrderAction, market_data: Mapping[str, Any] | None) -> float | None:
    if not market_data:
        return None
    if action in {OrderAction.BUY, OrderAction.COVER}:
        candidates = ("ask", "last", "mid_price", "price_proxy", "close")
    else:
        candidates = ("bid", "last", "mid_price", "price_proxy", "close")
    for key in candidates:
        value = _coerce_optional_float(market_data.get(key))
        if value is not None and value > 0:
            return value
    return None


def _slippage_bps(action: OrderAction, fill_price: float, market_data: Mapping[str, Any] | None) -> float:
    reference = _reference_price(action, market_data)
    if reference is None or reference <= 0:
        return 0.0
    if action in {OrderAction.BUY, OrderAction.COVER}:
        delta = (fill_price - reference) / reference
    else:
        delta = (reference - fill_price) / reference
    return float(delta * 10000.0)


def _latency_ms(start_value: str | None, end_value: str | None) -> float | None:
    start = _parse_timestamp(start_value)
    end = _parse_timestamp(end_value)
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() * 1000.0, 3)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass
    for pattern in ("%Y%m%d  %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(candidate, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


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


def _weighted_average(values: list[float], weights: list[int]) -> float | None:
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(int(weight) for weight in weights)
    if total_weight <= 0:
        return None
    weighted_sum = sum(float(value) * int(weight) for value, weight in zip(values, weights))
    return float(weighted_sum / total_weight)
