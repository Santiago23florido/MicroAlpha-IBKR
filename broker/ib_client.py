from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.execution import ExecutionFilter
from ibapi.order import Order
from ibapi.wrapper import EWrapper

from broker.contracts import create_stock_contract
from broker.orders import (
    create_bracket_order,
    create_limit_order,
    create_market_order,
    create_marketable_limit_order,
)
from storage.executions import ExecutionAuditStore

ACCOUNT_SUMMARY_TAGS = ",".join(
    [
        "AccountType",
        "NetLiquidation",
        "TotalCashValue",
        "BuyingPower",
        "AvailableFunds",
        "ExcessLiquidity",
        "Cushion",
    ]
)

INFO_ERROR_CODES = {2104, 2106, 2107, 2108, 2158}
NON_FATAL_REQUEST_CODES = {10167}
TERMINAL_ORDER_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
TICK_PRICE_FIELDS = {
    1: "bid",
    2: "ask",
    4: "last",
    6: "high",
    7: "low",
    9: "close",
    14: "open",
}
TICK_SIZE_FIELDS = {
    0: "bid_size",
    3: "ask_size",
    5: "last_size",
    8: "volume",
}
TICK_STRING_FIELDS = {
    45: "last_timestamp",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class IBClientError(RuntimeError):
    pass


class IBRequestTimeout(IBClientError):
    pass


class _IBGatewayApp(EWrapper, EClient):
    def __init__(self, logger: logging.Logger) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.logger = logger
        self.audit_store: ExecutionAuditStore | None = None
        self.lock = threading.Lock()
        self.connection_ready = threading.Event()
        self.server_time_event = threading.Event()
        self.account_summary_event = threading.Event()
        self.positions_event = threading.Event()
        self.open_orders_event = threading.Event()
        self.executions_event = threading.Event()
        self.connection_errors: list[str] = []
        self.request_errors: dict[int, list[str]] = defaultdict(list)
        self.request_notices: dict[int, list[str]] = defaultdict(list)
        self.snapshot_events: dict[int, threading.Event] = {}
        self.historical_data_events: dict[int, threading.Event] = {}
        self.order_events: dict[int, threading.Event] = {}
        self.order_terminal_events: dict[int, threading.Event] = {}
        self.snapshot_data: dict[int, dict[str, Any]] = {}
        self.historical_data_rows: dict[int, list[dict[str, Any]]] = {}
        self.order_statuses: dict[int, dict[str, Any]] = {}
        self.order_execution_details: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.account_summary_rows: dict[int, list[dict[str, str]]] = {}
        self.positions_rows: list[dict[str, Any]] = []
        self.open_orders_rows: list[dict[str, Any]] = []
        self.execution_rows: list[dict[str, Any]] = []
        self.current_server_time: int | None = None
        self.next_valid_order_id: int | None = None
        self.active_account_summary_req_id: int | None = None
        self.collecting_open_orders = False

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self.logger.info("IB connection handshake complete. Next valid order id: %s", orderId)
        self.next_valid_order_id = orderId
        self.connection_ready.set()

    def currentTime(self, time_value: int) -> None:  # noqa: N802
        self.current_server_time = time_value
        self.server_time_event.set()

    def accountSummary(  # noqa: N802
        self,
        reqId: int,
        account: str,
        tag: str,
        value: str,
        currency: str,
    ) -> None:
        with self.lock:
            rows = self.account_summary_rows.setdefault(reqId, [])
            rows.append(
                {
                    "account": account,
                    "tag": tag,
                    "value": value,
                    "currency": currency,
                }
            )

    def accountSummaryEnd(self, reqId: int) -> None:  # noqa: N802
        self.account_summary_event.set()

    def position(  # noqa: N802
        self,
        account: str,
        contract: Contract,
        position: float,
        avgCost: float,
    ) -> None:
        with self.lock:
            self.positions_rows.append(
                {
                    "account": account,
                    "symbol": contract.symbol,
                    "secType": contract.secType,
                    "exchange": contract.exchange,
                    "currency": contract.currency,
                    "position": position,
                    "avgCost": avgCost,
                }
            )

    def positionEnd(self) -> None:  # noqa: N802
        self.positions_event.set()

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:  # noqa: N802
        field_name = TICK_PRICE_FIELDS.get(tickType)
        if field_name is None:
            return

        with self.lock:
            payload = self.snapshot_data.setdefault(reqId, {})
            payload[field_name] = price

    def tickSize(self, reqId: int, tickType: int, size: float) -> None:  # noqa: N802
        field_name = TICK_SIZE_FIELDS.get(tickType)
        if field_name is None:
            return

        with self.lock:
            payload = self.snapshot_data.setdefault(reqId, {})
            payload[field_name] = size

    def tickString(self, reqId: int, tickType: int, value: str) -> None:  # noqa: N802
        field_name = TICK_STRING_FIELDS.get(tickType)
        if field_name is None:
            return

        with self.lock:
            payload = self.snapshot_data.setdefault(reqId, {})
            payload[field_name] = value

    def tickSnapshotEnd(self, reqId: int) -> None:  # noqa: N802
        event = self.snapshot_events.get(reqId)
        if event is not None:
            event.set()

    def historicalData(self, reqId: int, bar: Any) -> None:  # noqa: N802
        with self.lock:
            rows = self.historical_data_rows.setdefault(reqId, [])
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(int(bar.date), tz=timezone.utc).isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "bar_count": getattr(bar, "barCount", None),
                    "average": getattr(bar, "average", None),
                }
            )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
        event = self.historical_data_events.get(reqId)
        if event is not None:
            event.set()

    def openOrder(  # noqa: N802
        self,
        orderId: int,
        contract: Contract,
        order: Order,
        orderState: Any,
    ) -> None:
        now = _utc_now_iso()

        with self.lock:
            payload = self.order_statuses.setdefault(orderId, {"order_id": orderId})
            was_acknowledged = "acknowledged_at_utc" in payload
            payload.setdefault("created_at_utc", now)
            payload.update(
                {
                    "order_id": orderId,
                    "parent_order_id": getattr(order, "parentId", 0) or None,
                    "symbol": contract.symbol,
                    "action": order.action,
                    "quantity": order.totalQuantity,
                    "order_type": order.orderType,
                    "limit_price": getattr(order, "lmtPrice", None) or None,
                    "stop_price": getattr(order, "auxPrice", None) or None,
                    "status": getattr(orderState, "status", payload.get("status", "PendingSubmit")),
                    "transmit": order.transmit,
                    "updated_at_utc": now,
                }
            )
            payload.setdefault("filled_quantity", 0.0)
            payload.setdefault("remaining_quantity", order.totalQuantity)
            payload.setdefault("average_fill_price", None)
            payload.setdefault("last_fill_price", None)
            payload.setdefault("status_history", [])
            if not was_acknowledged:
                payload["acknowledged_at_utc"] = now
            snapshot = self._copy_order_status_locked(orderId)
            if self.collecting_open_orders:
                self.open_orders_rows.append(snapshot)

        if not was_acknowledged:
            self.logger.info(
                "Order %s acknowledged by IB: %s %s %s %s",
                orderId,
                snapshot.get("action"),
                snapshot.get("quantity"),
                snapshot.get("symbol"),
                snapshot.get("order_type"),
            )
            self._append_audit("acknowledged", snapshot)

        self._set_order_events(orderId, snapshot.get("status"))

    def openOrderEnd(self) -> None:  # noqa: N802
        self.open_orders_event.set()

    def orderStatus(  # noqa: N802
        self,
        orderId: int,
        status: str,
        filled: float,
        remaining: float,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float,
    ) -> None:
        now = _utc_now_iso()

        with self.lock:
            payload = self.order_statuses.setdefault(orderId, {"order_id": orderId, "created_at_utc": now})
            previous_status = payload.get("status")
            previous_filled = float(payload.get("filled_quantity", 0.0) or 0.0)

            payload.update(
                {
                    "order_id": orderId,
                    "parent_order_id": parentId or payload.get("parent_order_id"),
                    "status": status,
                    "filled_quantity": filled,
                    "remaining_quantity": remaining,
                    "average_fill_price": avgFillPrice or payload.get("average_fill_price"),
                    "last_fill_price": lastFillPrice or payload.get("last_fill_price"),
                    "updated_at_utc": now,
                }
            )
            payload.setdefault("status_history", []).append(
                {"timestamp_utc": now, "status": status, "filled_quantity": filled}
            )
            if status in {"PreSubmitted", "Submitted", "PendingSubmit"}:
                payload.setdefault("acknowledged_at_utc", now)
            if status in {"Cancelled", "ApiCancelled"}:
                payload["cancelled_at_utc"] = now
            if status in TERMINAL_ORDER_STATUSES:
                payload["final_at_utc"] = now

            snapshot = self._copy_order_status_locked(orderId)

        self._log_order_status_transition(
            snapshot=snapshot,
            previous_status=previous_status,
            previous_filled=previous_filled,
        )
        self._set_order_events(orderId, status)

    def execDetails(self, reqId: int, contract: Contract, execution: Any) -> None:  # noqa: N802
        now = _utc_now_iso()
        details = {
            "execution_id": execution.execId,
            "order_id": execution.orderId,
            "symbol": contract.symbol,
            "side": execution.side,
            "shares": execution.shares,
            "price": execution.price,
            "exchange": execution.exchange,
            "execution_time": execution.time,
            "timestamp_utc": now,
        }

        with self.lock:
            self.order_execution_details[execution.orderId].append(details)
            payload = self.order_statuses.setdefault(
                execution.orderId,
                {
                    "order_id": execution.orderId,
                    "symbol": contract.symbol,
                    "action": execution.side,
                    "quantity": execution.shares,
                    "order_type": "",
                    "created_at_utc": now,
                },
            )
            payload["updated_at_utc"] = now
            if execution.price:
                payload["average_fill_price"] = execution.price
                payload["last_fill_price"] = execution.price
            snapshot = self._copy_order_status_locked(execution.orderId)

        self.logger.info(
            "Order %s fill recorded: %s %s %s @ %s",
            execution.orderId,
            execution.side,
            execution.shares,
            contract.symbol,
            execution.price,
        )
        self._append_audit(
            "fill",
            {
                **snapshot,
                "execution_id": execution.execId,
                "execution_time": execution.time,
            },
        )
        self._set_order_events(execution.orderId, snapshot.get("status"))

        if reqId != -1:
            with self.lock:
                self.execution_rows.append(details)

    def execDetailsEnd(self, reqId: int) -> None:  # noqa: N802
        self.executions_event.set()

    def error(  # noqa: N802
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        message = f"IB error {errorCode} (reqId={reqId}): {errorString}"

        if errorCode in INFO_ERROR_CODES:
            self.logger.info(message)
            return

        if errorCode in NON_FATAL_REQUEST_CODES:
            self.logger.warning(message)
            if reqId != -1:
                self.request_notices[reqId].append(message)
            return

        if reqId == -1:
            self.connection_errors.append(message)
        else:
            self.request_errors[reqId].append(message)

        if reqId in self.order_events or reqId in self.order_statuses:
            now = _utc_now_iso()
            with self.lock:
                payload = self.order_statuses.setdefault(reqId, {"order_id": reqId, "created_at_utc": now})
                payload.update(
                    {
                        "status": "Rejected",
                        "updated_at_utc": now,
                        "final_at_utc": now,
                        "message": message,
                    }
                )
                snapshot = self._copy_order_status_locked(reqId)
            self._append_audit("rejected", snapshot)
            self._set_order_events(reqId, "Rejected")

        if errorCode >= 1000:
            self.logger.warning(message)
        else:
            self.logger.error(message)

        if reqId == self.active_account_summary_req_id:
            self.account_summary_event.set()

        snapshot_event = self.snapshot_events.get(reqId)
        if snapshot_event is not None:
            snapshot_event.set()

        historical_event = self.historical_data_events.get(reqId)
        if historical_event is not None:
            historical_event.set()

        order_event = self.order_events.get(reqId)
        if order_event is not None:
            order_event.set()

    def connectionClosed(self) -> None:  # noqa: N802
        self.logger.warning("IB connection closed.")
        self.connection_ready.clear()

    def _copy_order_status_locked(self, order_id: int) -> dict[str, Any]:
        payload = dict(self.order_statuses.get(order_id, {}))
        payload["executions"] = list(self.order_execution_details.get(order_id, []))
        return payload

    def _append_audit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.audit_store is not None:
            self.audit_store.append_event(event_type, payload)

    def _set_order_events(self, order_id: int, status: str | None) -> None:
        update_event = self.order_events.get(order_id)
        if update_event is not None:
            update_event.set()

        if status in TERMINAL_ORDER_STATUSES or status == "Rejected":
            terminal_event = self.order_terminal_events.get(order_id)
            if terminal_event is not None:
                terminal_event.set()

    def _log_order_status_transition(
        self,
        *,
        snapshot: dict[str, Any],
        previous_status: str | None,
        previous_filled: float,
    ) -> None:
        order_id = snapshot.get("order_id")
        status = snapshot.get("status")
        filled_quantity = float(snapshot.get("filled_quantity", 0.0) or 0.0)

        if status != previous_status:
            if status in {"Submitted", "PreSubmitted", "PendingSubmit"}:
                self.logger.info("Order %s submitted to IB with status %s.", order_id, status)
                self._append_audit("submitted", snapshot)
            elif status == "Filled":
                self.logger.info(
                    "Order %s filled: %s shares at avg price %s.",
                    order_id,
                    snapshot.get("filled_quantity"),
                    snapshot.get("average_fill_price"),
                )
                self._append_audit("filled", snapshot)
            elif status in {"Cancelled", "ApiCancelled"}:
                self.logger.info("Order %s cancelled.", order_id)
                self._append_audit("cancelled", snapshot)
            elif status == "Inactive":
                self.logger.warning("Order %s became inactive.", order_id)
                self._append_audit("inactive", snapshot)
            elif status == "Rejected":
                self.logger.warning("Order %s rejected.", order_id)
            else:
                self.logger.info("Order %s status update: %s.", order_id, status)

        if filled_quantity > previous_filled and status not in {"Filled"}:
            self.logger.info(
                "Order %s partially filled: %s filled, %s remaining.",
                order_id,
                snapshot.get("filled_quantity"),
                snapshot.get("remaining_quantity"),
            )
            self._append_audit("partial_fill", snapshot)


class IBClient:
    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        logger: logging.Logger,
        request_timeout: float = 15.0,
        order_follow_up_seconds: float = 5.0,
        account_summary_group: str = "All",
        exchange: str = "SMART",
        currency: str = "USD",
        audit_store: ExecutionAuditStore | None = None,
        app_factory: Callable[[logging.Logger], _IBGatewayApp] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.logger = logger
        self.request_timeout = request_timeout
        self.order_follow_up_seconds = order_follow_up_seconds
        self.account_summary_group = account_summary_group
        self.exchange = exchange
        self.currency = currency
        self.audit_store = audit_store
        self._app_factory = app_factory or _IBGatewayApp
        self._app = self._app_factory(logger)
        self._app.audit_store = audit_store
        self._thread: threading.Thread | None = None
        self._request_id = 0
        self._request_lock = threading.Lock()

    def _next_request_id(self) -> int:
        with self._request_lock:
            self._request_id += 1
            return self._request_id

    def _wait_for_event(
        self,
        event: threading.Event,
        description: str,
        timeout: float | None = None,
        req_id: int | None = None,
    ) -> None:
        effective_timeout = timeout or self.request_timeout
        deadline = time.monotonic() + effective_timeout

        while time.monotonic() < deadline:
            if event.wait(timeout=0.1):
                return

            if req_id is not None:
                request_errors = self._app.request_errors.get(req_id, [])
                if request_errors:
                    raise IBClientError(
                        f"IB request failed while waiting for {description}. "
                        + " | ".join(request_errors)
                    )
            elif self._app.connection_errors:
                raise IBClientError(
                    f"IB connection failed while waiting for {description}. "
                    + " | ".join(self._app.connection_errors)
                )

        details: list[str] = []
        if req_id is not None:
            details.extend(self._app.request_errors.get(req_id, []))
        else:
            details.extend(self._app.connection_errors)

        extra = f" Details: {' | '.join(details)}" if details else ""
        raise IBRequestTimeout(
            f"Timed out waiting for {description} after {effective_timeout:.1f}s. "
            f"Confirm IB Gateway paper is running at {self.host}:{self.port}, "
            "the API port is enabled, and clientId is not already in use."
            f"{extra}"
        )

    def connect(self, timeout: float | None = None) -> bool:
        if self.is_connected():
            self.logger.info("IB client is already connected.")
            return True

        self.logger.info(
            "Connecting to IB Gateway paper at %s:%s with clientId=%s",
            self.host,
            self.port,
            self.client_id,
        )
        self._app.connection_ready.clear()
        self._app.connection_errors.clear()
        self._app.connect(self.host, self.port, self.client_id)

        self._thread = threading.Thread(
            target=self._app.run,
            name="ibapi-network-loop",
            daemon=True,
        )
        self._thread.start()
        self._wait_for_event(self._app.connection_ready, "the IB connection handshake", timeout)

        if not self._app.isConnected():
            raise IBClientError("IB handshake completed, but the socket is not connected.")

        self.logger.info("IB connection is healthy.")
        return True

    def disconnect(self) -> None:
        try:
            if self._app.isConnected():
                self.logger.info("Disconnecting from IB Gateway.")
                self._app.disconnect()
        finally:
            self._app.connection_ready.clear()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._thread = None

    def is_connected(self) -> bool:
        return bool(self._app.isConnected() and self._app.connection_ready.is_set())

    def get_server_time(self) -> dict[str, str | int]:
        self._ensure_connected()
        self._app.server_time_event.clear()
        self._app.current_server_time = None
        self.logger.info("Requesting current server time.")
        self._app.reqCurrentTime()
        self._wait_for_event(self._app.server_time_event, "current server time")

        assert self._app.current_server_time is not None
        server_time = datetime.fromtimestamp(
            self._app.current_server_time,
            tz=timezone.utc,
        )
        return {
            "epoch": self._app.current_server_time,
            "iso_utc": server_time.isoformat(),
        }

    def get_account_summary(self) -> list[dict[str, str]]:
        self._ensure_connected()
        req_id = self._next_request_id()
        self._app.active_account_summary_req_id = req_id
        self._app.account_summary_rows[req_id] = []
        self._app.request_errors.pop(req_id, None)
        self._app.account_summary_event.clear()
        self.logger.info("Requesting account summary.")
        self._app.reqAccountSummary(req_id, self.account_summary_group, ACCOUNT_SUMMARY_TAGS)
        self._wait_for_event(self._app.account_summary_event, "account summary", req_id=req_id)
        self._app.cancelAccountSummary(req_id)

        rows = self._app.account_summary_rows.get(req_id, [])
        errors = self._app.request_errors.get(req_id, [])
        if errors and not rows:
            raise IBClientError("Account summary request failed. " + " | ".join(errors))

        return rows

    def get_positions(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        self._app.positions_rows = []
        self._app.positions_event.clear()
        self.logger.info("Requesting positions.")
        self._app.reqPositions()
        self._wait_for_event(self._app.positions_event, "positions")
        self._app.cancelPositions()
        return list(self._app.positions_rows)

    def get_market_snapshot(
        self,
        symbol: str,
        *,
        exchange: str | None = None,
        currency: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_connected()
        req_id = self._next_request_id()
        snapshot_event = threading.Event()
        self._app.snapshot_events[req_id] = snapshot_event
        self._app.snapshot_data[req_id] = {"symbol": symbol.upper()}
        self._app.request_errors.pop(req_id, None)
        self._app.request_notices.pop(req_id, None)
        self.logger.info("Requesting market snapshot for %s.", symbol.upper())
        self._app.reqMarketDataType(3)
        contract = self._build_contract(symbol, exchange=exchange, currency=currency)
        self._app.reqMktData(req_id, contract, "", True, False, [])
        self._wait_for_event(snapshot_event, f"market snapshot for {symbol.upper()}", req_id=req_id)
        self._app.cancelMktData(req_id)

        snapshot = self._app.snapshot_data.get(req_id, {})
        errors = self._app.request_errors.get(req_id, [])
        notices = self._app.request_notices.get(req_id, [])
        if errors and len(snapshot) <= 1:
            raise IBClientError(
                f"Market snapshot request for {symbol.upper()} failed. " + " | ".join(errors)
            )

        if len(snapshot) <= 1:
            details = (
                " Confirm the symbol is valid and that your IB paper account has "
                "live or delayed market data permissions for this instrument."
            )
            if notices:
                details += " Broker notices: " + " | ".join(notices)
            raise IBClientError(
                f"Market snapshot for {symbol.upper()} returned no price fields." + details
            )

        snapshot["snapshot_utc"] = _utc_now_iso()
        return snapshot

    def get_historical_bars(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        currency: str | None = None,
        duration: str = "1 D",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> list[dict[str, Any]]:
        self._ensure_connected()
        req_id = self._next_request_id()
        historical_event = threading.Event()
        self._app.historical_data_events[req_id] = historical_event
        self._app.historical_data_rows[req_id] = []
        self._app.request_errors.pop(req_id, None)
        contract = self._build_contract(symbol, exchange=exchange, currency=currency)
        self.logger.info("Requesting historical bars for %s.", symbol.upper())
        self._app.reqHistoricalData(
            req_id,
            contract,
            "",
            duration,
            bar_size,
            what_to_show,
            1 if use_rth else 0,
            2,
            False,
            [],
        )
        self._wait_for_event(historical_event, f"historical bars for {symbol.upper()}", req_id=req_id)
        rows = list(self._app.historical_data_rows.get(req_id, []))
        errors = self._app.request_errors.get(req_id, [])
        if errors and not rows:
            raise IBClientError(
                f"Historical data request for {symbol.upper()} failed. " + " | ".join(errors)
            )
        return rows

    def submit_market_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: int,
        exchange: str | None = None,
        currency: str | None = None,
    ) -> dict[str, Any]:
        contract = self._build_contract(symbol, exchange=exchange, currency=currency)
        order_id = self._reserve_order_ids(1)
        order = create_market_order(action=action, quantity=quantity)
        order.orderId = order_id
        return self._submit_orders(contract, [order], f"market order {action.upper()} {quantity} {symbol.upper()}")

    def submit_limit_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float,
        exchange: str | None = None,
        currency: str | None = None,
    ) -> dict[str, Any]:
        contract = self._build_contract(symbol, exchange=exchange, currency=currency)
        order_id = self._reserve_order_ids(1)
        order = create_limit_order(action=action, quantity=quantity, limit_price=limit_price)
        order.orderId = order_id
        return self._submit_orders(
            contract,
            [order],
            f"limit order {action.upper()} {quantity} {symbol.upper()} @ {limit_price}",
        )

    def submit_marketable_limit_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: int,
        bid: float | None,
        ask: float | None,
        last: float | None = None,
        buffer_bps: float = 2.0,
        exchange: str | None = None,
        currency: str | None = None,
    ) -> dict[str, Any]:
        contract = self._build_contract(symbol, exchange=exchange, currency=currency)
        order_id = self._reserve_order_ids(1)
        order = create_marketable_limit_order(
            action=action,
            quantity=quantity,
            bid=bid,
            ask=ask,
            last=last,
            buffer_bps=buffer_bps,
        )
        order.orderId = order_id
        return self._submit_orders(
            contract,
            [order],
            f"marketable limit order {action.upper()} {quantity} {symbol.upper()}",
        )

    def submit_bracket_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: int,
        entry_limit_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        exchange: str | None = None,
        currency: str | None = None,
    ) -> dict[str, Any]:
        contract = self._build_contract(symbol, exchange=exchange, currency=currency)
        parent_order_id = self._reserve_order_ids(3)
        orders = create_bracket_order(
            parent_order_id=parent_order_id,
            action=action,
            quantity=quantity,
            entry_limit_price=entry_limit_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
        )
        return self._submit_orders(
            contract,
            orders,
            (
                f"bracket order {action.upper()} {quantity} {symbol.upper()} "
                f"entry={entry_limit_price} take_profit={take_profit_price} stop={stop_loss_price}"
            ),
        )

    def cancel_order(self, order_id: int) -> dict[str, Any]:
        self._ensure_connected()
        self._prime_order_tracking([order_id])
        self.logger.warning("Cancelling order %s.", order_id)
        self._app.cancelOrder(order_id, "")
        self._wait_for_event(
            self._app.order_events[order_id],
            f"cancellation acknowledgement for order {order_id}",
            req_id=order_id,
        )
        self._wait_for_order_completion([order_id], timeout=self.order_follow_up_seconds)
        snapshot = self.get_order_status(order_id)
        if snapshot.get("status") not in {"Cancelled", "ApiCancelled", "Inactive", "Rejected", "Filled"}:
            raise IBClientError(
                f"Order {order_id} cancellation did not reach a terminal state. Current status: {snapshot.get('status')}."
            )
        return snapshot

    def get_open_orders(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        self._app.open_orders_rows = []
        self._app.open_orders_event.clear()
        self._app.collecting_open_orders = True
        self.logger.info("Requesting open orders.")
        self._app.reqOpenOrders()
        self._wait_for_event(self._app.open_orders_event, "open orders")
        self._app.collecting_open_orders = False

        rows = list(self._app.open_orders_rows)
        if rows:
            return rows

        return [
            self.get_order_status(order_id)
            for order_id, status in self._app.order_statuses.items()
            if status.get("status") not in TERMINAL_ORDER_STATUSES
        ]

    def request_recent_executions(self) -> list[dict[str, Any]]:
        self._ensure_connected()
        req_id = self._next_request_id()
        self._app.execution_rows = []
        self._app.executions_event.clear()
        self.logger.info("Requesting recent execution details.")
        self._app.reqExecutions(req_id, ExecutionFilter())
        self._wait_for_event(self._app.executions_event, "recent executions", req_id=req_id)
        return list(self._app.execution_rows)

    def close_position(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        currency: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_connected()
        positions = self.get_positions()
        matching_position = next(
            (row for row in positions if row["symbol"].upper() == symbol.upper()),
            None,
        )
        if matching_position is None or matching_position["position"] == 0:
            raise IBClientError(f"No open position found for {symbol.upper()}.")

        position_quantity = matching_position["position"]
        if not float(abs(position_quantity)).is_integer():
            raise IBClientError(
                f"Cannot close {symbol.upper()} because the position size is fractional: {position_quantity}."
            )

        quantity = int(abs(position_quantity))
        action = "SELL" if position_quantity > 0 else "BUY"
        self.logger.warning(
            "Submitting manual close for %s: %s %s shares.",
            symbol.upper(),
            action,
            quantity,
        )
        result = self.submit_market_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            exchange=exchange,
            currency=currency,
        )
        result["close_position"] = True
        result["position_before_close"] = matching_position
        if self.audit_store is not None:
            self.audit_store.append_event("manual_close", result)
        return result

    def get_order_status(self, order_id: int) -> dict[str, Any]:
        with self._app.lock:
            if order_id not in self._app.order_statuses:
                raise IBClientError(f"No tracked order data found for order id {order_id}.")
            return self._app._copy_order_status_locked(order_id)

    def _ensure_connected(self) -> None:
        if not self.is_connected():
            raise IBClientError(
                "IB client is not connected. Run the connection check first and confirm "
                "IB Gateway paper is reachable at 127.0.0.1:4002."
            )

    def _build_contract(
        self,
        symbol: str,
        *,
        exchange: str | None = None,
        currency: str | None = None,
    ) -> Contract:
        return create_stock_contract(
            symbol=symbol,
            exchange=exchange or self.exchange,
            currency=currency or self.currency,
        )

    def _reserve_order_ids(self, count: int) -> int:
        self._ensure_connected()
        if self._app.next_valid_order_id is None:
            raise IBClientError("No valid IB order id is available yet.")

        start_order_id = self._app.next_valid_order_id
        self._app.next_valid_order_id += count
        return start_order_id

    def _prime_order_tracking(self, order_ids: Iterable[int]) -> None:
        for order_id in order_ids:
            self._app.request_errors.pop(order_id, None)
            self._app.request_notices.pop(order_id, None)
            self._app.order_events[order_id] = threading.Event()
            self._app.order_terminal_events[order_id] = threading.Event()

    def _seed_order_state(self, order: Order, contract: Contract) -> None:
        now = _utc_now_iso()
        with self._app.lock:
            payload = self._app.order_statuses.setdefault(order.orderId, {"order_id": order.orderId})
            payload.update(
                {
                    "order_id": order.orderId,
                    "parent_order_id": getattr(order, "parentId", 0) or None,
                    "symbol": contract.symbol,
                    "action": order.action,
                    "quantity": order.totalQuantity,
                    "order_type": order.orderType,
                    "limit_price": getattr(order, "lmtPrice", None) or None,
                    "stop_price": getattr(order, "auxPrice", None) or None,
                    "status": payload.get("status", "PendingSubmit"),
                    "transmit": order.transmit,
                    "created_at_utc": payload.get("created_at_utc", now),
                    "updated_at_utc": now,
                }
            )
            payload.setdefault("filled_quantity", 0.0)
            payload.setdefault("remaining_quantity", order.totalQuantity)
            payload.setdefault("average_fill_price", None)
            payload.setdefault("last_fill_price", None)
            payload.setdefault("status_history", [])
            snapshot = self._app._copy_order_status_locked(order.orderId)

        if self.audit_store is not None:
            self.audit_store.append_event("submission_requested", snapshot)

    def _submit_orders(
        self,
        contract: Contract,
        orders: list[Order],
        description: str,
    ) -> dict[str, Any]:
        self._ensure_connected()

        order_ids = [order.orderId for order in orders]
        self._prime_order_tracking(order_ids)
        for order in orders:
            self._seed_order_state(order, contract)

        self.logger.warning("Submitting explicit paper %s.", description)
        for order in orders:
            self._app.placeOrder(order.orderId, contract, order)

        for order_id in order_ids:
            self._wait_for_event(
                self._app.order_events[order_id],
                f"order acknowledgement for order {order_id}",
                req_id=order_id,
            )

        self._wait_for_order_completion(order_ids, timeout=self.order_follow_up_seconds)
        results = [self.get_order_status(order_id) for order_id in order_ids]

        if len(results) == 1:
            return results[0]

        return {
            "parent_order_id": results[0]["order_id"],
            "status": results[0].get("status"),
            "orders": results,
        }

    def _wait_for_order_completion(self, order_ids: Iterable[int], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshots = [self._app.order_statuses.get(order_id, {}) for order_id in order_ids]
            if snapshots and all(snapshot.get("status") in TERMINAL_ORDER_STATUSES.union({"Rejected"}) for snapshot in snapshots):
                return
            time.sleep(0.25)
