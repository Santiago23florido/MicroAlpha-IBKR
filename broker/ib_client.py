from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

from broker.contracts import create_stock_contract
from broker.orders import create_market_order

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


class IBClientError(RuntimeError):
    pass


class IBRequestTimeout(IBClientError):
    pass


class _IBGatewayApp(EWrapper, EClient):
    def __init__(self, logger: logging.Logger) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.logger = logger
        self.lock = threading.Lock()
        self.connection_ready = threading.Event()
        self.server_time_event = threading.Event()
        self.account_summary_event = threading.Event()
        self.positions_event = threading.Event()
        self.connection_errors: list[str] = []
        self.request_errors: dict[int, list[str]] = defaultdict(list)
        self.request_notices: dict[int, list[str]] = defaultdict(list)
        self.snapshot_events: dict[int, threading.Event] = {}
        self.order_events: dict[int, threading.Event] = {}
        self.snapshot_data: dict[int, dict[str, Any]] = {}
        self.order_statuses: dict[int, dict[str, Any]] = {}
        self.account_summary_rows: dict[int, list[dict[str, str]]] = {}
        self.positions_rows: list[dict[str, Any]] = []
        self.current_server_time: int | None = None
        self.next_valid_order_id: int | None = None
        self.active_account_summary_req_id: int | None = None

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

    def openOrder(  # noqa: N802
        self,
        orderId: int,
        contract: Contract,
        order: Order,
        orderState: Any,
    ) -> None:
        with self.lock:
            self.order_statuses[orderId] = {
                "order_id": orderId,
                "symbol": contract.symbol,
                "action": order.action,
                "quantity": order.totalQuantity,
                "status": getattr(orderState, "status", "PendingSubmit"),
            }

        event = self.order_events.get(orderId)
        if event is not None:
            event.set()

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
        with self.lock:
            payload = self.order_statuses.setdefault(orderId, {"order_id": orderId})
            payload.update(
                {
                    "status": status,
                    "filled": filled,
                    "remaining": remaining,
                    "avg_fill_price": avgFillPrice,
                    "last_fill_price": lastFillPrice,
                }
            )

        event = self.order_events.get(orderId)
        if event is not None:
            event.set()

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

        if errorCode >= 1000:
            self.logger.warning(message)
        else:
            self.logger.error(message)

        if reqId == self.active_account_summary_req_id:
            self.account_summary_event.set()

        snapshot_event = self.snapshot_events.get(reqId)
        if snapshot_event is not None:
            snapshot_event.set()

        order_event = self.order_events.get(reqId)
        if order_event is not None:
            order_event.set()

    def connectionClosed(self) -> None:  # noqa: N802
        self.logger.warning("IB connection closed.")
        self.connection_ready.clear()


class IBClient:
    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        logger: logging.Logger,
        request_timeout: float = 15.0,
        account_summary_group: str = "All",
        app_factory: Callable[[logging.Logger], _IBGatewayApp] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.logger = logger
        self.request_timeout = request_timeout
        self.account_summary_group = account_summary_group
        self._app_factory = app_factory or _IBGatewayApp
        self._app = self._app_factory(logger)
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
            raise IBClientError(
                "Account summary request failed. "
                + " | ".join(errors)
            )

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

    def get_market_snapshot(self, symbol: str) -> dict[str, Any]:
        self._ensure_connected()
        req_id = self._next_request_id()
        snapshot_event = threading.Event()
        self._app.snapshot_events[req_id] = snapshot_event
        self._app.snapshot_data[req_id] = {"symbol": symbol.upper()}
        self._app.request_errors.pop(req_id, None)
        self._app.request_notices.pop(req_id, None)
        self.logger.info("Requesting market snapshot for %s.", symbol.upper())
        self._app.reqMarketDataType(3)
        contract = create_stock_contract(symbol)
        self._app.reqMktData(req_id, contract, "", True, False, [])
        self._wait_for_event(snapshot_event, f"market snapshot for {symbol.upper()}", req_id=req_id)
        self._app.cancelMktData(req_id)

        snapshot = self._app.snapshot_data.get(req_id, {})
        errors = self._app.request_errors.get(req_id, [])
        notices = self._app.request_notices.get(req_id, [])
        if errors and len(snapshot) <= 1:
            raise IBClientError(
                f"Market snapshot request for {symbol.upper()} failed. "
                + " | ".join(errors)
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

        return snapshot

    def submit_paper_test_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
    ) -> dict[str, Any]:
        self._ensure_connected()

        if self._app.next_valid_order_id is None:
            raise IBClientError("Cannot submit an order because no valid IB order id is available.")

        order_id = self._app.next_valid_order_id
        self._app.next_valid_order_id += 1
        self._app.request_errors.pop(order_id, None)
        self._app.order_statuses.pop(order_id, None)
        self._app.order_events[order_id] = threading.Event()

        contract = create_stock_contract(symbol)
        order = create_market_order(action, quantity)

        self.logger.warning(
            "Submitting explicit paper test order: %s %s %s",
            action.upper(),
            quantity,
            symbol.upper(),
        )
        self._app.placeOrder(order_id, contract, order)
        self._wait_for_event(
            self._app.order_events[order_id],
            f"paper order acknowledgement for order {order_id}",
            req_id=order_id,
        )

        errors = self._app.request_errors.get(order_id, [])
        if errors:
            raise IBClientError("Paper order submission failed. " + " | ".join(errors))

        payload = self._app.order_statuses.get(order_id, {"order_id": order_id})
        payload.setdefault("symbol", symbol.upper())
        payload.setdefault("action", action.upper())
        payload.setdefault("quantity", quantity)
        return payload

    def _ensure_connected(self) -> None:
        if not self.is_connected():
            raise IBClientError(
                "IB client is not connected. Run the connection check first and confirm "
                "IB Gateway paper is reachable at 127.0.0.1:4002."
            )
