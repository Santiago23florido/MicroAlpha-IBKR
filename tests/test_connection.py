from __future__ import annotations

import logging
import threading

import pytest

from broker.ib_client import IBClient, IBRequestTimeout


class FakeIBApp:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.audit_store = None
        self.connected = False
        self.connection_ready = threading.Event()
        self.server_time_event = threading.Event()
        self.account_summary_event = threading.Event()
        self.positions_event = threading.Event()
        self.open_orders_event = threading.Event()
        self.executions_event = threading.Event()
        self.connection_errors: list[str] = []
        self.request_errors: dict[int, list[str]] = {}
        self.request_notices: dict[int, list[str]] = {}
        self.snapshot_events: dict[int, threading.Event] = {}
        self.historical_data_events: dict[int, threading.Event] = {}
        self.order_events: dict[int, threading.Event] = {}
        self.order_terminal_events: dict[int, threading.Event] = {}
        self.snapshot_data: dict[int, dict[str, object]] = {}
        self.historical_data_rows: dict[int, list[dict[str, object]]] = {}
        self.order_statuses: dict[int, dict[str, object]] = {}
        self.order_execution_details: dict[int, list[dict[str, object]]] = {}
        self.account_summary_rows: dict[int, list[dict[str, str]]] = {}
        self.positions_rows: list[dict[str, object]] = []
        self.open_orders_rows: list[dict[str, object]] = []
        self.execution_rows: list[dict[str, object]] = []
        self.current_server_time: int | None = None
        self.next_valid_order_id: int | None = None
        self.active_account_summary_req_id: int | None = None
        self.collecting_open_orders = False
        self.lock = threading.Lock()

    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connected = True
        self.next_valid_order_id = 1000
        self.connection_ready.set()

    def disconnect(self) -> None:
        self.connected = False

    def isConnected(self) -> bool:  # noqa: N802
        return self.connected

    def run(self) -> None:
        return None

    def reqCurrentTime(self) -> None:  # noqa: N802
        self.current_server_time = 1_700_000_000
        self.server_time_event.set()

    def reqMarketDataType(self, market_data_type: int) -> None:  # noqa: N802
        self.market_data_type = market_data_type

    def reqMktData(self, reqId: int, contract, generic_tick_list: str, snapshot: bool, regulatory_snapshot: bool, mkt_data_options) -> None:  # noqa: N802
        self.snapshot_events[reqId].set()

    def cancelMktData(self, reqId: int) -> None:  # noqa: N802
        return None

    def reqHistoricalData(  # noqa: N802
        self,
        reqId: int,
        contract,
        endDateTime: str,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: int,
        formatDate: int,
        keepUpToDate: bool,
        chartOptions,
    ) -> None:
        self.historical_data_events[reqId].set()

    def _copy_order_status_locked(self, order_id: int) -> dict[str, object]:
        return dict(self.order_statuses.get(order_id, {}))


class SnapshotFallbackIBApp(FakeIBApp):
    def reqMktData(self, reqId: int, contract, generic_tick_list: str, snapshot: bool, regulatory_snapshot: bool, mkt_data_options) -> None:  # noqa: N802
        self.request_notices[reqId] = ["IB error 10167 (reqId=1): delayed market data."]
        self.snapshot_events[reqId].set()

    def reqHistoricalData(  # noqa: N802
        self,
        reqId: int,
        contract,
        endDateTime: str,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: int,
        formatDate: int,
        keepUpToDate: bool,
        chartOptions,
    ) -> None:
        self.historical_data_rows[reqId] = [
            {
                "timestamp": "2026-04-13T19:30:00+00:00",
                "open": 500.0,
                "high": 501.0,
                "low": 499.5,
                "close": 500.5,
                "volume": 12345,
            }
        ]
        self.historical_data_events[reqId].set()


class HangingIBApp(FakeIBApp):
    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connected = False


def build_client(app_factory):
    logger = logging.getLogger("test-ib-client")
    return IBClient(
        host="127.0.0.1",
        port=4002,
        client_id=1,
        logger=logger,
        request_timeout=0.01,
        app_factory=app_factory,
    )


def test_connect_and_disconnect_cycle() -> None:
    client = build_client(FakeIBApp)

    assert client.connect() is True
    assert client.is_connected() is True

    client.disconnect()
    assert client.is_connected() is False


def test_get_server_time_returns_utc_payload() -> None:
    client = build_client(FakeIBApp)
    client.connect()

    payload = client.get_server_time()

    assert payload["epoch"] == 1_700_000_000
    assert str(payload["iso_utc"]).endswith("+00:00")


def test_connect_timeout_is_actionable() -> None:
    client = build_client(HangingIBApp)

    with pytest.raises(IBRequestTimeout) as exc_info:
        client.connect()

    assert "127.0.0.1:4002" in str(exc_info.value)


def test_market_snapshot_falls_back_to_recent_bar_when_ticks_are_missing() -> None:
    client = build_client(SnapshotFallbackIBApp)
    client.connect()

    payload = client.get_market_snapshot("SPY")

    assert payload["symbol"] == "SPY"
    assert payload["source"] == "historical_bar_fallback"
    assert payload["last"] == 500.5
    assert payload["close"] == 500.5
    assert payload["market_data_notices"] == ["IB error 10167 (reqId=1): delayed market data."]
