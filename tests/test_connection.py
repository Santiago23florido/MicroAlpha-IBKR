from __future__ import annotations

import logging
import threading

import pytest

from broker.ib_client import IBClient, IBRequestTimeout


class FakeIBApp:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.connected = False
        self.connection_ready = threading.Event()
        self.server_time_event = threading.Event()
        self.account_summary_event = threading.Event()
        self.positions_event = threading.Event()
        self.connection_errors: list[str] = []
        self.request_errors: dict[int, list[str]] = {}
        self.request_notices: dict[int, list[str]] = {}
        self.snapshot_events: dict[int, threading.Event] = {}
        self.order_events: dict[int, threading.Event] = {}
        self.snapshot_data: dict[int, dict[str, object]] = {}
        self.order_statuses: dict[int, dict[str, object]] = {}
        self.account_summary_rows: dict[int, list[dict[str, str]]] = {}
        self.positions_rows: list[dict[str, object]] = []
        self.current_server_time: int | None = None
        self.next_valid_order_id: int | None = None
        self.active_account_summary_req_id: int | None = None

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
