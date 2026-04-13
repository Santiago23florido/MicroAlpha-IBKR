from __future__ import annotations

from typing import Any

from ui.actions import check_ib_connection, fetch_live_overview, resolve_ui_client_id


class FakeClient:
    def __init__(self) -> None:
        self.client_id = 101
        self.host = "127.0.0.1"
        self.port = 4002
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> bool:
        self.connect_calls += 1
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def get_account_summary(self) -> list[dict[str, Any]]:
        return [{"account": "DU123", "tag": "NetLiquidation", "value": "1000", "currency": "USD"}]

    def get_positions(self) -> list[dict[str, Any]]:
        return [{"symbol": "SPY", "position": 1}]


class FakeRuntime:
    def __init__(self) -> None:
        self.client = FakeClient()


def test_resolve_ui_client_id_uses_explicit_override() -> None:
    assert resolve_ui_client_id({"IB_CLIENT_ID": "1", "IB_UI_CLIENT_ID": "101"}) == 101


def test_resolve_ui_client_id_defaults_to_base_plus_100() -> None:
    assert resolve_ui_client_id({"IB_CLIENT_ID": "7"}) == 107


def test_check_ib_connection_uses_single_connect_cycle() -> None:
    runtime = FakeRuntime()

    payload = check_ib_connection(runtime)

    assert payload["connected"] is True
    assert payload["client_id"] == 101
    assert runtime.client.connect_calls == 1
    assert runtime.client.disconnect_calls == 1


def test_fetch_live_overview_returns_connection_and_account_state() -> None:
    runtime = FakeRuntime()

    payload = fetch_live_overview(runtime)

    assert payload["connection"]["connected"] is True
    assert payload["connection"]["client_id"] == 101
    assert payload["account_summary"][0]["account"] == "DU123"
    assert payload["positions"][0]["symbol"] == "SPY"
    assert runtime.client.connect_calls == 1
    assert runtime.client.disconnect_calls == 1
