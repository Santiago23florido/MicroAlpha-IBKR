from __future__ import annotations

import os
from contextlib import suppress
from typing import Any, Mapping

from engine.runtime import RuntimeServices


def resolve_ui_client_id(env: Mapping[str, str] | None = None) -> int:
    values = env or os.environ
    raw_value = values.get("IB_UI_CLIENT_ID")
    if raw_value is not None:
        return int(raw_value)
    base_client_id = int(values.get("IB_CLIENT_ID", "1"))
    return base_client_id + 100


def check_ib_connection(runtime: RuntimeServices) -> dict[str, Any]:
    try:
        runtime.client.connect()
        return {
            "connected": runtime.client.is_connected(),
            "client_id": runtime.client.client_id,
            "host": runtime.client.host,
            "port": runtime.client.port,
        }
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def fetch_live_overview(runtime: RuntimeServices) -> dict[str, Any]:
    try:
        runtime.client.connect()
        return {
            "connection": {
                "connected": runtime.client.is_connected(),
                "client_id": runtime.client.client_id,
                "host": runtime.client.host,
                "port": runtime.client.port,
            },
            "account_summary": runtime.client.get_account_summary(),
            "positions": runtime.client.get_positions(),
        }
    finally:
        with suppress(Exception):
            runtime.client.disconnect()
