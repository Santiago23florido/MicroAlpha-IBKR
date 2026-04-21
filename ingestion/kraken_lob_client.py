from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any


class KrakenLOBClientError(RuntimeError):
    pass


class KrakenLOBClient:
    """Minimal synchronous Kraken Spot L2 client for the public book channel."""

    def __init__(
        self,
        *,
        websocket_url: str = "wss://ws.kraken.com/v2",
        logger: Any | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.websocket_url = websocket_url
        self.logger = logger
        self.timeout = timeout
        self._ws: Any | None = None
        self._connected = False
        self._symbol: str | None = None
        self._depth = 10
        self._event_index = 0
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}

    def connect(self) -> bool:
        try:
            import websocket  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise KrakenLOBClientError(
                "websocket-client is required for Kraken LOB capture. "
                "Install dependencies from requirements.txt."
            ) from exc
        sslopt: dict[str, Any] = {}
        try:
            import certifi

            sslopt["ca_certs"] = certifi.where()
        except ImportError:
            pass
        self._ws = websocket.create_connection(self.websocket_url, timeout=self.timeout, sslopt=sslopt)
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._connected and self._ws is not None)

    def subscribe_market_depth(
        self,
        *,
        symbol: str,
        num_rows: int = 10,
        exchange: str | None = None,
        currency: str | None = None,
        is_smart_depth: bool = True,
    ) -> int:
        if self._ws is None:
            raise KrakenLOBClientError("Kraken WebSocket is not connected.")
        self._symbol = _normalize_kraken_symbol(symbol)
        self._depth = int(num_rows)
        self._bids.clear()
        self._asks.clear()
        self._event_index = 0
        payload = {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": [self._symbol],
                "depth": self._depth,
                "snapshot": True,
            },
        }
        self._ws.send(json.dumps(payload))
        if self.logger:
            self.logger.info("Subscribed to Kraken book channel for %s depth=%s.", self._symbol, self._depth)
        return 1

    def consume_market_depth_events(
        self,
        req_id: int,
        *,
        timeout: float = 1.0,
        max_events: int = 500,
    ) -> list[dict[str, Any]]:
        if self._ws is None:
            raise KrakenLOBClientError("Kraken WebSocket is not connected.")
        self._ws.settimeout(timeout)
        deadline = time.monotonic() + timeout
        rows: list[dict[str, Any]] = []
        while len(rows) < max_events and time.monotonic() <= deadline:
            try:
                raw = self._ws.recv()
            except TimeoutError:
                break
            except Exception as exc:
                name = exc.__class__.__name__.lower()
                if "timeout" in name:
                    break
                self._connected = False
                raise KrakenLOBClientError(f"Kraken WebSocket receive failed: {exc}") from exc
            if not raw:
                continue
            for row in self._parse_message(raw):
                rows.append(row)
                if len(rows) >= max_events:
                    break
        return rows

    def cancel_market_depth(self, req_id: int) -> None:
        if self._ws is None or self._symbol is None:
            return
        payload = {
            "method": "unsubscribe",
            "params": {
                "channel": "book",
                "symbol": [self._symbol],
                "depth": self._depth,
            },
        }
        with _suppress_ws_errors():
            self._ws.send(json.dumps(payload))

    def _parse_message(self, raw: str) -> list[dict[str, Any]]:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return []
        if payload.get("method") == "subscribe":
            if payload.get("success") is False:
                raise KrakenLOBClientError(str(payload.get("error") or "Kraken book subscription failed."))
            result = payload.get("result") or {}
            if isinstance(result, dict) and result.get("success") is False:
                raise KrakenLOBClientError(str(result.get("error") or "Kraken book subscription failed."))
            return []
        if payload.get("channel") != "book":
            return []
        message_type = str(payload.get("type") or "")
        data = payload.get("data") or []
        if not isinstance(data, list):
            return []
        rows: list[dict[str, Any]] = []
        for book_payload in data:
            if not isinstance(book_payload, dict):
                continue
            if message_type == "snapshot":
                self._replace_book(book_payload)
            elif message_type == "update":
                self._update_book(book_payload)
            else:
                continue
            rows.append(self._snapshot_row(book_payload))
        return rows

    def _replace_book(self, payload: dict[str, Any]) -> None:
        self._bids = _levels_to_map(payload.get("bids") or [])
        self._asks = _levels_to_map(payload.get("asks") or [])

    def _update_book(self, payload: dict[str, Any]) -> None:
        _apply_levels(self._bids, payload.get("bids") or [])
        _apply_levels(self._asks, payload.get("asks") or [])

    def _snapshot_row(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._event_index += 1
        symbol = _normalize_kraken_symbol(str(payload.get("symbol") or self._symbol or "BTC/EUR"))
        timestamp = str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat())
        row: dict[str, Any] = {
            "symbol": symbol,
            "event_ts_utc": timestamp,
            "event_type": "lob_snapshot",
            "provider": "kraken",
            "source": "kraken_spot_book",
            "update_source": "kraken_ws_v2_book",
            "event_index": self._event_index,
            "reset_count": 0,
            "is_smart_depth": False,
            "checksum": payload.get("checksum"),
            "observed_bid_levels": min(len(self._bids), self._depth),
            "observed_ask_levels": min(len(self._asks), self._depth),
        }
        bid_levels = sorted(self._bids.items(), key=lambda item: item[0], reverse=True)[: self._depth]
        ask_levels = sorted(self._asks.items(), key=lambda item: item[0])[: self._depth]
        _fill_side(row, "bid", bid_levels, self._depth)
        _fill_side(row, "ask", ask_levels, self._depth)
        return row


def _normalize_kraken_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("-", "/")


def _levels_to_map(levels: list[dict[str, Any]]) -> dict[float, float]:
    book: dict[float, float] = {}
    _apply_levels(book, levels)
    return book


def _apply_levels(book: dict[float, float], levels: list[dict[str, Any]]) -> None:
    for level in levels:
        if not isinstance(level, dict):
            continue
        price = float(level.get("price", 0.0) or 0.0)
        quantity = float(level.get("qty", 0.0) or 0.0)
        if price <= 0:
            continue
        if quantity <= 0:
            book.pop(price, None)
        else:
            book[price] = quantity


def _fill_side(row: dict[str, Any], prefix: str, levels: list[tuple[float, float]], depth: int) -> None:
    for index in range(1, depth + 1):
        if index <= len(levels):
            price, quantity = levels[index - 1]
            row[f"{prefix}_px_{index}"] = float(price)
            row[f"{prefix}_sz_{index}"] = float(quantity)
            row[f"{prefix}_mm_{index}"] = None
        else:
            row[f"{prefix}_px_{index}"] = 0.0
            row[f"{prefix}_sz_{index}"] = 0.0
            row[f"{prefix}_mm_{index}"] = None


class _suppress_ws_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return True
