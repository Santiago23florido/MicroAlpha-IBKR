from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from typing import Protocol

from broker.ib_client import IBClient
from config import Settings
from data.live_data import LiveDataService
from data.schemas import MarketSnapshot


class MarketDataClient(Protocol):
    logger: Logger

    def connect(self) -> bool: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def get_server_time(self) -> dict[str, str | int]: ...

    def fetch_market_snapshot(self, symbol: str) -> MarketSnapshot: ...


@dataclass
class CollectorIBClient:
    settings: Settings
    logger: Logger
    raw_client: IBClient

    def __post_init__(self) -> None:
        self._live_data = LiveDataService(self.raw_client, self.settings)

    @property
    def client_id(self) -> int:
        return self.raw_client.client_id

    def connect(self) -> bool:
        return self.raw_client.connect()

    def disconnect(self) -> None:
        self.raw_client.disconnect()

    def is_connected(self) -> bool:
        return self.raw_client.is_connected()

    def get_server_time(self) -> dict[str, str | int]:
        return self.raw_client.get_server_time()

    def fetch_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return self._live_data.fetch_market_snapshot(symbol)


def build_collector_ib_client(settings: Settings, logger: Logger) -> CollectorIBClient:
    raw_client = IBClient(
        host=settings.ib_host,
        port=settings.ib_port,
        client_id=settings.ib_collector_client_id,
        logger=logger,
        request_timeout=settings.request_timeout_seconds,
        order_follow_up_seconds=settings.order_follow_up_seconds,
        account_summary_group=settings.account_summary_group,
        exchange=settings.ib_exchange,
        currency=settings.ib_currency,
    )
    return CollectorIBClient(settings=settings, logger=logger, raw_client=raw_client)
