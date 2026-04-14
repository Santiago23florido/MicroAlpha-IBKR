from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from broker.ib_client import IBClient
from config import Settings
from data.schemas import MarketSnapshot


class LiveDataService:
    def __init__(self, client: IBClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def fetch_market_snapshot(self, symbol: str | None = None) -> MarketSnapshot:
        ticker = (symbol or self.settings.ib_symbol).upper()
        payload = self.client.get_market_snapshot(
            symbol=ticker,
            exchange=self.settings.ib_exchange,
            currency=self.settings.ib_currency,
        )
        timestamp = payload.get("snapshot_utc") or datetime.now(timezone.utc).isoformat()
        return MarketSnapshot(
            symbol=ticker,
            timestamp=timestamp,
            source=str(payload.get("source", "ib_snapshot")),
            bid=_coerce_float(payload.get("bid")),
            ask=_coerce_float(payload.get("ask")),
            last=_coerce_float(payload.get("last")),
            open=_coerce_float(payload.get("open")),
            high=_coerce_float(payload.get("high")),
            low=_coerce_float(payload.get("low")),
            close=_coerce_float(payload.get("close")),
            volume=_coerce_float(payload.get("volume")),
            bid_size=_coerce_float(payload.get("bid_size")),
            ask_size=_coerce_float(payload.get("ask_size")),
            last_size=_coerce_float(payload.get("last_size")),
            raw=payload,
        )

    def fetch_intraday_bars(
        self,
        symbol: str | None = None,
        duration: str = "1 D",
        bar_size: str = "1 min",
    ) -> pd.DataFrame:
        ticker = (symbol or self.settings.ib_symbol).upper()
        rows = self.client.get_historical_bars(
            symbol=ticker,
            exchange=self.settings.ib_exchange,
            currency=self.settings.ib_currency,
            duration=duration,
            bar_size=bar_size,
            what_to_show="TRADES",
            use_rth=True,
        )
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows)


def _coerce_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)
