from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from data.schemas import MarketSnapshot
from engine.market_clock import MarketClock


@dataclass(frozen=True)
class MarketDataRecord:
    timestamp: str
    symbol: str
    last_price: float | None
    bid: float | None
    ask: float | None
    spread: float | None
    bid_size: float | None
    ask_size: float | None
    last_size: float | None
    volume: float | None
    event_type: str
    source: str
    session_window: str
    is_market_open: bool
    exchange_time: str
    collected_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_market_snapshot(snapshot: MarketSnapshot, market_clock: MarketClock) -> MarketDataRecord:
    event_time = _coerce_timestamp(snapshot.timestamp)
    clock_state = market_clock.get_market_state(event_time)
    spread = None
    if snapshot.bid is not None and snapshot.ask is not None:
        spread = round(snapshot.ask - snapshot.bid, 10)

    return MarketDataRecord(
        timestamp=event_time.isoformat(),
        symbol=snapshot.symbol.upper(),
        last_price=snapshot.last,
        bid=snapshot.bid,
        ask=snapshot.ask,
        spread=spread,
        bid_size=snapshot.bid_size,
        ask_size=snapshot.ask_size,
        last_size=snapshot.last_size,
        volume=snapshot.volume,
        event_type="snapshot",
        source=snapshot.source,
        session_window=clock_state.session_window,
        is_market_open=clock_state.is_market_open,
        exchange_time=clock_state.exchange_time.isoformat(),
        collected_at=datetime.now(timezone.utc).isoformat(),
    )


def _coerce_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
