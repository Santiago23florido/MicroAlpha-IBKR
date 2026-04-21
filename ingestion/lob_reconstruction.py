from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LOBDepthUpdate:
    symbol: str
    timestamp_utc: str
    position: int
    operation: int
    side: int
    price: float
    size: float
    market_maker: str | None = None
    is_smart_depth: bool = True
    source: str = "updateMktDepthL2"


@dataclass
class LOBBookState:
    symbol: str
    depth_levels: int
    bids: list[dict[str, Any]] = field(default_factory=list)
    asks: list[dict[str, Any]] = field(default_factory=list)
    reset_count: int = 0
    event_count: int = 0

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.reset_count += 1

    def apply(self, update: LOBDepthUpdate) -> dict[str, Any]:
        levels = self.bids if update.side == 1 else self.asks
        self._apply_to_side(levels, update)
        self.event_count += 1
        return self.snapshot(update)

    def snapshot(self, update: LOBDepthUpdate) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "event_ts_utc": update.timestamp_utc,
            "event_type": "depth_update",
            "provider": "ibkr",
            "source": "ibkr_market_depth",
            "update_source": update.source,
            "event_position": update.position,
            "event_operation": update.operation,
            "event_side": "bid" if update.side == 1 else "ask",
            "event_price": update.price,
            "event_size": update.size,
            "event_market_maker": update.market_maker,
            "is_smart_depth": update.is_smart_depth,
            "event_index": self.event_count,
            "reset_count": self.reset_count,
            "observed_bid_levels": min(len(self.bids), self.depth_levels),
            "observed_ask_levels": min(len(self.asks), self.depth_levels),
        }
        self._fill_levels(payload, "bid", self.bids)
        self._fill_levels(payload, "ask", self.asks)
        return payload

    def _apply_to_side(self, side_levels: list[dict[str, Any]], update: LOBDepthUpdate) -> None:
        position = max(int(update.position), 0)
        level = {
            "price": float(update.price),
            "size": float(update.size),
            "market_maker": update.market_maker,
        }
        if update.operation == 0:
            if position >= len(side_levels):
                side_levels.append(level)
            else:
                side_levels.insert(position, level)
        elif update.operation == 1:
            while len(side_levels) <= position:
                side_levels.append({"price": 0.0, "size": 0.0, "market_maker": None})
            side_levels[position] = level
        elif update.operation == 2:
            if position < len(side_levels):
                side_levels.pop(position)
        else:
            raise ValueError(f"Unsupported market depth operation: {update.operation}")

        if len(side_levels) > self.depth_levels:
            del side_levels[self.depth_levels :]

    def _fill_levels(
        self,
        payload: dict[str, Any],
        prefix: str,
        levels: list[dict[str, Any]],
    ) -> None:
        for level_index in range(1, self.depth_levels + 1):
            if level_index <= len(levels):
                row = levels[level_index - 1]
                payload[f"{prefix}_px_{level_index}"] = float(row.get("price", 0.0) or 0.0)
                payload[f"{prefix}_sz_{level_index}"] = float(row.get("size", 0.0) or 0.0)
                payload[f"{prefix}_mm_{level_index}"] = row.get("market_maker")
            else:
                payload[f"{prefix}_px_{level_index}"] = 0.0
                payload[f"{prefix}_sz_{level_index}"] = 0.0
                payload[f"{prefix}_mm_{level_index}"] = None
