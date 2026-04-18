from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from config.phase7 import ExecutionBackendConfig
from execution.models import FillEvent, Order, OrderAction


@dataclass(frozen=True)
class FillSimulationResult:
    accepted: bool
    fills: list[FillEvent]
    rejection_reason: str | None = None
    metadata: dict[str, Any] | None = None


class FillSimulator:
    def __init__(self, config: ExecutionBackendConfig, *, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random(7)

    def simulate_fill(self, order: Order, market_data: Mapping[str, Any] | None = None) -> FillSimulationResult:
        market = dict(market_data or {})
        if self.config.reject_probability > 0 and self.rng.random() < self.config.reject_probability:
            return FillSimulationResult(
                accepted=False,
                fills=[],
                rejection_reason="mock_backend_reject_probability_triggered",
                metadata={"reject_probability": self.config.reject_probability},
            )

        if not self.config.simulate_immediate_fills:
            return FillSimulationResult(
                accepted=True,
                fills=[],
                metadata={"immediate_fill_disabled": True},
            )

        reference_price = self._reference_price(order.action, market)
        if reference_price is None or reference_price <= 0:
            return FillSimulationResult(
                accepted=False,
                fills=[],
                rejection_reason="missing_reference_price_for_fill_simulation",
            )

        if order.order_type.value == "LIMIT" and order.limit_price is not None:
            if not self._limit_is_marketable(order.action, order.limit_price, reference_price):
                return FillSimulationResult(
                    accepted=True,
                    fills=[],
                    metadata={"pending_reason": "limit_order_not_marketable"},
                )

        fill_plan = self._fill_quantities(order.quantity)
        fills: list[FillEvent] = []
        fill_time = self._parse_timestamp(order.updated_at)
        for index, fill_quantity in enumerate(fill_plan):
            fill_time = fill_time + timedelta(milliseconds=self.config.fill_delay_ms * index)
            fill_price = self._apply_slippage(reference_price, order.action)
            commission = round(
                float(self.config.commission_per_trade) + (float(self.config.commission_per_share) * fill_quantity),
                6,
            )
            fills.append(
                FillEvent(
                    fill_id=f"fill_{uuid4().hex[:12]}",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    action=order.action,
                    quantity=int(fill_quantity),
                    fill_price=round(float(fill_price), 6),
                    commission=commission,
                    filled_at=fill_time.isoformat(),
                    backend_name="mock",
                    source_model_name=order.source_model_name,
                    source_decision_id=order.source_decision_id,
                    slippage_bps=float(self.config.slippage_bps),
                    metadata={
                        "reference_price": round(float(reference_price), 6),
                        "spread_aware": bool(self.config.spread_aware_fills),
                        "fill_index": index,
                    },
                )
            )

        return FillSimulationResult(
            accepted=True,
            fills=fills,
            metadata={"fill_count": len(fills)},
        )

    def _fill_quantities(self, quantity: int) -> list[int]:
        if (
            not self.config.allow_partial_fills
            or quantity <= 1
            or self.config.partial_fill_probability <= 0
            or self.rng.random() >= self.config.partial_fill_probability
        ):
            return [int(quantity)]

        first_fill = max(1, min(quantity - 1, int(round(quantity * self.config.partial_fill_ratio))))
        second_fill = int(quantity) - first_fill
        if second_fill <= 0:
            return [int(quantity)]
        return [first_fill, second_fill]

    def _reference_price(self, action: OrderAction, market: Mapping[str, Any]) -> float | None:
        if self.config.spread_aware_fills:
            bid = _coerce_float(market.get("bid"))
            ask = _coerce_float(market.get("ask"))
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                return ask if action in {OrderAction.BUY, OrderAction.COVER} else bid

            spread_bps = _coerce_float(market.get("spread_bps"))
            mid = _coerce_float(market.get("mid_price"))
            if mid is None:
                mid = _coerce_float(market.get("price_proxy")) or _coerce_float(market.get("last")) or _coerce_float(market.get("close"))
            if mid is not None and spread_bps is not None:
                half_spread = mid * (spread_bps / 20000.0)
                synthetic_bid = mid - half_spread
                synthetic_ask = mid + half_spread
                return synthetic_ask if action in {OrderAction.BUY, OrderAction.COVER} else synthetic_bid

        for key in ("last", "price_proxy", "mid_price", "close", "open"):
            value = _coerce_float(market.get(key))
            if value is not None and value > 0:
                return value
        return None

    def _apply_slippage(self, reference_price: float, action: OrderAction) -> float:
        direction = 1.0 if action in {OrderAction.BUY, OrderAction.COVER} else -1.0
        return reference_price * (1.0 + ((direction * self.config.slippage_bps) / 10000.0))

    def _limit_is_marketable(self, action: OrderAction, limit_price: float, reference_price: float) -> bool:
        if action in {OrderAction.BUY, OrderAction.COVER}:
            return reference_price <= limit_price
        return reference_price >= limit_price

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime:
        if value:
            parsed = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.notna(parsed):
                return parsed.to_pydatetime()
        return datetime.now(timezone.utc)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
