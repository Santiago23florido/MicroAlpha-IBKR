from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from execution.models import FillApplicationResult, FillEvent, OrderAction, PortfolioSnapshot, PositionState


class PositionConsistencyError(ValueError):
    """Raised when a fill would leave the portfolio in an inconsistent state."""


class PositionManager:
    def __init__(
        self,
        *,
        initial_cash: float,
        positions: Mapping[str, PositionState] | None = None,
        cash: float | None = None,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(self.initial_cash if cash is None else cash)
        self.positions: dict[str, PositionState] = {symbol.upper(): state for symbol, state in (positions or {}).items()}

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any] | None, *, initial_cash: float) -> "PositionManager":
        if not snapshot:
            return cls(initial_cash=initial_cash)
        positions_payload = snapshot.get("positions", {}) or {}
        positions = {
            symbol.upper(): PositionState(
                symbol=str(payload.get("symbol", symbol)).upper(),
                quantity=int(payload.get("quantity", 0) or 0),
                average_entry_price=float(payload.get("average_entry_price", 0.0) or 0.0),
                realized_pnl=float(payload.get("realized_pnl", 0.0) or 0.0),
                unrealized_pnl=float(payload.get("unrealized_pnl", 0.0) or 0.0),
                total_commissions=float(payload.get("total_commissions", 0.0) or 0.0),
                trade_count=int(payload.get("trade_count", 0) or 0),
                last_price=_coerce_float(payload.get("last_price")),
                market_value=float(payload.get("market_value", 0.0) or 0.0),
                notional_exposure=float(payload.get("notional_exposure", 0.0) or 0.0),
                updated_at=payload.get("updated_at"),
            )
            for symbol, payload in positions_payload.items()
        }
        return cls(
            initial_cash=float(snapshot.get("initial_cash", initial_cash)),
            positions=positions,
            cash=float(snapshot.get("cash", initial_cash)),
        )

    def apply_fill(self, fill: FillEvent) -> FillApplicationResult:
        symbol = fill.symbol.upper()
        existing = self.positions.get(symbol, PositionState(symbol=symbol))
        delta = _signed_quantity(fill.action, fill.quantity)
        current_qty = int(existing.quantity)
        new_qty = current_qty + delta
        commission = float(fill.commission)
        realized_pnl_delta = 0.0
        realized_return_bps: float | None = None

        if current_qty == 0 or (current_qty > 0 and delta > 0) or (current_qty < 0 and delta < 0):
            average_entry_price = self._weighted_average_price(existing, fill.fill_price, delta)
        else:
            close_quantity = min(abs(current_qty), abs(delta))
            average_entry_price = existing.average_entry_price
            if current_qty > 0 and delta < 0:
                realized_pnl_delta += (fill.fill_price - existing.average_entry_price) * close_quantity
                realized_return_bps = _return_bps(existing.average_entry_price, fill.fill_price, is_short=False)
            elif current_qty < 0 and delta > 0:
                realized_pnl_delta += (existing.average_entry_price - fill.fill_price) * close_quantity
                realized_return_bps = _return_bps(existing.average_entry_price, fill.fill_price, is_short=True)

            if abs(delta) > abs(current_qty):
                average_entry_price = fill.fill_price
            elif new_qty == 0:
                average_entry_price = 0.0

        realized_pnl_delta -= commission
        self.cash -= (fill.fill_price * delta) + commission

        last_price = float(fill.fill_price)
        updated = replace(
            existing,
            quantity=int(new_qty),
            average_entry_price=float(average_entry_price),
            realized_pnl=float(existing.realized_pnl + realized_pnl_delta),
            total_commissions=float(existing.total_commissions + commission),
            trade_count=int(existing.trade_count + 1),
            last_price=last_price,
            updated_at=fill.filled_at,
        )
        updated = self._mark_position(updated, last_price)
        if updated.quantity == 0:
            updated = replace(updated, average_entry_price=0.0, market_value=0.0, notional_exposure=0.0, unrealized_pnl=0.0)
        self.positions[symbol] = updated
        self._validate_position(updated)

        portfolio = self.snapshot()
        return FillApplicationResult(
            position=updated,
            portfolio=portfolio,
            realized_pnl_delta=float(realized_pnl_delta),
            realized_return_bps=realized_return_bps,
        )

    def update_market_prices(self, price_map: Mapping[str, float | None]) -> PortfolioSnapshot:
        for symbol, price in price_map.items():
            normalized = symbol.upper()
            if normalized not in self.positions:
                continue
            if price is None:
                continue
            self.positions[normalized] = self._mark_position(self.positions[normalized], float(price))
        return self.snapshot()

    def snapshot(self) -> PortfolioSnapshot:
        positions = {symbol: state for symbol, state in self.positions.items()}
        realized_pnl = sum(state.realized_pnl for state in positions.values())
        unrealized_pnl = sum(state.unrealized_pnl for state in positions.values())
        total_commissions = sum(state.total_commissions for state in positions.values())
        trade_count = sum(state.trade_count for state in positions.values())
        market_value = sum(state.market_value for state in positions.values())
        open_position_count = sum(1 for state in positions.values() if state.quantity != 0)
        return PortfolioSnapshot(
            positions=positions,
            cash=float(self.cash),
            realized_pnl=float(realized_pnl),
            unrealized_pnl=float(unrealized_pnl),
            total_commissions=float(total_commissions),
            trade_count=int(trade_count),
            open_position_count=int(open_position_count),
            equity=float(self.cash + market_value),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_state_payload(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "initial_cash": self.initial_cash,
            "cash": snapshot.cash,
            "positions": {symbol: state.to_dict() for symbol, state in self.positions.items()},
            "portfolio": snapshot.to_dict(),
        }

    def current_quantity(self, symbol: str) -> int:
        return int(self.positions.get(symbol.upper(), PositionState(symbol=symbol.upper())).quantity)

    def _weighted_average_price(self, state: PositionState, fill_price: float, delta: int) -> float:
        current_qty = abs(int(state.quantity))
        incoming_qty = abs(int(delta))
        if current_qty <= 0:
            return float(fill_price)
        total_qty = current_qty + incoming_qty
        weighted_notional = (state.average_entry_price * current_qty) + (float(fill_price) * incoming_qty)
        return float(weighted_notional / total_qty)

    def _mark_position(self, state: PositionState, last_price: float) -> PositionState:
        if state.quantity == 0:
            return replace(
                state,
                last_price=last_price,
                market_value=0.0,
                notional_exposure=0.0,
                unrealized_pnl=0.0,
            )
        if state.quantity > 0:
            unrealized = (last_price - state.average_entry_price) * state.quantity
        else:
            unrealized = (state.average_entry_price - last_price) * abs(state.quantity)
        return replace(
            state,
            last_price=float(last_price),
            market_value=float(state.quantity * last_price),
            notional_exposure=float(abs(state.quantity * last_price)),
            unrealized_pnl=float(unrealized),
        )

    @staticmethod
    def _validate_position(state: PositionState) -> None:
        if state.quantity == 0 and abs(state.average_entry_price) > 1e-9:
            raise PositionConsistencyError(
                f"Flat position for {state.symbol} cannot keep a non-zero average entry price."
            )


def _signed_quantity(action: OrderAction, quantity: int) -> int:
    if action in {OrderAction.BUY, OrderAction.COVER}:
        return int(quantity)
    return -int(quantity)


def _return_bps(entry_price: float, exit_price: float, *, is_short: bool) -> float | None:
    if entry_price <= 0:
        return None
    raw_return = ((entry_price - exit_price) / entry_price) if is_short else ((exit_price - entry_price) / entry_price)
    return float(raw_return * 10000.0)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
