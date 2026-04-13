from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from data.schemas import RiskCheckResult
from risk.limits import ExecutionGateContext, StrategyRiskContext

SUPPORTED_ORDER_TYPES = {"market", "limit", "bracket"}
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")


@dataclass(frozen=True)
class OrderDecision:
    approved: bool
    submit_to_broker: bool
    reason: str


@dataclass(frozen=True)
class ExecutionRequest:
    symbol: str
    action: str
    quantity: int
    order_type: str
    explicit_command: bool
    limit_price: float | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None


class RiskManager:
    def __init__(
        self,
        safe_to_trade: bool,
        dry_run: bool,
        supported_symbols: Iterable[str],
        *,
        max_open_positions: int = 1,
        max_trades_per_day: int = 2,
        max_daily_loss_pct: float = 1.0,
        max_spread_bps: float = 8.0,
    ) -> None:
        self.safe_to_trade = safe_to_trade
        self.dry_run = dry_run
        self.supported_symbols = {symbol.upper() for symbol in supported_symbols}
        self.max_open_positions = max_open_positions
        self.max_trades_per_day = max_trades_per_day
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_spread_bps = max_spread_bps

    def evaluate_signal_risk(self, context: StrategyRiskContext) -> RiskCheckResult:
        checks = {
            "connection_healthy": context.connection_healthy,
            "market_is_open": context.market_is_open,
            "trading_window_allowed": context.trading_window_allowed,
            "flatten_window_clear": not context.flatten_required or context.action == "close",
            "open_position_limit_ok": (
                context.action == "close"
                or context.open_positions_count < context.max_open_positions
                or context.symbol_position != 0
            ),
            "max_trades_per_day_ok": context.trades_today < context.max_trades_per_day,
            "spread_ok": context.spread_bps is None or context.spread_bps <= context.max_spread_bps,
            "max_daily_loss_ok": self._daily_loss_ok(
                daily_realized_pnl=context.daily_realized_pnl,
                net_liquidation=context.net_liquidation,
                max_daily_loss_pct=context.max_daily_loss_pct,
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        failures.extend(context.failure_overrides)
        passed = not failures
        summary = (
            "Risk checks passed."
            if passed
            else "Risk rejected the trade: " + "; ".join(self._humanize_failure(item) for item in failures)
        )
        return RiskCheckResult(passed=passed, checks=checks, failures=failures, summary=summary)

    def evaluate_execution_gate(self, context: ExecutionGateContext) -> dict[str, bool]:
        return {
            "explicit_session_request": context.explicit_session_request,
            "session_execution_enabled": context.session_execution_enabled,
            "safe_to_trade": context.safe_to_trade,
            "dry_run_disabled": not context.dry_run,
        }

    def evaluate_execution_request(self, request: ExecutionRequest) -> OrderDecision:
        symbol = request.symbol.strip().upper()
        action = request.action.upper()
        order_type = request.order_type.lower()

        if not request.explicit_command:
            return OrderDecision(
                approved=False,
                submit_to_broker=False,
                reason="Order rejected: execution requires an explicit command.",
            )
        if not symbol:
            return OrderDecision(False, False, "Order rejected: symbol must not be empty.")
        if not SYMBOL_PATTERN.match(symbol):
            return OrderDecision(
                False,
                False,
                f"Order rejected: symbol {symbol!r} is not valid for this paper system.",
            )
        if self.supported_symbols and symbol not in self.supported_symbols:
            return OrderDecision(
                False,
                False,
                (
                    f"Order rejected: symbol {symbol} is not enabled. "
                    f"Supported symbols: {', '.join(sorted(self.supported_symbols))}."
                ),
            )
        if action not in {"BUY", "SELL"}:
            return OrderDecision(False, False, "Order rejected: action must be BUY or SELL.")
        if request.quantity <= 0:
            return OrderDecision(False, False, "Order rejected: quantity must be greater than zero.")
        if order_type not in SUPPORTED_ORDER_TYPES:
            return OrderDecision(
                False,
                False,
                (
                    f"Order rejected: unsupported order type {request.order_type!r}. "
                    f"Supported types: {', '.join(sorted(SUPPORTED_ORDER_TYPES))}."
                ),
            )

        specific_validation = self._validate_order_specific_fields(action, request)
        if specific_validation is not None:
            return specific_validation
        if not self.safe_to_trade:
            return OrderDecision(
                False,
                False,
                (
                    "Order rejected: SAFE_TO_TRADE is false. "
                    "Paper execution remains blocked until you explicitly set SAFE_TO_TRADE=true."
                ),
            )
        if self.dry_run:
            return OrderDecision(
                True,
                False,
                (
                    "Dry-run is enabled. The order passed validation but was not sent. "
                    "Set DRY_RUN=false only when you intentionally want a paper submission."
                ),
            )
        return OrderDecision(True, True, "Order approved for explicit paper submission.")

    def evaluate_position_close(
        self,
        *,
        symbol: str,
        position_quantity: float,
        explicit_command: bool,
    ) -> OrderDecision:
        if position_quantity == 0:
            return OrderDecision(
                False,
                False,
                f"Close rejected: there is no open position for {symbol.upper()}.",
            )
        if not float(abs(position_quantity)).is_integer():
            return OrderDecision(
                False,
                False,
                (
                    f"Close rejected: position size for {symbol.upper()} is fractional "
                    f"({position_quantity}) and only whole-share closes are supported."
                ),
            )
        action = "SELL" if position_quantity > 0 else "BUY"
        return self.evaluate_execution_request(
            ExecutionRequest(
                symbol=symbol.upper(),
                action=action,
                quantity=int(abs(position_quantity)),
                order_type="market",
                explicit_command=explicit_command,
            )
        )

    @staticmethod
    def _daily_loss_ok(
        *,
        daily_realized_pnl: float,
        net_liquidation: float | None,
        max_daily_loss_pct: float,
    ) -> bool:
        if net_liquidation in {None, 0}:
            return True
        if daily_realized_pnl >= 0:
            return True
        realized_loss_pct = abs(daily_realized_pnl) / float(net_liquidation) * 100.0
        return realized_loss_pct <= max_daily_loss_pct

    @staticmethod
    def _humanize_failure(name: str) -> str:
        return name.replace("_", " ")

    def _validate_order_specific_fields(
        self,
        action: str,
        request: ExecutionRequest,
    ) -> OrderDecision | None:
        order_type = request.order_type.lower()
        if order_type == "market":
            return None
        if request.limit_price is None or request.limit_price <= 0:
            return OrderDecision(
                False,
                False,
                f"Order rejected: {order_type} orders require a positive limit price.",
            )
        if order_type == "limit":
            return None
        if request.take_profit_price is None or request.take_profit_price <= 0:
            return OrderDecision(
                False,
                False,
                "Order rejected: bracket orders require a positive take-profit price.",
            )
        if request.stop_loss_price is None or request.stop_loss_price <= 0:
            return OrderDecision(
                False,
                False,
                "Order rejected: bracket orders require a positive stop-loss price.",
            )
        if action == "BUY":
            valid_bracket = request.take_profit_price > request.limit_price > request.stop_loss_price
        else:
            valid_bracket = request.take_profit_price < request.limit_price < request.stop_loss_price
        if not valid_bracket:
            return OrderDecision(
                False,
                False,
                (
                    "Order rejected: bracket prices are inconsistent. "
                    "For BUY use take-profit > entry limit > stop-loss. "
                    "For SELL use take-profit < entry limit < stop-loss."
                ),
            )
        return None
