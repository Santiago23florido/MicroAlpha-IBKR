from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderDecision:
    approved: bool
    submit_to_broker: bool
    reason: str


class RiskManager:
    def __init__(self, safe_to_trade: bool, dry_run: bool) -> None:
        self.safe_to_trade = safe_to_trade
        self.dry_run = dry_run

    def evaluate_manual_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
    ) -> OrderDecision:
        if not symbol.strip():
            return OrderDecision(
                approved=False,
                submit_to_broker=False,
                reason="Order rejected: symbol must not be empty.",
            )

        if action.upper() not in {"BUY", "SELL"}:
            return OrderDecision(
                approved=False,
                submit_to_broker=False,
                reason="Order rejected: action must be BUY or SELL.",
            )

        if quantity <= 0:
            return OrderDecision(
                approved=False,
                submit_to_broker=False,
                reason="Order rejected: quantity must be greater than zero.",
            )

        if not self.safe_to_trade:
            return OrderDecision(
                approved=False,
                submit_to_broker=False,
                reason=(
                    "Order rejected: SAFE_TO_TRADE is false. "
                    "Paper orders remain blocked until you explicitly set "
                    "SAFE_TO_TRADE=true in .env."
                ),
            )

        if self.dry_run:
            return OrderDecision(
                approved=True,
                submit_to_broker=False,
                reason=(
                    "Dry-run is enabled. The order was validated but was not sent. "
                    "Set DRY_RUN=false only when you intentionally want to submit "
                    "a paper order."
                ),
            )

        return OrderDecision(
            approved=True,
            submit_to_broker=True,
            reason="Order approved for explicit paper submission.",
        )
