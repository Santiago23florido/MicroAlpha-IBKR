from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from broker.ib_client import IBClient
from broker.orders import calculate_marketable_limit_price
from config import Settings
from data.schemas import DecisionRecord, MarketSnapshot, TradeLifecycleEvent
from storage.trades import TradeStore


class PaperExecutor:
    def __init__(
        self,
        settings: Settings,
        client: IBClient,
        trade_store: TradeStore,
    ) -> None:
        self.settings = settings
        self.client = client
        self.trade_store = trade_store
        self.logger = client.logger

    def execute_decision(
        self,
        decision: DecisionRecord,
        market_snapshot: MarketSnapshot,
    ) -> dict[str, Any]:
        if decision.final_action not in {"long", "short", "close"}:
            return {
                "submitted": False,
                "reason": f"Decision action {decision.final_action} is not executable.",
            }

        timestamp = datetime.now(timezone.utc).isoformat()
        if decision.final_action == "close":
            result = self.client.close_position(
                symbol=decision.symbol,
                exchange=self.settings.ib_exchange,
                currency=self.settings.ib_currency,
            )
            self.trade_store.append_trade(
                TradeLifecycleEvent(
                    timestamp=timestamp,
                    symbol=decision.symbol,
                    event_type="close_submitted",
                    action="SELL" if float(result["position_before_close"]["position"]) > 0 else "BUY",
                    quantity=abs(float(result["position_before_close"]["position"])),
                    status=str(result.get("status", "Submitted")),
                    order_id=result.get("order_id"),
                    message="Manual/system close submitted.",
                    payload=result,
                )
            )
            self.trade_store.append_execution_event(
                timestamp=timestamp,
                event_type="close_execution",
                payload=result,
                symbol=decision.symbol,
                order_id=result.get("order_id"),
                status=result.get("status"),
            )
            return {"submitted": True, "broker_result": result}

        action = "BUY" if decision.final_action == "long" else "SELL"
        entry_price = calculate_marketable_limit_price(
            action=action,
            bid=market_snapshot.bid,
            ask=market_snapshot.ask,
            last=market_snapshot.last,
            buffer_bps=self.settings.trading.entry_limit_buffer_bps,
        )
        orb_state = decision.orb_state
        if action == "BUY":
            stop_loss = float(orb_state["range_low"])
            risk_distance = max(entry_price - stop_loss, 0.01)
            take_profit = round(entry_price + (1.5 * risk_distance), 2)
        else:
            stop_loss = float(orb_state["range_high"])
            risk_distance = max(stop_loss - entry_price, 0.01)
            take_profit = round(entry_price - (1.5 * risk_distance), 2)

        result = self.client.submit_bracket_order(
            symbol=decision.symbol,
            action=action,
            quantity=self.settings.trading.default_order_quantity,
            entry_limit_price=entry_price,
            take_profit_price=take_profit,
            stop_loss_price=stop_loss,
            exchange=self.settings.ib_exchange,
            currency=self.settings.ib_currency,
        )
        self.trade_store.append_trade(
            TradeLifecycleEvent(
                timestamp=timestamp,
                symbol=decision.symbol,
                event_type="entry_submitted",
                action=action,
                quantity=self.settings.trading.default_order_quantity,
                status=str(result.get("status", "Submitted")),
                order_id=result.get("parent_order_id"),
                parent_order_id=result.get("parent_order_id"),
                price=entry_price,
                message="ORB paper bracket order submitted.",
                payload=result,
            )
        )
        self.trade_store.append_execution_event(
            timestamp=timestamp,
            event_type="entry_execution",
            payload=result,
            symbol=decision.symbol,
            order_id=result.get("parent_order_id"),
            status=result.get("status"),
        )
        return {"submitted": True, "broker_result": result}
