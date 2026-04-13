from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from broker.ib_client import IBClient, IBClientError
from config import Settings
from data.feature_store import FeatureStore
from data.live_data import LiveDataService
from data.schemas import DecisionRecord
from engine.market_clock import MarketClock
from execution.executor import PaperExecutor
from execution.tracking import ExecutionTracker
from models.inference import InferenceEngine
from models.registry import ModelRegistry
from risk.risk_manager import RiskManager
from storage.decisions import DecisionStore
from storage.trades import TradeStore
from strategy.orb import OpeningRangeBreakoutStrategy
from strategy.signal_engine import SignalEngine
from features.microstructure_features import build_feature_snapshot


class SessionEngine:
    def __init__(
        self,
        settings: Settings,
        client: IBClient,
        risk_manager: RiskManager,
        feature_store: FeatureStore,
        decision_store: DecisionStore,
        trade_store: TradeStore,
        model_registry: ModelRegistry,
    ) -> None:
        self.settings = settings
        self.client = client
        self.risk_manager = risk_manager
        self.feature_store = feature_store
        self.decision_store = decision_store
        self.trade_store = trade_store
        self.model_registry = model_registry
        self.market_clock = MarketClock(settings.session)
        self.live_data = LiveDataService(client, settings)
        self.orb_strategy = OpeningRangeBreakoutStrategy(settings, self.market_clock)
        self.inference_engine = InferenceEngine(settings, model_registry)
        self.signal_engine = SignalEngine(settings, risk_manager)
        self.executor = PaperExecutor(settings, client, trade_store)
        self.tracker = ExecutionTracker(trade_store)
        self.logger = client.logger

    def test_connection(self) -> dict[str, Any]:
        try:
            self.client.connect()
            return {"connected": self.client.is_connected()}
        finally:
            self.client.disconnect()

    def run_cycle(
        self,
        *,
        execute_requested: bool = False,
    ) -> dict[str, Any]:
        try:
            self.logger.info("Starting session cycle.")
            self.client.connect()
            server_time = self.client.get_server_time()
            reference_time = datetime.fromtimestamp(server_time["epoch"], tz=timezone.utc)
            clock_state = self.market_clock.get_market_state(reference_time)
            market_status = clock_state.to_dict() | {"connected": self.client.is_connected()}
            account_summary = self.client.get_account_summary()
            positions = self.client.get_positions()
            market_snapshot = self.live_data.fetch_market_snapshot(self.settings.ib_symbol)
            bars = self.live_data.fetch_intraday_bars(self.settings.ib_symbol)
            orb_state = self.orb_strategy.evaluate(
                symbol=self.settings.ib_symbol,
                bars=bars,
                market_snapshot=market_snapshot,
                reference_time=reference_time,
            )
            prior_feature_rows = self.feature_store.get_recent_sequence(
                self.settings.ib_symbol,
                limit=self.settings.models.sequence_length,
            )
            feature_snapshot = build_feature_snapshot(
                market_snapshot=market_snapshot,
                orb_state=orb_state,
                feature_history=prior_feature_rows,
                source_mode=self.settings.trading.data_mode,
            )
            self.feature_store.append(feature_snapshot)
            sequence = self.feature_store.get_recent_sequence(
                self.settings.ib_symbol,
                limit=self.settings.models.sequence_length,
            )
            baseline_prediction = self.inference_engine.predict_baseline(feature_snapshot)
            deep_prediction = self.inference_engine.predict_deep(sequence)
            session_date = clock_state.exchange_time.date().isoformat()
            trades_today = self.trade_store.get_daily_trade_count(session_date)
            daily_realized_pnl = self.trade_store.get_daily_realized_pnl(session_date)
            position_age_minutes = self._estimate_position_age_minutes(self.settings.ib_symbol, reference_time)

            decision = self.signal_engine.build_decision(
                market_snapshot=market_snapshot,
                orb_state=orb_state,
                feature_snapshot=feature_snapshot,
                baseline_prediction=baseline_prediction,
                deep_prediction=deep_prediction,
                market_status=market_status,
                account_summary=account_summary,
                positions=positions,
                trades_today=trades_today,
                daily_realized_pnl=daily_realized_pnl,
                position_age_minutes=position_age_minutes,
                explicit_session_request=execute_requested,
            )
            decision_id = self.decision_store.save_decision(decision)
            self.decision_store.save_config_snapshot(
                {
                    "created_at": reference_time.isoformat(),
                    "settings": self.settings.as_dict(),
                }
            )

            execution_result = self._execution_result(decision, market_snapshot, execute_requested)
            payload = {
                "connection": {"connected": self.client.is_connected()},
                "server_time": server_time,
                "market_status": market_status,
                "account_summary": account_summary,
                "positions": positions,
                "market_snapshot": market_snapshot.to_dict(),
                "orb_state": orb_state.to_dict(),
                "feature_snapshot": feature_snapshot.to_dict(),
                "decision_id": decision_id,
                "decision": decision.to_dict(),
                "execution": execution_result,
                "active_models": {
                    "baseline": self.model_registry.get_active_model("baseline"),
                    "deep": self.model_registry.get_active_model("deep"),
                },
            }
            self.logger.info("Session cycle completed.")
            return payload
        finally:
            self.client.disconnect()
            self.logger.info("Session cycle shutdown complete.")

    def explain_latest_decision(self) -> dict[str, Any]:
        latest = self.decision_store.get_latest_decision()
        if latest is None:
            return {"status": "empty", "message": "No decision has been stored yet."}
        return latest

    def list_recent_decisions(self, limit: int = 25) -> list[dict[str, Any]]:
        return self.decision_store.list_recent_decisions(limit=limit)

    def list_recent_trades(self, limit: int = 25) -> list[dict[str, Any]]:
        return self.trade_store.list_recent_trades(limit=limit)

    def list_recent_execution_events(self, limit: int = 25) -> list[dict[str, Any]]:
        return self.trade_store.list_recent_execution_events(limit=limit)

    def _execution_result(
        self,
        decision: DecisionRecord,
        market_snapshot,
        execute_requested: bool,
    ) -> dict[str, Any]:
        if decision.final_action not in {"long", "short", "close"}:
            return {
                "requested": execute_requested,
                "submitted": False,
                "reason": "Decision is not actionable.",
            }
        if not execute_requested:
            return {
                "requested": False,
                "submitted": False,
                "reason": "Actionable signal detected but explicit paper execution was not requested.",
            }
        if not decision.execution_allowed:
            return {
                "requested": True,
                "submitted": False,
                "reason": "Actionable signal exists but execution gates remain blocked.",
            }
        try:
            result = self.executor.execute_decision(decision, market_snapshot)
            return {
                "requested": True,
                "submitted": result.get("submitted", False),
                "broker_result": result.get("broker_result"),
            }
        except IBClientError as exc:
            return {
                "requested": True,
                "submitted": False,
                "reason": str(exc),
            }

    def _estimate_position_age_minutes(
        self,
        symbol: str,
        reference_time: datetime,
    ) -> float | None:
        trades = self.trade_store.list_recent_trades(limit=200)
        for trade in trades:
            if trade.get("symbol") != symbol:
                continue
            if trade.get("event_type") not in {"entry_submitted", "entry_filled"}:
                continue
            try:
                entered_at = datetime.fromisoformat(str(trade["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            if entered_at.tzinfo is None:
                entered_at = entered_at.replace(tzinfo=timezone.utc)
            return max((reference_time - entered_at.astimezone(timezone.utc)).total_seconds() / 60.0, 0.0)
        return None
