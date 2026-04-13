from __future__ import annotations

from typing import Any

from config import Settings
from data.schemas import DecisionRecord, FeatureSnapshot, MarketSnapshot, ModelPrediction, ORBState
from risk.limits import ExecutionGateContext, StrategyRiskContext
from risk.risk_manager import RiskManager
from strategy.decision_explainer import DecisionExplainer


class SignalEngine:
    def __init__(self, settings: Settings, risk_manager: RiskManager) -> None:
        self.settings = settings
        self.risk_manager = risk_manager
        self.explainer = DecisionExplainer()

    def build_decision(
        self,
        *,
        market_snapshot: MarketSnapshot,
        orb_state: ORBState,
        feature_snapshot: FeatureSnapshot,
        baseline_prediction: ModelPrediction,
        deep_prediction: ModelPrediction,
        market_status: dict[str, Any],
        account_summary: list[dict[str, str]],
        positions: list[dict[str, Any]],
        trades_today: int,
        daily_realized_pnl: float,
        position_age_minutes: float | None,
        explicit_session_request: bool,
    ) -> DecisionRecord:
        symbol = market_snapshot.symbol.upper()
        symbol_position = _get_symbol_position(positions, symbol)
        open_positions_count = sum(1 for row in positions if float(row.get("position", 0)) != 0)
        direction = orb_state.breakout_direction

        threshold_checks = {
            "orb_condition_valid": orb_state.range_complete and direction is not None,
            "orb_trading_allowed": orb_state.trading_allowed,
            "baseline_model_available": baseline_prediction.eligible,
            "deep_model_available": deep_prediction.eligible,
        }
        baseline_candidate_probability = _candidate_probability(baseline_prediction, direction)
        deep_candidate_probability = _candidate_probability(deep_prediction, direction)
        threshold_checks["baseline_probability_pass"] = baseline_candidate_probability >= self.settings.models.model_prob_threshold
        threshold_checks["deep_probability_pass"] = deep_candidate_probability >= self.settings.models.model_prob_threshold
        threshold_checks["baseline_direction_match"] = baseline_prediction.direction == direction
        threshold_checks["deep_direction_match"] = deep_prediction.direction == direction

        final_probability = None
        final_return = None
        if direction is not None and baseline_prediction.eligible and deep_prediction.eligible:
            final_probability = (
                self.settings.models.baseline_weight * baseline_candidate_probability
                + self.settings.models.deep_weight * deep_candidate_probability
            )
            combined_return = (
                self.settings.models.baseline_weight * baseline_prediction.predicted_return_bps
                + self.settings.models.deep_weight * deep_prediction.predicted_return_bps
            )
            final_return = combined_return if direction == "long" else -combined_return

        expected_edge = final_return
        estimated_cost = feature_snapshot.estimated_cost_bps + self.settings.trading.cost_buffer_bps
        threshold_checks["expected_edge_exceeds_cost"] = (
            expected_edge is not None and expected_edge > estimated_cost
        )

        hold_close_required = (
            symbol_position != 0
            and position_age_minutes is not None
            and position_age_minutes >= self.settings.trading.max_hold_minutes
        )
        failure_overrides = []
        if direction == "short" and not self.settings.trading.allow_shorts:
            failure_overrides.append("shorts_disabled")

        net_liquidation = _extract_net_liquidation(account_summary)
        risk_result = self.risk_manager.evaluate_signal_risk(
            StrategyRiskContext(
                symbol=symbol,
                current_date=market_status["exchange_time"][:10],
                connection_healthy=bool(market_status.get("connected", True)),
                market_is_open=bool(market_status.get("is_market_open", market_status.get("is_open", False))),
                trading_window_allowed=orb_state.trading_allowed,
                flatten_required=orb_state.flatten_required,
                spread_bps=feature_snapshot.feature_values.get("spread_bps"),
                open_positions_count=open_positions_count,
                symbol_position=symbol_position,
                trades_today=trades_today,
                daily_realized_pnl=daily_realized_pnl,
                net_liquidation=net_liquidation,
                action="close" if symbol_position != 0 and (orb_state.flatten_required or hold_close_required) else direction,
                max_trades_per_day=self.settings.risk.max_trades_per_day,
                max_daily_loss_pct=self.settings.risk.max_daily_loss_pct,
                max_open_positions=self.settings.risk.max_open_positions,
                max_spread_bps=self.settings.trading.max_spread_bps,
                failure_overrides=failure_overrides,
            )
        )
        execution_gates = self.risk_manager.evaluate_execution_gate(
            ExecutionGateContext(
                explicit_session_request=explicit_session_request,
                session_execution_enabled=self.settings.trading.allow_session_execution,
                safe_to_trade=self.settings.trading.safe_to_trade,
                dry_run=self.settings.trading.dry_run,
            )
        )

        final_action = "no_trade"
        if symbol_position != 0 and orb_state.flatten_required:
            final_action = "close"
        elif hold_close_required:
            final_action = "close"
        elif not threshold_checks["orb_condition_valid"]:
            final_action = "no_trade"
        elif not orb_state.trading_allowed:
            final_action = "reject"
        elif not all(
            threshold_checks[key]
            for key in [
                "baseline_model_available",
                "deep_model_available",
                "baseline_probability_pass",
                "deep_probability_pass",
                "baseline_direction_match",
                "deep_direction_match",
                "expected_edge_exceeds_cost",
            ]
        ):
            final_action = "reject"
        elif not risk_result.passed:
            final_action = "reject"
        elif direction == "long":
            final_action = "long"
        elif direction == "short":
            final_action = "short"

        execution_allowed = final_action in {"long", "short", "close"} and all(execution_gates.values())
        explanation_text, structured = self.explainer.build(
            orb_state=orb_state,
            baseline_prediction=baseline_prediction,
            deep_prediction=deep_prediction,
            threshold_checks=threshold_checks,
            risk_result=risk_result,
            execution_gates=execution_gates,
            final_action=final_action,
            expected_edge=expected_edge,
            estimated_cost=estimated_cost,
        )
        return DecisionRecord(
            symbol=symbol,
            timestamp=market_snapshot.timestamp,
            orb_state=orb_state.to_dict(),
            direction=direction,
            candidate_trigger_reason=orb_state.candidate_reason,
            baseline_model_output=baseline_prediction.to_dict(),
            deep_model_output=deep_prediction.to_dict(),
            selected_final_score=final_probability,
            expected_edge=expected_edge,
            estimated_cost=estimated_cost,
            risk_checks=risk_result.checks,
            risk_passed=risk_result.passed,
            final_action=final_action,
            explanation_text=explanation_text,
            structured_explanation_data=structured,
            feature_values=feature_snapshot.feature_values,
            threshold_checks=threshold_checks,
            market_status=market_status,
            execution_allowed=execution_allowed,
            model_confirmation_passed=all(
                threshold_checks[key]
                for key in [
                    "baseline_model_available",
                    "deep_model_available",
                    "baseline_probability_pass",
                    "deep_probability_pass",
                    "baseline_direction_match",
                    "deep_direction_match",
                ]
            ),
            cost_check_passed=threshold_checks["expected_edge_exceeds_cost"],
            orb_condition_passed=threshold_checks["orb_condition_valid"],
            metadata={
                "baseline_candidate_probability": baseline_candidate_probability,
                "deep_candidate_probability": deep_candidate_probability,
                "position_age_minutes": position_age_minutes,
                "daily_realized_pnl": daily_realized_pnl,
                "trades_today": trades_today,
            },
        )


def _candidate_probability(prediction: ModelPrediction, direction: str | None) -> float:
    if direction == "long":
        return prediction.probability_up
    if direction == "short":
        return prediction.probability_down
    return 0.0


def _get_symbol_position(positions: list[dict[str, Any]], symbol: str) -> float:
    for row in positions:
        if str(row.get("symbol", "")).upper() == symbol.upper():
            return float(row.get("position", 0))
    return 0.0


def _extract_net_liquidation(account_summary: list[dict[str, str]]) -> float | None:
    for row in account_summary:
        if row.get("tag") == "NetLiquidation":
            try:
                return float(row["value"])
            except (KeyError, TypeError, ValueError):
                return None
    return None
