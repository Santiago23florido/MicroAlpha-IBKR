from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from config.phase6 import RiskEngineConfig
from strategy.explainability import build_risk_reasons


@dataclass
class OperationalRiskState:
    session_date: str | None = None
    trades_in_session: int = 0
    daily_realized_pnl_bps: float = 0.0
    symbol_realized_pnl_bps: dict[str, float] = field(default_factory=dict)
    last_loss_timestamp_by_symbol: dict[str, str] = field(default_factory=dict)
    kill_switch_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskEvaluation:
    allowed: bool
    blocked_by_risk: bool
    checks: dict[str, bool]
    failures: list[str]
    reasons: list[str]
    state_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperationalRiskEngine:
    def __init__(self, config: RiskEngineConfig) -> None:
        self.config = config

    def evaluate(
        self,
        decision: dict[str, Any],
        feature_row: pd.Series | dict[str, Any],
        prediction: dict[str, Any],
        state: OperationalRiskState,
    ) -> RiskEvaluation:
        row = feature_row if isinstance(feature_row, pd.Series) else pd.Series(feature_row)
        state = self._roll_session_if_needed(row, state)
        if not self.config.enabled:
            return RiskEvaluation(
                allowed=True,
                blocked_by_risk=False,
                checks={"risk_enabled": False},
                failures=[],
                reasons=[],
                state_snapshot=state.to_dict(),
            )

        timestamp = _resolve_timestamp(row)
        symbol = str(decision.get("symbol") or row.get("symbol") or "").upper()
        spread_bps = _coerce_numeric(row.get("spread_bps"))
        estimated_cost_bps = _coerce_numeric(row.get("estimated_cost_bps"))
        prediction_valid = bool(prediction.get("valid", True))

        checks = {
            "kill_switch_clear": state.kill_switch_reason is None,
            "model_output_valid": prediction_valid,
            "max_trades_per_session_ok": state.trades_in_session < self.config.max_trades_per_session,
            "daily_loss_limit_ok": state.daily_realized_pnl_bps > -abs(self.config.daily_loss_limit_bps),
            "symbol_loss_limit_ok": state.symbol_realized_pnl_bps.get(symbol, 0.0) > -abs(self.config.symbol_loss_limit_bps),
            "spread_ok": spread_bps is None or spread_bps <= self.config.max_spread_bps,
            "estimated_cost_ok": estimated_cost_bps is None or estimated_cost_bps <= self.config.max_estimated_cost_bps,
            "cooldown_ok": self._cooldown_ok(timestamp, symbol, state),
        }
        failures = [name for name, passed in checks.items() if not passed]

        if not prediction_valid and self.config.kill_switch_on_invalid_model and state.kill_switch_reason is None:
            state.kill_switch_reason = "invalid_model_output"
            failures.append("kill_switch_triggered_invalid_model_output")
        if _prediction_anomalous(prediction) and self.config.kill_switch_on_anomalous_prediction and state.kill_switch_reason is None:
            state.kill_switch_reason = "anomalous_prediction_output"
            failures.append("kill_switch_triggered_anomalous_prediction_output")

        blocked_by_risk = bool(failures and decision.get("action") != "NO_TRADE")
        return RiskEvaluation(
            allowed=not blocked_by_risk,
            blocked_by_risk=blocked_by_risk,
            checks=checks,
            failures=failures,
            reasons=build_risk_reasons(failures),
            state_snapshot=state.to_dict(),
        )

    def apply(self, decision: dict[str, Any], evaluation: RiskEvaluation) -> dict[str, Any]:
        if not evaluation.blocked_by_risk:
            return {**decision, "risk_checks": evaluation.checks, "risk_failures": evaluation.failures}
        updated_reasons = list(decision.get("reasons", [])) + evaluation.reasons
        return {
            **decision,
            "action": "NO_TRADE",
            "size_suggestion": 0,
            "blocked_by_risk": True,
            "reasons": updated_reasons,
            "risk_checks": evaluation.checks,
            "risk_failures": evaluation.failures,
        }

    def record_post_decision(
        self,
        state: OperationalRiskState,
        decision: dict[str, Any],
        *,
        realized_net_return_bps: float | None = None,
    ) -> OperationalRiskState:
        if decision.get("action") == "NO_TRADE" or int(decision.get("size_suggestion", 0) or 0) <= 0:
            return state

        symbol = str(decision.get("symbol") or "").upper()
        state.trades_in_session += 1
        if realized_net_return_bps is None:
            return state

        size_multiplier = float(decision.get("size_suggestion", 1) or 1)
        realized = float(realized_net_return_bps) * size_multiplier
        state.daily_realized_pnl_bps += realized
        state.symbol_realized_pnl_bps[symbol] = state.symbol_realized_pnl_bps.get(symbol, 0.0) + realized
        if realized < 0 and decision.get("timestamp"):
            state.last_loss_timestamp_by_symbol[symbol] = str(decision["timestamp"])
        return state

    def _roll_session_if_needed(self, row: pd.Series, state: OperationalRiskState) -> OperationalRiskState:
        session_date = row.get("session_date")
        if session_date and state.session_date != session_date:
            state.session_date = str(session_date)
            state.trades_in_session = 0
            state.daily_realized_pnl_bps = 0.0
            state.symbol_realized_pnl_bps = {}
            state.last_loss_timestamp_by_symbol = {}
            state.kill_switch_reason = None
        return state

    def _cooldown_ok(self, timestamp: datetime | None, symbol: str, state: OperationalRiskState) -> bool:
        if timestamp is None or not symbol:
            return True
        last_loss = state.last_loss_timestamp_by_symbol.get(symbol)
        if not last_loss:
            return True
        parsed = pd.to_datetime(last_loss, utc=True, errors="coerce")
        if pd.isna(parsed):
            return True
        return timestamp >= parsed.to_pydatetime() + timedelta(minutes=self.config.cooldown_minutes)


def _resolve_timestamp(row: pd.Series) -> datetime | None:
    for column in ("exchange_timestamp", "timestamp", "collected_at"):
        if column in row.index and pd.notna(row.get(column)):
            value = pd.to_datetime(row.get(column), utc=True, errors="coerce")
            if pd.notna(value):
                return value.to_pydatetime()
    return None


def _coerce_numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prediction_anomalous(prediction: dict[str, Any]) -> bool:
    for key in ("score", "probability", "predicted_return_bps"):
        value = prediction.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return True
        if not pd.notna(numeric):
            return True
    return False
