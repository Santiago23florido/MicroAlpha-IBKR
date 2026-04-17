from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from typing import Any

import numpy as np
import pandas as pd

from config.phase6 import DecisionConfig, SizingConfig
from strategy.explainability import build_prediction_reasons, build_threshold_reason


@dataclass(frozen=True)
class DecisionResult:
    action: str
    model_name: str
    model_type: str
    run_id: str
    feature_set_name: str
    target_mode: str
    confidence: float
    score: float | None
    probability: float | None
    expected_return_bps: float | None
    expected_cost_bps: float
    net_edge_bps: float | None
    size_suggestion: int
    blocked_by_risk: bool
    reasons: list[str] = field(default_factory=list)
    timestamp: str | None = None
    symbol: str | None = None
    class_label: int | float | str | None = None
    predicted_quantiles: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionEngine:
    def __init__(self, decision_config: DecisionConfig, sizing_config: SizingConfig) -> None:
        self.decision_config = decision_config
        self.sizing_config = sizing_config

    def decide(self, feature_row: pd.Series | dict[str, Any], prediction: dict[str, Any]) -> DecisionResult:
        row = _as_series(feature_row)
        timestamp = _resolve_timestamp(row)
        symbol = str(row.get("symbol", "")).upper() or None
        reasons = build_prediction_reasons(prediction)

        missing_critical = [
            column
            for column in self.decision_config.critical_feature_columns
            if column not in row.index or pd.isna(row.get(column))
        ]
        if missing_critical:
            reasons.append(build_threshold_reason("critical_features", False, f"missing={missing_critical}"))
            return self._build_no_trade(
                prediction,
                row,
                reasons=reasons,
                expected_cost_bps=_resolve_expected_cost_bps(row),
                timestamp=timestamp,
                symbol=symbol,
            )

        if not bool(prediction.get("valid", True)):
            reasons.append(build_threshold_reason("prediction_valid", False, "model returned invalid output"))
            return self._build_no_trade(
                prediction,
                row,
                reasons=reasons,
                expected_cost_bps=_resolve_expected_cost_bps(row),
                timestamp=timestamp,
                symbol=symbol,
            )

        within_window = _is_within_allowed_window(
            row,
            timestamp,
            self.decision_config.allowed_trading_start,
            self.decision_config.allowed_trading_end,
        )
        reasons.append(
            build_threshold_reason(
                "trading_window",
                within_window,
                f"{self.decision_config.allowed_trading_start.isoformat(timespec='minutes')}..{self.decision_config.allowed_trading_end.isoformat(timespec='minutes')}",
            )
        )
        if not within_window:
            return self._build_no_trade(
                prediction,
                row,
                reasons=reasons,
                expected_cost_bps=_resolve_expected_cost_bps(row),
                timestamp=timestamp,
                symbol=symbol,
            )

        spread_bps = _resolve_numeric(row, "spread_bps")
        if spread_bps is not None:
            spread_ok = spread_bps <= self.decision_config.spread_max_bps
            reasons.append(
                build_threshold_reason(
                    "spread_limit",
                    spread_ok,
                    f"spread_bps={spread_bps:.4f} max={self.decision_config.spread_max_bps:.4f}",
                )
            )
            if not spread_ok:
                return self._build_no_trade(
                    prediction,
                    row,
                    reasons=reasons,
                    expected_cost_bps=_resolve_expected_cost_bps(row),
                    timestamp=timestamp,
                    symbol=symbol,
                )

        expected_cost_bps = _resolve_expected_cost_bps(row)
        cost_ok = expected_cost_bps <= self.decision_config.cost_max_bps
        reasons.append(
            build_threshold_reason(
                "estimated_cost_limit",
                cost_ok,
                f"estimated_cost_bps={expected_cost_bps:.4f} max={self.decision_config.cost_max_bps:.4f}",
            )
        )
        if not cost_ok:
            return self._build_no_trade(
                prediction,
                row,
                reasons=reasons,
                expected_cost_bps=expected_cost_bps,
                timestamp=timestamp,
                symbol=symbol,
            )

        candidate = self._candidate_from_prediction(prediction, reasons)
        expected_return_bps = candidate["expected_return_bps"]
        confidence = candidate["confidence"]
        action = candidate["action"]
        if action == "NO_TRADE":
            return self._build_no_trade(
                prediction,
                row,
                reasons=candidate["reasons"],
                expected_cost_bps=expected_cost_bps,
                timestamp=timestamp,
                symbol=symbol,
                confidence=confidence,
            )

        gross_edge_bps = abs(float(expected_return_bps))
        net_edge_bps = gross_edge_bps - expected_cost_bps
        edge_ok = net_edge_bps >= self.decision_config.net_edge_min_bps
        candidate["reasons"].append(
            build_threshold_reason(
                "net_edge",
                edge_ok,
                f"net_edge_bps={net_edge_bps:.4f} min={self.decision_config.net_edge_min_bps:.4f}",
            )
        )
        if not edge_ok:
            return self._build_no_trade(
                prediction,
                row,
                reasons=candidate["reasons"],
                expected_cost_bps=expected_cost_bps,
                timestamp=timestamp,
                symbol=symbol,
                expected_return_bps=expected_return_bps,
                net_edge_bps=net_edge_bps,
                confidence=confidence,
            )

        size_suggestion = _size_suggestion(
            sizing=self.sizing_config,
            confidence=confidence,
            net_edge_bps=net_edge_bps,
            min_edge_bps=max(self.decision_config.net_edge_min_bps, 0.01),
        )
        return DecisionResult(
            action=action,
            model_name=str(prediction.get("model_name")),
            model_type=str(prediction.get("model_type")),
            run_id=str(prediction.get("run_id")),
            feature_set_name=str(prediction.get("feature_set_name")),
            target_mode=str(prediction.get("target_mode")),
            confidence=confidence,
            score=_optional_float(prediction.get("score")),
            probability=_optional_float(prediction.get("probability")),
            expected_return_bps=_optional_float(expected_return_bps),
            expected_cost_bps=float(expected_cost_bps),
            net_edge_bps=float(net_edge_bps),
            size_suggestion=size_suggestion,
            blocked_by_risk=False,
            reasons=candidate["reasons"],
            timestamp=timestamp.isoformat() if timestamp is not None else None,
            symbol=symbol,
            class_label=prediction.get("class_label"),
            predicted_quantiles={str(key): float(value) for key, value in (prediction.get("predicted_quantiles") or {}).items()},
            metadata={
                "explain_features": {
                    column: row.get(column)
                    for column in self.decision_config.explain_feature_columns
                    if column in row.index
                },
                "raw_prediction_metadata": prediction.get("metadata", {}),
            },
        )

    def _candidate_from_prediction(self, prediction: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
        target_mode = str(prediction.get("target_mode"))
        probability = _optional_float(prediction.get("probability")) or 0.0
        score = _optional_float(prediction.get("score")) or 0.0
        predicted_return_bps = _optional_float(prediction.get("predicted_return_bps")) or 0.0
        quantiles = {str(key): float(value) for key, value in (prediction.get("predicted_quantiles") or {}).items()}
        action_bias = str(prediction.get("action_bias") or "NO_TRADE").upper()

        if target_mode == "classification_binary":
            probability_ok = probability >= self.decision_config.probability_threshold
            reasons.append(
                build_threshold_reason(
                    "probability_threshold",
                    probability_ok,
                    f"probability={probability:.4f} min={self.decision_config.probability_threshold:.4f}",
                )
            )
            if action_bias != "LONG":
                reasons.append("binary_classifier_did_not_signal_positive_class")
                return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": probability, "reasons": reasons}
            if not self.decision_config.allow_long:
                reasons.append("long_disabled_in_decision_config")
                return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": probability, "reasons": reasons}
            if not probability_ok:
                return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": probability, "reasons": reasons}
            return {"action": "LONG", "expected_return_bps": predicted_return_bps, "confidence": probability, "reasons": reasons}

        if target_mode == "quantile_regression":
            interval_width = None
            if "0.1" in quantiles and "0.9" in quantiles:
                interval_width = quantiles["0.9"] - quantiles["0.1"]
            elif len(quantiles) >= 2:
                ordered = sorted(quantiles.values())
                interval_width = ordered[-1] - ordered[0]
            if interval_width is not None:
                interval_ok = interval_width <= self.decision_config.max_quantile_interval_width_bps
                reasons.append(
                    build_threshold_reason(
                        "quantile_interval_width",
                        interval_ok,
                        f"interval_width_bps={interval_width:.4f} max={self.decision_config.max_quantile_interval_width_bps:.4f}",
                    )
                )
                if not interval_ok:
                    return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": _confidence_from_return(predicted_return_bps, self.decision_config.predicted_return_min_bps), "reasons": reasons}

        predicted_return_ok = abs(predicted_return_bps) >= self.decision_config.predicted_return_min_bps
        reasons.append(
            build_threshold_reason(
                "predicted_return_threshold",
                predicted_return_ok,
                f"predicted_return_bps={predicted_return_bps:.4f} min_abs={self.decision_config.predicted_return_min_bps:.4f}",
            )
        )
        if not predicted_return_ok:
            return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": _confidence_from_return(predicted_return_bps, self.decision_config.predicted_return_min_bps), "reasons": reasons}

        score_ok = abs(score) >= self.decision_config.score_threshold
        reasons.append(
            build_threshold_reason(
                "score_threshold",
                score_ok,
                f"score={score:.4f} min_abs={self.decision_config.score_threshold:.4f}",
            )
        )
        if not score_ok:
            return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": _confidence_from_return(predicted_return_bps, self.decision_config.predicted_return_min_bps), "reasons": reasons}

        if predicted_return_bps > 0:
            if not self.decision_config.allow_long:
                reasons.append("long_disabled_in_decision_config")
                return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": _confidence_from_return(predicted_return_bps, self.decision_config.predicted_return_min_bps), "reasons": reasons}
            return {"action": "LONG", "expected_return_bps": predicted_return_bps, "confidence": max(probability, _confidence_from_return(predicted_return_bps, self.decision_config.predicted_return_min_bps)), "reasons": reasons}
        if predicted_return_bps < 0:
            if not self.decision_config.allow_short:
                reasons.append("short_disabled_in_decision_config")
                return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": _confidence_from_return(predicted_return_bps, self.decision_config.predicted_return_min_bps), "reasons": reasons}
            return {"action": "SHORT", "expected_return_bps": predicted_return_bps, "confidence": max(probability, _confidence_from_return(predicted_return_bps, self.decision_config.predicted_return_min_bps)), "reasons": reasons}
        return {"action": "NO_TRADE", "expected_return_bps": predicted_return_bps, "confidence": 0.0, "reasons": reasons}

    def _build_no_trade(
        self,
        prediction: dict[str, Any],
        row: pd.Series,
        *,
        reasons: list[str],
        expected_cost_bps: float,
        timestamp: datetime | None,
        symbol: str | None,
        expected_return_bps: float | None = None,
        net_edge_bps: float | None = None,
        confidence: float | None = None,
    ) -> DecisionResult:
        return DecisionResult(
            action="NO_TRADE",
            model_name=str(prediction.get("model_name")),
            model_type=str(prediction.get("model_type")),
            run_id=str(prediction.get("run_id")),
            feature_set_name=str(prediction.get("feature_set_name")),
            target_mode=str(prediction.get("target_mode")),
            confidence=float(confidence or 0.0),
            score=_optional_float(prediction.get("score")),
            probability=_optional_float(prediction.get("probability")),
            expected_return_bps=_optional_float(expected_return_bps if expected_return_bps is not None else prediction.get("predicted_return_bps")),
            expected_cost_bps=float(expected_cost_bps),
            net_edge_bps=_optional_float(net_edge_bps),
            size_suggestion=0,
            blocked_by_risk=False,
            reasons=reasons,
            timestamp=timestamp.isoformat() if timestamp is not None else None,
            symbol=symbol,
            class_label=prediction.get("class_label"),
            predicted_quantiles={str(key): float(value) for key, value in (prediction.get("predicted_quantiles") or {}).items()},
            metadata={
                "explain_features": {
                    column: row.get(column)
                    for column in self.decision_config.explain_feature_columns
                    if column in row.index
                },
                "raw_prediction_metadata": prediction.get("metadata", {}),
            },
        )


def _as_series(feature_row: pd.Series | dict[str, Any]) -> pd.Series:
    return feature_row if isinstance(feature_row, pd.Series) else pd.Series(feature_row)


def _resolve_timestamp(row: pd.Series) -> datetime | None:
    for column in ("exchange_timestamp", "timestamp", "collected_at"):
        if column in row.index and pd.notna(row.get(column)):
            value = pd.to_datetime(row.get(column), utc=True, errors="coerce")
            if pd.notna(value):
                return value.to_pydatetime()
    return None


def _resolve_expected_cost_bps(row: pd.Series) -> float:
    for column in ("estimated_cost_bps", "spread_proxy_bps", "spread_bps"):
        value = _resolve_numeric(row, column)
        if value is not None:
            return float(max(value, 0.0))
    return 0.0


def _resolve_numeric(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row.get(column)):
        return None
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _is_within_allowed_window(row: pd.Series, timestamp: datetime | None, start: time, end: time) -> bool:
    current_time = None
    if "session_time" in row.index and pd.notna(row.get("session_time")):
        raw_value = row.get("session_time")
        if isinstance(raw_value, time):
            current_time = raw_value.replace(second=0, microsecond=0)
        else:
            parsed = pd.to_datetime(str(raw_value), errors="coerce")
            if pd.notna(parsed):
                current_time = parsed.time().replace(second=0, microsecond=0)
    if current_time is None and timestamp is not None:
        current_time = timestamp.time().replace(second=0, microsecond=0)
    if current_time is None:
        return False
    return start <= current_time <= end


def _confidence_from_return(predicted_return_bps: float, min_return_bps: float) -> float:
    scale = max(min_return_bps, 1.0)
    return float(min(max(abs(predicted_return_bps) / (scale * 2.0), 0.0), 1.0))


def _size_suggestion(
    *,
    sizing: SizingConfig,
    confidence: float,
    net_edge_bps: float,
    min_edge_bps: float,
) -> int:
    if confidence < sizing.min_size_confidence_floor or net_edge_bps <= 0:
        return 0
    confidence_multiplier = min(max(confidence / max(sizing.min_confidence_for_full_size, 1e-6), 0.0), 1.0)
    edge_multiplier = min(max(net_edge_bps / max(min_edge_bps * 2.0, 1e-6), 0.0), 1.0)
    scaled = sizing.default_position_size * max(confidence_multiplier, 0.25) * max(edge_multiplier, 0.25)
    return int(min(max(round(scaled), 1), sizing.max_position_size))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None
