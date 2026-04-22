from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from strategy.regime_detector import RegimeResult


@dataclass(frozen=True)
class AlphaSignal:
    name: str
    action: str
    estimated_edge_bps: float
    confidence: float
    score: float
    enabled: bool
    reasons: list[str] = field(default_factory=list)
    required_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlphaModel:
    name = "base"
    description = ""
    required_features: tuple[str, ...] = ()

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def evaluate(self, row: pd.Series, prediction: dict[str, Any], regime: RegimeResult) -> AlphaSignal:
        raise NotImplementedError

    def _missing_features(self, row: pd.Series) -> list[str]:
        return [column for column in self.required_features if column not in row.index or pd.isna(row.get(column))]

    def _threshold(self, key: str, default: float) -> float:
        try:
            return float(self.params.get(key, default))
        except (TypeError, ValueError):
            return float(default)


class LowEdgeNoTradeFilter(AlphaModel):
    name = "low_edge_no_trade_filter"
    description = "Blocks trading when regime/cost/time context suggests low expected edge."
    required_features = ("estimated_cost_bps", "spread_bps")

    def evaluate(self, row: pd.Series, prediction: dict[str, Any], regime: RegimeResult) -> AlphaSignal:
        flags = set(regime.flags)
        blocked_regimes = set(self.params.get("blocked_regimes", ("high_cost_regime", "low_liquidity_regime", "noisy_open", "low_edge_midday")))
        block = regime.regime_name in blocked_regimes
        reasons = [f"regime={regime.regime_name}", *[f"flag={flag}" for flag in sorted(flags)]]
        return AlphaSignal(
            name=self.name,
            action="NO_TRADE" if block else "PASS",
            estimated_edge_bps=0.0,
            confidence=regime.confidence if block else 0.0,
            score=regime.confidence if block else 0.0,
            enabled=True,
            reasons=reasons,
            required_features=list(self.required_features),
            metadata={"block_all": block},
        )


class ORBContinuationAlpha(AlphaModel):
    name = "orb_continuation"
    description = "Continuation alpha after the opening range only when breakout context is clean."
    required_features = ("orb_range_width_bps", "orb_relative_price_position", "relative_volume")

    def evaluate(self, row: pd.Series, prediction: dict[str, Any], regime: RegimeResult) -> AlphaSignal:
        missing = self._missing_features(row)
        if missing:
            return _disabled(self.name, self.description, missing)
        relative_volume = _num(row, "relative_volume", default=1.0)
        orb_position = _num(row, "orb_relative_price_position", default=0.5)
        predicted_return = _predicted_return(prediction)
        if regime.regime_name not in {"breakout_regime", "trend_day_candidate"}:
            return _no_trade(self.name, "regime_not_breakout_or_trend", self.required_features)
        if relative_volume < self._threshold("min_relative_volume", 0.9):
            return _no_trade(self.name, "relative_volume_too_low", self.required_features)
        action = "NO_TRADE"
        if orb_position >= 1.0 and predicted_return > 0:
            action = "LONG"
        elif orb_position <= 0.0 and predicted_return < 0:
            action = "SHORT"
        if action == "NO_TRADE":
            return _no_trade(self.name, "breakout_direction_not_confirmed_by_prediction", self.required_features)
        edge = abs(predicted_return) + max((relative_volume - 1.0) * 2.0, 0.0)
        return AlphaSignal(
            name=self.name,
            action=action,
            estimated_edge_bps=float(edge),
            confidence=min(max(abs(predicted_return) / 20.0, 0.0), 1.0),
            score=float(edge),
            enabled=True,
            reasons=["orb_continuation_confirmed", f"orb_relative_price_position={orb_position:.4f}", f"relative_volume={relative_volume:.4f}"],
            required_features=list(self.required_features),
        )


class VWAPMeanReversionAlpha(AlphaModel):
    name = "vwap_mean_reversion"
    description = "Intraday reversion alpha when price is extended from VWAP in range/mean-reversion regimes."
    required_features = ("distance_to_vwap_bps", "vwap_slope_bps", "spread_bps")

    def evaluate(self, row: pd.Series, prediction: dict[str, Any], regime: RegimeResult) -> AlphaSignal:
        missing = self._missing_features(row)
        if missing:
            return _disabled(self.name, self.description, missing)
        distance = _num(row, "distance_to_vwap_bps", default=0.0)
        vwap_slope = _num(row, "vwap_slope_bps", default=0.0)
        if regime.regime_name not in {"mean_reversion_regime", "range_day_candidate", "neutral_intraday"}:
            return _no_trade(self.name, "regime_not_mean_reversion_friendly", self.required_features)
        if abs(distance) < self._threshold("min_distance_to_vwap_bps", 8.0):
            return _no_trade(self.name, "distance_to_vwap_too_small", self.required_features)
        action = "SHORT" if distance > 0 else "LONG"
        edge = max(abs(distance) * 0.35 - abs(vwap_slope), 0.0)
        if edge <= 0:
            return _no_trade(self.name, "vwap_slope_offsets_reversion_edge", self.required_features)
        return AlphaSignal(
            name=self.name,
            action=action,
            estimated_edge_bps=float(edge),
            confidence=min(max(edge / 20.0, 0.0), 1.0),
            score=float(edge),
            enabled=True,
            reasons=[f"distance_to_vwap_bps={distance:.4f}", f"vwap_slope_bps={vwap_slope:.4f}", "vwap_reversion_edge_positive"],
            required_features=list(self.required_features),
        )


class LateSessionAlpha(AlphaModel):
    name = "late_session_alpha"
    description = "Selective late-session momentum/reversion alpha when data supports a clear directional edge."
    required_features = ("return_3_bps", "distance_to_vwap_bps", "relative_volume")

    def evaluate(self, row: pd.Series, prediction: dict[str, Any], regime: RegimeResult) -> AlphaSignal:
        missing = self._missing_features(row)
        if missing:
            return _disabled(self.name, self.description, missing)
        return_3 = _num(row, "return_3_bps", default=0.0)
        distance = _num(row, "distance_to_vwap_bps", default=0.0)
        relative_volume = _num(row, "relative_volume", default=1.0)
        predicted_return = _predicted_return(prediction)
        if relative_volume < self._threshold("min_relative_volume", 0.75):
            return _no_trade(self.name, "late_session_relative_volume_too_low", self.required_features)
        if abs(predicted_return) < self._threshold("min_predicted_return_bps", 2.0) or abs(return_3) < self._threshold("min_return_3_bps", 1.0):
            return _no_trade(self.name, "late_session_signal_too_weak", self.required_features)
        same_direction = (predicted_return > 0 and return_3 > 0) or (predicted_return < 0 and return_3 < 0)
        action = "LONG" if predicted_return > 0 else "SHORT"
        edge = abs(predicted_return) + (abs(return_3) * 0.25)
        if abs(distance) > self._threshold("max_distance_to_vwap_bps", 25.0) and same_direction:
            edge *= 0.5
        return AlphaSignal(
            name=self.name,
            action=action,
            estimated_edge_bps=float(edge),
            confidence=min(max(edge / 18.0, 0.0), 1.0),
            score=float(edge),
            enabled=True,
            reasons=[f"return_3_bps={return_3:.4f}", f"predicted_return_bps={predicted_return:.4f}", "late_session_directional_edge"],
            required_features=list(self.required_features),
            metadata={"same_direction_momentum": same_direction},
        )


def _predicted_return(prediction: dict[str, Any]) -> float:
    value = prediction.get("predicted_return_bps", prediction.get("expected_return_bps", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _num(row: pd.Series, column: str, *, default: float) -> float:
    if column not in row.index or pd.isna(row.get(column)):
        return default
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return default if pd.isna(value) else float(value)


def _disabled(name: str, description: str, missing: list[str]) -> AlphaSignal:
    return AlphaSignal(
        name=name,
        action="NO_TRADE",
        estimated_edge_bps=0.0,
        confidence=0.0,
        score=0.0,
        enabled=False,
        reasons=[f"missing_required_features={missing}", description],
        required_features=missing,
    )


def _no_trade(name: str, reason: str, required_features: tuple[str, ...]) -> AlphaSignal:
    return AlphaSignal(
        name=name,
        action="NO_TRADE",
        estimated_edge_bps=0.0,
        confidence=0.0,
        score=0.0,
        enabled=True,
        reasons=[reason],
        required_features=list(required_features),
    )
