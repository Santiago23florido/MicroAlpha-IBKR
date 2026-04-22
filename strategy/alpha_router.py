from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import pandas as pd

from strategy.alpha_models import AlphaSignal
from strategy.alpha_registry import build_alpha_registry
from strategy.regime_detector import RegimeResult


@dataclass(frozen=True)
class AlphaRoutingResult:
    selected_alpha: str
    action: str
    estimated_edge_bps: float
    confidence: float
    blocked: bool
    reasons: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlphaRouter:
    def __init__(
        self,
        *,
        enabled_alphas: tuple[str, ...],
        priority_order: tuple[str, ...],
        min_net_edge_bps_by_alpha: Mapping[str, float] | None = None,
        no_trade_filters: tuple[str, ...] = (),
        alpha_specific_thresholds: Mapping[str, Any] | None = None,
    ) -> None:
        alpha_thresholds = dict(alpha_specific_thresholds or {})
        low_edge_params = dict(alpha_thresholds.get("low_edge_no_trade_filter", {}))
        if no_trade_filters:
            low_edge_params["blocked_regimes"] = tuple(no_trade_filters)
            alpha_thresholds["low_edge_no_trade_filter"] = low_edge_params
        self.registry = build_alpha_registry(alpha_thresholds)
        self.enabled_alphas = tuple(enabled_alphas)
        self.priority_order = tuple(priority_order)
        self.min_net_edge_bps_by_alpha = dict(min_net_edge_bps_by_alpha or {})
        self.no_trade_filters = tuple(no_trade_filters)

    def route(
        self,
        feature_row: pd.Series | dict[str, Any],
        prediction: dict[str, Any],
        regime: RegimeResult,
        *,
        expected_cost_bps: float,
    ) -> AlphaRoutingResult:
        row = feature_row if isinstance(feature_row, pd.Series) else pd.Series(feature_row)
        candidates: list[AlphaSignal] = []
        reasons: list[str] = [f"regime={regime.regime_name}"]

        for alpha_name in self.priority_order:
            if alpha_name not in self.enabled_alphas:
                continue
            alpha = self.registry.get(alpha_name)
            if alpha is None:
                reasons.append(f"unknown_alpha={alpha_name}")
                continue
            signal = alpha.evaluate(row, prediction, regime)
            candidates.append(signal)
            if alpha_name == "low_edge_no_trade_filter" and signal.action == "NO_TRADE" and signal.metadata.get("block_all"):
                return AlphaRoutingResult(
                    selected_alpha=alpha_name,
                    action="NO_TRADE",
                    estimated_edge_bps=0.0,
                    confidence=signal.confidence,
                    blocked=True,
                    reasons=[*reasons, *signal.reasons, "alpha_router_blocked_by_no_trade_filter"],
                    candidates=[candidate.to_dict() for candidate in candidates],
                    metadata={"block_source": alpha_name},
                )

        best: AlphaSignal | None = None
        for signal in candidates:
            if signal.action not in {"LONG", "SHORT"} or not signal.enabled:
                continue
            min_edge = float(self.min_net_edge_bps_by_alpha.get(signal.name, 0.0))
            net_edge = signal.estimated_edge_bps - expected_cost_bps
            if net_edge < min_edge:
                reasons.append(f"{signal.name}:net_edge_below_alpha_min:{net_edge:.4f}<{min_edge:.4f}")
                continue
            if best is None or _candidate_rank(signal, self.priority_order) < _candidate_rank(best, self.priority_order):
                best = signal

        if best is None:
            return AlphaRoutingResult(
                selected_alpha="none",
                action="NO_TRADE",
                estimated_edge_bps=0.0,
                confidence=0.0,
                blocked=True,
                reasons=[*reasons, "no_alpha_with_positive_tradeable_edge"],
                candidates=[candidate.to_dict() for candidate in candidates],
            )

        return AlphaRoutingResult(
            selected_alpha=best.name,
            action=best.action,
            estimated_edge_bps=best.estimated_edge_bps,
            confidence=best.confidence,
            blocked=False,
            reasons=[*reasons, *best.reasons, f"selected_alpha={best.name}"],
            candidates=[candidate.to_dict() for candidate in candidates],
            metadata={"alpha_score": best.score},
        )


def _candidate_rank(signal: AlphaSignal, priority_order: tuple[str, ...]) -> int:
    try:
        return priority_order.index(signal.name)
    except ValueError:
        return len(priority_order)
