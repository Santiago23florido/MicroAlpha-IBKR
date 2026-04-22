from __future__ import annotations

from typing import Any, Mapping

from strategy.alpha_models import (
    AlphaModel,
    LateSessionAlpha,
    LowEdgeNoTradeFilter,
    ORBContinuationAlpha,
    VWAPMeanReversionAlpha,
)


def build_alpha_registry(alpha_specific_thresholds: Mapping[str, Any] | None = None) -> dict[str, AlphaModel]:
    thresholds = dict(alpha_specific_thresholds or {})
    alphas: list[AlphaModel] = [
        LowEdgeNoTradeFilter(dict(thresholds.get("low_edge_no_trade_filter", {}))),
        ORBContinuationAlpha(dict(thresholds.get("orb_continuation", {}))),
        VWAPMeanReversionAlpha(dict(thresholds.get("vwap_mean_reversion", {}))),
        LateSessionAlpha(dict(thresholds.get("late_session_alpha", {}))),
    ]
    return {alpha.name: alpha for alpha in alphas}
