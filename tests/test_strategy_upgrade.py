from __future__ import annotations

from datetime import time

import pandas as pd

from config.phase6 import DecisionConfig, SizingConfig, StrategyConfig
from strategy.alpha_router import AlphaRouter
from strategy.decision_engine import DecisionEngine
from strategy.regime_detector import RegimeDetector


def test_regime_detector_flags_high_cost_context() -> None:
    detector = RegimeDetector({"high_spread_bps": 12.0, "high_cost_bps": 18.0})
    regime = detector.detect(
        {
            "timestamp": "2026-04-21T14:00:00+00:00",
            "spread_bps": 15.0,
            "estimated_cost_bps": 20.0,
            "relative_volume": 1.0,
        }
    )

    assert regime.regime_name == "high_cost_regime"
    assert "high_cost" in regime.flags


def test_alpha_router_blocks_configured_no_trade_regime() -> None:
    router = AlphaRouter(
        enabled_alphas=("low_edge_no_trade_filter", "vwap_mean_reversion"),
        priority_order=("low_edge_no_trade_filter", "vwap_mean_reversion"),
        min_net_edge_bps_by_alpha={"vwap_mean_reversion": 1.0},
        no_trade_filters=("high_cost_regime",),
    )
    regime = RegimeDetector().detect({"spread_bps": 20.0, "estimated_cost_bps": 22.0})
    routing = router.route(
        pd.Series({"distance_to_vwap_bps": -25.0, "vwap_slope_bps": 0.0, "spread_bps": 20.0}),
        {"predicted_return_bps": 5.0},
        regime,
        expected_cost_bps=20.0,
    )

    assert routing.blocked is True
    assert routing.selected_alpha == "low_edge_no_trade_filter"
    assert routing.action == "NO_TRADE"


def test_decision_engine_selects_vwap_alpha_with_conservative_quantile_edge() -> None:
    engine = DecisionEngine(
        DecisionConfig(
            score_threshold=0.0,
            probability_threshold=0.5,
            predicted_return_min_bps=2.0,
            net_edge_min_bps=0.5,
            allow_long=True,
            allow_short=False,
            spread_max_bps=12.0,
            cost_max_bps=18.0,
            allowed_trading_start=time(9, 35),
            allowed_trading_end=time(15, 30),
            max_quantile_interval_width_bps=35.0,
            critical_feature_columns=("estimated_cost_bps", "spread_bps", "price_proxy"),
            explain_feature_columns=("distance_to_vwap_bps", "spread_bps", "estimated_cost_bps"),
        ),
        SizingConfig(
            default_position_size=1,
            max_position_size=3,
            min_confidence_for_full_size=0.72,
            min_size_confidence_floor=0.55,
        ),
        StrategyConfig(
            enabled_alphas=("low_edge_no_trade_filter", "vwap_mean_reversion"),
            alpha_priority_order=("low_edge_no_trade_filter", "vwap_mean_reversion"),
            alpha_router_mode="priority_conservative",
            regime_detection_enabled=True,
            conservative_decision_mode=True,
            min_net_edge_bps_by_alpha={"vwap_mean_reversion": 1.0},
            regime_thresholds={"mean_reversion_distance_bps": 12.0},
            no_trade_filters=("high_cost_regime", "low_liquidity_regime", "noisy_open", "low_edge_midday"),
            alpha_specific_thresholds={"vwap_mean_reversion": {"min_distance_to_vwap_bps": 8.0}},
        ),
    )
    decision = engine.decide(
        pd.Series(
            {
                "timestamp": "2026-04-21T14:30:00+00:00",
                "exchange_timestamp": "2026-04-21T14:30:00+00:00",
                "session_time": "10:30:00",
                "symbol": "SPY",
                "price_proxy": 100.0,
                "spread_bps": 1.0,
                "estimated_cost_bps": 0.5,
                "distance_to_vwap_bps": -25.0,
                "vwap_slope_bps": 0.1,
                "relative_volume": 1.0,
                "rolling_volatility_15": 3.0,
            }
        ),
        {
            "valid": True,
            "model_name": "test_model",
            "model_type": "quantile_regression",
            "run_id": "run_test",
            "feature_set_name": "hybrid_intraday",
            "target_mode": "quantile_regression",
            "score": 1.0,
            "probability": 0.8,
            "predicted_return_bps": 6.0,
            "predicted_quantiles": {"0.1": 3.0, "0.9": 12.0},
        },
    )

    assert decision.action == "LONG"
    assert decision.selected_alpha == "vwap_mean_reversion"
    assert decision.regime == "mean_reversion_regime"
    assert decision.conservative_return_bps == 3.0
    assert decision.net_edge_bps == 2.5
