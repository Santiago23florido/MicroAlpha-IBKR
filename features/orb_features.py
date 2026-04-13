from __future__ import annotations

from typing import Any

from data.schemas import ORBState


def build_orb_feature_map(orb_state: ORBState, current_price: float | None) -> dict[str, float | None]:
    if current_price is None:
        current_price = orb_state.breakout_price
    range_mid = orb_state.range_mid
    range_width = orb_state.range_width

    def _bps(delta: float | None, denominator: float | None) -> float | None:
        if delta is None or denominator in {None, 0}:
            return None
        return (delta / denominator) * 10000.0

    distance_from_high = None
    distance_from_low = None
    distance_from_mid = None
    if current_price is not None:
        if orb_state.range_high is not None:
            distance_from_high = current_price - orb_state.range_high
        if orb_state.range_low is not None:
            distance_from_low = current_price - orb_state.range_low
        if range_mid is not None:
            distance_from_mid = current_price - range_mid

    return {
        "orb_range_high": orb_state.range_high,
        "orb_range_low": orb_state.range_low,
        "orb_range_mid": range_mid,
        "orb_range_width": range_width,
        "orb_range_width_bps": _bps(range_width, range_mid),
        "price_vs_orb_high_bps": _bps(distance_from_high, orb_state.range_high),
        "price_vs_orb_low_bps": _bps(distance_from_low, orb_state.range_low),
        "price_vs_orb_mid_bps": _bps(distance_from_mid, range_mid),
        "breakout_distance": orb_state.breakout_distance,
        "breakout_distance_bps": _bps(orb_state.breakout_distance, range_mid),
        "is_primary_session": 1.0 if orb_state.session_window == "primary" else 0.0,
        "is_secondary_session": 1.0 if orb_state.session_window == "secondary" else 0.0,
        "orb_range_complete": 1.0 if orb_state.range_complete else 0.0,
        "orb_trading_allowed": 1.0 if orb_state.trading_allowed else 0.0,
    }
