from __future__ import annotations

from math import isnan

from data.schemas import FeatureSnapshot, MarketSnapshot, ORBState
from features.orb_features import build_orb_feature_map


def _valid(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        if isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def _bps(delta: float | None, base: float | None) -> float | None:
    if delta is None or base in {None, 0}:
        return None
    return (delta / base) * 10000.0


def build_feature_snapshot(
    *,
    market_snapshot: MarketSnapshot,
    orb_state: ORBState,
    feature_history: list[FeatureSnapshot],
    source_mode: str,
) -> FeatureSnapshot:
    bid = _valid(market_snapshot.bid)
    ask = _valid(market_snapshot.ask)
    last = _valid(market_snapshot.last)
    bid_size = _valid(market_snapshot.bid_size)
    ask_size = _valid(market_snapshot.ask_size)
    volume = _valid(market_snapshot.volume)

    spread = (ask - bid) if bid is not None and ask is not None else None
    mid = ((bid + ask) / 2.0) if bid is not None and ask is not None else last
    microprice = None
    if bid is not None and ask is not None and bid_size and ask_size:
        microprice = ((ask * bid_size) + (bid * ask_size)) / (bid_size + ask_size)
    depth_imbalance = None
    if bid_size is not None and ask_size is not None and (bid_size + ask_size) > 0:
        depth_imbalance = (bid_size - ask_size) / (bid_size + ask_size)

    previous = feature_history[-1] if feature_history else None
    previous_mid = previous.feature_values.get("mid_price") if previous else None
    previous_bid_size = previous.feature_values.get("best_bid_size") if previous else None
    previous_ask_size = previous.feature_values.get("best_ask_size") if previous else None

    return_1 = None
    if previous_mid not in {None, 0} and mid is not None:
        return_1 = (mid / float(previous_mid) - 1.0) * 10000.0

    recent_returns = [
        float(snapshot.feature_values["return_1_bps"])
        for snapshot in feature_history[-5:]
        if snapshot.feature_values.get("return_1_bps") is not None
    ]
    rolling_volatility = None
    if recent_returns:
        mean_return = sum(recent_returns) / len(recent_returns)
        rolling_volatility = (
            sum((item - mean_return) ** 2 for item in recent_returns) / len(recent_returns)
        ) ** 0.5

    ofi_proxy = None
    if (
        bid_size is not None
        and ask_size is not None
        and previous_bid_size is not None
        and previous_ask_size is not None
    ):
        ofi_proxy = (bid_size - float(previous_bid_size)) - (ask_size - float(previous_ask_size))

    recent_volume_imbalance = None
    if bid_size is not None and ask_size is not None and (bid_size + ask_size) > 0:
        recent_volume_imbalance = (bid_size - ask_size) / (bid_size + ask_size)

    multi_level_ofi = None
    bid_depth = 0.0
    ask_depth = 0.0
    for level in range(5):
        level_bid = market_snapshot.raw.get(f"bid_size_{level:02d}") or market_snapshot.raw.get(
            f"bid_sz_{level:02d}"
        )
        level_ask = market_snapshot.raw.get(f"ask_size_{level:02d}") or market_snapshot.raw.get(
            f"ask_sz_{level:02d}"
        )
        if level_bid is not None:
            bid_depth += float(level_bid)
        if level_ask is not None:
            ask_depth += float(level_ask)
    if (bid_depth + ask_depth) > 0:
        multi_level_ofi = (bid_depth - ask_depth) / (bid_depth + ask_depth)

    orb_features = build_orb_feature_map(orb_state, current_price=mid or last)
    spread_bps = _bps(spread, mid)
    estimated_cost_bps = (spread_bps or 0.0) + 0.5

    feature_values: dict[str, float | None] = {
        "best_bid": bid,
        "best_ask": ask,
        "best_bid_size": bid_size,
        "best_ask_size": ask_size,
        "last_price": last,
        "volume": volume,
        "spread": spread,
        "spread_bps": spread_bps,
        "mid_price": mid,
        "microprice": microprice,
        "depth_imbalance": depth_imbalance,
        "ofi_proxy": ofi_proxy,
        "multi_level_ofi": multi_level_ofi,
        "return_1_bps": return_1,
        "rolling_volatility_5": rolling_volatility,
        "recent_volume_imbalance": recent_volume_imbalance,
        "estimated_cost_bps": estimated_cost_bps,
        **orb_features,
    }
    missing_features = [key for key, value in feature_values.items() if value is None]

    return FeatureSnapshot(
        symbol=market_snapshot.symbol,
        timestamp=market_snapshot.timestamp,
        feature_values=feature_values,
        estimated_cost_bps=estimated_cost_bps,
        missing_features=missing_features,
        source_mode=source_mode,
    )
