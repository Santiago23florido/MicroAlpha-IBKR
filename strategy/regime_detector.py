from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class RegimeResult:
    regime_name: str
    confidence: float
    features_used: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RegimeDetector:
    def __init__(self, thresholds: Mapping[str, Any] | None = None) -> None:
        self.thresholds = dict(thresholds or {})

    def detect(self, feature_row: pd.Series | dict[str, Any]) -> RegimeResult:
        row = feature_row if isinstance(feature_row, pd.Series) else pd.Series(feature_row)
        values = {
            "spread_bps": _num(row, "spread_bps"),
            "estimated_cost_bps": _num(row, "estimated_cost_bps"),
            "relative_volume": _num(row, "relative_volume"),
            "orb_width_bps": _first_num(row, ("orb_width_bps", "orb_range_width_bps")),
            "distance_to_vwap_bps": _num(row, "distance_to_vwap_bps"),
            "vwap_slope_bps": _num(row, "vwap_slope_bps"),
            "rolling_volatility_5": _num(row, "rolling_volatility_5"),
            "rolling_volatility_15": _num(row, "rolling_volatility_15"),
            "price_vs_orb_high_bps": _num(row, "price_vs_orb_high_bps"),
            "price_vs_orb_low_bps": _num(row, "price_vs_orb_low_bps"),
            "orb_relative_price_position": _num(row, "orb_relative_price_position"),
        }
        flags: list[str] = []
        features_used = [key for key, value in values.items() if value is not None]

        spread = values["spread_bps"] or 0.0
        cost = values["estimated_cost_bps"] if values["estimated_cost_bps"] is not None else spread
        rel_volume = values["relative_volume"]
        orb_width = values["orb_width_bps"]
        distance_vwap = values["distance_to_vwap_bps"]
        volatility = values["rolling_volatility_15"] if values["rolling_volatility_15"] is not None else values["rolling_volatility_5"]
        session_time = _resolve_time(row)

        if spread >= _threshold(self.thresholds, "high_spread_bps", 12.0) or cost >= _threshold(self.thresholds, "high_cost_bps", 18.0):
            flags.append("high_cost")
        if rel_volume is not None and rel_volume <= _threshold(self.thresholds, "low_relative_volume", 0.65):
            flags.append("low_liquidity")
        if orb_width is not None and orb_width >= _threshold(self.thresholds, "noisy_open_orb_width_bps", 45.0) and _is_before(session_time, "10:15"):
            flags.append("noisy_open")
        if _is_between(session_time, "11:30", "13:30"):
            flags.append("midday_low_edge")
        if distance_vwap is not None and abs(distance_vwap) >= _threshold(self.thresholds, "mean_reversion_distance_bps", 12.0):
            flags.append("extended_from_vwap")
        if volatility is not None and volatility <= _threshold(self.thresholds, "low_volatility_bps", 1.5):
            flags.append("compressed")
        if volatility is not None and volatility >= _threshold(self.thresholds, "high_volatility_bps", 12.0):
            flags.append("expanded")

        if "high_cost" in flags:
            return self._result("high_cost_regime", 0.95, features_used, flags, values)
        if "low_liquidity" in flags:
            return self._result("low_liquidity_regime", 0.85, features_used, flags, values)
        if "noisy_open" in flags:
            return self._result("noisy_open", 0.80, features_used, flags, values)
        if "midday_low_edge" in flags and "extended_from_vwap" not in flags:
            return self._result("low_edge_midday", 0.75, features_used, flags, values)

        orb_position = values["orb_relative_price_position"]
        breakout_signal = (
            (orb_position is not None and (orb_position >= 1.02 or orb_position <= -0.02))
            or (values["price_vs_orb_high_bps"] is not None and values["price_vs_orb_high_bps"] > 0)
            or (values["price_vs_orb_low_bps"] is not None and values["price_vs_orb_low_bps"] < 0)
        )
        if breakout_signal and "expanded" in flags:
            return self._result("breakout_regime", 0.80, features_used, flags, values)
        if breakout_signal:
            return self._result("trend_day_candidate", 0.68, features_used, flags, values)
        if "extended_from_vwap" in flags:
            return self._result("mean_reversion_regime", 0.72, features_used, flags, values)
        if "compressed" in flags:
            return self._result("range_day_candidate", 0.62, features_used, flags, values)
        return self._result("neutral_intraday", 0.50, features_used, flags, values)

    @staticmethod
    def _result(
        name: str,
        confidence: float,
        features_used: list[str],
        flags: list[str],
        values: dict[str, float | None],
    ) -> RegimeResult:
        return RegimeResult(
            regime_name=name,
            confidence=float(confidence),
            features_used=features_used,
            flags=flags,
            metadata={"feature_values": {key: value for key, value in values.items() if value is not None}},
        )


def _num(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row.get(column)):
        return None
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _first_num(row: pd.Series, columns: tuple[str, ...]) -> float | None:
    for column in columns:
        value = _num(row, column)
        if value is not None:
            return value
    return None


def _threshold(thresholds: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(thresholds.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _resolve_time(row: pd.Series) -> time | None:
    raw = row.get("session_time") if "session_time" in row.index else None
    if isinstance(raw, time):
        return raw.replace(second=0, microsecond=0)
    timestamp = row.get("exchange_timestamp", row.get("timestamp", None))
    if timestamp is None or pd.isna(timestamp):
        return None
    parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.time().replace(second=0, microsecond=0)


def _parse_clock(value: str) -> time:
    hour, minute = value.split(":", maxsplit=1)
    return time(hour=int(hour), minute=int(minute))


def _is_before(value: time | None, limit: str) -> bool:
    return value is not None and value < _parse_clock(limit)


def _is_between(value: time | None, start: str, end: str) -> bool:
    return value is not None and _parse_clock(start) <= value <= _parse_clock(end)
