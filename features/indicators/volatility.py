from __future__ import annotations

import pandas as pd

from config import Settings
from features.definitions import IndicatorDefinition, IndicatorDependency
from features.indicators._utils import (
    high_column,
    low_column,
    price_column,
    rolling_mean_by_group,
    rolling_std_by_group,
    safe_divide,
    session_groupby,
)


def build_volatility_indicator_definitions() -> list[IndicatorDefinition]:
    return [
        IndicatorDefinition(
            name="true_range",
            family="volatility",
            description="True range from high, low, and previous close proxies.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("true_range",),
            output_type="numeric",
            default_params={},
            calculator=_calc_true_range_bundle,
        ),
        IndicatorDefinition(
            name="atr",
            family="volatility",
            description="Average true range.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("atr",),
            output_type="numeric",
            default_params={},
            calculator=_calc_true_range_bundle,
        ),
        IndicatorDefinition(
            name="rolling_volatility",
            family="volatility",
            description="Rolling volatility on intraday returns in basis points.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=(
                "rolling_volatility_short_bps",
                "rolling_volatility_medium_bps",
                "rolling_volatility_long_bps",
            ),
            output_type="numeric",
            default_params={},
            calculator=_calc_rolling_volatility,
        ),
        IndicatorDefinition(
            name="rolling_std_returns",
            family="volatility",
            description="Rolling standard deviation of returns in basis points.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=(
                "rolling_std_returns_short_bps",
                "rolling_std_returns_medium_bps",
                "rolling_std_returns_long_bps",
            ),
            output_type="numeric",
            default_params={},
            calculator=_calc_rolling_volatility,
        ),
        IndicatorDefinition(
            name="bollinger_mid",
            family="volatility",
            description="Bollinger band mid line.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("bollinger_mid",),
            output_type="numeric",
            default_params={"std_multiplier": 2.0},
            calculator=_calc_bollinger,
        ),
        IndicatorDefinition(
            name="bollinger_upper",
            family="volatility",
            description="Bollinger band upper line.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("bollinger_upper",),
            output_type="numeric",
            default_params={"std_multiplier": 2.0},
            calculator=_calc_bollinger,
        ),
        IndicatorDefinition(
            name="bollinger_lower",
            family="volatility",
            description="Bollinger band lower line.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("bollinger_lower",),
            output_type="numeric",
            default_params={"std_multiplier": 2.0},
            calculator=_calc_bollinger,
        ),
        IndicatorDefinition(
            name="bollinger_bandwidth",
            family="volatility",
            description="Bollinger bandwidth in basis points.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("bollinger_bandwidth_bps",),
            output_type="numeric",
            default_params={"std_multiplier": 2.0},
            calculator=_calc_bollinger,
        ),
        IndicatorDefinition(
            name="zscore_price",
            family="volatility",
            description="Rolling z-score of the price proxy.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("zscore_price",),
            output_type="numeric",
            default_params={},
            calculator=_calc_zscore_price,
        ),
        IndicatorDefinition(
            name="orb_width",
            family="volatility",
            description="Opening-range width and derived context.",
            required_inputs=(
                IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),
                IndicatorDependency("timestamp", ("exchange_timestamp", "timestamp")),
            ),
            output_columns=(
                "orb_high",
                "orb_low",
                "orb_width",
                "orb_width_bps",
                "orb_range_width",
                "orb_range_width_bps",
                "orb_relative_price_position",
                "breakout_distance",
                "breakout_distance_bps",
                "orb_range_complete",
            ),
            output_type="numeric",
            default_params={},
            calculator=_calc_orb_bundle,
        ),
    ]


def _window_map(settings: Settings) -> dict[str, int]:
    return {
        "short": int(settings.feature_pipeline.rolling_short_window),
        "medium": int(settings.feature_pipeline.rolling_medium_window),
        "long": int(settings.feature_pipeline.rolling_long_window),
    }


def _calc_true_range_bundle(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    high = frame[high_column(frame)]
    low = frame[low_column(frame)]
    close_col = price_column(frame)
    enriched = frame.copy()
    enriched["_close"] = frame[close_col]
    prev_close = session_groupby(enriched)["_close"].shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    temp = frame.copy()
    temp["_true_range"] = true_range
    window = int(params.get("window", settings.feature_pipeline.rolling_medium_window))
    atr = session_groupby(temp)["_true_range"].transform(lambda series: series.rolling(window, min_periods=1).mean())
    return pd.DataFrame({"true_range": true_range, "atr": atr}, index=frame.index)


def _calc_rolling_volatility(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price = frame[price_column(frame)]
    enriched = frame.copy()
    enriched["_returns"] = session_groupby(pd.concat([frame, price.rename("_price")], axis=1))["_price"].pct_change()
    windows = _window_map(settings)
    std_short = session_groupby(enriched)["_returns"].transform(lambda series: series.rolling(windows["short"], min_periods=2).std())
    std_medium = session_groupby(enriched)["_returns"].transform(lambda series: series.rolling(windows["medium"], min_periods=2).std())
    std_long = session_groupby(enriched)["_returns"].transform(lambda series: series.rolling(windows["long"], min_periods=2).std())
    return pd.DataFrame(
        {
            "rolling_volatility_short_bps": std_short * 10000.0,
            "rolling_volatility_medium_bps": std_medium * 10000.0,
            "rolling_volatility_long_bps": std_long * 10000.0,
            "rolling_std_returns_short_bps": std_short * 10000.0,
            "rolling_std_returns_medium_bps": std_medium * 10000.0,
            "rolling_std_returns_long_bps": std_long * 10000.0,
        },
        index=frame.index,
    )


def _calc_bollinger(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price_col = price_column(frame)
    window = int(params.get("window", settings.feature_pipeline.rolling_medium_window))
    std_multiplier = float(params.get("std_multiplier", 2.0))
    mid = rolling_mean_by_group(frame, price_col, window)
    std = rolling_std_by_group(frame, price_col, window)
    upper = mid + (std_multiplier * std)
    lower = mid - (std_multiplier * std)
    bandwidth = safe_divide(upper - lower, mid) * 10000.0
    return pd.DataFrame(
        {
            "bollinger_mid": mid,
            "bollinger_upper": upper,
            "bollinger_lower": lower,
            "bollinger_bandwidth_bps": bandwidth,
        },
        index=frame.index,
    )


def _calc_zscore_price(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price_col = price_column(frame)
    window = int(params.get("window", settings.feature_pipeline.rolling_medium_window))
    mean = rolling_mean_by_group(frame, price_col, window)
    std = rolling_std_by_group(frame, price_col, window)
    zscore = safe_divide(frame[price_col] - mean, std)
    return pd.DataFrame({"zscore_price": zscore}, index=frame.index)


def _calc_orb_bundle(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    enriched = frame.copy()
    price_col = price_column(frame)
    opening_mask = (
        (enriched["exchange_timestamp"].dt.time >= settings.session.orb_start)
        & (enriched["exchange_timestamp"].dt.time < settings.session.orb_end)
    )
    orb_stats = (
        enriched.loc[opening_mask]
        .groupby(["symbol", "session_date"], sort=False)[price_col]
        .agg(orb_high="max", orb_low="min")
        .reset_index()
    )
    enriched = enriched.merge(orb_stats, on=["symbol", "session_date"], how="left")
    enriched["orb_width"] = enriched["orb_high"] - enriched["orb_low"]
    enriched["orb_width_bps"] = safe_divide(enriched["orb_width"], enriched[price_col]) * 10000.0
    enriched["orb_range_width"] = enriched["orb_width"]
    enriched["orb_range_width_bps"] = enriched["orb_width_bps"]
    enriched["orb_relative_price_position"] = safe_divide(enriched[price_col] - enriched["orb_low"], enriched["orb_width"])
    upper_breakout = enriched[price_col] - enriched["orb_high"]
    lower_breakout = enriched[price_col] - enriched["orb_low"]
    enriched["breakout_distance"] = upper_breakout.where(upper_breakout.abs() <= lower_breakout.abs(), lower_breakout)
    enriched["breakout_distance_bps"] = safe_divide(enriched["breakout_distance"], enriched[price_col]) * 10000.0
    enriched["orb_range_complete"] = enriched["exchange_timestamp"].dt.time >= settings.session.orb_end
    return enriched[
        [
            "orb_high",
            "orb_low",
            "orb_width",
            "orb_width_bps",
            "orb_range_width",
            "orb_range_width_bps",
            "orb_relative_price_position",
            "breakout_distance",
            "breakout_distance_bps",
            "orb_range_complete",
        ]
    ]
