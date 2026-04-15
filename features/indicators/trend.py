from __future__ import annotations

import numpy as np
import pandas as pd

from config import Settings
from features.definitions import IndicatorDefinition, IndicatorDependency
from features.indicators._utils import (
    diff_by_group,
    ema_by_group,
    high_column,
    low_column,
    price_column,
    rolling_mean_by_group,
    safe_divide,
    session_groupby,
)


def build_trend_indicator_definitions() -> list[IndicatorDefinition]:
    return [
        IndicatorDefinition(
            name="sma",
            family="trend",
            description="Simple moving averages over short, medium, and long windows.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("sma_short", "sma_medium", "sma_long"),
            output_type="numeric",
            default_params={"windows": ("short", "medium", "long")},
            calculator=_calc_sma,
        ),
        IndicatorDefinition(
            name="ema",
            family="trend",
            description="Exponential moving averages over short, medium, and long windows.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("ema_short", "ema_medium", "ema_long"),
            output_type="numeric",
            default_params={"windows": ("short", "medium", "long")},
            calculator=_calc_ema,
        ),
        IndicatorDefinition(
            name="moving_average_distance",
            family="trend",
            description="Distance between price and moving averages in basis points.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=(
                "moving_average_distance_short_bps",
                "moving_average_distance_medium_bps",
                "moving_average_distance_long_bps",
            ),
            output_type="numeric",
            default_params={"windows": ("short", "medium", "long")},
            calculator=_calc_ma_distance,
        ),
        IndicatorDefinition(
            name="moving_average_slope",
            family="trend",
            description="One-step slopes of moving averages in basis points.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=(
                "moving_average_slope_short_bps",
                "moving_average_slope_medium_bps",
                "moving_average_slope_long_bps",
            ),
            output_type="numeric",
            default_params={"windows": ("short", "medium", "long")},
            calculator=_calc_ma_slope,
        ),
        IndicatorDefinition(
            name="ma_crossover_short_long",
            family="trend",
            description="Short versus long exponential moving-average crossover in basis points.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("ma_crossover_short_long_bps",),
            output_type="numeric",
            default_params={},
            calculator=_calc_ma_crossover,
        ),
        IndicatorDefinition(
            name="macd_line",
            family="trend",
            description="MACD line from the short and long exponential moving averages.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("macd_line",),
            output_type="numeric",
            default_params={},
            calculator=_calc_macd,
        ),
        IndicatorDefinition(
            name="macd_signal",
            family="trend",
            description="Signal line on top of MACD.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("macd_signal",),
            output_type="numeric",
            default_params={"signal_window": 9},
            calculator=_calc_macd,
        ),
        IndicatorDefinition(
            name="macd_histogram",
            family="trend",
            description="Histogram difference between MACD and signal lines.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("macd_histogram",),
            output_type="numeric",
            default_params={"signal_window": 9},
            calculator=_calc_macd,
        ),
        IndicatorDefinition(
            name="adx",
            family="trend",
            description="Average directional index based on price proxies.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("adx",),
            output_type="numeric",
            default_params={},
            calculator=_calc_adx_bundle,
        ),
        IndicatorDefinition(
            name="plus_di",
            family="trend",
            description="Positive directional indicator.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("plus_di",),
            output_type="numeric",
            default_params={},
            calculator=_calc_adx_bundle,
        ),
        IndicatorDefinition(
            name="minus_di",
            family="trend",
            description="Negative directional indicator.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("minus_di",),
            output_type="numeric",
            default_params={},
            calculator=_calc_adx_bundle,
        ),
    ]


def _window_map(settings: Settings, params: dict[str, object]) -> dict[str, int]:
    custom = params.get("window_values", {})
    return {
        "short": int(custom.get("short", settings.feature_pipeline.rolling_short_window)),
        "medium": int(custom.get("medium", settings.feature_pipeline.rolling_medium_window)),
        "long": int(custom.get("long", settings.feature_pipeline.rolling_long_window)),
    }


def _calc_sma(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price = price_column(frame)
    windows = _window_map(settings, params)
    return pd.DataFrame(
        {
            "sma_short": rolling_mean_by_group(frame, price, windows["short"]),
            "sma_medium": rolling_mean_by_group(frame, price, windows["medium"]),
            "sma_long": rolling_mean_by_group(frame, price, windows["long"]),
        },
        index=frame.index,
    )


def _calc_ema(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price = price_column(frame)
    windows = _window_map(settings, params)
    return pd.DataFrame(
        {
            "ema_short": ema_by_group(frame, price, windows["short"]),
            "ema_medium": ema_by_group(frame, price, windows["medium"]),
            "ema_long": ema_by_group(frame, price, windows["long"]),
        },
        index=frame.index,
    )


def _calc_ma_distance(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price = pd.Series(frame[price_column(frame)], index=frame.index)
    sma = _calc_sma(frame, settings, params)
    return pd.DataFrame(
        {
            "moving_average_distance_short_bps": safe_divide(price - sma["sma_short"], sma["sma_short"]) * 10000.0,
            "moving_average_distance_medium_bps": safe_divide(price - sma["sma_medium"], sma["sma_medium"]) * 10000.0,
            "moving_average_distance_long_bps": safe_divide(price - sma["sma_long"], sma["sma_long"]) * 10000.0,
        },
        index=frame.index,
    )


def _calc_ma_slope(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    sma = _calc_sma(frame, settings, params)
    group_keys = ["symbol", "session_date"] if "session_date" in frame.columns else ["symbol"]
    output = {}
    for label in ("short", "medium", "long"):
        column = f"sma_{label}"
        temp = pd.DataFrame(
            {
                **{key: frame[key] for key in group_keys},
                column: sma[column],
            }
        )
        diff = temp.groupby(group_keys, sort=False)[column].diff()
        output[f"moving_average_slope_{label}_bps"] = safe_divide(diff, sma[column]) * 10000.0
    return pd.DataFrame(output, index=frame.index)


def _calc_ma_crossover(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    ema = _calc_ema(frame, settings, params)
    price = pd.Series(frame[price_column(frame)], index=frame.index)
    crossover = safe_divide(ema["ema_short"] - ema["ema_long"], price) * 10000.0
    return pd.DataFrame({"ma_crossover_short_long_bps": crossover}, index=frame.index)


def _calc_macd(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    ema = _calc_ema(frame, settings, params)
    macd_line = ema["ema_short"] - ema["ema_long"]
    signal_window = int(params.get("signal_window", 9))
    enriched = pd.concat([frame, macd_line.rename("macd_line_base")], axis=1)
    signal = ema_by_group(enriched, "macd_line_base", signal_window)
    histogram = macd_line - signal
    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal,
            "macd_histogram": histogram,
        },
        index=frame.index,
    )


def _calc_adx_bundle(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    high = pd.Series(frame[high_column(frame)], index=frame.index)
    low = pd.Series(frame[low_column(frame)], index=frame.index)
    close_col = price_column(frame)
    close = pd.Series(frame[close_col], index=frame.index)
    window = int(params.get("window", settings.feature_pipeline.rolling_medium_window))

    enriched = frame.copy()
    enriched["_high"] = high
    enriched["_low"] = low
    enriched["_close"] = close
    grouped = session_groupby(enriched)

    high_diff = grouped["_high"].diff()
    low_diff = -grouped["_low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    prev_close = grouped["_close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    temp = enriched.copy()
    temp["_plus_dm"] = plus_dm
    temp["_minus_dm"] = minus_dm
    temp["_tr"] = tr

    plus_dm_smooth = session_groupby(temp)["_plus_dm"].transform(
        lambda series: series.rolling(window, min_periods=1).sum()
    )
    minus_dm_smooth = session_groupby(temp)["_minus_dm"].transform(
        lambda series: series.rolling(window, min_periods=1).sum()
    )
    tr_smooth = session_groupby(temp)["_tr"].transform(
        lambda series: series.rolling(window, min_periods=1).sum()
    )

    plus_di = safe_divide(plus_dm_smooth, tr_smooth) * 100.0
    minus_di = safe_divide(minus_dm_smooth, tr_smooth) * 100.0
    dx = safe_divide((plus_di - minus_di).abs(), plus_di + minus_di) * 100.0
    temp["_dx"] = dx
    adx = session_groupby(temp)["_dx"].transform(lambda series: series.rolling(window, min_periods=1).mean())

    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di}, index=frame.index)
