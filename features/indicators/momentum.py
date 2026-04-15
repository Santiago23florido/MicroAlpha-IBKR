from __future__ import annotations

import numpy as np
import pandas as pd

from config import Settings
from features.definitions import IndicatorDefinition, IndicatorDependency
from features.indicators._utils import (
    high_column,
    low_column,
    price_column,
    rolling_apply_by_group,
    rolling_max_by_group,
    rolling_mean_by_group,
    rolling_min_by_group,
    safe_divide,
    session_groupby,
)


def build_momentum_indicator_definitions() -> list[IndicatorDefinition]:
    return [
        IndicatorDefinition(
            name="rsi",
            family="momentum",
            description="Relative strength index on the selected price proxy.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("rsi",),
            output_type="numeric",
            default_params={},
            calculator=_calc_rsi,
        ),
        IndicatorDefinition(
            name="roc",
            family="momentum",
            description="Rate of change in basis points.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("roc_bps",),
            output_type="numeric",
            default_params={},
            calculator=_calc_roc,
        ),
        IndicatorDefinition(
            name="momentum_simple",
            family="momentum",
            description="Simple momentum change in basis points.",
            required_inputs=(IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),),
            output_columns=("momentum_simple_bps",),
            output_type="numeric",
            default_params={},
            calculator=_calc_momentum_simple,
        ),
        IndicatorDefinition(
            name="stochastic_k",
            family="momentum",
            description="Stochastic oscillator K line using price proxies.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("stochastic_k",),
            output_type="numeric",
            default_params={},
            calculator=_calc_stochastic,
        ),
        IndicatorDefinition(
            name="stochastic_d",
            family="momentum",
            description="Stochastic oscillator D line.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("stochastic_d",),
            output_type="numeric",
            default_params={"signal_window": 3},
            calculator=_calc_stochastic,
        ),
        IndicatorDefinition(
            name="williams_r",
            family="momentum",
            description="Williams %R oscillator.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("williams_r",),
            output_type="numeric",
            default_params={},
            calculator=_calc_williams_r,
        ),
        IndicatorDefinition(
            name="cci",
            family="momentum",
            description="Commodity channel index from the price proxies.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
            ),
            output_columns=("cci",),
            output_type="numeric",
            default_params={},
            calculator=_calc_cci,
        ),
    ]


def _window(settings: Settings, params: dict[str, object]) -> int:
    return int(params.get("window", settings.feature_pipeline.rolling_medium_window))


def _calc_rsi(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price = price_column(frame)
    window = _window(settings, params)
    enriched = frame.copy()
    enriched["_price"] = frame[price]
    delta = session_groupby(enriched)["_price"].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    enriched["_gain"] = gains
    enriched["_loss"] = losses
    avg_gain = session_groupby(enriched)["_gain"].transform(lambda series: series.rolling(window, min_periods=1).mean())
    avg_loss = session_groupby(enriched)["_loss"].transform(lambda series: series.rolling(window, min_periods=1).mean())
    rs = safe_divide(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.DataFrame({"rsi": rsi}, index=frame.index)


def _calc_roc(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price = price_column(frame)
    periods = int(params.get("periods", settings.feature_pipeline.rolling_short_window))
    enriched = frame.copy()
    enriched["_price"] = frame[price]
    prev_price = session_groupby(enriched)["_price"].shift(periods)
    roc = safe_divide(enriched["_price"] - prev_price, prev_price) * 10000.0
    return pd.DataFrame({"roc_bps": roc}, index=frame.index)


def _calc_momentum_simple(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price = price_column(frame)
    periods = int(params.get("periods", settings.feature_pipeline.rolling_short_window))
    enriched = frame.copy()
    enriched["_price"] = frame[price]
    momentum = session_groupby(enriched)["_price"].diff(periods)
    momentum_bps = safe_divide(momentum, enriched["_price"]) * 10000.0
    return pd.DataFrame({"momentum_simple_bps": momentum_bps}, index=frame.index)


def _calc_stochastic(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    high = high_column(frame)
    low = low_column(frame)
    close = price_column(frame)
    window = _window(settings, params)
    highest = rolling_max_by_group(frame, high, window)
    lowest = rolling_min_by_group(frame, low, window)
    k = safe_divide(frame[close] - lowest, highest - lowest) * 100.0
    enriched = frame.copy()
    enriched["_k"] = k
    d_window = int(params.get("signal_window", 3))
    d = session_groupby(enriched)["_k"].transform(lambda series: series.rolling(d_window, min_periods=1).mean())
    return pd.DataFrame({"stochastic_k": k, "stochastic_d": d}, index=frame.index)


def _calc_williams_r(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    high = high_column(frame)
    low = low_column(frame)
    close = price_column(frame)
    window = _window(settings, params)
    highest = rolling_max_by_group(frame, high, window)
    lowest = rolling_min_by_group(frame, low, window)
    williams = -100.0 * safe_divide(highest - frame[close], highest - lowest)
    return pd.DataFrame({"williams_r": williams}, index=frame.index)


def _calc_cci(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    high = frame[high_column(frame)]
    low = frame[low_column(frame)]
    close = frame[price_column(frame)]
    typical_price = (high + low + close) / 3.0
    enriched = frame.copy()
    enriched["_typical_price"] = typical_price
    window = _window(settings, params)
    sma = session_groupby(enriched)["_typical_price"].transform(lambda series: series.rolling(window, min_periods=1).mean())
    mad = session_groupby(enriched)["_typical_price"].transform(
        lambda series: series.rolling(window, min_periods=1).apply(
            lambda values: np.mean(np.abs(values - np.mean(values))),
            raw=True,
        )
    )
    cci = safe_divide(typical_price - sma, 0.015 * mad)
    return pd.DataFrame({"cci": cci}, index=frame.index)
