from __future__ import annotations

import numpy as np
import pandas as pd

from config import Settings
from features.definitions import IndicatorDefinition, IndicatorDependency
from features.indicators._utils import (
    cumulative_sum_by_group,
    high_column,
    low_column,
    price_column,
    rolling_mean_by_group,
    safe_divide,
    session_groupby,
    volume_column,
)


def build_volume_flow_indicator_definitions() -> list[IndicatorDefinition]:
    return [
        IndicatorDefinition(
            name="rolling_volume_mean",
            family="volume_flow",
            description="Rolling mean of trade volume.",
            required_inputs=(IndicatorDependency("volume", ("volume", "last_size")),),
            output_columns=("rolling_volume_mean",),
            output_type="numeric",
            default_params={},
            calculator=_calc_rolling_volume_mean,
        ),
        IndicatorDefinition(
            name="relative_volume",
            family="volume_flow",
            description="Volume relative to the rolling intraday mean.",
            required_inputs=(IndicatorDependency("volume", ("volume", "last_size")),),
            output_columns=("relative_volume",),
            output_type="numeric",
            default_params={},
            calculator=_calc_relative_volume,
        ),
        IndicatorDefinition(
            name="vwap",
            family="volume_flow",
            description="Rolling VWAP based on the price proxy and volume.",
            required_inputs=(
                IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),
                IndicatorDependency("volume", ("volume", "last_size")),
            ),
            output_columns=("vwap",),
            output_type="numeric",
            default_params={},
            calculator=_calc_vwap_bundle,
        ),
        IndicatorDefinition(
            name="distance_to_vwap",
            family="volume_flow",
            description="Distance between price and VWAP in basis points.",
            required_inputs=(
                IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),
                IndicatorDependency("volume", ("volume", "last_size")),
            ),
            output_columns=("distance_to_vwap_bps",),
            output_type="numeric",
            default_params={},
            calculator=_calc_vwap_bundle,
        ),
        IndicatorDefinition(
            name="vwap_slope",
            family="volume_flow",
            description="One-step VWAP slope in basis points.",
            required_inputs=(
                IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),
                IndicatorDependency("volume", ("volume", "last_size")),
            ),
            output_columns=("vwap_slope_bps",),
            output_type="numeric",
            default_params={},
            calculator=_calc_vwap_bundle,
        ),
        IndicatorDefinition(
            name="obv",
            family="volume_flow",
            description="On-balance volume.",
            required_inputs=(
                IndicatorDependency("price", ("price_proxy", "last_price", "mid_price", "close")),
                IndicatorDependency("volume", ("volume", "last_size")),
            ),
            output_columns=("obv",),
            output_type="numeric",
            default_params={},
            calculator=_calc_obv,
        ),
        IndicatorDefinition(
            name="volume_spike_flag",
            family="volume_flow",
            description="Boolean flag for volume spikes over the rolling mean.",
            required_inputs=(IndicatorDependency("volume", ("volume", "last_size")),),
            output_columns=("volume_spike_flag",),
            output_type="boolean",
            default_params={"spike_multiple": 1.5},
            calculator=_calc_volume_spike_flag,
        ),
        IndicatorDefinition(
            name="accumulation_distribution",
            family="volume_flow",
            description="Accumulation/distribution line using price proxies and volume.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
                IndicatorDependency("volume", ("volume", "last_size")),
            ),
            output_columns=("accumulation_distribution",),
            output_type="numeric",
            default_params={},
            calculator=_calc_accumulation_distribution,
        ),
        IndicatorDefinition(
            name="mfi",
            family="volume_flow",
            description="Money flow index using price proxies and volume.",
            required_inputs=(
                IndicatorDependency("high", ("high", "high_price_proxy", "ask", "last_price")),
                IndicatorDependency("low", ("low", "low_price_proxy", "bid", "last_price")),
                IndicatorDependency("close", ("price_proxy", "last_price", "mid_price", "close")),
                IndicatorDependency("volume", ("volume", "last_size")),
            ),
            output_columns=("mfi",),
            output_type="numeric",
            default_params={},
            calculator=_calc_mfi,
        ),
    ]


def _calc_rolling_volume_mean(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    volume = volume_column(frame)
    window = int(params.get("window", settings.feature_pipeline.volume_window))
    return pd.DataFrame(
        {"rolling_volume_mean": rolling_mean_by_group(frame, volume, window)},
        index=frame.index,
    )


def _calc_relative_volume(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    volume = volume_column(frame)
    rolling_volume = _calc_rolling_volume_mean(frame, settings, params)["rolling_volume_mean"]
    relative = safe_divide(frame[volume], rolling_volume)
    return pd.DataFrame({"relative_volume": relative}, index=frame.index)


def _calc_vwap_bundle(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price_col = price_column(frame)
    volume_col = volume_column(frame)
    window = int(params.get("window", settings.feature_pipeline.vwap_window))

    enriched = frame.copy()
    enriched["_price_volume"] = frame[price_col] * frame[volume_col]
    enriched["_volume"] = frame[volume_col]

    volume_sum = session_groupby(enriched)["_volume"].transform(lambda series: series.rolling(window, min_periods=1).sum())
    price_volume_sum = session_groupby(enriched)["_price_volume"].transform(
        lambda series: series.rolling(window, min_periods=1).sum()
    )
    vwap = safe_divide(price_volume_sum, volume_sum)
    enriched["_vwap"] = vwap
    vwap_slope = session_groupby(enriched)["_vwap"].transform(lambda series: series.diff())
    distance = safe_divide(frame[price_col] - vwap, vwap) * 10000.0
    slope_bps = safe_divide(vwap_slope, vwap) * 10000.0
    return pd.DataFrame(
        {
            "vwap": vwap,
            "distance_to_vwap_bps": distance,
            "vwap_slope_bps": slope_bps,
        },
        index=frame.index,
    )


def _calc_obv(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    price_col = price_column(frame)
    volume_col = volume_column(frame)
    enriched = frame.copy()
    enriched["_price"] = frame[price_col]
    enriched["_volume"] = frame[volume_col]
    direction = session_groupby(enriched)["_price"].diff().fillna(0.0).apply(np.sign)
    enriched["_signed_volume"] = direction * enriched["_volume"]
    obv = session_groupby(enriched)["_signed_volume"].transform(lambda series: series.cumsum())
    return pd.DataFrame({"obv": obv}, index=frame.index)


def _calc_volume_spike_flag(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    volume_col = volume_column(frame)
    rolling_volume = _calc_rolling_volume_mean(frame, settings, params)["rolling_volume_mean"]
    spike_multiple = float(params.get("spike_multiple", 1.5))
    flag = (frame[volume_col] > (rolling_volume * spike_multiple)).astype(int)
    return pd.DataFrame({"volume_spike_flag": flag}, index=frame.index)


def _calc_accumulation_distribution(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    high = frame[high_column(frame)]
    low = frame[low_column(frame)]
    close = frame[price_column(frame)]
    volume = frame[volume_column(frame)]
    money_flow_multiplier = safe_divide(((close - low) - (high - close)), (high - low))
    money_flow_volume = money_flow_multiplier * volume
    enriched = frame.copy()
    enriched["_mfv"] = money_flow_volume.fillna(0.0)
    adl = session_groupby(enriched)["_mfv"].transform(lambda series: series.cumsum())
    return pd.DataFrame({"accumulation_distribution": adl}, index=frame.index)


def _calc_mfi(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    high = frame[high_column(frame)]
    low = frame[low_column(frame)]
    close = frame[price_column(frame)]
    volume = frame[volume_column(frame)]
    typical = (high + low + close) / 3.0
    enriched = frame.copy()
    enriched["_typical"] = typical
    enriched["_volume"] = volume
    prev_typical = session_groupby(enriched)["_typical"].shift(1)
    positive_flow = (typical * volume).where(typical >= prev_typical, 0.0)
    negative_flow = (typical * volume).where(typical < prev_typical, 0.0)
    enriched["_positive_flow"] = positive_flow
    enriched["_negative_flow"] = negative_flow
    window = int(params.get("window", settings.feature_pipeline.rolling_medium_window))
    pos_sum = session_groupby(enriched)["_positive_flow"].transform(lambda series: series.rolling(window, min_periods=1).sum())
    neg_sum = session_groupby(enriched)["_negative_flow"].transform(lambda series: series.rolling(window, min_periods=1).sum())
    money_ratio = safe_divide(pos_sum, neg_sum)
    mfi = 100.0 - (100.0 / (1.0 + money_ratio))
    return pd.DataFrame({"mfi": mfi}, index=frame.index)
