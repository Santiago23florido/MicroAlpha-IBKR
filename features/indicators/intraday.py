from __future__ import annotations

import numpy as np
import pandas as pd

from config import Settings
from features.definitions import IndicatorDefinition, IndicatorDependency
from features.indicators._utils import safe_divide


def build_intraday_indicator_definitions() -> list[IndicatorDefinition]:
    time_dependency = (IndicatorDependency("timestamp", ("exchange_timestamp", "timestamp")),)
    return [
        IndicatorDefinition(
            name="minute_of_day",
            family="intraday_structure",
            description="Minute index within the exchange day.",
            required_inputs=time_dependency,
            output_columns=("minute_of_day", "minutes_since_open", "time_of_day_sin", "time_of_day_cos"),
            output_type="numeric",
            calculator=_calc_intraday_bundle,
        ),
        IndicatorDefinition(
            name="seconds_since_open",
            family="intraday_structure",
            description="Seconds elapsed since the regular market open.",
            required_inputs=time_dependency,
            output_columns=("seconds_since_open",),
            output_type="numeric",
            calculator=_calc_intraday_bundle,
        ),
        IndicatorDefinition(
            name="seconds_to_close",
            family="intraday_structure",
            description="Seconds until the regular market close.",
            required_inputs=time_dependency,
            output_columns=("seconds_to_close",),
            output_type="numeric",
            calculator=_calc_intraday_bundle,
        ),
        IndicatorDefinition(
            name="opening_session_flag",
            family="intraday_structure",
            description="Flag for the opening session window.",
            required_inputs=time_dependency,
            output_columns=("opening_session_flag",),
            output_type="boolean",
            calculator=_calc_intraday_bundle,
        ),
        IndicatorDefinition(
            name="midday_flag",
            family="intraday_structure",
            description="Flag for the middle of the session.",
            required_inputs=time_dependency,
            output_columns=("midday_flag",),
            output_type="boolean",
            calculator=_calc_intraday_bundle,
        ),
        IndicatorDefinition(
            name="closing_session_flag",
            family="intraday_structure",
            description="Flag for the closing session window.",
            required_inputs=time_dependency,
            output_columns=("closing_session_flag",),
            output_type="boolean",
            calculator=_calc_intraday_bundle,
        ),
        IndicatorDefinition(
            name="day_of_week",
            family="intraday_structure",
            description="Day-of-week encoded as integer.",
            required_inputs=time_dependency,
            output_columns=("day_of_week",),
            output_type="categorical",
            calculator=_calc_intraday_bundle,
        ),
        IndicatorDefinition(
            name="intraday_volume_percentile",
            family="intraday_structure",
            description="Within-session volume percentile rank.",
            required_inputs=(
                IndicatorDependency("timestamp", ("exchange_timestamp", "timestamp")),
                IndicatorDependency("volume", ("volume", "last_size")),
            ),
            output_columns=("intraday_volume_percentile",),
            output_type="numeric",
            calculator=_calc_intraday_bundle,
        ),
        IndicatorDefinition(
            name="intraday_spread_percentile",
            family="intraday_structure",
            description="Within-session spread percentile rank.",
            required_inputs=(
                IndicatorDependency("timestamp", ("exchange_timestamp", "timestamp")),
                IndicatorDependency("spread", ("spread", "spread_bps")),
            ),
            output_columns=("intraday_spread_percentile",),
            output_type="numeric",
            calculator=_calc_intraday_bundle,
        ),
    ]


def _calc_intraday_bundle(frame: pd.DataFrame, settings: Settings, params: dict[str, object]) -> pd.DataFrame:
    exchange_timestamp = frame["exchange_timestamp"]
    open_minutes = settings.session.regular_market_open.hour * 60 + settings.session.regular_market_open.minute
    close_minutes = settings.session.regular_market_close.hour * 60 + settings.session.regular_market_close.minute
    minute_of_day = exchange_timestamp.dt.hour * 60 + exchange_timestamp.dt.minute
    seconds_since_midnight = minute_of_day * 60 + exchange_timestamp.dt.second
    open_seconds = open_minutes * 60
    close_seconds = close_minutes * 60

    session_rank_groups = ["symbol", "session_date"] if "session_date" in frame.columns else ["symbol"]
    volume_col = "volume" if "volume" in frame.columns else ("last_size" if "last_size" in frame.columns else None)
    spread_col = "spread" if "spread" in frame.columns else ("spread_bps" if "spread_bps" in frame.columns else None)

    intraday_volume_percentile = (
        frame.groupby(session_rank_groups, sort=False)[volume_col].rank(pct=True) if volume_col else pd.Series(0.0, index=frame.index)
    )
    intraday_spread_percentile = (
        frame.groupby(session_rank_groups, sort=False)[spread_col].rank(pct=True) if spread_col else pd.Series(0.0, index=frame.index)
    )

    opening_end_seconds = (settings.session.orb_end.hour * 60 + settings.session.orb_end.minute) * 60
    closing_start_seconds = max(close_seconds - (60 * settings.feature_pipeline.rolling_medium_window), open_seconds)

    return pd.DataFrame(
        {
            "minute_of_day": minute_of_day,
            "minutes_since_open": (seconds_since_midnight - open_seconds) / 60.0,
            "seconds_since_open": seconds_since_midnight - open_seconds,
            "seconds_to_close": close_seconds - seconds_since_midnight,
            "opening_session_flag": ((seconds_since_midnight >= open_seconds) & (seconds_since_midnight < opening_end_seconds)).astype(int),
            "midday_flag": ((seconds_since_midnight >= opening_end_seconds) & (seconds_since_midnight < closing_start_seconds)).astype(int),
            "closing_session_flag": (seconds_since_midnight >= closing_start_seconds).astype(int),
            "day_of_week": exchange_timestamp.dt.dayofweek,
            "time_of_day_sin": pd.Series((2.0 * 3.141592653589793 * minute_of_day / 1440.0), index=frame.index).map(np.sin),
            "time_of_day_cos": pd.Series((2.0 * 3.141592653589793 * minute_of_day / 1440.0), index=frame.index).map(np.cos),
            "intraday_volume_percentile": intraday_volume_percentile.fillna(0.0),
            "intraday_spread_percentile": intraday_spread_percentile.fillna(0.0),
        },
        index=frame.index,
    )
