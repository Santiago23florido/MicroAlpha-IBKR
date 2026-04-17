from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def session_group_keys(frame: pd.DataFrame) -> list[str]:
    if "session_date" in frame.columns:
        return ["symbol", "session_date"]
    return ["symbol"]


def session_groupby(frame: pd.DataFrame) -> pd.core.groupby.generic.DataFrameGroupBy:
    return frame.groupby(session_group_keys(frame), sort=False)


def resolve_first_available(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns and pd.Series(frame[candidate]).notna().any():
            return candidate
    return None


def price_column(frame: pd.DataFrame) -> str | None:
    return resolve_first_available(frame, ("price_proxy", "last_price", "mid_price", "close"))


def high_column(frame: pd.DataFrame) -> str | None:
    return resolve_first_available(frame, ("high", "high_price_proxy", "ask", "last_price"))


def low_column(frame: pd.DataFrame) -> str | None:
    return resolve_first_available(frame, ("low", "low_price_proxy", "bid", "last_price"))


def volume_column(frame: pd.DataFrame) -> str | None:
    return resolve_first_available(frame, ("volume", "last_size"))


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    numerator_series = pd.Series(numerator)
    denominator_series = pd.Series(denominator)
    result = numerator_series / denominator_series.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_mean_by_group(frame: pd.DataFrame, column: str, window: int, *, min_periods: int = 1) -> pd.Series:
    return session_groupby(frame)[column].transform(
        lambda series: series.rolling(window, min_periods=min_periods).mean()
    )


def rolling_std_by_group(frame: pd.DataFrame, column: str, window: int, *, min_periods: int = 2) -> pd.Series:
    return session_groupby(frame)[column].transform(
        lambda series: series.rolling(window, min_periods=min_periods).std()
    )


def rolling_sum_by_group(frame: pd.DataFrame, column: str, window: int, *, min_periods: int = 1) -> pd.Series:
    return session_groupby(frame)[column].transform(
        lambda series: series.rolling(window, min_periods=min_periods).sum()
    )


def rolling_min_by_group(frame: pd.DataFrame, column: str, window: int, *, min_periods: int = 1) -> pd.Series:
    return session_groupby(frame)[column].transform(
        lambda series: series.rolling(window, min_periods=min_periods).min()
    )


def rolling_max_by_group(frame: pd.DataFrame, column: str, window: int, *, min_periods: int = 1) -> pd.Series:
    return session_groupby(frame)[column].transform(
        lambda series: series.rolling(window, min_periods=min_periods).max()
    )


def rolling_apply_by_group(
    frame: pd.DataFrame,
    column: str,
    window: int,
    func,
    *,
    min_periods: int = 1,
) -> pd.Series:
    return session_groupby(frame)[column].transform(
        lambda series: series.rolling(window, min_periods=min_periods).apply(func, raw=False)
    )


def ema_by_group(frame: pd.DataFrame, column: str, span: int) -> pd.Series:
    return session_groupby(frame)[column].transform(
        lambda series: series.ewm(span=span, adjust=False, min_periods=1).mean()
    )


def diff_by_group(frame: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    return session_groupby(frame)[column].transform(lambda series: series.diff(periods))


def pct_change_by_group(frame: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    return session_groupby(frame)[column].transform(lambda series: series.pct_change(periods))


def cumulative_sum_by_group(frame: pd.DataFrame, column: str) -> pd.Series:
    return session_groupby(frame)[column].transform(lambda series: series.cumsum())
