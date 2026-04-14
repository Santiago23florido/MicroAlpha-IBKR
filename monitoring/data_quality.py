from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import Settings


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    symbol_count: int
    missing_timestamp_count: int
    duplicate_count: int
    critical_null_count: int
    bid_gt_ask_count: int
    negative_spread_count: int
    absurd_spread_count: int
    outside_regular_hours_count: int
    large_gap_count: int
    max_gap_seconds_observed: float
    null_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_market_data_quality(frame: pd.DataFrame, settings: Settings) -> DataQualityReport:
    if frame.empty:
        return DataQualityReport(
            row_count=0,
            symbol_count=0,
            missing_timestamp_count=0,
            duplicate_count=0,
            critical_null_count=0,
            bid_gt_ask_count=0,
            negative_spread_count=0,
            absurd_spread_count=0,
            outside_regular_hours_count=0,
            large_gap_count=0,
            max_gap_seconds_observed=0.0,
            issues=["no_rows"],
        )

    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    missing_symbol_count = int(working["symbol"].isna().sum()) if "symbol" in working.columns else int(len(working))
    if "symbol" in working.columns:
        working["symbol"] = working["symbol"].astype(str).str.upper()
    else:
        working["symbol"] = "UNKNOWN"

    if "last_price" not in working.columns and "last" in working.columns:
        working["last_price"] = pd.to_numeric(working["last"], errors="coerce")
    for column in ["last_price", "bid", "ask", "volume", "bid_size", "ask_size"]:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")

    missing_timestamp_count = int(working["timestamp"].isna().sum())
    duplicate_count = int(working.duplicated(subset=["symbol", "timestamp"], keep=False).sum())
    critical_null_count = int(working[["last_price", "bid", "ask"]].isna().all(axis=1).sum())
    bid_gt_ask_count = int(((working["bid"].notna()) & (working["ask"].notna()) & (working["bid"] > working["ask"])).sum())

    spread = working["ask"] - working["bid"]
    mid_price = np.where(
        working["bid"].notna() & working["ask"].notna(),
        (working["bid"] + working["ask"]) / 2.0,
        working["last_price"],
    )
    spread_bps = np.where(mid_price > 0, spread / mid_price * 10000.0, np.nan)
    negative_spread_count = int((spread < 0).fillna(False).sum())
    absurd_spread_count = int((pd.Series(spread_bps).abs() > settings.feature_pipeline.max_abs_spread_bps).fillna(False).sum())

    exchange_timestamp = working["timestamp"].dt.tz_convert(settings.session.timezone)
    session_minutes = exchange_timestamp.dt.hour * 60 + exchange_timestamp.dt.minute
    open_minutes = settings.session.regular_market_open.hour * 60 + settings.session.regular_market_open.minute
    close_minutes = settings.session.regular_market_close.hour * 60 + settings.session.regular_market_close.minute
    outside_regular_hours_count = int(((session_minutes < open_minutes) | (session_minutes > close_minutes)).fillna(False).sum())

    diffs = (
        working.sort_values(["symbol", "timestamp"])
        .groupby("symbol")["timestamp"]
        .diff()
        .dt.total_seconds()
    )
    gap_threshold = settings.feature_pipeline.gap_threshold_seconds
    large_gap_mask = diffs > gap_threshold
    large_gap_count = int(large_gap_mask.fillna(False).sum())
    max_gap_seconds_observed = float(diffs.max()) if not diffs.dropna().empty else 0.0

    null_counts = {column: int(working[column].isna().sum()) for column in ["timestamp", "last_price", "bid", "ask", "volume"]}
    null_counts["symbol"] = missing_symbol_count
    issues: list[str] = []
    if missing_timestamp_count:
        issues.append("missing_timestamps")
    if duplicate_count:
        issues.append("duplicate_rows")
    if critical_null_count:
        issues.append("critical_null_rows")
    if bid_gt_ask_count:
        issues.append("bid_gt_ask")
    if absurd_spread_count:
        issues.append("absurd_spreads")
    if outside_regular_hours_count:
        issues.append("outside_regular_hours")
    if large_gap_count:
        issues.append("large_gaps")

    return DataQualityReport(
        row_count=int(len(working)),
        symbol_count=int(working["symbol"].nunique()),
        missing_timestamp_count=missing_timestamp_count,
        duplicate_count=duplicate_count,
        critical_null_count=critical_null_count,
        bid_gt_ask_count=bid_gt_ask_count,
        negative_spread_count=negative_spread_count,
        absurd_spread_count=absurd_spread_count,
        outside_regular_hours_count=outside_regular_hours_count,
        large_gap_count=large_gap_count,
        max_gap_seconds_observed=max_gap_seconds_observed,
        null_counts=null_counts,
        issues=issues,
    )
