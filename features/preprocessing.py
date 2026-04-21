from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import Settings

FEATURE_COLUMNS = [
    "spread_bps",
    "mid_price",
    "microprice",
    "depth_imbalance",
    "ofi_proxy",
    "multi_level_ofi",
    "return_1_bps",
    "return_3_bps",
    "rolling_volatility_5",
    "recent_volume_imbalance",
    "orb_range_width_bps",
    "price_vs_orb_high_bps",
    "price_vs_orb_low_bps",
    "price_vs_orb_mid_bps",
    "breakout_distance_bps",
    "is_primary_session",
    "is_secondary_session",
    "estimated_cost_bps",
]

BAR_FEATURE_COLUMNS = [
    "mid_price",
    "return_1_bps",
    "return_3_bps",
    "return_10_bps",
    "rolling_volatility_5",
    "rolling_volatility_15",
    "hl_range_bps",
    "open_close_return_bps",
    "close_position_in_bar",
    "orb_range_width_bps",
    "price_vs_orb_high_bps",
    "price_vs_orb_low_bps",
    "price_vs_orb_mid_bps",
    "breakout_distance_bps",
    "is_primary_session",
    "is_secondary_session",
    "estimated_cost_bps",
]


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    class_threshold_bps: float
    target_horizon_minutes: int
    training_profile: str


def prepare_training_dataframe(raw_frame: pd.DataFrame, settings: Settings) -> PreparedDataset:
    frame = raw_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["session_timestamp"] = frame["timestamp"].dt.tz_convert(settings.session.timezone)
    frame["mid_price"] = np.where(
        frame["bid"].notna() & frame["ask"].notna(),
        (frame["bid"] + frame["ask"]) / 2.0,
        frame["last"],
    )
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["spread_bps"] = np.where(
        frame["mid_price"] > 0,
        frame["spread"] / frame["mid_price"] * 10000.0,
        np.nan,
    )
    frame["microprice"] = np.where(
        (frame["bid_size"] + frame["ask_size"]) > 0,
        ((frame["ask"] * frame["bid_size"]) + (frame["bid"] * frame["ask_size"]))
        / (frame["bid_size"] + frame["ask_size"]),
        frame["mid_price"],
    )
    frame["depth_imbalance"] = np.where(
        (frame["bid_size"] + frame["ask_size"]) > 0,
        (frame["bid_size"] - frame["ask_size"]) / (frame["bid_size"] + frame["ask_size"]),
        0.0,
    )
    frame["ofi_proxy"] = frame["bid_size"].diff().fillna(0.0) - frame["ask_size"].diff().fillna(0.0)
    multi_level_columns = [
        column
        for column in frame.columns
        if column.startswith("bid_size_") or column.startswith("ask_size_")
    ]
    if multi_level_columns:
        bid_columns = [column for column in multi_level_columns if column.startswith("bid_size_")]
        ask_columns = [column for column in multi_level_columns if column.startswith("ask_size_")]
        total_bid_depth = frame[bid_columns].sum(axis=1)
        total_ask_depth = frame[ask_columns].sum(axis=1)
        frame["multi_level_ofi"] = np.where(
            (total_bid_depth + total_ask_depth) > 0,
            (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth),
            0.0,
    )
    else:
        frame["multi_level_ofi"] = 0.0
    frame["return_1_bps"] = frame["mid_price"].pct_change().fillna(0.0) * 10000.0
    frame["return_3_bps"] = frame["mid_price"].pct_change(3).fillna(0.0) * 10000.0
    frame["return_10_bps"] = frame["mid_price"].pct_change(10).fillna(0.0) * 10000.0
    frame["rolling_volatility_5"] = frame["return_1_bps"].rolling(5).std().fillna(0.0)
    frame["rolling_volatility_15"] = frame["return_1_bps"].rolling(15).std().fillna(0.0)
    frame["recent_volume_imbalance"] = frame["depth_imbalance"]
    frame["hl_range_bps"] = np.where(
        frame["mid_price"] > 0,
        (frame["high"] - frame["low"]) / frame["mid_price"] * 10000.0,
        0.0,
    )
    frame["open_close_return_bps"] = np.where(
        frame["open"] > 0,
        (frame["close"] / frame["open"] - 1.0) * 10000.0,
        0.0,
    )
    frame["close_position_in_bar"] = np.where(
        (frame["high"] - frame["low"]) > 0,
        (frame["close"] - frame["low"]) / (frame["high"] - frame["low"]),
        0.5,
    )

    frame = _add_orb_context(frame, settings)
    training_profile = _resolve_training_profile(frame)
    if training_profile == "bar_bootstrap":
        frame["estimated_cost_bps"] = np.maximum(frame["spread_bps"].fillna(0.0), settings.trading.cost_buffer_bps)
        target_horizon_minutes = max(settings.models.target_horizon_minutes, 10)
        class_threshold_bps = max(4.0, settings.trading.cost_buffer_bps * 1.5)
        selected_feature_columns = list(BAR_FEATURE_COLUMNS)
    else:
        frame["estimated_cost_bps"] = frame["spread_bps"].fillna(0.0) + 0.5
        target_horizon_minutes = settings.models.target_horizon_minutes
        class_threshold_bps = max(1.0, settings.trading.cost_buffer_bps / 2.0)
        selected_feature_columns = list(FEATURE_COLUMNS)

    frame["future_mid_price"] = frame["mid_price"].shift(-target_horizon_minutes)
    frame["future_return_bps"] = (
        (frame["future_mid_price"] / frame["mid_price"]) - 1.0
    ) * 10000.0
    frame["future_net_return_bps"] = np.where(
        frame["future_return_bps"].notna(),
        frame["future_return_bps"] - (np.sign(frame["future_return_bps"]) * frame["estimated_cost_bps"]),
        np.nan,
    )
    frame["target_class"] = np.select(
        [
            frame["future_net_return_bps"] > class_threshold_bps,
            frame["future_net_return_bps"] < -class_threshold_bps,
        ],
        [1, -1],
        default=0,
    )
    frame = frame.dropna(subset=["future_return_bps"]).reset_index(drop=True)
    frame[selected_feature_columns] = frame[selected_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return PreparedDataset(
        frame=frame,
        feature_columns=selected_feature_columns,
        class_threshold_bps=class_threshold_bps,
        target_horizon_minutes=target_horizon_minutes,
        training_profile=training_profile,
    )


def build_feature_vector(feature_values: dict[str, float | None], feature_columns: list[str]) -> np.ndarray:
    return np.asarray([float(feature_values.get(column) or 0.0) for column in feature_columns], dtype=float)


def _add_orb_context(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    exchange_frame = frame["session_timestamp"]
    frame["session_date"] = exchange_frame.dt.date
    frame["session_time"] = exchange_frame.dt.time
    range_mask = (
        (frame["session_time"] >= settings.session.orb_start)
        & (frame["session_time"] < settings.session.orb_end)
    )
    range_summary = (
        frame.loc[range_mask]
        .groupby("session_date")
        .agg(orb_range_high=("high", "max"), orb_range_low=("low", "min"))
    )
    frame = frame.merge(range_summary, how="left", left_on="session_date", right_index=True)
    frame["orb_range_mid"] = (frame["orb_range_high"] + frame["orb_range_low"]) / 2.0
    frame["orb_range_width"] = frame["orb_range_high"] - frame["orb_range_low"]
    frame["orb_range_width_bps"] = np.where(
        frame["orb_range_mid"] > 0,
        frame["orb_range_width"] / frame["orb_range_mid"] * 10000.0,
        0.0,
    )
    frame["price_vs_orb_high_bps"] = np.where(
        frame["orb_range_high"] > 0,
        (frame["mid_price"] - frame["orb_range_high"]) / frame["orb_range_high"] * 10000.0,
        0.0,
    )
    frame["price_vs_orb_low_bps"] = np.where(
        frame["orb_range_low"] > 0,
        (frame["mid_price"] - frame["orb_range_low"]) / frame["orb_range_low"] * 10000.0,
        0.0,
    )
    frame["price_vs_orb_mid_bps"] = np.where(
        frame["orb_range_mid"] > 0,
        (frame["mid_price"] - frame["orb_range_mid"]) / frame["orb_range_mid"] * 10000.0,
        0.0,
    )
    frame["breakout_distance_bps"] = np.where(
        frame["price_vs_orb_high_bps"] > 0,
        frame["price_vs_orb_high_bps"],
        np.where(frame["price_vs_orb_low_bps"] < 0, -frame["price_vs_orb_low_bps"], 0.0),
    )
    frame["is_primary_session"] = (
        (frame["session_time"] >= settings.session.orb_end)
        & (frame["session_time"] <= settings.session.primary_session_end)
    ).astype(float)
    frame["is_secondary_session"] = (
        settings.session.enable_secondary_session
        & (frame["session_time"] >= settings.session.secondary_session_start)
        & (frame["session_time"] <= settings.session.secondary_session_end)
    ).astype(float)
    return frame


def _resolve_training_profile(frame: pd.DataFrame) -> str:
    provider = str(frame.get("provider", pd.Series([""])).iloc[0]).lower() if "provider" in frame.columns else ""
    what_to_show = (
        str(frame.get("what_to_show", pd.Series([""])).iloc[0]).upper() if "what_to_show" in frame.columns else ""
    )
    synthetic_bid_ask = bool(frame.get("synthetic_bid_ask_flag", pd.Series([False])).astype(bool).any()) if "synthetic_bid_ask_flag" in frame.columns else False
    synthetic_depth = bool(frame.get("synthetic_depth_flag", pd.Series([False])).astype(bool).any()) if "synthetic_depth_flag" in frame.columns else False
    invalid_trade_fields = False
    if "volume" in frame.columns:
        invalid_trade_fields = invalid_trade_fields or pd.to_numeric(frame["volume"], errors="coerce").fillna(-1).le(0).all()
    if "count" in frame.columns:
        invalid_trade_fields = invalid_trade_fields or pd.to_numeric(frame["count"], errors="coerce").fillna(-1).lt(0).all()
    if provider == "ibkr" and what_to_show == "MIDPOINT":
        return "bar_bootstrap"
    if synthetic_bid_ask or synthetic_depth or invalid_trade_fields:
        return "bar_bootstrap"
    return "microstructure"
