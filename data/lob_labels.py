from __future__ import annotations

import numpy as np
import pandas as pd


def attach_lob_mid_price_labels(
    frame: pd.DataFrame,
    *,
    horizon_events: int,
    stationary_threshold_bps: float,
    estimated_roundtrip_cost_bps: float = 0.0,
    edge_buffer_bps: float = 0.0,
) -> pd.DataFrame:
    labeled = frame.copy()
    if "session_date" not in labeled.columns:
        labeled["session_date"] = pd.to_datetime(labeled["event_ts_utc"], utc=True).dt.date.astype(str)

    labeled["mid_price"] = np.where(
        (pd.to_numeric(labeled["bid_px_1"], errors="coerce") > 0)
        & (pd.to_numeric(labeled["ask_px_1"], errors="coerce") > 0),
        (pd.to_numeric(labeled["bid_px_1"], errors="coerce") + pd.to_numeric(labeled["ask_px_1"], errors="coerce")) / 2.0,
        np.nan,
    )
    labeled["future_mid_price"] = labeled.groupby("session_date", sort=False)["mid_price"].shift(-horizon_events)
    labeled["future_return_bps"] = (
        (labeled["future_mid_price"] / labeled["mid_price"]) - 1.0
    ) * 10000.0
    labeled["spread_bps"] = np.where(
        labeled["mid_price"] > 0,
        ((pd.to_numeric(labeled["ask_px_1"], errors="coerce") - pd.to_numeric(labeled["bid_px_1"], errors="coerce")) / labeled["mid_price"]) * 10000.0,
        np.nan,
    )
    labeled["estimated_roundtrip_cost_bps"] = float(estimated_roundtrip_cost_bps)
    labeled["edge_buffer_bps"] = float(edge_buffer_bps)
    labeled["net_future_return_bps"] = labeled["future_return_bps"] - labeled["estimated_roundtrip_cost_bps"]
    labeled["cost_aware_threshold_bps"] = labeled["estimated_roundtrip_cost_bps"] + labeled["edge_buffer_bps"]
    labeled["target_class"] = np.select(
        [
            labeled["future_return_bps"] > stationary_threshold_bps,
            labeled["future_return_bps"] < -stationary_threshold_bps,
        ],
        [1, -1],
        default=0,
    )
    labeled["target_class_cost_aware"] = np.select(
        [
            labeled["future_return_bps"] > labeled["cost_aware_threshold_bps"],
            labeled["future_return_bps"] < -labeled["cost_aware_threshold_bps"],
        ],
        [1, -1],
        default=0,
    )
    labeled = attach_lob_intraday_momentum_features(labeled)
    labeled = labeled.dropna(subset=["mid_price", "future_mid_price", "future_return_bps"]).reset_index(drop=True)
    return labeled


def attach_lob_intraday_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "event_ts_utc" in enriched.columns:
        enriched["event_ts_utc"] = pd.to_datetime(enriched["event_ts_utc"], utc=True)
    for window in [10, 50, 100]:
        grouped_mid = enriched.groupby("session_date", sort=False)["mid_price"]
        enriched[f"momentum_{window}_events_bps"] = grouped_mid.pct_change(periods=window) * 10000.0
        enriched[f"rolling_volatility_{window}_events_bps"] = grouped_mid.pct_change().groupby(enriched["session_date"], sort=False).rolling(window).std().reset_index(level=0, drop=True) * 10000.0
    bid_size = pd.to_numeric(enriched.get("bid_sz_1", 0.0), errors="coerce").fillna(0.0)
    ask_size = pd.to_numeric(enriched.get("ask_sz_1", 0.0), errors="coerce").fillna(0.0)
    denominator = (bid_size + ask_size).replace(0.0, np.nan)
    enriched["top_level_imbalance"] = ((bid_size - ask_size) / denominator).fillna(0.0)
    enriched["spread_regime_bps"] = pd.to_numeric(enriched["spread_bps"], errors="coerce").fillna(0.0)
    momentum_columns = [
        "momentum_10_events_bps",
        "momentum_50_events_bps",
        "momentum_100_events_bps",
        "rolling_volatility_10_events_bps",
        "rolling_volatility_50_events_bps",
        "rolling_volatility_100_events_bps",
    ]
    enriched[momentum_columns] = enriched[momentum_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return enriched
