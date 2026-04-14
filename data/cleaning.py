from __future__ import annotations

import numpy as np
import pandas as pd

from config import Settings


CLEAN_NUMERIC_COLUMNS = [
    "last_price",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "last_size",
    "volume",
]


def clean_market_data(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")
    cleaned["symbol"] = cleaned["symbol"].astype(str).str.upper()

    for column in CLEAN_NUMERIC_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["timestamp", "symbol"]).copy()
    cleaned = cleaned.sort_values(["symbol", "timestamp", "collected_at"]).reset_index(drop=True)
    cleaned = cleaned.drop_duplicates(subset=["symbol", "timestamp"], keep="last").reset_index(drop=True)

    cleaned["exchange_timestamp"] = cleaned["timestamp"].dt.tz_convert(settings.session.timezone)
    cleaned["session_date"] = cleaned["exchange_timestamp"].dt.strftime("%Y-%m-%d")
    cleaned["session_time"] = cleaned["exchange_timestamp"].dt.time

    grouped = cleaned.groupby(["symbol", "session_date"], sort=False)
    ffill_columns = ["bid", "ask", "bid_size", "ask_size", "last_price"]
    for column in ffill_columns:
        cleaned[column] = grouped[column].transform(lambda series: series.ffill(limit=settings.feature_pipeline.forward_fill_limit))

    cleaned["mid_price"] = np.where(
        cleaned["bid"].notna() & cleaned["ask"].notna(),
        (cleaned["bid"] + cleaned["ask"]) / 2.0,
        cleaned["last_price"],
    )
    cleaned["last_price"] = cleaned["last_price"].fillna(cleaned["mid_price"])
    cleaned["volume"] = cleaned["volume"].fillna(0.0)
    cleaned["spread"] = cleaned["ask"] - cleaned["bid"]
    cleaned["spread_bps"] = np.where(
        cleaned["mid_price"] > 0,
        cleaned["spread"] / cleaned["mid_price"] * 10000.0,
        np.nan,
    )

    cleaned = cleaned[~(cleaned["bid"].notna() & cleaned["ask"].notna() & (cleaned["bid"] > cleaned["ask"]))].copy()
    cleaned = cleaned[~(cleaned["spread_bps"].abs() > settings.feature_pipeline.max_abs_spread_bps)].copy()

    if settings.feature_pipeline.drop_outside_regular_hours:
        session_minutes = cleaned["exchange_timestamp"].dt.hour * 60 + cleaned["exchange_timestamp"].dt.minute
        open_minutes = settings.session.regular_market_open.hour * 60 + settings.session.regular_market_open.minute
        close_minutes = settings.session.regular_market_close.hour * 60 + settings.session.regular_market_close.minute
        cleaned = cleaned[(session_minutes >= open_minutes) & (session_minutes <= close_minutes)].copy()

    cleaned["is_market_open"] = cleaned["is_market_open"].fillna(True).astype(bool)
    cleaned = cleaned.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return cleaned
