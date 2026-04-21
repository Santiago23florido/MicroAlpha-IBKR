from __future__ import annotations

import numpy as np
import pandas as pd


def attach_lob_mid_price_labels(
    frame: pd.DataFrame,
    *,
    horizon_events: int,
    stationary_threshold_bps: float,
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
    labeled["target_class"] = np.select(
        [
            labeled["future_return_bps"] > stationary_threshold_bps,
            labeled["future_return_bps"] < -stationary_threshold_bps,
        ],
        [1, -1],
        default=0,
    )
    labeled = labeled.dropna(subset=["mid_price", "future_mid_price", "future_return_bps"]).reset_index(drop=True)
    return labeled
