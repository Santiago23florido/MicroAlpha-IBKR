from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import Settings


def normalize_historical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ts_event": "timestamp",
        "ts_recv": "received_timestamp",
        "symbol": "symbol",
        "bid_px_00": "bid",
        "ask_px_00": "ask",
        "bid_sz_00": "bid_size",
        "ask_sz_00": "ask_size",
        "price": "last",
        "size": "last_size",
    }
    normalized = frame.rename(columns={key: value for key, value in rename_map.items() if key in frame.columns})
    if "timestamp" not in normalized.columns:
        raise ValueError("Historical dataset must include a timestamp or ts_event column.")
    if "symbol" not in normalized.columns:
        normalized["symbol"] = "SPY"
    if "last" not in normalized.columns:
        normalized["last"] = normalized.get("close") or normalized.get("mid")
    if "bid" not in normalized.columns and "last" in normalized.columns:
        normalized["bid"] = normalized["last"] - 0.01
    if "ask" not in normalized.columns and "last" in normalized.columns:
        normalized["ask"] = normalized["last"] + 0.01
    if "bid_size" not in normalized.columns:
        normalized["bid_size"] = 100.0
    if "ask_size" not in normalized.columns:
        normalized["ask_size"] = 100.0
    if "volume" not in normalized.columns:
        normalized["volume"] = normalized.get("size", 0.0)
    if "high" not in normalized.columns:
        normalized["high"] = normalized[["last", "ask"]].max(axis=1)
    if "low" not in normalized.columns:
        normalized["low"] = normalized[["last", "bid"]].min(axis=1)
    if "open" not in normalized.columns:
        normalized["open"] = normalized["last"]
    if "close" not in normalized.columns:
        normalized["close"] = normalized["last"]
    return normalized


def load_historical_dataset(settings: Settings, path: str | None = None) -> pd.DataFrame:
    if path:
        dataset_path = Path(path)
    else:
        dataset_path = Path("data/sample/spy_microstructure_sample.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found at {dataset_path}. "
            "Provide --data-path or add a local CSV/Parquet dataset."
        )

    if dataset_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(dataset_path)
    else:
        frame = pd.read_csv(dataset_path)
    return normalize_historical_frame(frame)
