from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.io import write_json


BAR_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "last",
    "volume",
    "count",
    "wap",
    "source",
    "provider",
    "bar_size",
    "what_to_show",
    "collected_at",
]

TICK_COLUMNS = [
    "timestamp",
    "symbol",
    "price",
    "size",
    "what_to_show",
    "source",
    "provider",
    "collected_at",
]


def normalize_ibkr_bar_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    bar_size: str,
    what_to_show: str,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("IBKR historical bar frame is empty.")
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    working["symbol"] = str(symbol).upper()
    working["last"] = pd.to_numeric(working.get("last", working.get("close")), errors="coerce")
    working["close"] = pd.to_numeric(working["close"], errors="coerce")
    working["last"] = working["last"].where(working["last"].notna(), working["close"])
    for column in ("open", "high", "low", "volume"):
        working[column] = pd.to_numeric(working.get(column, working["close"]), errors="coerce")
    working["count"] = pd.to_numeric(working.get("count", working.get("bar_count")), errors="coerce")
    working["wap"] = pd.to_numeric(working.get("wap", working.get("average")), errors="coerce")
    working["source"] = "ibkr_historical_backfill"
    working["provider"] = "ibkr"
    working["bar_size"] = bar_size
    working["what_to_show"] = what_to_show
    working["collected_at"] = datetime.now(timezone.utc).isoformat()
    normalized = working.loc[:, [column for column in BAR_COLUMNS if column in working.columns]].copy()
    normalized = normalized.sort_values(["timestamp", "symbol"]).drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return normalized.reset_index(drop=True)


def export_ibkr_training_csv(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    symbol: str,
    synthetic_spread_bps: float,
    default_depth_size: float,
    write_parquet: bool,
    write_manifest: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot export an empty IBKR training dataset.")
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    working["symbol"] = str(symbol).upper()
    working["last"] = pd.to_numeric(working.get("last", working["close"]), errors="coerce").where(lambda s: s.notna(), working["close"])
    close = pd.to_numeric(working["close"], errors="coerce")
    half_spread = synthetic_spread_bps / 20000.0
    working["bid"] = close * (1.0 - half_spread)
    working["ask"] = close * (1.0 + half_spread)
    working["bid_size"] = float(default_depth_size)
    working["ask_size"] = float(default_depth_size)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    working.to_csv(target, index=False)

    parquet_path = None
    if write_parquet:
        parquet_target = target.with_suffix(".parquet")
        working.to_parquet(parquet_target, index=False)
        parquet_path = str(parquet_target)

    manifest_path = None
    if write_manifest:
        manifest_payload = {
            "provider": "ibkr",
            "source": "ibkr_historical_backfill",
            "symbol": str(symbol).upper(),
            "row_count": int(len(working)),
            "synthetic_columns": ["bid", "ask", "bid_size", "ask_size"],
            "real_columns": [column for column in working.columns if column not in {"bid", "ask", "bid_size", "ask_size"}],
            "output_path": str(target),
            "parquet_path": parquet_path,
            **(metadata or {}),
        }
        manifest_path = write_json(target.with_suffix(".manifest.json"), manifest_payload)

    return {
        "status": "ok",
        "output_path": str(target),
        "parquet_path": parquet_path,
        "manifest_path": manifest_path,
        "row_count": int(len(working)),
        "columns": list(working.columns),
    }
