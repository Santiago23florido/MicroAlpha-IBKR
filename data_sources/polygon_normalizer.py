from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config.polygon import PolygonBootstrapConfig


CANONICAL_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "last",
    "volume",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "source",
    "provider",
    "interval",
    "event_type",
    "collected_at",
    "bootstrap_source",
    "market_data_mode",
    "synthetic_bid_ask_flag",
    "synthetic_depth_flag",
]


def normalize_polygon_frame(
    frame: pd.DataFrame,
    *,
    symbol: str | None,
    interval: str,
    config: PolygonBootstrapConfig,
    source_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        raise ValueError("Cannot normalize an empty Polygon dataset.")
    working = frame.copy()
    column_mapping = _detect_polygon_columns(working.columns)
    if "timestamp" not in column_mapping or "close" not in column_mapping:
        raise ValueError("Polygon dataset must contain a timestamp-like column and a close-like column.")
    present_fields = set(column_mapping)
    working = working.rename(columns={source: target for target, source in column_mapping.items()})
    working["timestamp"] = _normalize_timestamp_series(working["timestamp"])
    if working["timestamp"].isna().all():
        raise ValueError("Polygon dataset has no usable timestamp values after normalization.")

    resolved_symbol = str(symbol or working.get("symbol", pd.Series([""])).iloc[0] or config.default_symbol).upper()
    if not resolved_symbol or not resolved_symbol.replace(".", "").isalnum():
        raise ValueError(f"Invalid symbol for normalization: {resolved_symbol!r}")
    working["symbol"] = resolved_symbol
    real_columns = {"timestamp", "close"}
    if "symbol" in present_fields:
        real_columns.add("symbol")
    synthetic_columns = set()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in working.columns:
            if column == "open":
                working[column] = working["close"]
            elif column == "high":
                working[column] = working["close"]
            elif column == "low":
                working[column] = working["close"]
            elif column == "volume":
                working[column] = 0.0
            synthetic_columns.add(column)
        elif column != "close":
            real_columns.add(column)

    working["last"] = pd.to_numeric(working.get("last", working["close"]), errors="coerce")
    working["close"] = pd.to_numeric(working["close"], errors="coerce")
    for column in ("open", "high", "low", "volume"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if working["close"].isna().all():
        raise ValueError("Polygon dataset has no usable close values after normalization.")
    if "last" in present_fields:
        real_columns.add("last")
    else:
        synthetic_columns.add("last")
    working["last"] = working["last"].where(working["last"].notna(), working["close"])

    bid_ask_synthetic = True
    depth_synthetic = True

    if "bid" in working.columns and "ask" in working.columns:
        working["bid"] = pd.to_numeric(working["bid"], errors="coerce")
        working["ask"] = pd.to_numeric(working["ask"], errors="coerce")
        bid_ask_synthetic = False
        real_columns.update({"bid", "ask"})
    else:
        half_spread_fraction = config.synthetic_spread_bps / 20000.0
        working["bid"] = working["close"] * (1.0 - half_spread_fraction)
        working["ask"] = working["close"] * (1.0 + half_spread_fraction)
        synthetic_columns.update({"bid", "ask"})

    if "bid_size" in working.columns and "ask_size" in working.columns:
        working["bid_size"] = pd.to_numeric(working["bid_size"], errors="coerce")
        working["ask_size"] = pd.to_numeric(working["ask_size"], errors="coerce")
        depth_synthetic = False
        real_columns.update({"bid_size", "ask_size"})
    else:
        working["bid_size"] = float(config.default_depth_size)
        working["ask_size"] = float(config.default_depth_size)
        synthetic_columns.update({"bid_size", "ask_size"})

    collected_at = str(working.get("collected_at", pd.Series([datetime.now(timezone.utc).isoformat()])).iloc[0])
    working["source"] = "polygon_basic"
    working["provider"] = "polygon"
    working["interval"] = interval
    working["event_type"] = "bar"
    working["collected_at"] = collected_at
    working["bootstrap_source"] = True
    working["market_data_mode"] = "ohlcv_bootstrap"
    working["synthetic_bid_ask_flag"] = bid_ask_synthetic
    working["synthetic_depth_flag"] = depth_synthetic

    normalized = working.loc[:, [column for column in CANONICAL_COLUMNS if column in working.columns]].copy()
    normalized = normalized.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    metadata = {
        "provider": "polygon",
        "source_mode": source_mode,
        "interval": interval,
        "symbol": resolved_symbol,
        "row_count": int(len(normalized)),
        "real_columns": sorted(real_columns),
        "synthetic_columns": sorted(synthetic_columns),
        "synthetic_bid_ask_flag": bid_ask_synthetic,
        "synthetic_depth_flag": depth_synthetic,
        "synthetic_spread_bps": config.synthetic_spread_bps,
        "default_depth_size": config.default_depth_size,
    }
    return normalized, metadata


def _detect_polygon_columns(columns: pd.Index) -> dict[str, str]:
    candidates = {str(column).strip().lower(): str(column) for column in columns}
    mapping: dict[str, str] = {}
    for target, aliases in {
        "timestamp": ("timestamp", "t", "window_start", "datetime", "date", "time"),
        "symbol": ("symbol", "ticker"),
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c"),
        "last": ("last", "price"),
        "volume": ("volume", "v"),
        "bid": ("bid", "bid_price"),
        "ask": ("ask", "ask_price"),
        "bid_size": ("bid_size", "bid_sz", "bidsize"),
        "ask_size": ("ask_size", "ask_sz", "asksize"),
        "collected_at": ("collected_at",),
    }.items():
        for alias in aliases:
            if alias in candidates:
                mapping[target] = candidates[alias]
                break
    return mapping


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        normalized = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        if normalized.notna().any():
            return normalized.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
