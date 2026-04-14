from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from config import Settings


RAW_MARKET_COLUMNS = [
    "timestamp",
    "symbol",
    "last_price",
    "bid",
    "ask",
    "spread",
    "bid_size",
    "ask_size",
    "last_size",
    "volume",
    "event_type",
    "source",
    "session_window",
    "is_market_open",
    "exchange_time",
    "collected_at",
]


def load_market_data(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    input_root: str | Path | None = None,
) -> pd.DataFrame:
    root_dir = Path(input_root or settings.paths.market_raw_dir)
    files = list_market_data_files(root_dir, symbols=symbols or settings.supported_symbols, start_date=start_date, end_date=end_date)
    if not files:
        raise FileNotFoundError(
            f"No market parquet files found under {root_dir}. "
            "Run pull-from-pc2 on PC1, or pass --input-root with a populated market directory."
        )

    frames = [_read_market_file(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined["collected_at"] = pd.to_datetime(combined["collected_at"], utc=True, errors="coerce")
    combined["exchange_time"] = pd.to_datetime(combined["exchange_time"], utc=True, errors="coerce")
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.sort_values(["symbol", "timestamp", "source_file"]).reset_index(drop=True)
    return combined


def list_market_data_files(
    root_dir: str | Path,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []

    requested_symbols = {symbol.upper() for symbol in symbols} if symbols else None
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)

    files: list[Path] = []
    for date_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        session_date = _coerce_date(date_dir.name)
        if session_date is None:
            continue
        if start and session_date < start:
            continue
        if end and session_date > end:
            continue
        files.extend(_list_date_partition_files(date_dir, requested_symbols))

    return sorted(dict.fromkeys(files))


def _list_date_partition_files(date_dir: Path, requested_symbols: set[str] | None) -> list[Path]:
    files: list[Path] = []
    top_level_files = [path for path in date_dir.glob("*.parquet") if path.is_file()]
    for path in top_level_files:
        symbol = path.stem.upper()
        if requested_symbols is None or symbol in requested_symbols:
            files.append(path)

    for symbol_dir in sorted(path for path in date_dir.iterdir() if path.is_dir()):
        symbol = symbol_dir.name.upper()
        if requested_symbols is not None and symbol not in requested_symbols:
            continue
        files.extend(sorted(path for path in symbol_dir.glob("*.parquet") if path.is_file()))

    return files


def _read_market_file(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = _standardize_market_frame(frame, path)
    frame["source_file"] = str(path)
    return frame


def _standardize_market_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    normalized = frame.copy()
    rename_map = {
        "last": "last_price",
        "bid_sz": "bid_size",
        "ask_sz": "ask_size",
    }
    normalized = normalized.rename(columns={key: value for key, value in rename_map.items() if key in normalized.columns})

    inferred_symbol = path.parent.name.upper() if _coerce_date(path.parent.name) is None else path.stem.upper()
    if "symbol" not in normalized.columns:
        normalized["symbol"] = inferred_symbol

    if "timestamp" not in normalized.columns:
        raise ValueError(f"Market file {path} is missing the required 'timestamp' column.")

    for column in RAW_MARKET_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    return normalized[RAW_MARKET_COLUMNS]


def _coerce_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None
