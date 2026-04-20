from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class HistoricalChunk:
    index: int
    start_utc: str
    end_utc: str
    duration_str: str
    end_datetime_ib: str


def plan_historical_bar_chunks(
    *,
    earliest_timestamp: str,
    latest_timestamp: str | None = None,
    start_timestamp: str | None = None,
    bar_size: str,
    chunk_days_1m: int,
    chunk_days_intraday_fallback: int,
) -> list[HistoricalChunk]:
    start = _parse_iso(earliest_timestamp)
    if start_timestamp:
        start = max(start, _parse_iso(start_timestamp))
    end = _parse_iso(latest_timestamp) if latest_timestamp else datetime.now(timezone.utc)
    if end <= start:
        raise ValueError("Latest timestamp must be after the earliest timestamp.")
    chunk_days = chunk_days_1m if bar_size.strip().lower() == "1 min" else chunk_days_intraday_fallback
    chunk_span = timedelta(days=max(1, chunk_days))

    chunks: list[HistoricalChunk] = []
    chunk_end = end
    index = 0
    while chunk_end > start:
        chunk_start = max(start, chunk_end - chunk_span)
        chunks.append(
            HistoricalChunk(
                index=index,
                start_utc=chunk_start.isoformat(),
                end_utc=chunk_end.isoformat(),
                duration_str=f"{max(1, int((chunk_end - chunk_start).total_seconds() // 86400) or 1)} D",
                end_datetime_ib=_to_ib_datetime(chunk_end),
            )
        )
        chunk_end = chunk_start
        index += 1
    chunks.reverse()
    return chunks


def validate_what_to_show(value: str) -> str:
    normalized = str(value).strip().upper()
    allowed = {"TRADES", "MIDPOINT", "BID", "ASK", "BID_ASK"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported whatToShow value: {value!r}. Allowed: {sorted(allowed)}")
    return normalized


def validate_bar_size(value: str) -> str:
    normalized = " ".join(str(value).strip().split())
    allowed = {
        "1 min",
        "2 mins",
        "3 mins",
        "5 mins",
        "15 mins",
        "30 mins",
        "1 hour",
        "1 day",
    }
    if normalized not in allowed:
        raise ValueError(f"Unsupported bar size: {value!r}. Allowed: {sorted(allowed)}")
    return normalized


def validate_symbol(value: str) -> str:
    normalized = str(value).strip().upper()
    if not normalized or not normalized.replace(".", "").isalnum():
        raise ValueError(f"Invalid symbol: {value!r}")
    return normalized


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_ib_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d-%H:%M:%S")
