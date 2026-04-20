from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

import pandas as pd

from config.polygon import PolygonBootstrapConfig


@dataclass(frozen=True)
class PolygonRequestSpec:
    symbol: str
    start_date: str
    end_date: str
    multiplier: int
    timespan: str


class PolygonClient:
    def __init__(self, config: PolygonBootstrapConfig) -> None:
        self.config = config

    def fetch_aggregates(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame:
        if not self.config.api_key:
            raise ValueError("POLYGON_API_KEY is required when using Polygon API mode.")
        spec = self._build_request_spec(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval)
        rows: list[dict[str, Any]] = []
        next_url = self._build_initial_url(spec)
        while next_url:
            payload = self._request_json(next_url)
            results = payload.get("results", [])
            rows.extend(results if isinstance(results, list) else [])
            next_url = payload.get("next_url")
            if next_url:
                separator = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{separator}apiKey={self.config.api_key}"
                time.sleep(max(self.config.api_rate_limit_sleep_seconds, 0.0))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("Polygon returned an empty dataset for the requested range.")
        frame["symbol"] = spec.symbol
        frame["source_interval"] = interval
        frame["collected_at"] = datetime.now(timezone.utc).isoformat()
        return frame

    def _build_request_spec(self, *, symbol: str, start_date: str, end_date: str, interval: str) -> PolygonRequestSpec:
        cleaned_symbol = str(symbol).strip().upper()
        if not cleaned_symbol or not cleaned_symbol.replace(".", "").isalnum():
            raise ValueError(f"Invalid symbol for Polygon download: {symbol!r}")
        _parse_date(start_date, label="start_date")
        _parse_date(end_date, label="end_date")
        multiplier, timespan = parse_polygon_interval(interval)
        return PolygonRequestSpec(
            symbol=cleaned_symbol,
            start_date=start_date,
            end_date=end_date,
            multiplier=multiplier,
            timespan=timespan,
        )

    def _build_initial_url(self, spec: PolygonRequestSpec) -> str:
        path = f"/v2/aggs/ticker/{spec.symbol}/range/{spec.multiplier}/{spec.timespan}/{spec.start_date}/{spec.end_date}"
        query = urlencode(
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": self.config.api_key,
            }
        )
        return urljoin(f"{self.config.api_base_url}/", path.lstrip("/")) + f"?{query}"

    def _request_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": "MicroAlpha-IBKR Polygon Bootstrap/1.0"})
        with urlopen(request, timeout=self.config.request_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Polygon API returned a non-object payload.")
        status = str(payload.get("status") or "").upper()
        if status and status not in {"OK", "DELAYED"}:
            raise ValueError(f"Polygon API returned status={status!r}: {payload}")
        return payload


def parse_polygon_interval(interval: str) -> tuple[int, str]:
    value = str(interval).strip().lower()
    aliases = {
        "m": "minute",
        "min": "minute",
        "minute": "minute",
        "minutes": "minute",
        "h": "hour",
        "hour": "hour",
        "d": "day",
        "day": "day",
    }
    digits = "".join(character for character in value if character.isdigit())
    unit = "".join(character for character in value if character.isalpha())
    if not digits:
        raise ValueError(f"Invalid Polygon interval {interval!r}. Expected forms like '1m' or '5minute'.")
    timespan = aliases.get(unit)
    if timespan is None:
        raise ValueError(f"Unsupported Polygon interval unit {unit!r}.")
    return int(digits), timespan


def _parse_date(value: str, *, label: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {value!r}. Use YYYY-MM-DD.") from exc
