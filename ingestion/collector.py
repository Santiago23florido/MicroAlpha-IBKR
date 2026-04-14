from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from data.live_data import LiveDataService


def collect_market_data(
    settings: Settings,
    client: Any,
    *,
    symbol: str | None = None,
    duration: str = "1 D",
    bar_size: str = "1 min",
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    ticker = (symbol or settings.ib_symbol).upper()
    collector_root = Path(output_root or settings.paths.raw_dir) / "collector" / ticker.lower()

    try:
        client.connect()
        live_data = LiveDataService(client, settings)
        snapshot = live_data.fetch_market_snapshot(ticker)
        bars = live_data.fetch_intraday_bars(ticker, duration=duration, bar_size=bar_size)
        saved_paths = persist_collection_payload(collector_root, snapshot.to_dict(), bars)
        result = {
            "status": "ok",
            "symbol": ticker,
            "snapshot_source": snapshot.source,
            "bars_rows": int(len(bars)),
            "output_root": str(collector_root),
            **saved_paths,
        }
        client.logger.info(
            "Collector captured %s snapshot_source=%s bars_rows=%s",
            ticker,
            snapshot.source,
            len(bars),
        )
        return result
    finally:
        with suppress(Exception):
            client.disconnect()


def persist_collection_payload(
    collector_root: Path,
    snapshot_payload: dict[str, Any],
    bars: pd.DataFrame,
) -> dict[str, str]:
    collector_root.mkdir(parents=True, exist_ok=True)
    timestamp_token = _timestamp_token(str(snapshot_payload.get("timestamp") or snapshot_payload.get("snapshot_utc") or "latest"))
    snapshot_path = collector_root / f"{timestamp_token}_snapshot.json"
    bars_path = collector_root / f"{timestamp_token}_bars.csv"
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2, sort_keys=True), encoding="utf-8")
    bars.to_csv(bars_path, index=False)
    return {"snapshot_path": str(snapshot_path), "bars_path": str(bars_path)}


def _timestamp_token(value: str) -> str:
    return value.replace(":", "").replace("-", "").replace("+", "_").replace("T", "_")
