from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from broker.ib_client import IBClientError
from config import Settings
from engine.market_clock import MarketClock
from ingestion.ibkr_client import MarketDataClient, build_collector_ib_client
from ingestion.market_data import normalize_market_snapshot
from ingestion.persistence import ParquetMarketDataSink
from monitoring.logging import setup_logger


@dataclass(frozen=True)
class CollectorSummary:
    status: str
    symbols: tuple[str, ...]
    cycles: int
    records_collected: int
    records_persisted: int
    flushes: int
    reconnect_attempts: int
    output_root: str
    poll_interval_seconds: float
    flush_interval_seconds: float
    batch_size: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "symbols": list(self.symbols),
            "cycles": self.cycles,
            "records_collected": self.records_collected,
            "records_persisted": self.records_persisted,
            "flushes": self.flushes,
            "reconnect_attempts": self.reconnect_attempts,
            "output_root": self.output_root,
            "poll_interval_seconds": self.poll_interval_seconds,
            "flush_interval_seconds": self.flush_interval_seconds,
            "batch_size": self.batch_size,
            "message": self.message,
        }


class MarketDataCollector:
    def __init__(
        self,
        settings: Settings,
        client: MarketDataClient,
        sink: ParquetMarketDataSink,
        *,
        symbols: Sequence[str] | None = None,
        logger=None,
        poll_interval_seconds: float | None = None,
        health_log_interval_seconds: float | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.sink = sink
        self.logger = logger or client.logger
        self.symbols = tuple(symbol.upper() for symbol in (symbols or settings.supported_symbols))
        self.poll_interval_seconds = poll_interval_seconds or settings.collector.poll_interval_seconds
        self.health_log_interval_seconds = (
            health_log_interval_seconds or settings.collector.health_log_interval_seconds
        )
        self.market_clock = MarketClock(settings.session)

    def run(
        self,
        *,
        once: bool = False,
        max_cycles: int | None = None,
        max_runtime_seconds: float | None = None,
    ) -> dict[str, Any]:
        cycles = 0
        records_collected = 0
        reconnect_attempts = 0
        last_health_log_at = time.monotonic()
        started_at = time.monotonic()
        status = "ok"
        message = "Collector completed successfully."

        self.logger.info(
            "Starting market data collector: mode=%s symbols=%s poll_interval=%ss flush_interval=%ss batch_size=%s output_root=%s",
            self.settings.collector.mode,
            list(self.symbols),
            self.poll_interval_seconds,
            self.sink.flush_interval_seconds,
            self.sink.batch_size,
            self.sink.root_dir,
        )

        try:
            while True:
                if max_runtime_seconds is not None and (time.monotonic() - started_at) >= max_runtime_seconds:
                    message = "Collector stopped after reaching the configured runtime limit."
                    break

                cycle_started_at = time.monotonic()
                try:
                    self._ensure_connection()
                    records = self.poll_once()
                except IBClientError as exc:
                    reconnect_attempts += 1
                    self.logger.error("Collector cycle failed due to IBKR error: %s", exc)
                    if not self._should_retry(reconnect_attempts):
                        status = "error"
                        message = "Collector stopped after exceeding reconnect attempts."
                        break
                    self._reconnect(reconnect_attempts)
                    continue

                records_collected += len(records)
                cycles += 1
                if records:
                    self.sink.extend(records)
                    self.sink.flush_if_due()

                if time.monotonic() - last_health_log_at >= self.health_log_interval_seconds:
                    self.logger.info(
                        "Collector heartbeat: cycles=%s records_collected=%s pending_buffer=%s connected=%s",
                        cycles,
                        records_collected,
                        self.sink.pending_count,
                        self.client.is_connected(),
                    )
                    last_health_log_at = time.monotonic()

                if once:
                    message = "Collector completed one polling cycle."
                    break
                if max_cycles is not None and cycles >= max_cycles:
                    message = "Collector stopped after reaching the configured cycle limit."
                    break

                elapsed = time.monotonic() - cycle_started_at
                sleep_for = max(self.poll_interval_seconds - elapsed, 0.0)
                if sleep_for:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            status = "interrupted"
            message = "Collector interrupted by user."
            self.logger.warning(message)
        except Exception as exc:
            status = "error"
            message = f"Collector stopped after an unexpected error: {exc}"
            self.logger.exception(message)
        else:
            self.logger.info(message)
        finally:
            final_flush = self.sink.close()
            if final_flush:
                self.logger.info(
                    "Collector final flush persisted %s records to %s files.",
                    final_flush["records"],
                    len(final_flush["files"]),
                )
            with suppress(Exception):
                self.client.disconnect()
            self.logger.info("Collector shutdown complete.")

        summary = CollectorSummary(
            status=status,
            symbols=self.symbols,
            cycles=cycles,
            records_collected=records_collected,
            records_persisted=self.sink.persisted_records,
            flushes=self.sink.flush_count,
            reconnect_attempts=reconnect_attempts,
            output_root=str(self.sink.root_dir),
            poll_interval_seconds=self.poll_interval_seconds,
            flush_interval_seconds=self.sink.flush_interval_seconds,
            batch_size=self.sink.batch_size,
            message=message,
        )
        return summary.to_dict()

    def poll_once(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for symbol in self.symbols:
            try:
                snapshot = self.client.fetch_market_snapshot(symbol)
            except IBClientError as exc:
                self.logger.warning("Collector snapshot failed for %s: %s", symbol, exc)
                if not self.client.is_connected():
                    raise
                continue
            record = normalize_market_snapshot(snapshot, self.market_clock).to_dict()
            records.append(record)
        return records

    def describe(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "mode": self.settings.collector.mode,
            "poll_interval_seconds": self.poll_interval_seconds,
            "flush_interval_seconds": self.sink.flush_interval_seconds,
            "batch_size": self.sink.batch_size,
            "output_root": str(self.sink.root_dir),
        }

    def _ensure_connection(self) -> None:
        if self.client.is_connected():
            return
        self.logger.info(
            "Connecting collector client to IBKR at %s:%s with clientId=%s",
            self.settings.ib_host,
            self.settings.ib_port,
            getattr(self.client, "client_id", self.settings.ib_collector_client_id),
        )
        self.client.connect()

    def _should_retry(self, reconnect_attempts: int) -> bool:
        max_attempts = self.settings.collector.max_reconnect_attempts
        return max_attempts < 0 or reconnect_attempts <= max_attempts

    def _reconnect(self, reconnect_attempts: int) -> None:
        with suppress(Exception):
            self.client.disconnect()
        self.logger.warning(
            "Collector reconnect attempt %s/%s after %ss.",
            reconnect_attempts,
            self.settings.collector.max_reconnect_attempts,
            self.settings.collector.reconnect_delay_seconds,
        )
        time.sleep(self.settings.collector.reconnect_delay_seconds)


def collect_market_data(
    settings: Settings,
    client: MarketDataClient | None = None,
    *,
    symbols: Sequence[str] | None = None,
    symbol: str | None = None,
    once: bool = False,
    max_cycles: int | None = None,
    max_runtime_seconds: float | None = None,
    output_root: str | Path | None = None,
    poll_interval_seconds: float | None = None,
    flush_interval_seconds: float | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    logger = getattr(client, "logger", None) or setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.collector")
    collector_client = client or build_collector_ib_client(settings, logger)
    selected_symbols = tuple(
        dict.fromkeys(
            item.upper()
            for item in (
                [symbol] if symbol else []
            ) + list(symbols or settings.supported_symbols)
            if item
        )
    )
    sink = ParquetMarketDataSink(
        output_root or settings.paths.market_raw_dir,
        logger,
        batch_size=batch_size or settings.collector.batch_size,
        flush_interval_seconds=flush_interval_seconds or settings.collector.flush_interval_seconds,
    )
    collector = MarketDataCollector(
        settings,
        collector_client,
        sink,
        symbols=selected_symbols,
        logger=logger,
        poll_interval_seconds=poll_interval_seconds,
    )
    return collector.run(
        once=once,
        max_cycles=max_cycles,
        max_runtime_seconds=max_runtime_seconds,
    )


def persist_collection_payload(
    collector_root: Path,
    snapshot_payload: dict[str, Any],
    bars: pd.DataFrame,
) -> dict[str, str]:
    """Compatibility helper kept for the phase-1 tests and ad hoc debugging."""
    collector_root.mkdir(parents=True, exist_ok=True)
    timestamp_token = _timestamp_token(
        str(snapshot_payload.get("timestamp") or snapshot_payload.get("snapshot_utc") or "latest")
    )
    snapshot_path = collector_root / f"{timestamp_token}_snapshot.json"
    bars_path = collector_root / f"{timestamp_token}_bars.csv"
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2, sort_keys=True), encoding="utf-8")
    bars.to_csv(bars_path, index=False)
    return {"snapshot_path": str(snapshot_path), "bars_path": str(bars_path)}


def _timestamp_token(value: str) -> str:
    return value.replace(":", "").replace("-", "").replace("+", "_").replace("T", "_")
