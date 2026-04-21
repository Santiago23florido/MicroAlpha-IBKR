from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from broker.ib_client import IBClient, IBClientError
from config import Settings
from engine.market_clock import MarketClock
from ingestion.lob_reconstruction import LOBBookState, LOBDepthUpdate
from monitoring.logging import setup_logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LOBMarketDataClient(Protocol):
    def connect(self) -> bool: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def subscribe_market_depth(
        self,
        *,
        symbol: str,
        num_rows: int = 10,
        exchange: str | None = None,
        currency: str | None = None,
        is_smart_depth: bool = True,
    ) -> int: ...

    def consume_market_depth_events(
        self,
        req_id: int,
        *,
        timeout: float = 1.0,
        max_events: int = 500,
    ) -> list[dict[str, Any]]: ...

    def cancel_market_depth(self, req_id: int) -> None: ...


@dataclass
class LOBParquetSink:
    root_dir: Path
    batch_size: int
    flush_interval_seconds: float
    logger: Any

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._persisted_rows = 0
        self._flush_count = 0
        self._last_flush_at = time.monotonic()

    @property
    def pending_count(self) -> int:
        return len(self._buffer)

    @property
    def persisted_rows(self) -> int:
        return self._persisted_rows

    @property
    def flush_count(self) -> int:
        return self._flush_count

    def append(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)

    def flush_if_due(self, *, force: bool = False) -> dict[str, Any] | None:
        if not self._buffer:
            return None
        age = time.monotonic() - self._last_flush_at
        if not force and len(self._buffer) < self.batch_size and age < self.flush_interval_seconds:
            return None
        return self.flush()

    def flush(self) -> dict[str, Any] | None:
        if not self._buffer:
            return None

        frame = pd.DataFrame(self._buffer)
        written_files: list[str] = []
        manifest_paths: list[str] = []
        min_ts = None
        max_ts = None
        for session_date, group in frame.groupby("session_date", dropna=False):
            date_dir = self.root_dir / str(group["symbol"].iloc[0]).upper() / str(session_date)
            chunk_dir = date_dir / "chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output_path = chunk_dir / f"lob_{token}_{self._flush_count + 1:05d}.parquet"
            group.to_parquet(output_path, index=False)
            written_files.append(str(output_path))
            manifest_path = _update_lob_manifest(date_dir / "manifest.json", group, output_path)
            manifest_paths.append(str(manifest_path))
            date_min = str(group["event_ts_utc"].min())
            date_max = str(group["event_ts_utc"].max())
            min_ts = date_min if min_ts is None else min(min_ts, date_min)
            max_ts = date_max if max_ts is None else max(max_ts, date_max)

        persisted = len(self._buffer)
        self._persisted_rows += persisted
        self._flush_count += 1
        self._buffer.clear()
        self._last_flush_at = time.monotonic()
        payload = {
            "records": persisted,
            "files": written_files,
            "manifest_paths": manifest_paths,
            "persisted_rows": self._persisted_rows,
            "first_timestamp": min_ts,
            "last_timestamp": max_ts,
            "flush_count": self._flush_count,
        }
        self.logger.info(
            "LOB flush completed: records=%s files=%s flush_count=%s",
            persisted,
            len(written_files),
            self._flush_count,
        )
        return payload


class LOBCaptureCollector:
    def __init__(
        self,
        settings: Settings,
        *,
        symbol: str,
        depth_levels: int,
        rth_only: bool,
        session_id: str,
        logger,
        client: LOBMarketDataClient | None = None,
    ) -> None:
        self.settings = settings
        self.symbol = symbol.upper()
        self.depth_levels = depth_levels
        self.rth_only = rth_only
        self.session_id = session_id
        self.logger = logger
        self.client = client or _build_lob_client(settings, logger)
        self.market_clock = MarketClock(settings.session)
        self.book = LOBBookState(symbol=self.symbol, depth_levels=depth_levels)
        self.sink = LOBParquetSink(
            root_dir=Path(settings.lob_capture.output_root),
            batch_size=settings.lob_capture.batch_size,
            flush_interval_seconds=settings.lob_capture.flush_interval_seconds,
            logger=logger,
        )
        self.state_path = _lob_state_path(settings, self.symbol)
        self.session_path = _lob_session_path(settings, session_id)
        self._stop_requested = False
        self._current_req_id: int | None = None

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self, *, max_events: int | None = None) -> dict[str, Any]:
        processed_rows = 0
        reconnect_attempts = 0
        levels_observed = 0
        _write_json(
            self.session_path,
            {
                "session_id": self.session_id,
                "symbol": self.symbol,
                "status": "running",
                "started_at": _utc_now_iso(),
                "pid": os.getpid(),
                "depth_levels_requested": self.depth_levels,
                "rth_only": self.rth_only,
            },
        )
        self._write_state(
            status="running",
            connected=False,
            row_count=0,
            first_persisted_timestamp=None,
            last_persisted_timestamp=None,
            last_event_timestamp=None,
            active_session_id=self.session_id,
            levels_observed=0,
            reconnect_attempts=0,
            last_resubscription_at=None,
            last_reset_at=None,
            reset_count=0,
        )
        self._install_signal_handlers()
        try:
            self._connect_and_subscribe()
            while not self._stop_requested:
                try:
                    if not self.client.is_connected():
                        raise IBClientError("LOB capture lost the IBKR connection.")
                    events = self.client.consume_market_depth_events(
                        self._current_req_id or 0,
                        timeout=1.0,
                        max_events=1000,
                    )
                except IBClientError as exc:
                    reconnect_attempts += 1
                    if reconnect_attempts > self.settings.lob_capture.max_reconnect_attempts:
                        raise IBClientError(
                            f"LOB capture exceeded reconnect attempts for {self.symbol}: {exc}"
                        ) from exc
                    self.logger.warning(
                        "LOB capture reconnect %s/%s after error: %s",
                        reconnect_attempts,
                        self.settings.lob_capture.max_reconnect_attempts,
                        exc,
                    )
                    self._reconnect()
                    self._write_state(
                        status="running",
                        connected=True,
                        row_count=self.sink.persisted_rows + self.sink.pending_count,
                        first_persisted_timestamp=None,
                        last_persisted_timestamp=None,
                        last_event_timestamp=None,
                        active_session_id=self.session_id,
                        levels_observed=levels_observed,
                        reconnect_attempts=reconnect_attempts,
                        last_resubscription_at=_utc_now_iso(),
                        last_reset_at=None,
                        reset_count=self.book.reset_count,
                    )
                    continue

                if not events:
                    self._flush_if_needed(levels_observed=levels_observed)
                    continue

                for event in events:
                    event_type = event.get("event_type")
                    if event_type == "depth_reset":
                        self.book.reset()
                        self._write_state(
                            status="running",
                            connected=True,
                            row_count=self.sink.persisted_rows + self.sink.pending_count,
                            first_persisted_timestamp=None,
                            last_persisted_timestamp=None,
                            last_event_timestamp=event.get("timestamp_utc"),
                            active_session_id=self.session_id,
                            levels_observed=levels_observed,
                            reconnect_attempts=reconnect_attempts,
                            last_resubscription_at=None,
                            last_reset_at=event.get("timestamp_utc"),
                            reset_count=self.book.reset_count,
                        )
                        continue
                    if event_type == "connection_closed":
                        raise IBClientError(str(event.get("message") or "LOB capture connection closed."))
                    update = LOBDepthUpdate(
                        symbol=self.symbol,
                        timestamp_utc=str(event["timestamp_utc"]),
                        position=int(event["position"]),
                        operation=int(event["operation"]),
                        side=int(event["side"]),
                        price=float(event["price"]),
                        size=float(event["size"]),
                        market_maker=event.get("market_maker"),
                        is_smart_depth=bool(event.get("is_smart_depth", True)),
                        source=str(event.get("source", "updateMktDepthL2")),
                    )
                    market_state = self.market_clock.get_market_state()
                    if self.rth_only and not market_state.is_market_open:
                        continue
                    snapshot = self.book.apply(update)
                    snapshot["capture_session_id"] = self.session_id
                    snapshot["exchange_ts"] = market_state.exchange_time.isoformat()
                    snapshot["session_date"] = market_state.exchange_time.date().isoformat()
                    snapshot["session_window"] = market_state.session_window
                    snapshot["rth_only"] = self.rth_only
                    self.sink.append(snapshot)
                    processed_rows += 1
                    levels_observed = max(
                        levels_observed,
                        int(snapshot.get("observed_bid_levels", 0)),
                        int(snapshot.get("observed_ask_levels", 0)),
                    )
                    if max_events is not None and processed_rows >= max_events:
                        self._stop_requested = True
                        break
                self._flush_if_needed(levels_observed=levels_observed)
        except KeyboardInterrupt:
            self.logger.warning("LOB capture interrupted by user.")
            status = "interrupted"
        except Exception as exc:
            self.logger.exception("LOB capture failed: %s", exc)
            status = "error"
            error_message = str(exc)
        else:
            status = "stopped"
        finally:
            flushed = self.sink.flush_if_due(force=True)
            if flushed:
                self._write_state(
                    status="running",
                    connected=self.client.is_connected(),
                    row_count=flushed["persisted_rows"],
                    first_persisted_timestamp=flushed["first_timestamp"],
                    last_persisted_timestamp=flushed["last_timestamp"],
                    last_event_timestamp=flushed["last_timestamp"],
                    active_session_id=self.session_id,
                    levels_observed=levels_observed,
                    reconnect_attempts=reconnect_attempts,
                    last_resubscription_at=None,
                    last_reset_at=None,
                    reset_count=self.book.reset_count,
                )
            with suppress(Exception):
                if self._current_req_id is not None:
                    self.client.cancel_market_depth(self._current_req_id)
            with suppress(Exception):
                self.client.disconnect()
            latest = self._read_state()
            row_count = int(latest.get("row_count", 0))
            summary = {
                "status": status,
                "symbol": self.symbol,
                "provider": "ibkr",
                "session_id": self.session_id,
                "row_count": row_count,
                "persisted_rows": self.sink.persisted_rows,
                "flush_count": self.sink.flush_count,
                "state_path": str(self.state_path),
                "session_path": str(self.session_path),
                "output_root": self.settings.lob_capture.output_root,
                "depth_levels_requested": self.depth_levels,
                "levels_observed": levels_observed,
            }
            if flushed:
                summary["last_flush"] = flushed
            if status == "error":
                summary["message"] = error_message
            self._finalize_state(status=status, levels_observed=levels_observed)
            self._finalize_session(summary)
        return summary

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_args: self.request_stop())
        signal.signal(signal.SIGINT, lambda *_args: self.request_stop())

    def _connect_and_subscribe(self) -> None:
        self.client.connect()
        self._current_req_id = self.client.subscribe_market_depth(
            symbol=self.symbol,
            num_rows=self.depth_levels,
            exchange=self.settings.ib_exchange,
            currency=self.settings.ib_currency,
            is_smart_depth=True,
        )
        self._write_state(
            status="running",
            connected=True,
            row_count=self.sink.persisted_rows + self.sink.pending_count,
            first_persisted_timestamp=None,
            last_persisted_timestamp=None,
            last_event_timestamp=None,
            active_session_id=self.session_id,
            levels_observed=0,
            reconnect_attempts=0,
            last_resubscription_at=_utc_now_iso(),
            last_reset_at=None,
            reset_count=self.book.reset_count,
        )

    def _reconnect(self) -> None:
        with suppress(Exception):
            if self._current_req_id is not None:
                self.client.cancel_market_depth(self._current_req_id)
        with suppress(Exception):
            self.client.disconnect()
        self.book.reset()
        time.sleep(self.settings.lob_capture.reconnect_delay_seconds)
        self.client.connect()
        self._current_req_id = self.client.subscribe_market_depth(
            symbol=self.symbol,
            num_rows=self.depth_levels,
            exchange=self.settings.ib_exchange,
            currency=self.settings.ib_currency,
            is_smart_depth=True,
        )

    def _flush_if_needed(self, *, levels_observed: int) -> None:
        flushed = self.sink.flush_if_due()
        if not flushed:
            return
        self._write_state(
            status="running",
            connected=self.client.is_connected(),
            row_count=flushed["persisted_rows"],
            first_persisted_timestamp=flushed["first_timestamp"],
            last_persisted_timestamp=flushed["last_timestamp"],
            last_event_timestamp=flushed["last_timestamp"],
            active_session_id=self.session_id,
            levels_observed=levels_observed,
            reconnect_attempts=0,
            last_resubscription_at=None,
            last_reset_at=None,
            reset_count=self.book.reset_count,
        )

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(
        self,
        *,
        status: str,
        connected: bool,
        row_count: int,
        first_persisted_timestamp: str | None,
        last_persisted_timestamp: str | None,
        last_event_timestamp: str | None,
        active_session_id: str | None,
        levels_observed: int,
        reconnect_attempts: int,
        last_resubscription_at: str | None,
        last_reset_at: str | None,
        reset_count: int,
    ) -> None:
        current = self._read_state()
        _write_json(
            self.state_path,
            {
                **current,
                "symbol": self.symbol,
                "status": status,
                "pid": os.getpid(),
                "pid_alive": True,
                "connected": connected,
                "levels_requested": self.depth_levels,
                "levels_observed": max(int(current.get("levels_observed", 0)), levels_observed),
                "active_session_id": active_session_id,
                "row_count": max(int(current.get("row_count", 0)), row_count),
                "first_persisted_timestamp": first_persisted_timestamp or current.get("first_persisted_timestamp"),
                "last_persisted_timestamp": last_persisted_timestamp or current.get("last_persisted_timestamp"),
                "last_event_timestamp": last_event_timestamp or current.get("last_event_timestamp"),
                "last_resubscription_at": last_resubscription_at or current.get("last_resubscription_at"),
                "last_reset_at": last_reset_at or current.get("last_reset_at"),
                "reset_count": max(int(current.get("reset_count", 0)), reset_count),
                "reconnect_attempts": max(int(current.get("reconnect_attempts", 0)), reconnect_attempts),
                "rth_only": self.rth_only,
                "output_root": self.settings.lob_capture.output_root,
                "state_path": str(self.state_path),
                "session_path": str(self.session_path),
                "updated_at": _utc_now_iso(),
            },
        )

    def _finalize_state(self, *, status: str, levels_observed: int) -> None:
        current = self._read_state()
        _write_json(
            self.state_path,
            {
                **current,
                "status": status,
                "pid": None,
                "pid_alive": False,
                "connected": False,
                "active_session_id": None,
                "levels_observed": max(int(current.get("levels_observed", 0)), levels_observed),
                "updated_at": _utc_now_iso(),
            },
        )

    def _finalize_session(self, summary: dict[str, Any]) -> None:
        current = json.loads(self.session_path.read_text(encoding="utf-8"))
        _write_json(
            self.session_path,
            {
                **current,
                **summary,
                "ended_at": _utc_now_iso(),
            },
        )


def start_lob_capture(
    settings: Settings,
    *,
    symbol: str,
    levels: int | None = None,
    rth_only: bool | None = None,
) -> dict[str, Any]:
    ticker = symbol.upper()
    state_path = _lob_state_path(settings, ticker)
    current = _read_json(state_path)
    current_pid = current.get("pid")
    if isinstance(current_pid, int) and _pid_is_alive(current_pid):
        return {
            "status": "already_running",
            "symbol": ticker,
            "pid": current_pid,
            "state_path": str(state_path),
            "active_session_id": current.get("active_session_id"),
        }

    requested_levels = levels or settings.lob_capture.depth_levels
    requested_rth = settings.lob_capture.rth_only if rth_only is None else bool(rth_only)
    session_id = f"lob-{ticker}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    log_path = Path(settings.paths.log_dir) / f"lob_capture_{ticker.lower()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "app.py",
        "--env-file",
        settings.env_file or ".env",
        "--config-dir",
        settings.paths.config_dir,
        "--environment",
        settings.environment,
        "_lob-capture-runner",
        "--symbol",
        ticker,
        "--levels",
        str(requested_levels),
        "--session-id",
        session_id,
        "--rth",
        "true" if requested_rth else "false",
    ]
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=settings.paths.project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_json(
        state_path,
        {
            "symbol": ticker,
            "status": "starting",
            "pid": process.pid,
            "pid_alive": True,
            "connected": False,
            "levels_requested": requested_levels,
            "levels_observed": 0,
            "active_session_id": session_id,
            "row_count": int(current.get("row_count", 0)),
            "first_persisted_timestamp": current.get("first_persisted_timestamp"),
            "last_persisted_timestamp": current.get("last_persisted_timestamp"),
            "last_event_timestamp": current.get("last_event_timestamp"),
            "rth_only": requested_rth,
            "state_path": str(state_path),
            "session_path": str(_lob_session_path(settings, session_id)),
            "output_root": settings.lob_capture.output_root,
            "log_path": str(log_path),
            "updated_at": _utc_now_iso(),
        },
    )
    time.sleep(settings.lob_capture.startup_wait_seconds)
    started_state = _read_json(state_path)
    if process.poll() is not None and started_state.get("status") == "starting":
        return {
            "status": "error",
            "symbol": ticker,
            "message": f"LOB capture process exited immediately. Check {log_path}.",
            "state_path": str(state_path),
            "log_path": str(log_path),
        }
    return {
        "status": "ok",
        "symbol": ticker,
        "pid": process.pid,
        "session_id": session_id,
        "state_path": str(state_path),
        "session_path": str(_lob_session_path(settings, session_id)),
        "log_path": str(log_path),
        "command": command,
    }


def stop_lob_capture(settings: Settings, *, symbol: str) -> dict[str, Any]:
    ticker = symbol.upper()
    state_path = _lob_state_path(settings, ticker)
    state = _read_json(state_path)
    pid = state.get("pid")
    if not isinstance(pid, int):
        return {
            "status": "already_stopped",
            "symbol": ticker,
            "state_path": str(state_path),
        }
    _write_json(
        state_path,
        {
            **state,
            "status": "stopping",
            "stop_requested_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        },
    )
    if _pid_is_alive(pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and _pid_is_alive(pid):
            time.sleep(0.25)
    final_state = _read_json(state_path)
    return {
        "status": "ok",
        "symbol": ticker,
        "state_path": str(state_path),
        "state": final_state,
    }


def lob_capture_status(settings: Settings, *, symbol: str) -> dict[str, Any]:
    ticker = symbol.upper()
    state_path = _lob_state_path(settings, ticker)
    state = _read_json(state_path)
    if not state:
        return {
            "status": "missing",
            "symbol": ticker,
            "state_path": str(state_path),
        }
    pid = state.get("pid")
    state["pid_alive"] = isinstance(pid, int) and _pid_is_alive(pid)
    return {
        "status": "ok",
        "symbol": ticker,
        "state_path": str(state_path),
        "state": state,
    }


def run_lob_capture_loop(
    settings: Settings,
    *,
    symbol: str,
    levels: int | None = None,
    session_id: str | None = None,
    rth_only: bool | None = None,
    client: LOBMarketDataClient | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    ticker = symbol.upper()
    capture_session_id = session_id or f"lob-{ticker}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    logger = setup_logger(
        settings.log_level,
        Path(settings.paths.log_dir) / f"lob_capture_{ticker.lower()}.log",
        logger_name="microalpha.ibkr.lob_capture",
    )
    collector = LOBCaptureCollector(
        settings,
        symbol=ticker,
        depth_levels=levels or settings.lob_capture.depth_levels,
        rth_only=settings.lob_capture.rth_only if rth_only is None else bool(rth_only),
        session_id=capture_session_id,
        logger=logger,
        client=client,
    )
    return collector.run(max_events=max_events)


def _build_lob_client(settings: Settings, logger) -> IBClient:
    return IBClient(
        host=settings.ib_host,
        port=settings.ib_port,
        client_id=settings.ib_collector_client_id,
        logger=logger,
        request_timeout=settings.request_timeout_seconds,
        order_follow_up_seconds=settings.order_follow_up_seconds,
        account_summary_group=settings.account_summary_group,
        exchange=settings.ib_exchange,
        currency=settings.ib_currency,
    )


def _lob_state_path(settings: Settings, symbol: str) -> Path:
    root = Path(settings.lob_capture.state_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{symbol.upper()}.json"


def _lob_session_path(settings: Settings, session_id: str) -> Path:
    root = Path(settings.lob_capture.session_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{session_id}.json"


def _update_lob_manifest(path: Path, frame: pd.DataFrame, output_path: Path) -> Path:
    payload = _read_json(path)
    files = list(payload.get("files", []))
    files.append(str(output_path))
    observed_levels = max(
        int(frame.get("observed_bid_levels", pd.Series([0])).max()),
        int(frame.get("observed_ask_levels", pd.Series([0])).max()),
        int(payload.get("observed_levels", 0)),
    )
    manifest = {
        "provider": "ibkr",
        "source": "ibkr_market_depth",
        "symbol": str(frame["symbol"].iloc[0]).upper(),
        "session_date": str(frame["session_date"].iloc[0]),
        "row_count": int(payload.get("row_count", 0)) + int(len(frame)),
        "first_timestamp": min(
            str(frame["event_ts_utc"].min()),
            str(payload.get("first_timestamp") or frame["event_ts_utc"].min()),
        ),
        "last_timestamp": max(
            str(frame["event_ts_utc"].max()),
            str(payload.get("last_timestamp") or frame["event_ts_utc"].max()),
        ),
        "observed_levels": observed_levels,
        "files": files,
        "updated_at": _utc_now_iso(),
    }
    _write_json(path, manifest)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _pid_is_alive(pid: int) -> bool:
    with suppress(ProcessLookupError):
        os.kill(pid, 0)
        return True
    return False
