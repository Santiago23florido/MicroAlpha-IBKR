from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from broker.ib_client import IBClient, IBClientError
from config import Settings, load_ibkr_historical_config
from config.ibkr_history import IBKRHistoricalConfig
from data.historical_export import export_ibkr_training_csv, normalize_ibkr_bar_frame
from evaluation.io import write_json
from ingestion.ibkr_history_planner import (
    HistoricalChunk,
    plan_historical_bar_chunks,
    validate_bar_size,
    validate_symbol,
    validate_what_to_show,
)
from ingestion.ibkr_rate_limiter import IBKRHistoricalRateLimiter
from ingestion.ibkr_resume_store import IBKRBackfillResumeStore
from monitoring.logging import setup_logger


@dataclass
class HistoricalBackfillServices:
    settings: Settings
    config: IBKRHistoricalConfig
    client: IBClient
    rate_limiter: IBKRHistoricalRateLimiter
    resume_store: IBKRBackfillResumeStore


def ibkr_head_timestamp(
    settings: Settings,
    *,
    symbol: str,
    what_to_show: str,
    use_rth: bool | None = None,
) -> dict[str, Any]:
    config = load_ibkr_historical_config(settings)
    services = _build_services(settings, config)
    resolved_symbol = validate_symbol(symbol)
    resolved_what = validate_what_to_show(what_to_show)
    resolved_use_rth = config.use_rth if use_rth is None else bool(use_rth)
    _ensure_historical_enabled(config)
    try:
        services.client.connect()
        payload = services.client.get_head_timestamp(
            symbol=resolved_symbol,
            exchange=settings.ib_exchange,
            currency=settings.ib_currency,
            what_to_show=resolved_what,
            use_rth=resolved_use_rth,
        )
        payload["status"] = "ok"
        payload["provider"] = "ibkr"
        return payload
    finally:
        services.client.disconnect()


def ibkr_backfill(
    settings: Settings,
    *,
    symbol: str,
    what_to_show: str,
    bar_size: str,
    use_rth: bool | None = None,
    resume: bool = False,
    earliest_timestamp: str | None = None,
) -> dict[str, Any]:
    config = load_ibkr_historical_config(settings)
    services = _build_services(settings, config)
    _ensure_historical_enabled(config)
    resolved_symbol = validate_symbol(symbol)
    resolved_what = validate_what_to_show(what_to_show)
    resolved_bar_size = validate_bar_size(bar_size)
    resolved_use_rth = config.use_rth if use_rth is None else bool(use_rth)
    handle = services.resume_store.resolve(symbol=resolved_symbol, what_to_show=resolved_what, bar_size=resolved_bar_size)
    state = services.resume_store.load(handle) if (resume and config.enable_resume) else {}
    now = datetime.now(timezone.utc).isoformat()

    try:
        services.client.connect()
        if earliest_timestamp is None:
            head = services.client.get_head_timestamp(
                symbol=resolved_symbol,
                exchange=settings.ib_exchange,
                currency=settings.ib_currency,
                what_to_show=resolved_what,
                use_rth=resolved_use_rth,
            )
            earliest = head["head_timestamp"]
        else:
            earliest = earliest_timestamp
        chunks = plan_historical_bar_chunks(
            earliest_timestamp=earliest,
            latest_timestamp=state.get("planned_latest_timestamp"),
            bar_size=resolved_bar_size,
            chunk_days_1m=config.chunk_days_1m,
            chunk_days_intraday_fallback=config.chunk_days_intraday_fallback,
        )
        completed_indices = set(state.get("completed_chunk_indices", [])) if resume else set()
        existing_frame = _load_existing_raw_frame(handle.raw_path)
        all_frames = [existing_frame] if not existing_frame.empty else []
        request_count = int(state.get("request_count", 0))
        retry_count = int(state.get("retry_count", 0))
        pacing_wait_seconds = float(state.get("pacing_wait_seconds", 0.0))

        for chunk in chunks:
            if chunk.index in completed_indices:
                continue
            request_key = f"{resolved_symbol}:{resolved_what}:{resolved_bar_size}"
            request_signature = f"{request_key}:{chunk.end_datetime_ib}:{chunk.duration_str}:{resolved_use_rth}"
            cost = 2 if resolved_what == "BID_ASK" else 1
            pacing_wait_seconds += services.rate_limiter.wait_for_slot(
                request_key=request_key,
                request_signature=request_signature,
                cost=cost,
            )
            rows, retries_used = _request_chunk(
                services.client,
                symbol=resolved_symbol,
                what_to_show=resolved_what,
                bar_size=resolved_bar_size,
                use_rth=resolved_use_rth,
                chunk=chunk,
                settings=settings,
                retry_limit=config.retry_limit,
                backoff_seconds=config.backoff_seconds,
            )
            request_count += cost
            retry_count += retries_used
            if rows:
                frame = normalize_ibkr_bar_frame(
                    pd.DataFrame(rows),
                    symbol=resolved_symbol,
                    bar_size=resolved_bar_size,
                    what_to_show=resolved_what,
                )
                all_frames.append(frame)
            completed_indices.add(chunk.index)
            latest_downloaded = max((c.end_utc for c in chunks if c.index in completed_indices), default=earliest)
            state = {
                "status": "running",
                "symbol": resolved_symbol,
                "provider": "ibkr",
                "what_to_show": resolved_what,
                "bar_size": resolved_bar_size,
                "use_rth": resolved_use_rth,
                "earliest_timestamp": earliest,
                "planned_latest_timestamp": chunks[-1].end_utc if chunks else earliest,
                "latest_timestamp_downloaded": latest_downloaded,
                "completed_chunk_indices": sorted(completed_indices),
                "chunk_count": len(chunks),
                "request_count": request_count,
                "retry_count": retry_count,
                "pacing_wait_seconds": round(pacing_wait_seconds, 4),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "raw_output_path": str(handle.raw_path),
                "manifest_path": str(handle.manifest_path),
            }
            services.resume_store.save(handle, state)

        merged = _merge_frames(all_frames)
        _write_raw_frame(merged, handle.raw_path)
        manifest = {
            "symbol": resolved_symbol,
            "provider": "ibkr",
            "source": "ibkr_historical_backfill",
            "what_to_show": resolved_what,
            "bar_size": resolved_bar_size,
            "use_rth": resolved_use_rth,
            "requested_range": {
                "earliest_timestamp": earliest,
                "latest_timestamp": chunks[-1].end_utc if chunks else earliest,
            },
            "returned_range": {
                "min_timestamp": None if merged.empty else str(merged["timestamp"].min()),
                "max_timestamp": None if merged.empty else str(merged["timestamp"].max()),
            },
            "row_count": int(len(merged)),
            "chunk_count": len(chunks),
            "request_count": request_count,
            "retry_count": retry_count,
            "pacing_wait_seconds": round(pacing_wait_seconds, 4),
            "output_files": {
                "raw_parquet": str(handle.raw_path),
                "state_path": str(handle.state_path),
            },
            "status": "completed",
            "collected_at": now,
        }
        write_json(handle.manifest_path, manifest)
        state.update({"status": "completed", "row_count": int(len(merged)), "manifest_path": str(handle.manifest_path)})
        services.resume_store.save(handle, state)
        return state
    finally:
        services.client.disconnect()


def ibkr_backfill_status(
    settings: Settings,
    *,
    symbol: str,
    what_to_show: str | None = None,
    bar_size: str | None = None,
) -> dict[str, Any]:
    config = load_ibkr_historical_config(settings)
    resume_store = IBKRBackfillResumeStore(config.state_root, config.output_root)
    handle = resume_store.resolve(
        symbol=validate_symbol(symbol),
        what_to_show=validate_what_to_show(what_to_show or config.default_what_to_show),
        bar_size=validate_bar_size(bar_size or config.default_bar_size),
    )
    state = resume_store.load(handle)
    return {
        "status": "ok",
        "provider": "ibkr",
        "state_path": str(handle.state_path),
        "raw_output_path": str(handle.raw_path),
        "manifest_path": str(handle.manifest_path),
        "state": state,
    }


def export_training_csv_from_backfill(
    settings: Settings,
    *,
    symbol: str,
    what_to_show: str,
    bar_size: str,
    output_path: str | Path,
) -> dict[str, Any]:
    config = load_ibkr_historical_config(settings)
    resume_store = IBKRBackfillResumeStore(config.state_root, config.output_root)
    resolved_symbol = validate_symbol(symbol)
    resolved_what = validate_what_to_show(what_to_show)
    resolved_bar = validate_bar_size(bar_size)
    handle = resume_store.resolve(symbol=resolved_symbol, what_to_show=resolved_what, bar_size=resolved_bar)
    frame = _load_existing_raw_frame(handle.raw_path)
    if frame.empty:
        raise ValueError(f"No backfilled data available for {resolved_symbol} {resolved_what} {resolved_bar}.")
    return export_ibkr_training_csv(
        frame,
        output_path=output_path,
        symbol=resolved_symbol,
        synthetic_spread_bps=config.synthetic_spread_bps,
        default_depth_size=config.default_depth_size,
        write_parquet=config.write_parquet,
        write_manifest=config.write_manifest,
        metadata={
            "what_to_show": resolved_what,
            "bar_size": resolved_bar,
            "raw_backfill_path": str(handle.raw_path),
        },
    )


def prepare_ibkr_training_data(
    settings: Settings,
    *,
    symbol: str,
    what_to_show: str,
    bar_size: str,
    use_rth: bool | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_ibkr_historical_config(settings)
    resolved_symbol = validate_symbol(symbol)
    resolved_what = validate_what_to_show(what_to_show)
    resolved_bar = validate_bar_size(bar_size)
    resolved_output = Path(output_path) if output_path else Path(config.export_root) / f"{resolved_symbol}_{resolved_bar.replace(' ', '_')}_{resolved_what}.csv"
    head = ibkr_head_timestamp(settings, symbol=resolved_symbol, what_to_show=resolved_what, use_rth=use_rth)
    backfill = ibkr_backfill(
        settings,
        symbol=resolved_symbol,
        what_to_show=resolved_what,
        bar_size=resolved_bar,
        use_rth=use_rth,
        resume=config.enable_resume,
        earliest_timestamp=str(head["head_timestamp"]),
    )
    export_result = export_training_csv_from_backfill(
        settings,
        symbol=resolved_symbol,
        what_to_show=resolved_what,
        bar_size=resolved_bar,
        output_path=resolved_output,
    )
    return {
        "status": "ok",
        "provider": "ibkr",
        "head_timestamp": head,
        "backfill": backfill,
        "export": export_result,
    }


def _request_chunk(
    client: IBClient,
    *,
    symbol: str,
    what_to_show: str,
    bar_size: str,
    use_rth: bool,
    chunk: HistoricalChunk,
    settings: Settings,
    retry_limit: int,
    backoff_seconds: float,
) -> tuple[list[dict[str, Any]], int]:
    attempt = 0
    while True:
        try:
            rows = client.get_historical_bars(
                symbol=symbol,
                exchange=settings.ib_exchange,
                currency=settings.ib_currency,
                duration=chunk.duration_str,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
                end_datetime=chunk.end_datetime_ib,
            )
            return rows, attempt
        except IBClientError:
            attempt += 1
            if attempt > retry_limit:
                raise
            time.sleep(backoff_seconds * attempt)


def _load_existing_raw_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_raw_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError("Backfill produced an empty dataset.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _merge_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    merged = pd.concat(usable, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="raise")
    merged = merged.sort_values(["timestamp", "symbol"]).drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return merged.reset_index(drop=True)


def _build_services(settings: Settings, config: IBKRHistoricalConfig) -> HistoricalBackfillServices:
    logger = setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.ibkr.backfill")
    client = IBClient(
        host=settings.ib_host,
        port=settings.ib_port,
        client_id=settings.ib_client_id,
        logger=logger,
        request_timeout=settings.request_timeout_seconds,
        order_follow_up_seconds=settings.order_follow_up_seconds,
        account_summary_group=settings.account_summary_group,
        exchange=settings.ib_exchange,
        currency=settings.ib_currency,
    )
    return HistoricalBackfillServices(
        settings=settings,
        config=config,
        client=client,
        rate_limiter=IBKRHistoricalRateLimiter(
            max_requests_per_10_min=config.max_requests_per_10_min,
            max_same_contract_requests_per_2_sec=config.max_same_contract_requests_per_2_sec,
            dedupe_window_seconds=config.dedupe_window_seconds,
        ),
        resume_store=IBKRBackfillResumeStore(config.state_root, config.output_root),
    )


def _ensure_historical_enabled(config: IBKRHistoricalConfig) -> None:
    if not config.enabled:
        raise ValueError("IBKR historical backfill is disabled by configuration.")
