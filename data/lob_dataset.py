from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from data.lob_labels import attach_lob_mid_price_labels


def build_lob_dataset(
    settings: Settings,
    *,
    symbol: str,
    from_date: str,
    to_date: str | None = None,
    horizon_events: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    ticker = symbol.upper()
    raw_frame, source_files = load_lob_capture_frame(
        settings,
        symbol=ticker,
        from_date=from_date,
        to_date=to_date,
    )
    if raw_frame.empty:
        raise ValueError(
            f"No LOB capture data available for {ticker} between {from_date} and {to_date or from_date}."
        )

    resolved_horizon = horizon_events or settings.models.lob_horizon_events
    labeled = attach_lob_mid_price_labels(
        raw_frame,
        horizon_events=resolved_horizon,
        stationary_threshold_bps=settings.models.lob_stationary_threshold_bps,
    )
    labeled["dataset_type"] = "ibkr_lob_depth"
    labeled["provider"] = "ibkr"
    labeled["source"] = "ibkr_market_depth"
    labeled["depth_levels"] = settings.models.lob_depth_levels
    labeled["horizon_events"] = resolved_horizon
    labeled["stationary_threshold_bps"] = settings.models.lob_stationary_threshold_bps

    target_path = _default_dataset_path(settings, ticker, from_date, to_date, resolved_horizon)
    if output_path is not None:
        target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(target_path, index=False)

    manifest_path = target_path.with_suffix(".manifest.json")
    daily_summary_path = target_path.with_suffix(".daily.json")
    manifest = {
        "status": "ok",
        "dataset_type": "ibkr_lob_depth",
        "symbol": ticker,
        "provider": "ibkr",
        "source": "ibkr_market_depth",
        "from_date": from_date,
        "to_date": to_date or str(labeled["session_date"].max()),
        "depth_levels": settings.models.lob_depth_levels,
        "sequence_length": settings.models.lob_sequence_length,
        "horizon_events": resolved_horizon,
        "stationary_threshold_bps": settings.models.lob_stationary_threshold_bps,
        "normalization": "price_rel_to_last_mid_bps_and_log1p_sizes",
        "row_count": int(len(labeled)),
        "observed_levels": int(max(labeled["observed_bid_levels"].max(), labeled["observed_ask_levels"].max())),
        "first_timestamp": str(labeled["event_ts_utc"].min()),
        "last_timestamp": str(labeled["event_ts_utc"].max()),
        "source_files": source_files,
        "output_path": str(target_path),
    }
    daily_summary = (
        labeled.groupby("session_date", sort=True)
        .agg(
            row_count=("event_ts_utc", "size"),
            first_timestamp=("event_ts_utc", "min"),
            last_timestamp=("event_ts_utc", "max"),
            up_ratio=("target_class", lambda values: float((values == 1).mean())),
            down_ratio=("target_class", lambda values: float((values == -1).mean())),
            flat_ratio=("target_class", lambda values: float((values == 0).mean())),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    for row in daily_summary:
        row["first_timestamp"] = str(row["first_timestamp"])
        row["last_timestamp"] = str(row["last_timestamp"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    daily_summary_path.write_text(json.dumps(daily_summary, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "dataset_path": str(target_path),
        "manifest_path": str(manifest_path),
        "daily_summary_path": str(daily_summary_path),
        "row_count": int(len(labeled)),
        "dataset_type": "ibkr_lob_depth",
        "symbol": ticker,
    }


def load_lob_capture_frame(
    settings: Settings,
    *,
    symbol: str,
    from_date: str,
    to_date: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    ticker = symbol.upper()
    root = Path(settings.lob_capture.output_root) / ticker
    if not root.exists():
        return pd.DataFrame(), []

    end_date = to_date or "9999-12-31"
    files: list[Path] = []
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        session_date = date_dir.name
        if session_date < from_date or session_date > end_date:
            continue
        chunk_dir = date_dir / "chunks"
        if not chunk_dir.exists():
            continue
        files.extend(sorted(chunk_dir.glob("*.parquet")))
    if not files:
        return pd.DataFrame(), []

    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    frame["event_ts_utc"] = pd.to_datetime(frame["event_ts_utc"], utc=True)
    frame = frame.sort_values(["event_ts_utc", "event_index"]).drop_duplicates(
        subset=["event_ts_utc", "event_index", "capture_session_id"],
        keep="last",
    )
    frame = frame.reset_index(drop=True)
    return frame, [str(path) for path in files]


def _default_dataset_path(
    settings: Settings,
    symbol: str,
    from_date: str,
    to_date: str | None,
    horizon_events: int,
) -> Path:
    end_token = to_date or "latest"
    return (
        Path(settings.lob_capture.dataset_root)
        / symbol.upper()
        / f"{symbol.upper()}_{from_date}_{end_token}_k{horizon_events}.parquet"
    )
