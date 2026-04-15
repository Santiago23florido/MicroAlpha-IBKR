from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import Settings
from data.loader import RAW_MARKET_COLUMNS, list_market_data_files
from monitoring.logging import setup_logger


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    symbol_count: int
    missing_timestamp_count: int
    duplicate_count: int
    critical_null_count: int
    bid_gt_ask_count: int
    negative_spread_count: int
    absurd_spread_count: int
    outside_regular_hours_count: int
    large_gap_count: int
    max_gap_seconds_observed: float
    null_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MINIMUM_IMPORT_COLUMNS = ("timestamp", "symbol")
OPTIONAL_RENAME_MAP = {
    "last": "last_price",
    "bid_sz": "bid_size",
    "ask_sz": "ask_size",
}


def validate_imports(
    settings: Settings,
    *,
    input_root: str | Path | None = None,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    logger=None,
) -> dict[str, Any]:
    quality_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.validate_imports",
    )
    root_dir = Path(input_root or settings.paths.import_market_dir)
    files = list_market_data_files(
        root_dir,
        symbols=symbols or settings.supported_symbols,
        start_date=start_date,
        end_date=end_date,
    )
    if not files:
        raise FileNotFoundError(
            f"No imported parquet files found under {root_dir}. Run pull-from-pc2 first or pass --input-root."
        )

    quality_logger.info(
        "Validating imported market data: root=%s files=%s symbols=%s range=%s..%s",
        root_dir,
        len(files),
        list(symbols or settings.supported_symbols),
        start_date or "min",
        end_date or "max",
    )

    readable_files = 0
    unreadable_files = 0
    files_with_missing_columns = 0
    file_results: list[dict[str, Any]] = []
    normalized_frames: list[pd.DataFrame] = []

    for path in files:
        item = {
            "path": str(path),
            "session_date": _infer_session_date(path),
            "symbol": _infer_import_symbol(path),
            "status": "ok",
            "missing_columns": [],
            "row_count": 0,
        }
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            item["status"] = "unreadable"
            item["error"] = str(exc)
            unreadable_files += 1
            file_results.append(item)
            quality_logger.error("Imported parquet is unreadable: path=%s error=%s", path, exc)
            continue

        frame = frame.rename(columns={key: value for key, value in OPTIONAL_RENAME_MAP.items() if key in frame.columns})
        missing_columns = [column for column in MINIMUM_IMPORT_COLUMNS if column not in frame.columns]
        has_price_field = any(column in frame.columns for column in ("last_price", "bid", "ask"))
        if not has_price_field:
            missing_columns.append("price_field")
        if missing_columns:
            item["status"] = "missing_columns"
            item["missing_columns"] = missing_columns
            files_with_missing_columns += 1
            file_results.append(item)
            quality_logger.warning("Imported file is missing required columns: path=%s missing=%s", path, missing_columns)
            continue

        standardized = frame.copy()
        if "symbol" not in standardized.columns:
            standardized["symbol"] = _infer_import_symbol(path)
        standardized["symbol"] = standardized["symbol"].astype(str).str.upper()
        standardized["source_file"] = str(path)
        for column in RAW_MARKET_COLUMNS:
            if column not in standardized.columns:
                standardized[column] = pd.NA
        extra_columns = [
            column
            for column in standardized.columns
            if column not in RAW_MARKET_COLUMNS and column != "source_file"
        ]
        standardized = standardized[[*RAW_MARKET_COLUMNS, *extra_columns, "source_file"]]

        item["row_count"] = int(len(standardized))
        item["column_count"] = int(len(frame.columns))
        item["duplicate_count"] = int(standardized.duplicated(subset=["symbol", "timestamp"], keep=False).sum())
        readable_files += 1
        file_results.append(item)
        normalized_frames.append(standardized)

    if not normalized_frames:
        report = {
            "status": "error",
            "input_root": str(root_dir),
            "file_count": len(files),
            "readable_files": readable_files,
            "unreadable_files": unreadable_files,
            "files_with_missing_columns": files_with_missing_columns,
            "results": file_results,
            "issues": ["no_valid_import_files"],
        }
        report["report_path"] = _write_import_validation_report(settings, report)
        return report

    combined = pd.concat(normalized_frames, ignore_index=True)
    quality_report = assess_market_data_quality(combined, settings)
    summary_by_partition = _build_partition_summary(combined)
    report = {
        "status": "ok" if unreadable_files == 0 and files_with_missing_columns == 0 else "error",
        "input_root": str(root_dir),
        "file_count": len(files),
        "readable_files": readable_files,
        "unreadable_files": unreadable_files,
        "files_with_missing_columns": files_with_missing_columns,
        "rows": int(len(combined)),
        "symbols": sorted(combined["symbol"].dropna().astype(str).unique().tolist()),
        "quality": quality_report.to_dict(),
        "summary_by_partition": summary_by_partition,
        "results": file_results,
    }
    report["report_path"] = _write_import_validation_report(settings, report)
    quality_logger.info(
        "Import validation complete: files=%s readable=%s unreadable=%s missing_columns=%s issues=%s report=%s",
        len(files),
        readable_files,
        unreadable_files,
        files_with_missing_columns,
        quality_report.issues,
        report["report_path"],
    )
    return report


def assess_market_data_quality(frame: pd.DataFrame, settings: Settings) -> DataQualityReport:
    if frame.empty:
        return DataQualityReport(
            row_count=0,
            symbol_count=0,
            missing_timestamp_count=0,
            duplicate_count=0,
            critical_null_count=0,
            bid_gt_ask_count=0,
            negative_spread_count=0,
            absurd_spread_count=0,
            outside_regular_hours_count=0,
            large_gap_count=0,
            max_gap_seconds_observed=0.0,
            issues=["no_rows"],
        )

    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    missing_symbol_count = int(working["symbol"].isna().sum()) if "symbol" in working.columns else int(len(working))
    if "symbol" in working.columns:
        working["symbol"] = working["symbol"].astype(str).str.upper()
    else:
        working["symbol"] = "UNKNOWN"

    if "last_price" not in working.columns and "last" in working.columns:
        working["last_price"] = pd.to_numeric(working["last"], errors="coerce")
    for column in ["last_price", "bid", "ask", "volume", "bid_size", "ask_size"]:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")

    missing_timestamp_count = int(working["timestamp"].isna().sum())
    duplicate_count = int(working.duplicated(subset=["symbol", "timestamp"], keep=False).sum())
    critical_null_count = int(working[["last_price", "bid", "ask"]].isna().all(axis=1).sum())
    bid_gt_ask_count = int(((working["bid"].notna()) & (working["ask"].notna()) & (working["bid"] > working["ask"])).sum())

    spread = working["ask"] - working["bid"]
    mid_price = np.where(
        working["bid"].notna() & working["ask"].notna(),
        (working["bid"] + working["ask"]) / 2.0,
        working["last_price"],
    )
    spread_bps = np.where(mid_price > 0, spread / mid_price * 10000.0, np.nan)
    negative_spread_count = int((spread < 0).fillna(False).sum())
    absurd_spread_count = int((pd.Series(spread_bps).abs() > settings.feature_pipeline.max_abs_spread_bps).fillna(False).sum())

    exchange_timestamp = working["timestamp"].dt.tz_convert(settings.session.timezone)
    session_minutes = exchange_timestamp.dt.hour * 60 + exchange_timestamp.dt.minute
    open_minutes = settings.session.regular_market_open.hour * 60 + settings.session.regular_market_open.minute
    close_minutes = settings.session.regular_market_close.hour * 60 + settings.session.regular_market_close.minute
    outside_regular_hours_count = int(((session_minutes < open_minutes) | (session_minutes > close_minutes)).fillna(False).sum())

    diffs = (
        working.sort_values(["symbol", "timestamp"])
        .groupby("symbol")["timestamp"]
        .diff()
        .dt.total_seconds()
    )
    gap_threshold = settings.feature_pipeline.gap_threshold_seconds
    large_gap_mask = diffs > gap_threshold
    large_gap_count = int(large_gap_mask.fillna(False).sum())
    max_gap_seconds_observed = float(diffs.max()) if not diffs.dropna().empty else 0.0

    null_counts = {column: int(working[column].isna().sum()) for column in ["timestamp", "last_price", "bid", "ask", "volume"]}
    null_counts["symbol"] = missing_symbol_count
    issues: list[str] = []
    if missing_timestamp_count:
        issues.append("missing_timestamps")
    if duplicate_count:
        issues.append("duplicate_rows")
    if critical_null_count:
        issues.append("critical_null_rows")
    if bid_gt_ask_count:
        issues.append("bid_gt_ask")
    if absurd_spread_count:
        issues.append("absurd_spreads")
    if outside_regular_hours_count:
        issues.append("outside_regular_hours")
    if large_gap_count:
        issues.append("large_gaps")

    return DataQualityReport(
        row_count=int(len(working)),
        symbol_count=int(working["symbol"].nunique()),
        missing_timestamp_count=missing_timestamp_count,
        duplicate_count=duplicate_count,
        critical_null_count=critical_null_count,
        bid_gt_ask_count=bid_gt_ask_count,
        negative_spread_count=negative_spread_count,
        absurd_spread_count=absurd_spread_count,
        outside_regular_hours_count=outside_regular_hours_count,
        large_gap_count=large_gap_count,
        max_gap_seconds_observed=max_gap_seconds_observed,
        null_counts=null_counts,
        issues=issues,
    )


def _infer_session_date(path: Path) -> str | None:
    for part in path.parts:
        try:
            datetime.fromisoformat(part)
            return part
        except ValueError:
            continue
    return None


def _infer_import_symbol(path: Path) -> str:
    if path.parent.name and _looks_like_date(path.parent.parent.name):
        return path.parent.name.upper()
    return path.stem.upper()


def _build_partition_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    exchange_timestamp = working["timestamp"]
    if exchange_timestamp.notna().any():
        working["session_date"] = exchange_timestamp.dt.strftime("%Y-%m-%d")
    else:
        working["session_date"] = "unknown"
    summaries: list[dict[str, Any]] = []
    for (session_date, symbol), group in working.groupby(["session_date", "symbol"], sort=True):
        summaries.append(
            {
                "session_date": str(session_date),
                "symbol": str(symbol).upper(),
                "rows": int(len(group)),
                "duplicates": int(group.duplicated(subset=["symbol", "timestamp"], keep=False).sum()),
                "min_timestamp": str(group["timestamp"].min()) if group["timestamp"].notna().any() else None,
                "max_timestamp": str(group["timestamp"].max()) if group["timestamp"].notna().any() else None,
            }
        )
    return summaries


def _write_import_validation_report(settings: Settings, report: dict[str, Any]) -> str:
    report_dir = Path(settings.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"import_validation_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(report_path)


def _looks_like_date(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False
