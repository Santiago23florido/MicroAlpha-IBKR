from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import Settings
from data.loader import list_market_data_files
from monitoring.logging import setup_logger


@dataclass(frozen=True)
class FeatureValidationReport:
    row_count: int
    feature_count: int
    empty_feature_count: int
    constant_feature_count: int
    excessive_nan_feature_count: int
    infinite_value_count: int
    duplicate_column_count: int
    feature_columns: list[str] = field(default_factory=list)
    empty_features: list[str] = field(default_factory=list)
    constant_features: list[str] = field(default_factory=list)
    excessive_nan_features: dict[str, float] = field(default_factory=dict)
    duplicate_columns: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_feature_store(
    settings: Settings,
    *,
    feature_root: str | Path | None = None,
    feature_set_name: str | None = None,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    logger=None,
) -> dict[str, Any]:
    validation_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.validate_features",
    )
    root_dir = Path(feature_root or settings.paths.feature_dir)
    if feature_set_name and feature_root is None:
        root_dir = root_dir / feature_set_name
    files = list_market_data_files(
        root_dir,
        symbols=symbols or settings.supported_symbols,
        start_date=start_date,
        end_date=end_date,
    )
    if not files:
        raise FileNotFoundError(
            f"No feature parquet files found under {root_dir}. Run build-features first or pass --feature-root."
        )

    validation_logger.info(
        "Validating feature store: root=%s files=%s symbols=%s range=%s..%s",
        root_dir,
        len(files),
        list(symbols or settings.supported_symbols),
        start_date or "min",
        end_date or "max",
    )

    frames = [pd.read_parquet(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    report = assess_feature_quality(combined, settings)
    payload = {
        "status": "ok" if not report.issues else "warning",
        "feature_root": str(root_dir),
        "feature_set_name": feature_set_name,
        "file_count": len(files),
        "rows": int(len(combined)),
        "symbols": sorted(combined["symbol"].dropna().astype(str).unique().tolist()) if "symbol" in combined.columns else [],
        "quality": report.to_dict(),
    }
    payload["report_path"] = _write_feature_validation_report(settings, payload)
    validation_logger.info(
        "Feature validation complete: rows=%s features=%s issues=%s report=%s",
        report.row_count,
        report.feature_count,
        report.issues,
        payload["report_path"],
    )
    return payload


def assess_feature_quality(frame: pd.DataFrame, settings: Settings) -> FeatureValidationReport:
    if frame.empty:
        return FeatureValidationReport(
            row_count=0,
            feature_count=0,
            empty_feature_count=0,
            constant_feature_count=0,
            excessive_nan_feature_count=0,
            infinite_value_count=0,
            duplicate_column_count=0,
            issues=["no_rows"],
        )

    excluded = {
        "timestamp",
        "exchange_timestamp",
        "exchange_time",
        "collected_at",
        "symbol",
        "session_date",
        "session_time",
        "source_file",
        "event_type",
        "source",
        "session_window",
        "is_market_open",
    }
    feature_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]

    duplicate_columns = sorted({column for column in frame.columns if list(frame.columns).count(column) > 1})
    empty_features: list[str] = []
    constant_features: list[str] = []
    excessive_nan_features: dict[str, float] = {}
    infinite_value_count = 0

    for column in feature_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        infinite_value_count += int(np.isinf(series.fillna(np.nan).to_numpy(dtype=float)).sum())
        if series.notna().sum() == 0:
            empty_features.append(column)
            continue
        nan_ratio = float(series.isna().mean())
        if nan_ratio > settings.feature_pipeline.validation_max_nan_ratio:
            excessive_nan_features[column] = round(nan_ratio, 6)
        if series.dropna().nunique() <= 1:
            constant_features.append(column)

    issues: list[str] = []
    if empty_features:
        issues.append("empty_features")
    if constant_features:
        issues.append("constant_features")
    if excessive_nan_features:
        issues.append("excessive_nan_features")
    if infinite_value_count:
        issues.append("infinite_values")
    if duplicate_columns:
        issues.append("duplicate_columns")

    return FeatureValidationReport(
        row_count=int(len(frame)),
        feature_count=int(len(feature_columns)),
        empty_feature_count=len(empty_features),
        constant_feature_count=len(constant_features),
        excessive_nan_feature_count=len(excessive_nan_features),
        infinite_value_count=infinite_value_count,
        duplicate_column_count=len(duplicate_columns),
        feature_columns=feature_columns,
        empty_features=empty_features,
        constant_features=constant_features,
        excessive_nan_features=excessive_nan_features,
        duplicate_columns=duplicate_columns,
        issues=issues,
    )


def _write_feature_validation_report(settings: Settings, report: dict[str, Any]) -> str:
    report_dir = Path(settings.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"feature_validation_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(report_path)
