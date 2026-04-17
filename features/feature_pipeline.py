from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import Settings
from data.cleaning import clean_market_data
from data.loader import load_market_data
from features.registry import (
    build_indicator_registry,
    default_feature_set_name,
    inspect_feature_dependencies,
    list_feature_sets,
)
from features.validation import assess_feature_quality
from monitoring.data_quality import DataQualityReport, assess_market_data_quality
from monitoring.logging import setup_logger


LEGACY_COMPATIBILITY_COLUMNS = (
    "bid_ask_imbalance",
    "return_1_bps",
    "return_short_bps",
    "return_medium_bps",
    "vwap_approx",
    "micro_price",
    "estimated_cost_bps",
    "spread_proxy_bps",
    "slippage_proxy_bps",
)


@dataclass(frozen=True)
class FeatureBuildResult:
    input_rows: int
    cleaned_rows: int
    feature_rows: int
    feature_set_name: str
    feature_columns: tuple[str, ...]
    symbols: tuple[str, ...]
    dates: tuple[str, ...]
    written_files: list[str]
    raw_quality: dict[str, Any]
    cleaned_quality: dict[str, Any]
    feature_quality: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: str
    output_root: str
    report_path: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "cleaned_rows": self.cleaned_rows,
            "feature_rows": self.feature_rows,
            "feature_set_name": self.feature_set_name,
            "feature_columns": list(self.feature_columns),
            "symbols": list(self.symbols),
            "dates": list(self.dates),
            "written_files": self.written_files,
            "raw_quality": self.raw_quality,
            "cleaned_quality": self.cleaned_quality,
            "feature_quality": self.feature_quality,
            "manifest": self.manifest,
            "manifest_path": self.manifest_path,
            "output_root": self.output_root,
            "report_path": self.report_path,
            "duration_seconds": self.duration_seconds,
        }


def run_feature_build_pipeline(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    feature_set_name: str | None = None,
    logger=None,
) -> dict[str, Any]:
    pipeline_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.feature_pipeline",
    )
    started_at = time.monotonic()
    resolved_feature_set = feature_set_name or settings.feature_pipeline.default_feature_set or default_feature_set_name(settings)
    pipeline_logger.info("Starting feature build pipeline with feature_set=%s.", resolved_feature_set)

    raw_frame = load_market_data(
        settings,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        input_root=input_root,
    )
    pipeline_logger.info(
        "Loaded raw market data: rows=%s symbols=%s range=%s..%s",
        len(raw_frame),
        sorted(raw_frame["symbol"].dropna().astype(str).unique().tolist()),
        start_date or "min",
        end_date or "max",
    )
    raw_quality = assess_market_data_quality(raw_frame, settings)
    if raw_quality.issues:
        pipeline_logger.warning("Raw data quality issues detected: %s", ", ".join(raw_quality.issues))

    cleaned_frame = clean_market_data(raw_frame, settings)
    pipeline_logger.info("Cleaned market data rows=%s", len(cleaned_frame))
    cleaned_quality = assess_market_data_quality(cleaned_frame, settings)
    if cleaned_quality.issues:
        pipeline_logger.warning("Post-cleaning quality issues remain: %s", ", ".join(cleaned_quality.issues))

    feature_frame, manifest = build_feature_frame_with_manifest(
        cleaned_frame,
        settings,
        feature_set_name=resolved_feature_set,
    )
    feature_quality = assess_feature_quality(feature_frame, settings)
    if feature_quality.issues:
        pipeline_logger.warning("Feature quality warnings: %s", ", ".join(feature_quality.issues))

    target_output_root = resolve_feature_output_root(
        settings,
        feature_set_name=resolved_feature_set,
        output_root=output_root,
    )
    pipeline_logger.info(
        "Generated feature frame rows=%s columns=%s days=%s compatible_indicators=%s omitted_indicators=%s",
        len(feature_frame),
        len(feature_frame.columns),
        sorted(feature_frame["session_date"].dropna().astype(str).unique().tolist()),
        len(manifest["compatible_indicators"]),
        len(manifest["omitted_indicators"]),
    )
    write_summary = persist_feature_frame(feature_frame, target_output_root)
    report_paths = persist_feature_report(
        settings,
        raw_quality=raw_quality,
        cleaned_quality=cleaned_quality,
        feature_quality=feature_quality.to_dict(),
        feature_frame=feature_frame,
        manifest=manifest,
        output_root=target_output_root,
    )

    duration_seconds = round(time.monotonic() - started_at, 3)
    pipeline_logger.info(
        "Feature build pipeline complete: feature_set=%s input_rows=%s cleaned_rows=%s feature_rows=%s files=%s duration=%ss report=%s",
        resolved_feature_set,
        len(raw_frame),
        len(cleaned_frame),
        len(feature_frame),
        len(write_summary["written_files"]),
        duration_seconds,
        report_paths["report_path"],
    )

    result = FeatureBuildResult(
        input_rows=int(len(raw_frame)),
        cleaned_rows=int(len(cleaned_frame)),
        feature_rows=int(len(feature_frame)),
        feature_set_name=resolved_feature_set,
        feature_columns=tuple(manifest["feature_columns"]),
        symbols=tuple(sorted(feature_frame["symbol"].dropna().astype(str).unique().tolist())),
        dates=tuple(sorted(feature_frame["session_date"].dropna().astype(str).unique().tolist())),
        written_files=write_summary["written_files"],
        raw_quality=raw_quality.to_dict(),
        cleaned_quality=cleaned_quality.to_dict(),
        feature_quality=feature_quality.to_dict(),
        manifest=manifest,
        manifest_path=report_paths["manifest_path"],
        output_root=str(target_output_root),
        report_path=report_paths["report_path"],
        duration_seconds=duration_seconds,
    )
    return result.to_dict()


def build_feature_frame(
    clean_frame: pd.DataFrame,
    settings: Settings,
    *,
    feature_set_name: str | None = None,
) -> pd.DataFrame:
    feature_frame, _ = build_feature_frame_with_manifest(clean_frame, settings, feature_set_name=feature_set_name)
    return feature_frame


def build_feature_frame_with_manifest(
    clean_frame: pd.DataFrame,
    settings: Settings,
    *,
    feature_set_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = prepare_feature_inputs(clean_frame, settings)
    plan = inspect_feature_dependencies(prepared, settings, feature_set_name=feature_set_name)
    registry = build_indicator_registry()
    feature_frame = prepared.copy()
    calculated_indicators: list[dict[str, Any]] = []

    for resolution in plan.compatible_indicators:
        definition = registry[resolution.name]
        params = {**definition.default_params, **plan.feature_set.params.get(definition.name, {})}
        computed = definition.calculator(feature_frame, settings, params) if definition.calculator else pd.DataFrame(index=feature_frame.index)
        missing_outputs = [column for column in resolution.output_columns if column not in computed.columns]
        if missing_outputs:
            raise ValueError(
                f"Indicator {definition.name!r} did not produce the expected columns: {missing_outputs}."
            )
        computed = computed.loc[:, list(dict.fromkeys(resolution.output_columns))]
        computed = computed.replace([np.inf, -np.inf], np.nan)
        for column in computed.columns:
            feature_frame[column] = computed[column]
        calculated_indicators.append(
            {
                "name": definition.name,
                "family": definition.family,
                "output_columns": list(computed.columns),
                "params": params,
            }
        )

    feature_frame = _add_compatibility_columns(feature_frame, settings)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    feature_columns = _select_feature_columns(feature_frame, plan.planned_feature_columns)
    feature_frame.attrs["feature_columns"] = feature_columns
    feature_frame.attrs["feature_set_name"] = plan.feature_set.name
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set_name": plan.feature_set.name,
        "feature_set_description": plan.feature_set.description,
        "families_activated": list(plan.feature_set.families),
        "indicators_requested": list(plan.feature_set.indicators),
        "indicators_calculated": calculated_indicators,
        "compatible_indicators": [item.to_dict() for item in plan.compatible_indicators],
        "omitted_indicators": [item.to_dict() for item in plan.omitted_indicators],
        "parameters_used": {item["name"]: item["params"] for item in calculated_indicators},
        "input_columns_detected": list(plan.input_columns),
        "non_null_columns_detected": list(plan.non_null_columns),
        "feature_columns": feature_columns,
        "row_count": int(len(feature_frame)),
        "symbols": sorted(feature_frame["symbol"].dropna().astype(str).unique().tolist()),
        "date_range": {
            "start": str(feature_frame["session_date"].min()) if "session_date" in feature_frame.columns else None,
            "end": str(feature_frame["session_date"].max()) if "session_date" in feature_frame.columns else None,
        },
    }
    return feature_frame, manifest


def prepare_feature_inputs(clean_frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    frame = clean_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    if "exchange_timestamp" not in frame.columns or frame["exchange_timestamp"].isna().all():
        frame["exchange_timestamp"] = frame["timestamp"].dt.tz_convert(settings.session.timezone)
    else:
        frame["exchange_timestamp"] = pd.to_datetime(frame["exchange_timestamp"], utc=True, errors="coerce").dt.tz_convert(
            settings.session.timezone
        )

    frame["session_date"] = frame["exchange_timestamp"].dt.strftime("%Y-%m-%d")
    frame["session_time"] = frame["exchange_timestamp"].dt.time

    numeric_candidates = ["last_price", "bid", "ask", "bid_size", "ask_size", "last_size", "volume", "spread", "spread_bps"]
    for column in numeric_candidates:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "mid_price" not in frame.columns:
        frame["mid_price"] = np.nan
    frame["mid_price"] = np.where(
        frame["bid"].notna() & frame["ask"].notna(),
        (frame["bid"] + frame["ask"]) / 2.0,
        frame["mid_price"],
    )
    frame["mid_price"] = pd.Series(frame["mid_price"], index=frame.index).fillna(frame.get("last_price"))
    frame["price_proxy"] = frame["last_price"].fillna(frame["mid_price"]) if "last_price" in frame.columns else frame["mid_price"]
    frame["high_price_proxy"] = frame[[column for column in ("high", "ask", "last_price", "mid_price") if column in frame.columns]].max(axis=1, skipna=True)
    frame["low_price_proxy"] = frame[[column for column in ("low", "bid", "last_price", "mid_price") if column in frame.columns]].min(axis=1, skipna=True)

    if "spread" not in frame.columns or frame["spread"].isna().all():
        frame["spread"] = frame["ask"] - frame["bid"]
    if "spread_bps" not in frame.columns or frame["spread_bps"].isna().all():
        frame["spread_bps"] = np.where(
            frame["mid_price"] > 0,
            frame["spread"] / frame["mid_price"] * 10000.0,
            np.nan,
        )
    if "bid_size" not in frame.columns:
        frame["bid_size"] = np.nan
    if "ask_size" not in frame.columns:
        frame["ask_size"] = np.nan
    if "volume" not in frame.columns:
        frame["volume"] = np.nan
    if "last_size" not in frame.columns:
        frame["last_size"] = np.nan

    return frame


def list_available_feature_sets(settings: Settings) -> dict[str, Any]:
    return list_feature_sets(settings)


def inspect_feature_dependencies_for_build(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    input_root: str | Path | None = None,
    feature_set_name: str | None = None,
) -> dict[str, Any]:
    raw_frame = load_market_data(
        settings,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        input_root=input_root,
    )
    cleaned_frame = clean_market_data(raw_frame, settings)
    prepared = prepare_feature_inputs(cleaned_frame, settings)
    plan = inspect_feature_dependencies(prepared, settings, feature_set_name=feature_set_name)
    return plan.to_dict()


def resolve_feature_output_root(
    settings: Settings,
    *,
    feature_set_name: str,
    output_root: str | Path | None = None,
) -> Path:
    base_root = Path(output_root or settings.paths.feature_dir)
    if output_root is not None:
        return base_root
    return base_root / feature_set_name


def persist_feature_frame(feature_frame: pd.DataFrame, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    written_files: list[str] = []

    for (session_date, symbol), group in feature_frame.groupby(["session_date", "symbol"], sort=True):
        target_dir = root / str(session_date)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{str(symbol).upper()}.parquet"
        group.to_parquet(target_path, index=False)
        written_files.append(str(target_path))

    return {"written_files": written_files, "output_root": str(root)}


def persist_feature_report(
    settings: Settings,
    *,
    raw_quality: DataQualityReport,
    cleaned_quality: DataQualityReport,
    feature_quality: dict[str, Any],
    feature_frame: pd.DataFrame,
    manifest: dict[str, Any],
    output_root: str | Path,
) -> dict[str, str]:
    report_dir = Path(settings.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = report_dir / f"feature_manifest_{timestamp_token}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path = report_dir / f"feature_build_report_{timestamp_token}.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_quality": raw_quality.to_dict(),
        "cleaned_quality": cleaned_quality.to_dict(),
        "feature_quality": feature_quality,
        "feature_rows": int(len(feature_frame)),
        "feature_columns": list(feature_frame.attrs.get("feature_columns", [])),
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"report_path": str(report_path), "manifest_path": str(manifest_path)}


def _add_compatibility_columns(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    enriched = frame.copy()
    group_keys = ["symbol", "session_date"] if "session_date" in enriched.columns else ["symbol"]
    price_col = "price_proxy"

    grouped_price = enriched.groupby(group_keys, sort=False)[price_col]
    enriched["return_1_bps"] = grouped_price.pct_change(1) * 10000.0
    enriched["return_short_bps"] = grouped_price.pct_change(settings.feature_pipeline.rolling_short_window) * 10000.0
    enriched["return_medium_bps"] = grouped_price.pct_change(settings.feature_pipeline.rolling_medium_window) * 10000.0

    if "imbalance" in enriched.columns:
        enriched["bid_ask_imbalance"] = enriched["imbalance"]
    if "microprice_proxy" in enriched.columns:
        enriched["micro_price"] = enriched["microprice_proxy"]
    if "vwap" in enriched.columns:
        enriched["vwap_approx"] = enriched["vwap"]

    enriched["spread_proxy_bps"] = enriched["spread_bps"] if "spread_bps" in enriched.columns else np.nan
    imbalance_abs = enriched["imbalance"].abs() if "imbalance" in enriched.columns else 0.0
    spread_proxy = enriched["spread_proxy_bps"].fillna(0.0)
    enriched["slippage_proxy_bps"] = spread_proxy * (0.5 + imbalance_abs.fillna(0.0))
    enriched["estimated_cost_bps"] = spread_proxy.fillna(0.0) + enriched["slippage_proxy_bps"].fillna(0.0)

    for column in LEGACY_COMPATIBILITY_COLUMNS:
        if column in enriched.columns:
            enriched[column] = enriched[column].replace([np.inf, -np.inf], np.nan)
    return enriched


def _select_feature_columns(frame: pd.DataFrame, planned_feature_columns: Sequence[str]) -> list[str]:
    columns = [column for column in planned_feature_columns if column in frame.columns]
    columns.extend(column for column in LEGACY_COMPATIBILITY_COLUMNS if column in frame.columns and column not in columns)
    return list(dict.fromkeys(columns))


def _default_feature_columns() -> list[str]:
    dummy_settings = type(
        "_DummySettings",
        (),
        {
            "paths": type("_DummyPaths", (), {"config_dir": str(Path(__file__).resolve().parents[1] / "config")})(),
            "feature_pipeline": type("_DummyFeaturePipeline", (), {"default_feature_set": "hybrid_intraday"})(),
        },
    )()
    try:
        registry = build_indicator_registry()
        plan_names = inspect_feature_dependencies(
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2026-01-01T14:30:00Z")],
                    "exchange_timestamp": [pd.Timestamp("2026-01-01T14:30:00Z")],
                    "symbol": ["SPY"],
                    "session_date": ["2026-01-01"],
                    "last_price": [1.0],
                    "price_proxy": [1.0],
                    "mid_price": [1.0],
                    "bid": [0.99],
                    "ask": [1.01],
                    "bid_size": [10.0],
                    "ask_size": [11.0],
                    "volume": [100.0],
                    "last_size": [1.0],
                    "spread": [0.02],
                    "spread_bps": [200.0],
                    "high_price_proxy": [1.01],
                    "low_price_proxy": [0.99],
                }
            ),
            dummy_settings,  # type: ignore[arg-type]
            feature_set_name="hybrid_intraday",
        )
        columns = list(plan_names.planned_feature_columns)
    except Exception:
        columns = []
    columns.extend(list(LEGACY_COMPATIBILITY_COLUMNS))
    return list(dict.fromkeys(columns))


FEATURE_COLUMNS = _default_feature_columns()
