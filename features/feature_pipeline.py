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
from monitoring.data_quality import DataQualityReport, assess_market_data_quality
from monitoring.logging import setup_logger


FEATURE_COLUMNS = [
    "last_price",
    "bid",
    "ask",
    "mid_price",
    "micro_price",
    "spread",
    "spread_bps",
    "bid_ask_imbalance",
    "rolling_spread_mean_bps",
    "rolling_spread_std_bps",
    "rolling_imbalance_mean",
    "rolling_imbalance_std",
    "return_1_bps",
    "return_short_bps",
    "return_medium_bps",
    "rolling_volatility_short_bps",
    "rolling_volatility_medium_bps",
    "rolling_volatility_long_bps",
    "vwap_approx",
    "distance_to_vwap_bps",
    "relative_volume",
    "minutes_since_open",
    "time_of_day_sin",
    "time_of_day_cos",
    "orb_high",
    "orb_low",
    "orb_range_width",
    "orb_range_width_bps",
    "orb_relative_price_position",
    "breakout_distance",
    "breakout_distance_bps",
    "orb_range_complete",
    "estimated_cost_bps",
    "spread_proxy_bps",
    "slippage_proxy_bps",
]


@dataclass(frozen=True)
class FeatureBuildResult:
    input_rows: int
    cleaned_rows: int
    feature_rows: int
    symbols: tuple[str, ...]
    dates: tuple[str, ...]
    written_files: list[str]
    raw_quality: dict[str, Any]
    cleaned_quality: dict[str, Any]
    output_root: str
    report_path: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "cleaned_rows": self.cleaned_rows,
            "feature_rows": self.feature_rows,
            "symbols": list(self.symbols),
            "dates": list(self.dates),
            "written_files": self.written_files,
            "raw_quality": self.raw_quality,
            "cleaned_quality": self.cleaned_quality,
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
    logger=None,
) -> dict[str, Any]:
    pipeline_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.feature_pipeline",
    )
    started_at = time.monotonic()
    pipeline_logger.info("Starting feature build pipeline.")

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
    feature_frame = build_feature_frame(cleaned_frame, settings)
    pipeline_logger.info(
        "Generated feature frame rows=%s columns=%s days=%s",
        len(feature_frame),
        len(feature_frame.columns),
        sorted(feature_frame["session_date"].dropna().astype(str).unique().tolist()),
    )
    write_summary = persist_feature_frame(feature_frame, output_root or settings.paths.feature_dir)
    report_path = persist_feature_report(
        settings,
        raw_quality=raw_quality,
        cleaned_quality=cleaned_quality,
        feature_frame=feature_frame,
        output_root=output_root or settings.paths.feature_dir,
    )

    duration_seconds = round(time.monotonic() - started_at, 3)
    pipeline_logger.info(
        "Feature build pipeline complete: input_rows=%s cleaned_rows=%s feature_rows=%s files=%s duration=%ss report=%s",
        len(raw_frame),
        len(cleaned_frame),
        len(feature_frame),
        len(write_summary["written_files"]),
        duration_seconds,
        report_path,
    )

    result = FeatureBuildResult(
        input_rows=int(len(raw_frame)),
        cleaned_rows=int(len(cleaned_frame)),
        feature_rows=int(len(feature_frame)),
        symbols=tuple(sorted(feature_frame["symbol"].dropna().astype(str).unique().tolist())),
        dates=tuple(sorted(feature_frame["session_date"].dropna().astype(str).unique().tolist())),
        written_files=write_summary["written_files"],
        raw_quality=raw_quality.to_dict(),
        cleaned_quality=cleaned_quality.to_dict(),
        output_root=str(output_root or settings.paths.feature_dir),
        report_path=report_path,
        duration_seconds=duration_seconds,
    )
    return result.to_dict()


def build_feature_frame(clean_frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    frame = clean_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    frame["exchange_timestamp"] = frame["timestamp"].dt.tz_convert(settings.session.timezone)
    frame["session_date"] = frame["exchange_timestamp"].dt.strftime("%Y-%m-%d")
    frame["session_time"] = frame["exchange_timestamp"].dt.time

    frame["mid_price"] = np.where(
        frame["bid"].notna() & frame["ask"].notna(),
        (frame["bid"] + frame["ask"]) / 2.0,
        frame["last_price"],
    )
    frame["micro_price"] = np.where(
        (frame["bid_size"] + frame["ask_size"]) > 0,
        ((frame["ask"] * frame["bid_size"]) + (frame["bid"] * frame["ask_size"]))
        / (frame["bid_size"] + frame["ask_size"]),
        frame["mid_price"],
    )
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["spread_bps"] = np.where(
        frame["mid_price"] > 0,
        frame["spread"] / frame["mid_price"] * 10000.0,
        np.nan,
    )
    frame["bid_ask_imbalance"] = np.where(
        (frame["bid_size"] + frame["ask_size"]) > 0,
        (frame["bid_size"] - frame["ask_size"]) / (frame["bid_size"] + frame["ask_size"]),
        0.0,
    )

    frame = _add_orb_features(frame, settings)
    frame = _add_microstructure_features(frame, settings)
    frame = _add_intraday_features(frame, settings)
    frame = _add_cost_features(frame)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].fillna(0.0)
    return frame


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
    feature_frame: pd.DataFrame,
    output_root: str | Path,
) -> str:
    report_dir = Path(settings.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"feature_build_report_{timestamp_token}.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_quality": raw_quality.to_dict(),
        "cleaned_quality": cleaned_quality.to_dict(),
        "feature_rows": int(len(feature_frame)),
        "feature_columns": list(FEATURE_COLUMNS),
        "output_root": str(output_root),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(report_path)


def _add_orb_features(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    enriched = frame.copy()
    reference_price = enriched["last_price"].fillna(enriched["mid_price"])
    opening_mask = (
        (enriched["exchange_timestamp"].dt.time >= settings.session.orb_start)
        & (enriched["exchange_timestamp"].dt.time < settings.session.orb_end)
    )
    orb_stats = (
        enriched.loc[opening_mask]
        .assign(reference_price=reference_price.loc[opening_mask])
        .groupby(["symbol", "session_date"], sort=False)
        .agg(
            orb_high=("reference_price", "max"),
            orb_low=("reference_price", "min"),
        )
    )
    enriched = enriched.merge(orb_stats, how="left", left_on=["symbol", "session_date"], right_index=True)
    enriched["orb_range_width"] = enriched["orb_high"] - enriched["orb_low"]
    enriched["orb_range_mid"] = (enriched["orb_high"] + enriched["orb_low"]) / 2.0
    enriched["orb_range_width_bps"] = np.where(
        enriched["orb_range_mid"] > 0,
        enriched["orb_range_width"] / enriched["orb_range_mid"] * 10000.0,
        np.nan,
    )
    enriched["orb_relative_price_position"] = np.where(
        enriched["orb_range_width"] > 0,
        (reference_price - enriched["orb_low"]) / enriched["orb_range_width"],
        np.nan,
    )
    enriched["breakout_distance"] = np.where(
        reference_price > enriched["orb_high"],
        reference_price - enriched["orb_high"],
        np.where(reference_price < enriched["orb_low"], reference_price - enriched["orb_low"], 0.0),
    )
    enriched["breakout_distance_bps"] = np.where(
        enriched["orb_range_mid"] > 0,
        enriched["breakout_distance"] / enriched["orb_range_mid"] * 10000.0,
        np.nan,
    )
    session_minutes = (
        enriched["exchange_timestamp"].dt.hour * 60
        + enriched["exchange_timestamp"].dt.minute
        + enriched["exchange_timestamp"].dt.second / 60.0
    )
    open_minutes = settings.session.regular_market_open.hour * 60 + settings.session.regular_market_open.minute
    orb_end_minutes = settings.session.orb_end.hour * 60 + settings.session.orb_end.minute
    enriched["minutes_since_open"] = session_minutes - open_minutes
    enriched["orb_range_complete"] = (session_minutes >= orb_end_minutes).astype(float)
    return enriched


def _add_microstructure_features(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    enriched = frame.copy()
    group_keys = ["symbol", "session_date"]
    short_window = settings.feature_pipeline.rolling_short_window

    enriched["rolling_spread_mean_bps"] = enriched.groupby(group_keys, sort=False)["spread_bps"].transform(
        lambda series: series.rolling(short_window, min_periods=1).mean()
    )
    enriched["rolling_spread_std_bps"] = enriched.groupby(group_keys, sort=False)["spread_bps"].transform(
        lambda series: series.rolling(short_window, min_periods=2).std()
    )
    enriched["rolling_imbalance_mean"] = enriched.groupby(group_keys, sort=False)["bid_ask_imbalance"].transform(
        lambda series: series.rolling(short_window, min_periods=1).mean()
    )
    enriched["rolling_imbalance_std"] = enriched.groupby(group_keys, sort=False)["bid_ask_imbalance"].transform(
        lambda series: series.rolling(short_window, min_periods=2).std()
    )
    return enriched


def _add_intraday_features(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    enriched = frame.copy()
    group_keys = ["symbol", "session_date"]
    short_window = settings.feature_pipeline.rolling_short_window
    medium_window = settings.feature_pipeline.rolling_medium_window
    long_window = settings.feature_pipeline.rolling_long_window
    vwap_window = settings.feature_pipeline.vwap_window
    volume_window = settings.feature_pipeline.volume_window

    enriched["return_1_bps"] = enriched.groupby(group_keys, sort=False)["mid_price"].pct_change(1) * 10000.0
    enriched["return_short_bps"] = enriched.groupby(group_keys, sort=False)["mid_price"].pct_change(short_window) * 10000.0
    enriched["return_medium_bps"] = enriched.groupby(group_keys, sort=False)["mid_price"].pct_change(medium_window) * 10000.0

    enriched["rolling_volatility_short_bps"] = enriched.groupby(group_keys, sort=False)["return_1_bps"].transform(
        lambda series: series.rolling(short_window, min_periods=2).std()
    )
    enriched["rolling_volatility_medium_bps"] = enriched.groupby(group_keys, sort=False)["return_1_bps"].transform(
        lambda series: series.rolling(medium_window, min_periods=2).std()
    )
    enriched["rolling_volatility_long_bps"] = enriched.groupby(group_keys, sort=False)["return_1_bps"].transform(
        lambda series: series.rolling(long_window, min_periods=2).std()
    )

    vwap_chunks: list[pd.Series] = []
    for _, group in enriched.groupby(group_keys, sort=False):
        vwap_chunks.append(_rolling_vwap(group["last_price"], group["volume"], vwap_window))
    enriched["vwap_approx"] = pd.concat(vwap_chunks).sort_index()
    enriched["distance_to_vwap_bps"] = np.where(
        enriched["vwap_approx"] > 0,
        (enriched["mid_price"] - enriched["vwap_approx"]) / enriched["vwap_approx"] * 10000.0,
        np.nan,
    )

    rolling_volume = enriched.groupby(group_keys, sort=False)["volume"].transform(
        lambda series: series.rolling(volume_window, min_periods=1).mean()
    )
    enriched["relative_volume"] = np.where(rolling_volume > 0, enriched["volume"] / rolling_volume, np.nan)

    minutes_since_midnight = (
        enriched["exchange_timestamp"].dt.hour * 60
        + enriched["exchange_timestamp"].dt.minute
        + enriched["exchange_timestamp"].dt.second / 60.0
    )
    session_span_minutes = max(
        (
            (settings.session.regular_market_close.hour * 60 + settings.session.regular_market_close.minute)
            - (settings.session.regular_market_open.hour * 60 + settings.session.regular_market_open.minute)
        ),
        1,
    )
    normalized_time = (minutes_since_midnight - (settings.session.regular_market_open.hour * 60 + settings.session.regular_market_open.minute)) / session_span_minutes
    enriched["time_of_day_sin"] = np.sin(2 * np.pi * normalized_time)
    enriched["time_of_day_cos"] = np.cos(2 * np.pi * normalized_time)
    return enriched


def _add_cost_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["spread_proxy_bps"] = enriched["spread_bps"].fillna(enriched["rolling_spread_mean_bps"])
    enriched["slippage_proxy_bps"] = (
        enriched["spread_proxy_bps"].fillna(0.0) * 0.25
        + enriched["rolling_volatility_short_bps"].abs().fillna(0.0) * 0.10
    )
    enriched["estimated_cost_bps"] = (
        enriched["spread_proxy_bps"].fillna(0.0) * 0.50
        + enriched["slippage_proxy_bps"].fillna(0.0)
    )
    return enriched


def _rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    safe_price = price.ffill().fillna(0.0)
    safe_volume = volume.fillna(0.0)
    price_volume = safe_price * safe_volume
    rolling_pv = price_volume.rolling(window, min_periods=1).sum()
    rolling_volume = safe_volume.rolling(window, min_periods=1).sum()
    vwap = rolling_pv / rolling_volume.replace(0, np.nan)
    return vwap.fillna(safe_price)
