from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import Settings
from data.feature_loader import load_feature_data, resolve_feature_root
from models.config import TargetConfig, resolve_target_config
from monitoring.logging import setup_logger


@dataclass(frozen=True)
class LabelBuildResult:
    feature_set_name: str
    target_mode: str
    row_count: int
    labeled_row_count: int
    written_files: list[str]
    label_distribution: dict[str, int]
    report_path: str
    output_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_labels(
    settings: Settings,
    *,
    feature_set_name: str,
    target_mode: str,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_root: str | Path | None = None,
    output_root: str | Path | None = None,
    logger=None,
) -> dict[str, Any]:
    label_logger = logger or setup_logger(
        settings.log_level,
        settings.log_file,
        logger_name="microalpha.labeling",
    )
    target_config = resolve_target_config(settings, target_mode)
    label_logger.info(
        "Starting label build pipeline: feature_set=%s target_mode=%s symbols=%s range=%s..%s",
        feature_set_name,
        target_mode,
        list(symbols or settings.supported_symbols),
        start_date or "min",
        end_date or "max",
    )

    feature_frame = load_feature_data(
        settings,
        feature_set_name=feature_set_name,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        feature_root=feature_root,
    )
    labeled_frame, metadata = generate_labeled_frame(feature_frame, settings, target_config)
    target_column = metadata["target_column"]
    labeled_frame = labeled_frame.dropna(subset=[target_column]).reset_index(drop=True)

    target_base_root = Path(output_root or Path(settings.paths.processed_dir) / "labels")
    label_output_root = target_base_root / feature_set_name / target_mode
    written_files = _persist_labeled_frame(labeled_frame, label_output_root)
    report_path = _write_label_report(
        settings,
        feature_set_name=feature_set_name,
        target_mode=target_mode,
        metadata=metadata,
        labeled_frame=labeled_frame,
        output_root=label_output_root,
    )

    label_distribution = (
        labeled_frame[target_column].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if target_column in labeled_frame.columns
        else {}
    )
    label_logger.info(
        "Label build complete: feature_set=%s target_mode=%s rows=%s labeled_rows=%s files=%s report=%s",
        feature_set_name,
        target_mode,
        len(feature_frame),
        len(labeled_frame),
        len(written_files),
        report_path,
    )
    return LabelBuildResult(
        feature_set_name=feature_set_name,
        target_mode=target_mode,
        row_count=int(len(feature_frame)),
        labeled_row_count=int(len(labeled_frame)),
        written_files=written_files,
        label_distribution={str(key): int(value) for key, value in label_distribution.items()},
        report_path=report_path,
        output_root=str(label_output_root),
    ).to_dict()


def generate_labeled_frame(
    feature_frame: pd.DataFrame,
    settings: Settings,
    target_config: TargetConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feature_frame.empty:
        raise ValueError("Feature frame is empty. Build features before generating labels.")

    frame = feature_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    group_columns = ["symbol", "session_date"] if "session_date" in frame.columns else ["symbol"]

    price_column = _resolve_price_column(frame)
    future_price = _resolve_future_price(frame, target_config, price_column, group_columns)
    frame["future_price_proxy"] = future_price
    frame["future_return_bps"] = ((future_price / frame[price_column]) - 1.0) * 10000.0

    cost_component = float(target_config.cost_adjustment_bps)
    if "estimated_cost_bps" in frame.columns:
        cost_component = cost_component + (
            target_config.cost_adjustment_multiplier * frame["estimated_cost_bps"].fillna(0.0)
        )
    frame["target_cost_adjustment_bps"] = cost_component
    frame["future_net_return_bps"] = frame["future_return_bps"] - frame["target_cost_adjustment_bps"]

    target_column = f"target_{target_config.name}"
    if target_config.task_type == "classification":
        threshold = float(target_config.threshold_bps or 0.0)
        frame[target_column] = np.where(
            frame["future_net_return_bps"] > threshold,
            int(target_config.positive_label),
            int(target_config.negative_label),
        )
    elif target_config.task_type in {"ordinal", "distribution_bins"}:
        if not target_config.bin_edges_bps:
            raise ValueError(f"Target mode {target_config.name!r} requires bin_edges_bps.")
        labels = list(target_config.class_labels) if target_config.class_labels else list(range(len(target_config.bin_edges_bps) - 1))
        frame[target_column] = pd.cut(
            frame["future_net_return_bps"],
            bins=list(target_config.bin_edges_bps),
            labels=labels,
            include_lowest=True,
            right=False,
        ).astype(float)
    elif target_config.task_type in {"regression", "quantile_regression"}:
        frame[target_column] = frame["future_net_return_bps"]
    else:
        raise ValueError(f"Unsupported target task type {target_config.task_type!r}.")

    metadata = {
        "target_mode": target_config.name,
        "target_column": target_column,
        "task_type": target_config.task_type,
        "price_column": price_column,
        "horizon_bars": target_config.horizon_bars,
        "horizon_minutes": target_config.horizon_minutes,
        "threshold_bps": target_config.threshold_bps,
        "negative_threshold_bps": target_config.negative_threshold_bps,
        "bin_edges_bps": list(target_config.bin_edges_bps),
        "class_labels": list(target_config.class_labels),
        "quantiles": list(target_config.quantiles),
        "cost_adjustment_bps": target_config.cost_adjustment_bps,
        "cost_adjustment_multiplier": target_config.cost_adjustment_multiplier,
    }
    return frame, metadata


def _resolve_price_column(frame: pd.DataFrame) -> str:
    for column in ("price_proxy", "mid_price", "last_price", "close"):
        if column in frame.columns and frame[column].notna().any():
            return column
    raise ValueError("Could not resolve a price proxy for label generation.")


def _resolve_future_price(
    frame: pd.DataFrame,
    target_config: TargetConfig,
    price_column: str,
    group_columns: Sequence[str],
) -> pd.Series:
    if target_config.horizon_bars:
        return frame.groupby(list(group_columns), sort=False)[price_column].shift(-int(target_config.horizon_bars))

    if target_config.horizon_minutes:
        horizon_delta = timedelta(minutes=int(target_config.horizon_minutes))
        future_series = pd.Series(np.nan, index=frame.index, dtype=float)
        for _, group in frame.groupby(list(group_columns), sort=False):
            reference = group[["timestamp", price_column]].copy()
            reference["future_timestamp"] = reference["timestamp"]
            targets = group[["timestamp"]].copy()
            targets["lookup_timestamp"] = targets["timestamp"] + horizon_delta
            merged = pd.merge_asof(
                targets.sort_values("lookup_timestamp"),
                reference.sort_values("future_timestamp"),
                left_on="lookup_timestamp",
                right_on="future_timestamp",
                direction="forward",
            )
            future_series.loc[group.index] = merged[price_column].to_numpy()
        return future_series

    raise ValueError(f"Target mode {target_config.name!r} requires either horizon_bars or horizon_minutes.")


def _persist_labeled_frame(frame: pd.DataFrame, output_root: Path) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    written_files: list[str] = []
    for (session_date, symbol), group in frame.groupby(["session_date", "symbol"], sort=True):
        target_dir = output_root / str(session_date)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{str(symbol).upper()}.parquet"
        group.to_parquet(target_path, index=False)
        written_files.append(str(target_path))
    return written_files


def _write_label_report(
    settings: Settings,
    *,
    feature_set_name: str,
    target_mode: str,
    metadata: dict[str, Any],
    labeled_frame: pd.DataFrame,
    output_root: Path,
) -> str:
    report_dir = Path(settings.paths.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"label_build_report_{feature_set_name}_{target_mode}_{timestamp_token}.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set_name": feature_set_name,
        "target_mode": target_mode,
        "metadata": metadata,
        "row_count": int(len(labeled_frame)),
        "symbols": sorted(labeled_frame["symbol"].dropna().astype(str).unique().tolist()),
        "date_range": {
            "start": str(labeled_frame["session_date"].min()) if "session_date" in labeled_frame.columns else None,
            "end": str(labeled_frame["session_date"].max()) if "session_date" in labeled_frame.columns else None,
        },
        "target_distribution": labeled_frame[metadata["target_column"]].value_counts(dropna=False).sort_index().to_dict(),
        "output_root": str(output_root),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(report_path)
