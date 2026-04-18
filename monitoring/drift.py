from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import Settings
from data.feature_loader import load_feature_data
from evaluation.io import list_phase7_parquet_paths, load_phase7_frame
from labels.dataset_builder import load_labeled_data


def build_drift_report(
    settings: Settings,
    *,
    decision_frame: pd.DataFrame,
    phase7_summary: dict[str, Any],
    phase8_config,
) -> dict[str, Any]:
    model_name = str(phase7_summary.get("active_model", {}).get("model_name") or "")
    feature_set_name = str(phase7_summary.get("active_model", {}).get("feature_set_name") or "")
    target_mode = str(phase7_summary.get("active_model", {}).get("target_mode") or "")
    phase7_report_dir = Path(phase7_summary.get("summary_path", "")).resolve().parent if phase7_summary.get("summary_path") else Path("data/reports/phase7")

    timestamps = pd.to_datetime(decision_frame.get("timestamp"), utc=True, errors="coerce")
    timestamps = timestamps.dropna()
    if timestamps.empty:
        return {
            "alerts": ["decision_frame_missing_timestamps"],
            "data_drift": {"status": "unavailable"},
            "prediction_drift": {"status": "unavailable"},
            "label_drift": {"status": "unavailable"},
        }

    current_start = timestamps.min().date()
    current_end = timestamps.max().date()
    reference_start = current_start - timedelta(days=phase8_config.evaluation_window.reference_days)
    reference_end = current_start - timedelta(days=1)

    data_drift = detect_data_drift(
        settings,
        feature_set_name=feature_set_name,
        current_start_date=current_start.isoformat(),
        current_end_date=current_end.isoformat(),
        reference_start_date=reference_start.isoformat(),
        reference_end_date=reference_end.isoformat(),
        thresholds=phase8_config.drift_thresholds,
        min_samples=phase8_config.evaluation_window.min_samples_for_drift,
    )
    prediction_drift = detect_prediction_drift(
        decision_frame,
        phase7_report_dir=phase7_report_dir,
        current_summary_path=phase7_summary.get("summary_path"),
        model_name=model_name,
        feature_set_name=feature_set_name,
        compare_run_limit=phase8_config.evaluation_window.compare_run_limit,
        thresholds=phase8_config.drift_thresholds,
        min_samples=phase8_config.evaluation_window.min_samples_for_drift,
    )
    label_drift = detect_label_drift(
        settings,
        feature_set_name=feature_set_name,
        target_mode=target_mode,
        current_start_date=current_start.isoformat(),
        current_end_date=current_end.isoformat(),
        reference_start_date=reference_start.isoformat(),
        reference_end_date=reference_end.isoformat(),
        thresholds=phase8_config.drift_thresholds,
        min_samples=phase8_config.evaluation_window.min_samples_for_drift,
    )

    alerts = [
        *data_drift.get("alerts", []),
        *prediction_drift.get("alerts", []),
        *label_drift.get("alerts", []),
    ]
    return {
        "alerts": alerts,
        "data_drift": data_drift,
        "prediction_drift": prediction_drift,
        "label_drift": label_drift,
    }


def detect_data_drift(
    settings: Settings,
    *,
    feature_set_name: str,
    current_start_date: str,
    current_end_date: str,
    reference_start_date: str,
    reference_end_date: str,
    thresholds,
    min_samples: int,
) -> dict[str, Any]:
    try:
        current_frame = load_feature_data(
            settings,
            feature_set_name=feature_set_name,
            start_date=current_start_date,
            end_date=current_end_date,
        )
        reference_frame = load_feature_data(
            settings,
            feature_set_name=feature_set_name,
            start_date=reference_start_date,
            end_date=reference_end_date,
        )
    except FileNotFoundError as exc:
        return {"status": "unavailable", "alerts": [f"data_drift_input_missing:{exc}"], "details": []}

    numeric_columns = [
        column
        for column in current_frame.columns
        if pd.api.types.is_numeric_dtype(current_frame[column]) and column in reference_frame.columns
    ]
    details = _drift_details(
        reference_frame,
        current_frame,
        columns=numeric_columns,
        psi_warning=thresholds.feature_psi_warning,
        mean_shift_sigma_warning=thresholds.mean_shift_sigma_warning,
        min_samples=min_samples,
    )
    return _drift_summary("data_drift", details, thresholds.feature_psi_warning, thresholds.feature_psi_critical)


def detect_prediction_drift(
    decision_frame: pd.DataFrame,
    *,
    phase7_report_dir: Path,
    current_summary_path: str | None,
    model_name: str,
    feature_set_name: str,
    compare_run_limit: int,
    thresholds,
    min_samples: int,
) -> dict[str, Any]:
    current_parquet_path = None
    if current_summary_path:
        summary_payload = Path(current_summary_path)
        if summary_payload.exists():
            current_parquet_path = Path(_safe_json_read(summary_payload).get("parquet_path", ""))

    reference_frames: list[pd.DataFrame] = []
    for path in reversed(list_phase7_parquet_paths(phase7_report_dir, limit=compare_run_limit * 3)):
        if current_parquet_path is not None and path.resolve() == current_parquet_path.resolve():
            continue
        frame = load_phase7_frame(path)
        if frame.empty:
            continue
        if model_name and str(frame.get("model_name").iloc[0]) != model_name:
            continue
        if feature_set_name and str(frame.get("feature_set_name").iloc[0]) != feature_set_name:
            continue
        reference_frames.append(frame)
        if len(reference_frames) >= compare_run_limit:
            break
    if not reference_frames:
        return {"status": "unavailable", "alerts": ["prediction_drift_reference_runs_missing"], "details": []}

    reference_frame = pd.concat(reference_frames, ignore_index=True)
    columns = [column for column in ("score", "probability", "expected_return_bps", "net_edge_bps") if column in decision_frame.columns and column in reference_frame.columns]
    details = _drift_details(
        reference_frame,
        decision_frame,
        columns=columns,
        psi_warning=thresholds.prediction_psi_warning,
        mean_shift_sigma_warning=thresholds.mean_shift_sigma_warning,
        min_samples=min_samples,
    )
    return _drift_summary("prediction_drift", details, thresholds.prediction_psi_warning, thresholds.prediction_psi_warning)


def detect_label_drift(
    settings: Settings,
    *,
    feature_set_name: str,
    target_mode: str,
    current_start_date: str,
    current_end_date: str,
    reference_start_date: str,
    reference_end_date: str,
    thresholds,
    min_samples: int,
) -> dict[str, Any]:
    try:
        current_frame = load_labeled_data(
            settings,
            feature_set_name=feature_set_name,
            target_mode=target_mode,
            start_date=current_start_date,
            end_date=current_end_date,
        )
        reference_frame = load_labeled_data(
            settings,
            feature_set_name=feature_set_name,
            target_mode=target_mode,
            start_date=reference_start_date,
            end_date=reference_end_date,
        )
    except FileNotFoundError:
        return {"status": "unavailable", "alerts": ["label_drift_label_store_missing"], "details": []}

    columns = [column for column in ("future_return_bps", "future_net_return_bps") if column in current_frame.columns and column in reference_frame.columns]
    if not columns:
        return {"status": "unavailable", "alerts": ["label_drift_columns_missing"], "details": []}
    details = _drift_details(
        reference_frame,
        current_frame,
        columns=columns,
        psi_warning=thresholds.label_psi_warning,
        mean_shift_sigma_warning=thresholds.mean_shift_sigma_warning,
        min_samples=min_samples,
    )
    return _drift_summary("label_drift", details, thresholds.label_psi_warning, thresholds.label_psi_warning)


def population_stability_index(reference: Iterable[float], current: Iterable[float], *, bucket_count: int = 10) -> float | None:
    ref = pd.Series(list(reference), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    cur = pd.Series(list(current), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if ref.empty or cur.empty:
        return None
    if ref.nunique() <= 1 and cur.nunique() <= 1:
        return 0.0 if float(ref.iloc[0]) == float(cur.iloc[0]) else float("inf")

    quantiles = np.linspace(0, 1, min(bucket_count, max(ref.nunique(), 2)) + 1)
    bins = np.unique(ref.quantile(quantiles).to_numpy())
    if len(bins) < 3:
        bins = np.array([ref.min() - 1e-9, ref.median(), ref.max() + 1e-9])
    bins[0] = min(bins[0], float(cur.min())) - 1e-9
    bins[-1] = max(bins[-1], float(cur.max())) + 1e-9

    ref_hist = pd.cut(ref, bins=bins, include_lowest=True).value_counts(normalize=True, sort=False)
    cur_hist = pd.cut(cur, bins=bins, include_lowest=True).value_counts(normalize=True, sort=False)
    psi = 0.0
    for bucket in ref_hist.index:
        ref_ratio = float(ref_hist.get(bucket, 0.0) or 0.0)
        cur_ratio = float(cur_hist.get(bucket, 0.0) or 0.0)
        if not np.isfinite(ref_ratio):
            ref_ratio = 0.0
        if not np.isfinite(cur_ratio):
            cur_ratio = 0.0
        ref_ratio = max(ref_ratio, 1e-6)
        cur_ratio = max(cur_ratio, 1e-6)
        psi += (cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)
    return float(psi)


def _drift_details(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    *,
    columns: list[str],
    psi_warning: float,
    mean_shift_sigma_warning: float,
    min_samples: int,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for column in columns:
        reference = pd.to_numeric(reference_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        current = pd.to_numeric(current_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(reference) < min_samples or len(current) < min_samples:
            details.append(
                {
                    "column": column,
                    "status": "insufficient_sample",
                    "reference_count": int(len(reference)),
                    "current_count": int(len(current)),
                }
            )
            continue
        reference_std = float(reference.std(ddof=0))
        current_std = float(current.std(ddof=0))
        mean_shift_sigma = 0.0 if reference_std <= 0 else abs(float(current.mean() - reference.mean())) / reference_std
        psi = population_stability_index(reference, current)
        alerts: list[str] = []
        if psi is not None and psi >= psi_warning:
            alerts.append(f"psi_warning:{psi:.6f}>={psi_warning:.6f}")
        if mean_shift_sigma >= mean_shift_sigma_warning:
            alerts.append(f"mean_shift_sigma_warning:{mean_shift_sigma:.6f}>={mean_shift_sigma_warning:.6f}")
        details.append(
            {
                "column": column,
                "status": "ok" if not alerts else "warning",
                "reference_count": int(len(reference)),
                "current_count": int(len(current)),
                "reference_mean": float(reference.mean()),
                "current_mean": float(current.mean()),
                "reference_std": reference_std,
                "current_std": current_std,
                "mean_shift_sigma": mean_shift_sigma,
                "psi": psi,
                "alerts": alerts,
            }
        )
    return details


def _drift_summary(name: str, details: list[dict[str, Any]], warning_threshold: float, critical_threshold: float) -> dict[str, Any]:
    psi_values = [float(item["psi"]) for item in details if item.get("psi") is not None and np.isfinite(item.get("psi"))]
    top_details = sorted(
        [item for item in details if item.get("psi") is not None and np.isfinite(item.get("psi"))],
        key=lambda item: float(item.get("psi", 0.0)),
        reverse=True,
    )[:10]
    alerts: list[str] = []
    if psi_values:
        max_psi = max(psi_values)
        if max_psi >= critical_threshold:
            alerts.append(f"{name}_critical:max_psi={max_psi:.6f}")
        elif max_psi >= warning_threshold:
            alerts.append(f"{name}_warning:max_psi={max_psi:.6f}")
    for item in details:
        alerts.extend(item.get("alerts", []))
    return {
        "status": "ok" if not alerts else "warning",
        "column_count": int(len(details)),
        "max_psi": None if not psi_values else float(max(psi_values)),
        "alerts": alerts,
        "details": details,
        "top_drift_columns": top_details,
    }


def _safe_json_read(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
