from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def evaluate_execution_metrics(
    decision_frame: pd.DataFrame,
    *,
    orders: list[Mapping[str, Any]] | None = None,
    fills: list[Mapping[str, Any]] | None = None,
    reports: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    frame = decision_frame.copy()
    if frame.empty:
        empty = pd.DataFrame()
        return {
            "summary": {
                "order_count": 0,
                "fill_event_count": 0,
                "rejection_rate": 0.0,
                "cancel_rate": 0.0,
                "fill_ratio_mean": 0.0,
            },
            "alerts": ["empty_execution_frame"],
            "orders": empty,
        }

    orders_frame = pd.DataFrame(list(orders or []))
    fills_frame = pd.DataFrame(list(fills or []))
    reports_frame = pd.DataFrame(list(reports or []))

    order_rows = frame.loc[frame["order_id"].notna()].copy() if "order_id" in frame.columns else pd.DataFrame()
    if order_rows.empty:
        return {
            "summary": {
                "order_count": 0,
                "fill_event_count": int(len(fills_frame)),
                "rejection_rate": 0.0,
                "cancel_rate": 0.0,
                "fill_ratio_mean": 0.0,
            },
            "alerts": ["no_orders_found"],
            "orders": pd.DataFrame(),
        }

    order_rows["decision_to_submit_ms"] = _numeric(order_rows, "decision_to_submit_ms")
    order_rows["submit_to_ack_ms"] = _numeric(order_rows, "submit_to_ack_ms")
    order_rows["submit_to_first_fill_ms"] = _numeric(order_rows, "submit_to_first_fill_ms")
    order_rows["submit_to_final_fill_ms"] = _numeric(order_rows, "submit_to_final_fill_ms")
    order_rows["fill_ratio"] = _numeric(order_rows, "fill_ratio")
    order_rows["average_execution_slippage_bps"] = _numeric(order_rows, "average_execution_slippage_bps")
    order_rows["expected_cost_bps"] = _numeric(order_rows, "expected_cost_bps")
    order_rows["realized_pnl_delta"] = _numeric(order_rows, "realized_pnl_delta")
    order_rows["execution_status"] = order_rows.get("execution_status", pd.Series(dtype=str)).astype(str)
    order_rows["backend_name"] = order_rows.get("backend_name", pd.Series(dtype=str)).astype(str)
    order_rows["execution_discrepancy_flag_count"] = order_rows.get(
        "execution_discrepancy_flags",
        pd.Series([[]] * len(order_rows)),
    ).map(lambda value: len(value) if isinstance(value, list) else 0)

    rejected_mask = order_rows["execution_status"].isin(["REJECTED", "FAILED", "FAILED_VALIDATION"])
    cancelled_mask = order_rows["execution_status"].isin(["CANCELLED"])
    filled_mask = order_rows["execution_status"].isin(["FILLED", "PARTIALLY_FILLED"])

    summary = {
        "order_count": int(len(order_rows)),
        "fill_event_count": int(len(fills_frame)),
        "filled_order_count": int(filled_mask.sum()),
        "rejected_order_count": int(rejected_mask.sum()),
        "cancelled_order_count": int(cancelled_mask.sum()),
        "rejection_rate": _ratio(rejected_mask.sum(), len(order_rows)),
        "cancel_rate": _ratio(cancelled_mask.sum(), len(order_rows)),
        "fill_ratio_mean": _series_mean(order_rows.get("fill_ratio")),
        "avg_decision_to_submit_ms": _series_mean(order_rows.get("decision_to_submit_ms")),
        "avg_submit_to_ack_ms": _series_mean(order_rows.get("submit_to_ack_ms")),
        "avg_submit_to_first_fill_ms": _series_mean(order_rows.get("submit_to_first_fill_ms")),
        "avg_submit_to_final_fill_ms": _series_mean(order_rows.get("submit_to_final_fill_ms")),
        "avg_execution_slippage_bps": _series_mean(order_rows.get("average_execution_slippage_bps")),
        "avg_expected_cost_bps": _series_mean(order_rows.get("expected_cost_bps")),
        "avg_realized_pnl_delta": _series_mean(order_rows.get("realized_pnl_delta")),
        "avg_discrepancy_flag_count": _series_mean(order_rows.get("execution_discrepancy_flag_count")),
        "backend_names": sorted({value for value in order_rows["backend_name"].dropna().astype(str) if value}),
        "report_count": int(len(reports_frame)),
    }

    alerts: list[str] = []
    if summary["rejection_rate"] is not None and summary["rejection_rate"] > 0.25:
        alerts.append(f"high_rejection_rate:{summary['rejection_rate']:.4f}")
    if summary["cancel_rate"] is not None and summary["cancel_rate"] > 0.25:
        alerts.append(f"high_cancel_rate:{summary['cancel_rate']:.4f}")
    if summary["fill_ratio_mean"] is not None and summary["fill_ratio_mean"] < 0.8:
        alerts.append(f"low_fill_ratio:{summary['fill_ratio_mean']:.4f}")

    return {
        "summary": summary,
        "alerts": alerts,
        "orders": order_rows.reset_index(drop=True),
    }


def compare_mock_vs_real_execution(
    report_root: str | Path,
    *,
    current_run_report: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(report_root)
    current_backend = str(current_run_report.get("backend_name") or "")
    current_run_id = str(current_run_report.get("run_id") or "")
    if not current_backend or not current_run_id:
        return {"status": "no_comparable_run", "reason": "missing_backend_or_run_id"}

    candidates: list[dict[str, Any]] = []
    for path in root.glob("*/run_report.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_id") or "") != current_run_id:
            continue
        if str(payload.get("backend_name") or "") == current_backend:
            continue
        candidates.append(payload)

    if not candidates:
        return {"status": "no_comparable_run", "reason": "no_matching_backend_run_found"}

    reference = sorted(candidates, key=lambda item: str(item.get("generated_at_utc") or ""), reverse=True)[0]
    current_execution = dict(current_run_report.get("execution_summary", {}) or {})
    reference_execution = dict(reference.get("execution_summary", {}) or {})
    current_performance = dict(current_run_report.get("performance_summary", {}) or {})
    reference_performance = dict(reference.get("performance_summary", {}) or {})

    return {
        "status": "ok",
        "current_backend": current_backend,
        "reference_backend": reference.get("backend_name"),
        "reference_run_report_path": reference.get("report_paths", {}).get("run_report_path") or reference.get("run_report_path"),
        "current_run_label": current_run_report.get("run_label"),
        "reference_run_label": reference.get("run_label"),
        "execution_deltas": {
            "fill_ratio_mean_delta": _delta(current_execution, reference_execution, "fill_ratio_mean"),
            "avg_submit_to_ack_ms_delta": _delta(current_execution, reference_execution, "avg_submit_to_ack_ms"),
            "avg_submit_to_final_fill_ms_delta": _delta(current_execution, reference_execution, "avg_submit_to_final_fill_ms"),
            "rejection_rate_delta": _delta(current_execution, reference_execution, "rejection_rate"),
            "avg_execution_slippage_bps_delta": _delta(current_execution, reference_execution, "avg_execution_slippage_bps"),
        },
        "performance_deltas": {
            "session_total_pnl_delta": _delta(current_performance, reference_performance, "session_total_pnl"),
            "portfolio_total_pnl_delta": _delta(current_performance, reference_performance, "portfolio_total_pnl"),
            "max_drawdown_delta": _delta(current_performance, reference_performance, "max_drawdown"),
        },
    }


def _series_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return float(numerator / denominator)


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float | None:
    left_value = _to_float(left.get(key))
    right_value = _to_float(right.get(key))
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

