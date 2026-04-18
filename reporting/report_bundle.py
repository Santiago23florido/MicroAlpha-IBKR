from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from config.phase7 import load_phase7_config
from config.phase8 import load_phase8_config
from evaluation.compare_runs import compare_runs, update_economic_leaderboard
from evaluation.io import filter_records, flatten_decision_metadata, load_phase7_frame, read_jsonl, resolve_phase7_paths, write_json
from evaluation.performance import analyze_trade_logs, evaluate_performance, performance_by_segments
from evaluation.signal_analysis import analyze_signal_quality
from monitoring.drift import build_drift_report


def evaluate_performance_report(
    settings: Settings,
    *,
    summary_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    context = _build_report_context(settings, summary_path=summary_path, parquet_path=parquet_path)
    performance = evaluate_performance(
        context["decision_frame"],
        fills=context["fills"],
        final_portfolio=context["phase7_summary"].get("portfolio_final"),
        thresholds=context["phase8"].performance_thresholds,
    )
    segments = performance_by_segments(context["decision_frame"])
    payload = {
        "status": "ok",
        "summary": performance["summary"],
        "alerts": performance["alerts"],
        "outcome_column": segments["outcome_column"],
        "segment_names": sorted(segments["segment_tables"].keys()),
        "closed_trade_count": performance["summary"]["closed_trade_count"],
        "open_trade_count": performance["summary"]["open_trade_count"],
        "phase7_summary_path": context["summary_path"],
        "phase7_parquet_path": context["parquet_path"],
    }
    if persist:
        bundle = _write_performance_outputs(context, performance, segments)
        payload["report_paths"] = bundle
    return payload


def analyze_signal_report(
    settings: Settings,
    *,
    summary_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    context = _build_report_context(settings, summary_path=summary_path, parquet_path=parquet_path)
    signal = analyze_signal_quality(
        context["decision_frame"],
        degenerate_output_std_floor=context["phase8"].drift_thresholds.degenerate_output_std_floor,
    )
    payload = {
        "status": "ok",
        "summary": signal["summary"],
        "alerts": signal["alerts"],
        "table_names": sorted(signal["tables"].keys()),
        "phase7_summary_path": context["summary_path"],
        "phase7_parquet_path": context["parquet_path"],
    }
    if persist:
        payload["report_paths"] = _write_signal_outputs(context, signal)
    return payload


def detect_drift_report(
    settings: Settings,
    *,
    summary_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    context = _build_report_context(settings, summary_path=summary_path, parquet_path=parquet_path)
    drift = build_drift_report(
        settings,
        decision_frame=context["decision_frame"],
        phase7_summary=context["phase7_summary"],
        phase8_config=context["phase8"],
    )
    payload = {
        "status": "ok",
        "alerts": drift["alerts"],
        "data_drift": {
            "status": drift["data_drift"].get("status"),
            "max_psi": drift["data_drift"].get("max_psi"),
        },
        "prediction_drift": {
            "status": drift["prediction_drift"].get("status"),
            "max_psi": drift["prediction_drift"].get("max_psi"),
        },
        "label_drift": {
            "status": drift["label_drift"].get("status"),
            "max_psi": drift["label_drift"].get("max_psi"),
        },
        "phase7_summary_path": context["summary_path"],
        "phase7_parquet_path": context["parquet_path"],
    }
    if persist:
        payload["report_paths"] = _write_drift_outputs(context, drift)
    return payload


def generate_report(
    settings: Settings,
    *,
    summary_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    auto_generated: bool = False,
) -> dict[str, Any]:
    context = _build_report_context(settings, summary_path=summary_path, parquet_path=parquet_path)
    performance = evaluate_performance(
        context["decision_frame"],
        fills=context["fills"],
        final_portfolio=context["phase7_summary"].get("portfolio_final"),
        thresholds=context["phase8"].performance_thresholds,
    )
    segments = performance_by_segments(context["decision_frame"])
    signal = analyze_signal_quality(
        context["decision_frame"],
        degenerate_output_std_floor=context["phase8"].drift_thresholds.degenerate_output_std_floor,
    )
    drift = build_drift_report(
        settings,
        decision_frame=context["decision_frame"],
        phase7_summary=context["phase7_summary"],
        phase8_config=context["phase8"],
    )
    trade_analysis = analyze_trade_logs(
        context["decision_frame"],
        orders=context["orders"],
        fills=context["fills"],
        reports=context["reports"],
    )

    report_dir = _bundle_dir(context)
    report_paths = {
        **_write_performance_outputs(context, performance, segments),
        **_write_signal_outputs(context, signal),
        **_write_drift_outputs(context, drift),
        **_write_trade_outputs(context, trade_analysis),
    }

    metrics_report_path = write_json(
        report_dir / "metrics_report.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "performance_summary": performance["summary"],
            "signal_summary": signal["summary"],
            "trade_summary": trade_analysis["summary"],
        },
    )
    alerts = [
        *performance["alerts"],
        *segments["alerts"],
        *signal["alerts"],
        *drift["alerts"],
    ]
    alerts = _apply_alert_flags(alerts, context["phase8"].alert_flags)
    run_report = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": context["phase7_summary"].get("active_model", {}).get("run_id"),
        "run_label": Path(context["parquet_path"]).stem,
        "phase7_summary_path": str(context["summary_path"]),
        "phase7_parquet_path": str(context["parquet_path"]),
        "model_name": context["phase7_summary"].get("active_model", {}).get("model_name"),
        "feature_set_name": context["phase7_summary"].get("active_model", {}).get("feature_set_name"),
        "target_mode": context["phase7_summary"].get("active_model", {}).get("target_mode"),
        "auto_generated": auto_generated,
        "alerts": alerts,
        "performance_summary": performance["summary"],
        "signal_summary": signal["summary"],
        "drift_summary": {
            "data_drift_max_psi": drift["data_drift"].get("max_psi"),
            "prediction_drift_max_psi": drift["prediction_drift"].get("max_psi"),
            "label_drift_max_psi": drift["label_drift"].get("max_psi"),
        },
        "trade_summary": trade_analysis["summary"],
        "report_paths": {**report_paths, "metrics_report_path": metrics_report_path},
    }
    run_report_path = write_json(report_dir / "run_report.json", run_report)

    economic_leaderboard = update_economic_leaderboard(
        context["phase8"].report_paths.report_dir,
        output_path=context["phase8"].report_paths.economic_leaderboard_path,
    )

    return {
        "status": "ok",
        "report_dir": str(report_dir),
        "run_report_path": run_report_path,
        "metrics_report_path": metrics_report_path,
        "alerts": alerts,
        "economic_leaderboard": economic_leaderboard,
    }


def _apply_alert_flags(alerts: list[str], flags) -> list[str]:
    filtered: list[str] = []
    for alert in alerts:
        if alert.startswith("session_total_pnl_below_threshold") and not flags.alert_on_negative_pnl:
            continue
        if alert.startswith("win_rate_below_threshold") and not flags.alert_on_low_win_rate:
            continue
        if alert.startswith("max_drawdown_exceeds_threshold") and not flags.alert_on_high_drawdown:
            continue
        if ("data_drift" in alert or "feature" in alert or "mean_shift_sigma_warning" in alert) and not flags.alert_on_feature_drift:
            continue
        if "prediction_drift" in alert and not flags.alert_on_prediction_drift:
            continue
        if "label_drift" in alert and not flags.alert_on_label_drift:
            continue
        if alert.startswith("degenerate_") and not flags.alert_on_degenerate_outputs:
            continue
        filtered.append(alert)
    return filtered


def full_evaluation_run(
    settings: Settings,
    *,
    summary_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
) -> dict[str, Any]:
    return generate_report(settings, summary_path=summary_path, parquet_path=parquet_path)


def _build_report_context(
    settings: Settings,
    *,
    summary_path: str | Path | None,
    parquet_path: str | Path | None,
) -> dict[str, Any]:
    phase7 = load_phase7_config(settings)
    phase8 = load_phase8_config(settings)
    resolved_summary_path, resolved_parquet_path, phase7_summary = resolve_phase7_paths(
        report_dir=phase7.logging.report_dir,
        summary_path=summary_path,
        parquet_path=parquet_path,
    )
    decision_frame = flatten_decision_metadata(load_phase7_frame(resolved_parquet_path))
    decision_ids = {str(value) for value in decision_frame.get("decision_id", pd.Series(dtype=str)).dropna().astype(str)}
    order_ids = {str(value) for value in decision_frame.get("order_id", pd.Series(dtype=str)).dropna().astype(str)}
    journal_root = Path(phase7.logging.journal_dir)

    orders = filter_records(read_jsonl(journal_root / "orders.jsonl"), decision_ids=decision_ids, order_ids=order_ids)
    fills = filter_records(read_jsonl(journal_root / "fills.jsonl"), decision_ids=decision_ids, order_ids=order_ids)
    reports = filter_records(read_jsonl(journal_root / "reports.jsonl"), decision_ids=decision_ids, order_ids=order_ids)

    return {
        "phase7": phase7,
        "phase8": phase8,
        "summary_path": resolved_summary_path,
        "parquet_path": resolved_parquet_path,
        "phase7_summary": phase7_summary,
        "decision_frame": decision_frame,
        "orders": orders,
        "fills": fills,
        "reports": reports,
    }


def _bundle_dir(context: dict[str, Any]) -> Path:
    phase8 = context["phase8"]
    return Path(phase8.report_paths.report_dir) / Path(context["parquet_path"]).stem


def _write_performance_outputs(context: dict[str, Any], performance: dict[str, Any], segments: dict[str, Any]) -> dict[str, str]:
    report_dir = _bundle_dir(context)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(report_dir / "performance_summary.json", performance["summary"])
    alerts_path = write_json(report_dir / "performance_alerts.json", {"alerts": performance["alerts"]})
    closed_trades_path = report_dir / "closed_trades.csv"
    open_trades_path = report_dir / "open_trades.csv"
    equity_curve_path = report_dir / "equity_curve.csv"
    performance["closed_trades"].to_csv(closed_trades_path, index=False)
    performance["open_trades"].to_csv(open_trades_path, index=False)
    performance["equity_curve"].to_csv(equity_curve_path, index=False)

    segment_paths: dict[str, str] = {}
    for name, frame in segments["segment_tables"].items():
        path = report_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        segment_paths[f"{name}_path"] = str(path)

    return {
        "performance_summary_path": summary_path,
        "performance_alerts_path": alerts_path,
        "closed_trades_path": str(closed_trades_path),
        "open_trades_path": str(open_trades_path),
        "equity_curve_path": str(equity_curve_path),
        **segment_paths,
    }


def _write_signal_outputs(context: dict[str, Any], signal: dict[str, Any]) -> dict[str, str]:
    report_dir = _bundle_dir(context)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(report_dir / "signal_quality.json", {"summary": signal["summary"], "alerts": signal["alerts"]})
    table_paths: dict[str, str] = {}
    for name, frame in signal["tables"].items():
        path = report_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        table_paths[f"{name}_path"] = str(path)
    return {"signal_quality_path": summary_path, **table_paths}


def _write_drift_outputs(context: dict[str, Any], drift: dict[str, Any]) -> dict[str, str]:
    report_dir = _bundle_dir(context)
    report_dir.mkdir(parents=True, exist_ok=True)
    drift_path = write_json(report_dir / "drift_report.json", drift)
    detail_paths: dict[str, str] = {}
    for name in ("data_drift", "prediction_drift", "label_drift"):
        detail_frame = pd.DataFrame(drift.get(name, {}).get("details", []))
        path = report_dir / f"{name}_details.csv"
        detail_frame.to_csv(path, index=False)
        detail_paths[f"{name}_details_path"] = str(path)
    return {"drift_report_path": drift_path, **detail_paths}


def _write_trade_outputs(context: dict[str, Any], trade_analysis: dict[str, Any]) -> dict[str, str]:
    report_dir = _bundle_dir(context)
    report_dir.mkdir(parents=True, exist_ok=True)
    trade_path = write_json(report_dir / "trade_analysis.json", trade_analysis["summary"])
    orders_path = report_dir / "trade_orders.csv"
    fills_path = report_dir / "trade_fills.csv"
    reports_path = report_dir / "trade_reports.csv"
    rejections_path = report_dir / "trade_rejections.csv"
    inconsistencies_path = report_dir / "trade_inconsistencies.csv"
    trade_analysis["orders"].to_csv(orders_path, index=False)
    trade_analysis["fills"].to_csv(fills_path, index=False)
    trade_analysis["reports"].to_csv(reports_path, index=False)
    trade_analysis["rejections"].to_csv(rejections_path, index=False)
    trade_analysis["inconsistencies"].to_csv(inconsistencies_path, index=False)
    return {
        "trade_analysis_path": trade_path,
        "trade_orders_path": str(orders_path),
        "trade_fills_path": str(fills_path),
        "trade_reports_path": str(reports_path),
        "trade_rejections_path": str(rejections_path),
        "trade_inconsistencies_path": str(inconsistencies_path),
    }
