from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import Settings
from config.phase6 import (
    load_active_model_selection,
    load_phase6_config,
    required_phase5_artifact_files,
    resolve_phase5_artifact,
    show_active_model_status,
)
from data.feature_loader import load_feature_data
from labels.dataset_builder import load_labeled_data
from models.inference import OperationalInferenceEngine
from monitoring.logging import setup_logger
from risk.risk_engine import OperationalRiskEngine, OperationalRiskState
from storage.decision_logs import DecisionLogStore
from strategy.decision_engine import DecisionEngine


def show_active_model(settings: Settings) -> dict[str, Any]:
    return show_active_model_status(settings)


def risk_check(settings: Settings) -> dict[str, Any]:
    selection = load_active_model_selection(settings)
    phase6_config = load_phase6_config(settings)
    artifact = resolve_phase5_artifact(settings, run_id=selection.run_id, artifact_dir=selection.artifact_dir)
    required_files = required_phase5_artifact_files(artifact)
    errors: list[str] = []
    inference_status: dict[str, Any]
    try:
        engine = OperationalInferenceEngine(settings, artifact)
        inference_status = {
            "loadable": True,
            "description": engine.describe(),
        }
    except Exception as exc:
        inference_status = {
            "loadable": False,
            "error": str(exc),
        }
        errors.append(str(exc))
    return {
        "status": "ok" if not errors else "error",
        "active_model": selection.to_dict(),
        "artifact_status": {
            "required_files": required_files,
            "ready": all(item["exists"] for item in required_files.values()),
        },
        "inference": inference_status,
        "decision_config": phase6_config.decision.to_dict(),
        "risk_config": phase6_config.risk.to_dict(),
        "sizing_config": phase6_config.sizing.to_dict(),
        "logging_config": phase6_config.logging.to_dict(),
        "errors": errors,
    }


def run_decisions_offline(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    feature_root: str | Path | None = None,
    label_root: str | Path | None = None,
    decision_log_path: str | None = None,
) -> dict[str, Any]:
    logger = setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.phase6.offline_runner")
    selection = load_active_model_selection(settings)
    phase6_config = load_phase6_config(settings)
    artifact = resolve_phase5_artifact(settings, run_id=selection.run_id, artifact_dir=selection.artifact_dir)
    inference = OperationalInferenceEngine(settings, artifact)
    feature_frame = load_feature_data(
        settings,
        feature_set_name=selection.feature_set_name,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        feature_root=feature_root,
    )
    if limit:
        feature_frame = feature_frame.sort_values(["timestamp", "symbol"]).tail(limit).reset_index(drop=True)
    enriched_frame, label_status = _attach_realized_outcomes(
        settings,
        feature_frame,
        feature_set_name=selection.feature_set_name,
        target_mode=selection.target_mode,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        label_root=label_root,
    )
    result = _execute_operational_run(
        settings,
        feature_frame=enriched_frame,
        phase6_config=phase6_config,
        inference=inference,
        logger=logger,
        decision_log_path=decision_log_path or phase6_config.logging.decision_log_path,
        run_label="offline",
    )
    result["labels_attached"] = label_status
    return result


def run_session(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    latest_per_symbol: int = 1,
    decision_log_path: str | None = None,
    execute_requested: bool = False,
) -> dict[str, Any]:
    logger = setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.phase6.session_runner")
    selection = load_active_model_selection(settings)
    phase6_config = load_phase6_config(settings)
    artifact = resolve_phase5_artifact(settings, run_id=selection.run_id, artifact_dir=selection.artifact_dir)
    inference = OperationalInferenceEngine(settings, artifact)
    feature_frame = load_feature_data(
        settings,
        feature_set_name=selection.feature_set_name,
        symbols=symbols,
        feature_root=feature_root,
    )
    latest_rows = (
        feature_frame.sort_values(["symbol", "timestamp"])
        .groupby("symbol", group_keys=False, sort=False)
        .tail(max(int(latest_per_symbol), 1))
        .reset_index(drop=True)
    )
    result = _execute_operational_run(
        settings,
        feature_frame=latest_rows,
        phase6_config=phase6_config,
        inference=inference,
        logger=logger,
        decision_log_path=decision_log_path or phase6_config.logging.decision_log_path,
        run_label="session",
    )
    result["execution_requested"] = bool(execute_requested)
    result["orders_sent"] = False
    result["message"] = "Session runner completed. No real or paper orders were sent in Phase 6."
    return result


def _execute_operational_run(
    settings: Settings,
    *,
    feature_frame: pd.DataFrame,
    phase6_config,
    inference: OperationalInferenceEngine,
    logger,
    decision_log_path: str,
    run_label: str,
) -> dict[str, Any]:
    if feature_frame.empty:
        raise ValueError("The input feature frame is empty. Build features before running Phase 6 inference.")

    decision_engine = DecisionEngine(phase6_config.decision, phase6_config.sizing, phase6_config.strategy)
    risk_engine = OperationalRiskEngine(phase6_config.risk)
    decision_store = DecisionLogStore(decision_log_path, enabled=phase6_config.logging.enabled)
    state = OperationalRiskState()
    records: list[dict[str, Any]] = []

    ordered_frame = feature_frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    for _, row in ordered_frame.iterrows():
        prediction = inference.predict_row(row).to_dict()
        decision = decision_engine.decide(row, prediction).to_dict()
        evaluation = risk_engine.evaluate(decision, row, prediction, state)
        final_decision = risk_engine.apply(decision, evaluation)
        realized_net_return_bps = _coerce_optional_float(row.get("future_net_return_bps"))
        state = risk_engine.record_post_decision(
            state,
            final_decision,
            realized_net_return_bps=realized_net_return_bps,
        )
        record = {
            "timestamp": final_decision.get("timestamp"),
            "symbol": final_decision.get("symbol"),
            "model_name": final_decision.get("model_name"),
            "feature_set_name": final_decision.get("feature_set_name"),
            "target_mode": final_decision.get("target_mode"),
            "score": final_decision.get("score"),
            "probability": final_decision.get("probability"),
            "expected_return_bps": final_decision.get("expected_return_bps"),
            "expected_cost_bps": final_decision.get("expected_cost_bps"),
            "net_edge_bps": final_decision.get("net_edge_bps"),
            "conservative_return_bps": final_decision.get("conservative_return_bps"),
            "selected_alpha": final_decision.get("selected_alpha"),
            "regime": final_decision.get("regime"),
            "action": final_decision.get("action"),
            "size_suggestion": final_decision.get("size_suggestion"),
            "blocked_by_risk": final_decision.get("blocked_by_risk"),
            "reasons": final_decision.get("reasons", []),
            "risk_checks": final_decision.get("risk_checks", {}),
            "risk_failures": final_decision.get("risk_failures", []),
            "predicted_quantiles": final_decision.get("predicted_quantiles", {}),
            "realized_net_return_bps": realized_net_return_bps,
            "state_after": state.to_dict(),
            "decision_metadata": final_decision.get("metadata", {}),
        }
        decision_store.append(record)
        records.append(record)

    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(phase6_config.logging.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    details_frame = pd.DataFrame(records)
    parquet_path = report_dir / f"{run_label}_decisions_{timestamp_token}.parquet"
    csv_path = report_dir / f"{run_label}_decisions_{timestamp_token}.csv"
    summary_path = report_dir / f"{run_label}_summary_{timestamp_token}.json"
    parquet_frame = _serialize_nested_columns(details_frame)
    parquet_frame.to_parquet(parquet_path, index=False)
    parquet_frame.to_csv(csv_path, index=False)

    action_counts = details_frame["action"].value_counts(dropna=False).to_dict() if not details_frame.empty else {}
    blocked_count = int(details_frame["blocked_by_risk"].fillna(False).astype(bool).sum()) if "blocked_by_risk" in details_frame.columns else 0
    summary = {
        "status": "ok",
        "run_type": run_label,
        "row_count": int(len(details_frame)),
        "action_counts": {str(key): int(value) for key, value in action_counts.items()},
        "blocked_by_risk_count": blocked_count,
        "active_model": inference.describe(),
        "decision_log_path": decision_log_path,
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "state_final": state.to_dict(),
        "mean_expected_return_bps": _frame_mean(details_frame, "expected_return_bps"),
        "mean_net_edge_bps": _frame_mean(details_frame, "net_edge_bps"),
        "mean_realized_net_return_bps": _frame_mean(details_frame, "realized_net_return_bps"),
        "no_trade_rate": _rate(details_frame, "action", "NO_TRADE"),
        "trade_selectivity": 1.0 - (_rate(details_frame, "action", "NO_TRADE") or 0.0),
        "average_net_edge_bps": _frame_mean(details_frame, "net_edge_bps"),
        "alpha_activation_count": _value_counts(details_frame, "selected_alpha"),
        "alpha_block_rate": _alpha_block_rate(details_frame),
        "regime_distribution": _value_counts(details_frame, "regime"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info(
        "Phase6 %s runner complete: rows=%s actions=%s blocked=%s summary=%s",
        run_label,
        len(details_frame),
        summary["action_counts"],
        blocked_count,
        summary_path,
    )
    return summary


def _attach_realized_outcomes(
    settings: Settings,
    feature_frame: pd.DataFrame,
    *,
    feature_set_name: str,
    target_mode: str,
    symbols: Sequence[str] | None,
    start_date: str | None,
    end_date: str | None,
    label_root: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        labeled_frame = load_labeled_data(
            settings,
            feature_set_name=feature_set_name,
            target_mode=target_mode,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            label_root=label_root,
        )
    except FileNotFoundError:
        return feature_frame, {"attached": False, "reason": "label_store_missing"}

    merge_columns = [column for column in ("timestamp", "symbol", "session_date") if column in labeled_frame.columns and column in feature_frame.columns]
    attach_columns = [
        column
        for column in ("future_return_bps", "future_net_return_bps", "target_cost_adjustment_bps", f"target_{target_mode}")
        if column in labeled_frame.columns
    ]
    if not merge_columns or not attach_columns:
        return feature_frame, {"attached": False, "reason": "no_merge_columns_or_attach_columns"}

    merged = feature_frame.merge(
        labeled_frame.loc[:, [*merge_columns, *attach_columns]],
        on=merge_columns,
        how="left",
        suffixes=("", "_label"),
    )
    return merged, {
        "attached": True,
        "columns": attach_columns,
        "merge_columns": merge_columns,
    }


def _frame_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if series.empty else float(series.mean())


def _rate(frame: pd.DataFrame, column: str, value: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    return float((frame[column].astype(str) == value).mean())


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns or frame.empty:
        return {}
    return {str(key): int(value) for key, value in frame[column].fillna("unknown").astype(str).value_counts().items()}


def _alpha_block_rate(frame: pd.DataFrame) -> float | None:
    if "selected_alpha" not in frame.columns or "action" not in frame.columns or frame.empty:
        return None
    alpha_rows = frame[frame["selected_alpha"].fillna("none").astype(str) != "none"]
    if alpha_rows.empty:
        return None
    return float((alpha_rows["action"].astype(str) == "NO_TRADE").mean())


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _serialize_nested_columns(frame: pd.DataFrame) -> pd.DataFrame:
    serialized = frame.copy()
    for column in serialized.columns:
        if serialized[column].dtype != "object":
            continue
        if serialized[column].map(lambda value: isinstance(value, (dict, list))).any():
            serialized[column] = serialized[column].map(
                lambda value: json.dumps(value, sort_keys=True, default=str)
                if isinstance(value, (dict, list))
                else value
            )
    return serialized
