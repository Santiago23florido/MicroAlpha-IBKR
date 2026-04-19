from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from config import Settings
from config.phase6 import load_active_model_selection, load_phase6_config, resolve_phase5_artifact, show_active_model_status
from config.phase7 import load_phase7_config
from data.feature_loader import load_feature_data
from execution import ExecutionJournal, ModelTrace, OrderManager, OrderValidationError, build_execution_backend
from execution.position_manager import PositionManager
from labels.dataset_builder import load_labeled_data
from models.inference import OperationalInferenceEngine
from monitoring.logging import setup_logger
from reporting.report_bundle import generate_report
from risk.risk_engine import OperationalRiskEngine, OperationalRiskState
from storage.decision_logs import DecisionLogStore
from strategy.decision_engine import DecisionEngine


def show_execution_backend(settings: Settings) -> dict[str, Any]:
    phase7 = load_phase7_config(settings)
    backend = build_execution_backend(phase7, settings=settings)
    return {
        "status": "ok",
        "active_execution_backend": phase7.execution.active_execution_backend,
        "paper_mode": phase7.execution.paper_mode,
        "backend": backend.describe(),
        "execution_config": phase7.execution.to_dict(),
        "logging_config": phase7.logging.to_dict(),
        "session_config": phase7.session.to_dict(),
        "ibkr_paper_config": phase7.ibkr_paper.to_dict(),
    }


def execution_status(settings: Settings, *, limit: int | None = None) -> dict[str, Any]:
    phase7 = load_phase7_config(settings)
    backend = build_execution_backend(phase7, settings=settings)
    journal = ExecutionJournal(phase7.logging)
    state = journal.load_state()
    recent_limit = int(limit or phase7.session.recent_event_limit)
    orders = state.get("orders", []) or []
    terminal_statuses = {"REJECTED", "FILLED", "CANCELLED", "EXPIRED", "FAILED"}
    open_orders = [order for order in orders if str(order.get("status")) not in terminal_statuses]
    broker_health: dict[str, Any] | None = None
    broker_open_orders: list[dict[str, Any]] = []
    broker_positions: list[dict[str, Any]] = []
    broker_error: str | None = None
    if backend.name == "ibkr_paper":
        try:
            broker_health = backend.healthcheck()
            broker_open_orders = backend.get_open_orders()
            broker_positions = backend.get_positions()
        except Exception as exc:  # pragma: no cover - depends on broker availability
            broker_error = str(exc)
        finally:
            backend.disconnect()
    return {
        "status": "ok",
        "active_model": show_active_model_status(settings).get("active_model", {}),
        "active_execution_backend": phase7.execution.active_execution_backend,
        "backend": backend.describe(),
        "paper_mode": phase7.execution.paper_mode,
        "portfolio": (((state.get("portfolio_state") or {}).get("portfolio")) or {}),
        "risk_state": state.get("risk_state", {}),
        "open_orders": open_orders,
        "recent_orders": journal.recent_orders(recent_limit),
        "recent_fills": journal.recent_fills(recent_limit),
        "recent_reports": journal.recent_reports(recent_limit),
        "recent_backend_events": journal.recent_backend_events(recent_limit),
        "recent_reconciliation": journal.recent_reconciliation(recent_limit),
        "broker_health": broker_health,
        "broker_open_orders": broker_open_orders,
        "broker_positions": broker_positions,
        "broker_error": broker_error,
        "paths": {
            "journal_dir": phase7.logging.journal_dir,
            "state_path": phase7.logging.state_path,
            "report_dir": phase7.logging.report_dir,
        },
        "execution_config": phase7.execution.to_dict(),
        "ibkr_paper_config": phase7.ibkr_paper.to_dict(),
        "phase6_risk_config": load_phase6_config(settings).risk.to_dict(),
    }


def run_paper_sim_offline(
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
    logger = setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.phase7.paper_sim_offline")
    selection = load_active_model_selection(settings)
    phase6 = load_phase6_config(settings)
    phase7 = load_phase7_config(settings)
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
    summary = _execute_phase7_run(
        settings,
        feature_frame=enriched_frame,
        inference=inference,
        run_label="paper_sim_offline",
        logger=logger,
        decision_log_path=decision_log_path or phase6.logging.decision_log_path,
        reset_state=phase7.session.reset_state_on_offline_run,
        backend_name_override="mock",
    )
    summary["labels_attached"] = label_status
    return summary


def run_paper_session(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    latest_per_symbol: int | None = None,
    decision_log_path: str | None = None,
) -> dict[str, Any]:
    logger = setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.phase7.paper_session")
    selection = load_active_model_selection(settings)
    phase6 = load_phase6_config(settings)
    phase7 = load_phase7_config(settings)
    artifact = resolve_phase5_artifact(settings, run_id=selection.run_id, artifact_dir=selection.artifact_dir)
    inference = OperationalInferenceEngine(settings, artifact)
    feature_frame = load_feature_data(
        settings,
        feature_set_name=selection.feature_set_name,
        symbols=symbols,
        feature_root=feature_root,
    )
    rows_per_symbol = int(latest_per_symbol or phase7.session.latest_per_symbol)
    latest_rows = (
        feature_frame.sort_values(["symbol", "timestamp"])
        .groupby("symbol", group_keys=False, sort=False)
        .tail(max(rows_per_symbol, 1))
        .reset_index(drop=True)
    )
    summary = _execute_phase7_run(
        settings,
        feature_frame=latest_rows,
        inference=inference,
        run_label="paper_session",
        logger=logger,
        decision_log_path=decision_log_path or phase6.logging.decision_log_path,
        reset_state=phase7.session.reset_state_on_session_run,
        backend_name_override="mock",
    )
    summary["message"] = "Paper session completed using the mock execution backend."
    return summary


def broker_healthcheck(settings: Settings) -> dict[str, Any]:
    logger = setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.phase9.broker_healthcheck")
    phase7 = load_phase7_config(settings)
    backend = build_execution_backend(
        phase7,
        settings=settings,
        logger=logger,
        backend_name="ibkr_paper",
    )
    try:
        payload = backend.healthcheck()
        payload["active_execution_backend"] = phase7.execution.active_execution_backend
        payload["active_model"] = show_active_model_status(settings).get("active_model", {})
        payload["symbol_universe"] = list(settings.supported_symbols)
        return payload
    finally:
        backend.disconnect()


def run_paper_session_real(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    latest_per_symbol: int | None = None,
    decision_log_path: str | None = None,
) -> dict[str, Any]:
    logger = setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.phase9.paper_session_real")
    selection = load_active_model_selection(settings)
    phase6 = load_phase6_config(settings)
    phase7 = load_phase7_config(settings)
    if _normalize_backend_name(phase7.execution.active_execution_backend) != "ibkr_paper":
        raise ValueError(
            "run-paper-session-real requires ACTIVE_EXECUTION_BACKEND=ibkr_paper. "
            "Use show-execution-backend to confirm the current backend."
        )
    artifact = resolve_phase5_artifact(settings, run_id=selection.run_id, artifact_dir=selection.artifact_dir)
    inference = OperationalInferenceEngine(settings, artifact)
    feature_frame = load_feature_data(
        settings,
        feature_set_name=selection.feature_set_name,
        symbols=symbols,
        feature_root=feature_root,
    )
    rows_per_symbol = int(latest_per_symbol or phase7.session.latest_per_symbol)
    latest_rows = (
        feature_frame.sort_values(["symbol", "timestamp"])
        .groupby("symbol", group_keys=False, sort=False)
        .tail(max(rows_per_symbol, 1))
        .reset_index(drop=True)
    )
    summary = _execute_phase7_run(
        settings,
        feature_frame=latest_rows,
        inference=inference,
        run_label="paper_session_real",
        logger=logger,
        decision_log_path=decision_log_path or phase6.logging.decision_log_path,
        reset_state=False,
        backend_name_override="ibkr_paper",
    )
    summary["message"] = "Paper session completed using the IBKR Paper execution backend."
    return summary


def _execute_phase7_run(
    settings: Settings,
    *,
    feature_frame: pd.DataFrame,
    inference: OperationalInferenceEngine,
    run_label: str,
    logger,
    decision_log_path: str,
    reset_state: bool,
    backend_name_override: str | None = None,
) -> dict[str, Any]:
    if feature_frame.empty:
        raise ValueError("The input feature frame is empty. Build features before running Phase 7.")

    phase6 = load_phase6_config(settings)
    phase7 = load_phase7_config(settings)
    selection = load_active_model_selection(settings)
    journal = ExecutionJournal(phase7.logging)
    previous_state = {} if reset_state else journal.load_state()
    position_manager = PositionManager.from_snapshot(
        (previous_state.get("portfolio_state") or {}),
        initial_cash=phase7.session.initial_cash,
    )
    backend = build_execution_backend(
        phase7,
        settings=settings,
        logger=logger,
        backend_name=backend_name_override,
    )
    broker_health = None
    broker_positions_before: list[dict[str, Any]] = []
    broker_open_orders_before: list[dict[str, Any]] = []
    if backend.name == "ibkr_paper":
        broker_health = backend.healthcheck()
        broker_positions_before = backend.get_positions()
        broker_open_orders_before = backend.get_open_orders()
    order_manager = OrderManager(
        phase7,
        backend=backend,
        journal=journal,
        position_manager=position_manager,
    )
    order_manager.restore_orders(previous_state.get("orders", []))
    decision_engine = DecisionEngine(phase6.decision, phase6.sizing)
    risk_engine = OperationalRiskEngine(phase6.risk)
    risk_state = _restore_risk_state(previous_state.get("risk_state"))
    decision_store = DecisionLogStore(decision_log_path, enabled=phase6.logging.enabled)
    model_trace = _model_trace_from_selection(selection)
    records: list[dict[str, Any]] = []
    orders_attempted = 0
    orders_accepted = 0

    try:
        ordered_frame = feature_frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        for index, row in ordered_frame.iterrows():
            row_data = row.to_dict()
            symbol = str(row_data.get("symbol", "")).upper()
            mark_price = _resolve_mark_price(row_data)
            portfolio_before = (
                position_manager.update_market_prices({symbol: mark_price})
                if symbol and mark_price is not None
                else position_manager.snapshot()
            )
            prediction = inference.predict_row(row).to_dict()
            decision = decision_engine.decide(row, prediction).to_dict()
            evaluation = risk_engine.evaluate(decision, row, prediction, risk_state)
            final_decision = risk_engine.apply(decision, evaluation)
            final_decision["decision_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
            decision_metadata = dict(final_decision.get("metadata", {}) or {})
            decision_metadata["decision_generated_at_utc"] = final_decision["decision_generated_at_utc"]
            final_decision["metadata"] = decision_metadata
            decision_id = _build_decision_id(final_decision, model_trace, index)

            execution_result = None
            execution_status = "SKIPPED"
            execution_error: str | None = None

            if final_decision.get("action") != "NO_TRADE" and orders_attempted >= phase7.session.max_orders_per_run:
                execution_status = "SKIPPED_MAX_ORDERS_PER_RUN"
                execution_error = "max_orders_per_run_reached"
            elif final_decision.get("action") != "NO_TRADE":
                orders_attempted += 1
                try:
                    execution_result = order_manager.process_decision(
                        final_decision,
                        model_trace=model_trace,
                        decision_id=decision_id,
                        market_data=_row_to_market_context(row_data),
                    )
                    execution_status = execution_result.order.status.value
                    if execution_result.accepted:
                        orders_accepted += 1
                        risk_state = risk_engine.record_post_decision(
                            risk_state,
                            final_decision,
                            realized_net_return_bps=execution_result.realized_return_bps,
                        )
                    elif execution_result.errors:
                        execution_error = execution_result.errors[0]
                except OrderValidationError as exc:
                    execution_status = "FAILED_VALIDATION"
                    execution_error = str(exc)

            portfolio_after = (
                position_manager.update_market_prices({symbol: mark_price})
                if symbol and mark_price is not None
                else position_manager.snapshot()
            )
            latency_payload = (
                {}
                if execution_result is None
                else dict((execution_result.order.metadata or {}).get("latency_ms", {}) or {})
            )
            comparison_payload = (
                {}
                if execution_result is None
                else dict((execution_result.order.metadata or {}).get("decision_vs_execution", {}) or {})
            )
            record = {
                "timestamp": final_decision.get("timestamp"),
                "symbol": final_decision.get("symbol"),
                "decision_id": decision_id,
                "decision_generated_at_utc": final_decision.get("decision_generated_at_utc"),
                "run_id": model_trace.run_id,
                "model_name": final_decision.get("model_name"),
                "model_type": final_decision.get("model_type"),
                "feature_set_name": final_decision.get("feature_set_name"),
                "target_mode": final_decision.get("target_mode"),
                "artifact_dir": model_trace.artifact_dir,
                "score": final_decision.get("score"),
                "probability": final_decision.get("probability"),
                "expected_return_bps": final_decision.get("expected_return_bps"),
                "expected_cost_bps": final_decision.get("expected_cost_bps"),
                "net_edge_bps": final_decision.get("net_edge_bps"),
                "action": final_decision.get("action"),
                "size_suggestion": final_decision.get("size_suggestion"),
                "blocked_by_risk": final_decision.get("blocked_by_risk"),
                "reasons": final_decision.get("reasons", []),
                "risk_checks": final_decision.get("risk_checks", {}),
                "risk_failures": final_decision.get("risk_failures", []),
                "predicted_quantiles": final_decision.get("predicted_quantiles", {}),
                "future_net_return_bps": _coerce_optional_float(row_data.get("future_net_return_bps")),
                "spread_bps_observed": _coerce_optional_float(row_data.get("spread_bps")),
                "estimated_cost_bps_observed": _coerce_optional_float(row_data.get("estimated_cost_bps")),
                "relative_volume_observed": _coerce_optional_float(row_data.get("relative_volume")),
                "volume_observed": _coerce_optional_float(row_data.get("volume")),
                "day_of_week_observed": row_data.get("day_of_week"),
                "minute_of_day_observed": _coerce_optional_float(row_data.get("minute_of_day")),
                "backend_name": backend.name,
                "broker_status": None if execution_result is None else (execution_result.order.metadata or {}).get("broker_status"),
                "execution_status": execution_status,
                "execution_error": execution_error,
                "order_id": None if execution_result is None else execution_result.order.order_id,
                "broker_order_id": None if execution_result is None else execution_result.order.broker_order_id,
                "broker_perm_id": None if execution_result is None else execution_result.order.broker_perm_id,
                "order_action": None if execution_result is None else execution_result.order.action.value,
                "order_type": None if execution_result is None else execution_result.order.order_type.value,
                "order_quantity": None if execution_result is None else execution_result.order.quantity,
                "filled_quantity": None if execution_result is None else execution_result.order.filled_quantity,
                "average_fill_price": None if execution_result is None else execution_result.order.average_fill_price,
                "fill_count": 0 if execution_result is None else len(execution_result.fills),
                "first_fill_at_utc": latency_payload.get("first_fill_at_utc"),
                "final_fill_at_utc": latency_payload.get("final_fill_at_utc"),
                "decision_to_submit_ms": latency_payload.get("decision_to_submit_ms"),
                "submit_to_ack_ms": latency_payload.get("submit_to_ack_ms"),
                "submit_to_first_fill_ms": latency_payload.get("submit_to_first_fill_ms"),
                "submit_to_final_fill_ms": latency_payload.get("submit_to_final_fill_ms"),
                "fill_ratio": (
                    None
                    if execution_result is None
                    else (execution_result.order.filled_quantity / execution_result.order.quantity if execution_result.order.quantity else 0.0)
                ),
                "execution_discrepancy_flags": comparison_payload.get("discrepancy_flags", []),
                "average_execution_slippage_bps": comparison_payload.get("average_slippage_bps"),
                "realized_pnl_delta": None if execution_result is None else execution_result.realized_pnl_delta,
                "portfolio_before": portfolio_before.to_dict(),
                "portfolio_after": portfolio_after.to_dict(),
                "decision_metadata": final_decision.get("metadata", {}),
            }
            records.append(record)
            decision_store.append(record)
            journal.save_state(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "active_model": selection.to_dict(),
                    "backend": backend.describe(),
                    "broker_health": broker_health,
                    "broker_positions_before": broker_positions_before,
                    "broker_open_orders_before": broker_open_orders_before,
                    "risk_state": risk_state.to_dict(),
                    "orders": order_manager.snapshot_orders(),
                    "portfolio_state": position_manager.to_state_payload(),
                    "last_run": {
                        "run_label": run_label,
                        "decision_id": decision_id,
                        "symbol": symbol,
                        "execution_status": execution_status,
                    },
                }
            )
    finally:
        if backend.name == "ibkr_paper":
            backend.disconnect()

    return _write_phase7_reports(
        settings=settings,
        phase7_report_dir=Path(phase7.logging.report_dir),
        run_label=run_label,
        records=records,
        active_model=inference.describe(),
        decision_log_path=decision_log_path,
        backend_name=backend.name,
        paper_mode=phase7.execution.paper_mode,
        orders_attempted=orders_attempted,
        orders_accepted=orders_accepted,
        portfolio=position_manager.snapshot(),
        risk_state=risk_state,
        broker_health=broker_health,
        broker_positions_before=broker_positions_before,
        broker_open_orders_before=broker_open_orders_before,
        logger=logger,
    )


def _write_phase7_reports(
    *,
    settings: Settings,
    phase7_report_dir: Path,
    run_label: str,
    records: list[dict[str, Any]],
    active_model: dict[str, Any],
    decision_log_path: str,
    backend_name: str,
    paper_mode: bool,
    orders_attempted: int,
    orders_accepted: int,
    portfolio,
    risk_state: OperationalRiskState,
    broker_health: Mapping[str, Any] | None,
    broker_positions_before: Sequence[Mapping[str, Any]] | None,
    broker_open_orders_before: Sequence[Mapping[str, Any]] | None,
    logger,
) -> dict[str, Any]:
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    phase7_report_dir.mkdir(parents=True, exist_ok=True)
    details_frame = pd.DataFrame(records)
    parquet_path = phase7_report_dir / f"{run_label}_{timestamp_token}.parquet"
    csv_path = phase7_report_dir / f"{run_label}_{timestamp_token}.csv"
    summary_path = phase7_report_dir / f"{run_label}_summary_{timestamp_token}.json"
    serialized = _serialize_nested_columns(details_frame)
    serialized.to_parquet(parquet_path, index=False)
    serialized.to_csv(csv_path, index=False)

    action_counts = details_frame["action"].value_counts(dropna=False).to_dict() if not details_frame.empty else {}
    execution_counts = details_frame["execution_status"].value_counts(dropna=False).to_dict() if not details_frame.empty else {}
    blocked_count = int(details_frame["blocked_by_risk"].fillna(False).astype(bool).sum()) if "blocked_by_risk" in details_frame.columns else 0
    summary = {
        "status": "ok",
        "run_type": run_label,
        "row_count": int(len(details_frame)),
        "orders_attempted": int(orders_attempted),
        "orders_accepted": int(orders_accepted),
        "action_counts": {str(key): int(value) for key, value in action_counts.items()},
        "execution_status_counts": {str(key): int(value) for key, value in execution_counts.items()},
        "blocked_by_risk_count": blocked_count,
        "active_model": active_model,
        "backend_name": backend_name,
        "paper_mode": paper_mode,
        "decision_log_path": decision_log_path,
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "portfolio_final": portfolio.to_dict(),
        "risk_state_final": risk_state.to_dict(),
        "broker_health": dict(broker_health or {}),
        "broker_positions_before": list(broker_positions_before or []),
        "broker_open_orders_before": list(broker_open_orders_before or []),
        "mean_expected_return_bps": _frame_mean(details_frame, "expected_return_bps"),
        "mean_realized_pnl_delta": _frame_mean(details_frame, "realized_pnl_delta"),
        "mean_future_net_return_bps": _frame_mean(details_frame, "future_net_return_bps"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    try:
        phase8_report = generate_report(
            settings,
            summary_path=summary_path,
            parquet_path=parquet_path,
            auto_generated=True,
        )
        summary["phase8_report"] = phase8_report
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        logger.warning("Phase8 auto report generation failed for %s: %s", run_label, exc)
        summary["phase8_report_error"] = str(exc)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info(
        "Phase7 %s complete: rows=%s orders_attempted=%s accepted=%s summary=%s",
        run_label,
        len(details_frame),
        orders_attempted,
        orders_accepted,
        summary_path,
    )
    return summary


def _model_trace_from_selection(selection) -> ModelTrace:
    return ModelTrace(
        model_name=selection.model_name,
        model_type=selection.model_type,
        run_id=selection.run_id,
        feature_set_name=selection.feature_set_name,
        target_mode=selection.target_mode,
        artifact_dir=selection.artifact_dir,
        selection_reason=selection.selection_reason,
        source_leaderboard=selection.source_leaderboard,
        updated_at_utc=selection.updated_at_utc,
    )


def _restore_risk_state(payload: Mapping[str, Any] | None) -> OperationalRiskState:
    if not payload:
        return OperationalRiskState()
    normalized = dict(payload)
    return OperationalRiskState(
        session_date=normalized.get("session_date"),
        trades_in_session=int(normalized.get("trades_in_session", 0) or 0),
        daily_realized_pnl_bps=float(normalized.get("daily_realized_pnl_bps", 0.0) or 0.0),
        symbol_realized_pnl_bps={
            str(key): float(value)
            for key, value in dict(normalized.get("symbol_realized_pnl_bps", {}) or {}).items()
        },
        last_loss_timestamp_by_symbol={
            str(key): str(value)
            for key, value in dict(normalized.get("last_loss_timestamp_by_symbol", {}) or {}).items()
        },
        kill_switch_reason=normalized.get("kill_switch_reason"),
    )


def _build_decision_id(decision: Mapping[str, Any], model_trace: ModelTrace, index: int) -> str:
    symbol = str(decision.get("symbol") or "NA").upper()
    timestamp = str(decision.get("timestamp") or f"row_{index}")
    safe_timestamp = timestamp.replace(":", "").replace("-", "").replace("+", "").replace("T", "_")
    run_suffix = model_trace.run_id.split("_")[-1]
    return f"dec_{safe_timestamp}_{symbol}_{index:04d}_{run_suffix}"


def _row_to_market_context(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": row.get("symbol"),
        "timestamp": row.get("timestamp"),
        "bid": _coerce_optional_float(row.get("bid")),
        "ask": _coerce_optional_float(row.get("ask")),
        "last": _coerce_optional_float(row.get("last")),
        "close": _coerce_optional_float(row.get("close")),
        "open": _coerce_optional_float(row.get("open")),
        "price_proxy": _coerce_optional_float(row.get("price_proxy")),
        "mid_price": _coerce_optional_float(row.get("mid_price")),
        "spread_bps": _coerce_optional_float(row.get("spread_bps")),
    }


def _resolve_mark_price(row: Mapping[str, Any]) -> float | None:
    for key in ("last", "price_proxy", "mid_price", "close", "open"):
        value = _coerce_optional_float(row.get(key))
        if value is not None and value > 0:
            return value
    return None


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


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _normalize_backend_name(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_")


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
