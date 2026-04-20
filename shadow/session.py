from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pandas as pd

from config import Settings
from config.phase6 import load_active_model_selection, load_phase6_config, resolve_phase5_artifact
from config.phase12_14 import load_phase12_14_config, resolve_runtime_profile
from data.feature_loader import load_feature_data
from evaluation.io import write_json
from labels.dataset_builder import load_labeled_data
from models.inference import OperationalInferenceEngine
from risk.risk_engine import OperationalRiskEngine, OperationalRiskState
from storage.decision_logs import DecisionLogStore
from strategy.decision_engine import DecisionEngine
from shadow.comparison import compare_shadow_to_paper_and_market


def run_shadow_session(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    latest_per_symbol: int | None = None,
    decision_log_path: str | None = None,
) -> dict[str, Any]:
    phase12_14 = load_phase12_14_config(settings)
    profile = resolve_runtime_profile(settings, profile_name="shadow")
    if not profile.shadow_mode_enabled:
        raise ValueError("run-shadow-session requires the shadow runtime profile to have shadow_mode_enabled=true.")
    if profile.paper_order_submission_enabled:
        raise ValueError("Shadow mode cannot enable paper_order_submission_enabled.")

    selection = load_active_model_selection(settings)
    phase6 = load_phase6_config(settings)
    artifact = resolve_phase5_artifact(settings, run_id=selection.run_id, artifact_dir=selection.artifact_dir)
    inference = OperationalInferenceEngine(settings, artifact)

    feature_frame = load_feature_data(
        settings,
        feature_set_name=selection.feature_set_name,
        symbols=symbols,
        feature_root=feature_root,
    )
    rows_per_symbol = int(latest_per_symbol or 1)
    latest_rows = (
        feature_frame.sort_values(["symbol", "timestamp"])
        .groupby("symbol", group_keys=False, sort=False)
        .tail(max(rows_per_symbol, 1))
        .reset_index(drop=True)
    )
    latest_rows = _attach_outcomes(
        settings,
        latest_rows,
        feature_set_name=selection.feature_set_name,
        target_mode=selection.target_mode,
        symbols=symbols,
    )

    shadow_dir = Path(phase12_14.deployment_paths.shadow_dir)
    report_dir = Path(phase12_14.deployment_paths.shadow_report_dir)
    shadow_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    session_id = f"shadow_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

    decision_engine = DecisionEngine(phase6.decision, phase6.sizing)
    risk_engine = OperationalRiskEngine(phase6.risk)
    decision_store = DecisionLogStore(decision_log_path or phase6.logging.decision_log_path, enabled=phase6.logging.enabled)
    state = OperationalRiskState()
    intents: list[dict[str, Any]] = []

    ordered_frame = latest_rows.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    for index, row in ordered_frame.iterrows():
        prediction = inference.predict_row(row).to_dict()
        decision = decision_engine.decide(row, prediction).to_dict()
        evaluation = risk_engine.evaluate(decision, row, prediction, state)
        final_decision = risk_engine.apply(decision, evaluation)
        state = risk_engine.record_post_decision(
            state,
            final_decision,
            realized_net_return_bps=_coerce_float(row.get("future_net_return_bps")),
        )
        decision_store.append({"shadow_only": True, "session_id": session_id, **final_decision})
        intent = {
            "intent_id": f"shint_{uuid4().hex[:12]}",
            "decision_id": f"sdec_{session_id}_{index:04d}",
            "timestamp": final_decision.get("timestamp"),
            "session_id": session_id,
            "model_name": final_decision.get("model_name"),
            "run_id": final_decision.get("run_id"),
            "feature_set": final_decision.get("feature_set_name"),
            "target_mode": final_decision.get("target_mode"),
            "symbol": final_decision.get("symbol"),
            "action": final_decision.get("action"),
            "confidence": final_decision.get("confidence"),
            "score": final_decision.get("score"),
            "expected_return_bps": final_decision.get("expected_return_bps"),
            "expected_cost_bps": final_decision.get("expected_cost_bps"),
            "net_edge_bps": final_decision.get("net_edge_bps"),
            "suggested_size": final_decision.get("size_suggestion"),
            "reasons": final_decision.get("reasons", []),
            "blocked_by_risk": final_decision.get("blocked_by_risk"),
            "shadow_only_flag": True,
            "future_net_return_bps": _coerce_float(row.get("future_net_return_bps")),
            "decision_metadata": final_decision.get("metadata", {}),
        }
        intents.append(intent)

    intents_frame = pd.DataFrame(intents)
    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = report_dir / f"shadow_session_{timestamp_token}.csv"
    parquet_path = report_dir / f"shadow_session_{timestamp_token}.parquet"
    summary_path = report_dir / f"shadow_session_summary_{timestamp_token}.json"
    if intents_frame.empty:
        intents_frame = pd.DataFrame(columns=["timestamp", "symbol", "action", "confidence"])
    serialized = _serialize_nested_columns(intents_frame)
    serialized.to_csv(csv_path, index=False)
    serialized.to_parquet(parquet_path, index=False)
    _append_jsonl(Path(phase12_14.deployment_paths.shadow_intents_path), intents)

    comparison = compare_shadow_to_paper_and_market(
        intents_frame,
        shadow_report_dir=report_dir,
        phase7_report_dir=Path(settings.paths.report_dir) / "phase7",
        session_id=session_id,
    )
    summary = {
        "status": "ok",
        "run_type": "shadow_session",
        "session_id": session_id,
        "runtime_profile": profile.name,
        "shadow_mode_enabled": True,
        "paper_order_submission_enabled": False,
        "row_count": int(len(ordered_frame)),
        "intent_count": int(len(intents_frame)),
        "action_counts": {str(key): int(value) for key, value in intents_frame.get("action", pd.Series(dtype=object)).value_counts(dropna=False).to_dict().items()},
        "blocked_by_risk_count": int(pd.to_numeric(intents_frame.get("blocked_by_risk"), errors="coerce").fillna(0).astype(bool).sum()) if "blocked_by_risk" in intents_frame.columns else 0,
        "active_model": inference.describe(),
        "state_final": state.to_dict(),
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
        "summary_path": str(summary_path),
        "comparison": comparison,
    }
    write_json(summary_path, summary)
    return summary


def _attach_outcomes(
    settings: Settings,
    feature_frame: pd.DataFrame,
    *,
    feature_set_name: str,
    target_mode: str,
    symbols: Sequence[str] | None,
) -> pd.DataFrame:
    try:
        labeled_frame = load_labeled_data(
            settings,
            feature_set_name=feature_set_name,
            target_mode=target_mode,
            symbols=symbols,
        )
    except FileNotFoundError:
        return feature_frame
    merge_columns = [column for column in ("timestamp", "symbol", "session_date") if column in feature_frame.columns and column in labeled_frame.columns]
    attach_columns = [column for column in ("future_net_return_bps", "future_return_bps", f"target_{target_mode}") if column in labeled_frame.columns]
    if not merge_columns or not attach_columns:
        return feature_frame
    return feature_frame.merge(labeled_frame.loc[:, [*merge_columns, *attach_columns]], on=merge_columns, how="left")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def _serialize_nested_columns(frame: pd.DataFrame) -> pd.DataFrame:
    serialized = frame.copy()
    for column in serialized.columns:
        if serialized[column].dtype != "object":
            continue
        if serialized[column].map(lambda value: isinstance(value, (dict, list))).any():
            serialized[column] = serialized[column].map(lambda value: json.dumps(value, sort_keys=True, default=str))
    return serialized


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
