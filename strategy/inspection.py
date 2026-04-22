from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import Settings
from config.phase6 import load_active_model_selection, load_phase6_config, resolve_phase5_artifact
from data.feature_loader import load_feature_data
from models.inference import OperationalInferenceEngine
from strategy.decision_engine import DecisionEngine
from strategy.regime_detector import RegimeDetector


def inspect_regimes(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    limit: int | None = 50,
) -> dict[str, Any]:
    selection = load_active_model_selection(settings)
    phase6 = load_phase6_config(settings)
    frame = _load_inspection_frame(settings, selection.feature_set_name, symbols=symbols, feature_root=feature_root, limit=limit)
    detector = RegimeDetector(phase6.strategy.regime_thresholds)
    rows = []
    for _, row in frame.iterrows():
        regime = detector.detect(row)
        rows.append(
            {
                "timestamp": row.get("timestamp"),
                "symbol": row.get("symbol"),
                **regime.to_dict(),
            }
        )
    return _persist_inspection(settings, "regimes", rows)


def inspect_alpha_routing(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    limit: int | None = 50,
) -> dict[str, Any]:
    selection = load_active_model_selection(settings)
    phase6 = load_phase6_config(settings)
    artifact = resolve_phase5_artifact(settings, run_id=selection.run_id, artifact_dir=selection.artifact_dir)
    inference = OperationalInferenceEngine(settings, artifact)
    decision_engine = DecisionEngine(phase6.decision, phase6.sizing, phase6.strategy)
    frame = _load_inspection_frame(settings, selection.feature_set_name, symbols=symbols, feature_root=feature_root, limit=limit)
    rows = []
    for _, row in frame.iterrows():
        prediction = inference.predict_row(row).to_dict()
        decision = decision_engine.decide(row, prediction).to_dict()
        rows.append(
            {
                "timestamp": decision.get("timestamp"),
                "symbol": decision.get("symbol"),
                "action": decision.get("action"),
                "selected_alpha": decision.get("selected_alpha"),
                "regime": decision.get("regime"),
                "confidence": decision.get("confidence"),
                "expected_return_bps": decision.get("expected_return_bps"),
                "conservative_return_bps": decision.get("conservative_return_bps"),
                "expected_cost_bps": decision.get("expected_cost_bps"),
                "net_edge_bps": decision.get("net_edge_bps"),
                "reasons": decision.get("reasons", []),
            }
        )
    return _persist_inspection(settings, "alpha_routing", rows)


def compare_alpha_families(
    settings: Settings,
    *,
    symbols: Sequence[str] | None = None,
    feature_root: str | Path | None = None,
    limit: int | None = 200,
) -> dict[str, Any]:
    inspection = inspect_alpha_routing(settings, symbols=symbols, feature_root=feature_root, limit=limit)
    frame = pd.DataFrame(inspection.get("rows", []))
    if frame.empty:
        return {**inspection, "alpha_summary": [], "regime_summary": []}
    alpha_summary = _summary_table(frame, "selected_alpha")
    regime_summary = _summary_table(frame, "regime")
    payload = {
        **inspection,
        "alpha_summary": alpha_summary,
        "regime_summary": regime_summary,
    }
    summary_path = Path(inspection["report_dir"]) / f"alpha_family_summary_{inspection['timestamp_token']}.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    payload["summary_path"] = str(summary_path)
    return payload


def _load_inspection_frame(
    settings: Settings,
    feature_set_name: str,
    *,
    symbols: Sequence[str] | None,
    feature_root: str | Path | None,
    limit: int | None,
) -> pd.DataFrame:
    frame = load_feature_data(settings, feature_set_name=feature_set_name, symbols=symbols, feature_root=feature_root)
    if limit:
        frame = frame.sort_values(["timestamp", "symbol"]).tail(max(int(limit), 1)).reset_index(drop=True)
    return frame


def _summary_table(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(column, dropna=False, sort=True):
        trade_mask = group["action"].astype(str) != "NO_TRADE"
        rows.append(
            {
                column: str(key),
                "row_count": int(len(group)),
                "trade_count": int(trade_mask.sum()),
                "no_trade_rate": float((~trade_mask).mean()),
                "average_net_edge_bps": _mean(group, "net_edge_bps"),
                "average_conservative_return_bps": _mean(group, "conservative_return_bps"),
            }
        )
    return rows


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _persist_inspection(settings: Settings, report_type: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    report_dir = Path(settings.paths.report_dir) / "strategy"
    report_dir.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"{report_type}_{token}.json"
    csv_path = report_dir / f"{report_type}_{token}.csv"
    payload = {
        "status": "ok",
        "report_type": report_type,
        "timestamp_token": token,
        "row_count": int(len(rows)),
        "rows": rows,
        "report_dir": str(report_dir),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    pd.json_normalize(rows, sep=".").to_csv(csv_path, index=False)
    return payload
