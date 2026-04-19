from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


JSONISH_COLUMNS = {
    "reasons",
    "risk_checks",
    "risk_failures",
    "predicted_quantiles",
    "portfolio_before",
    "portfolio_after",
    "decision_metadata",
    "execution_discrepancy_flags",
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_phase7_frame(parquet_path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(parquet_path)
    frame = frame.copy()
    for column in set(frame.columns).intersection(JSONISH_COLUMNS):
        frame[column] = frame[column].map(_maybe_json_loads)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame


def list_phase7_summary_paths(report_dir: str | Path, limit: int | None = None) -> list[Path]:
    paths = sorted(Path(report_dir).glob("*_summary_*.json"))
    if limit is not None and limit > 0:
        paths = paths[-limit:]
    return paths


def list_phase7_parquet_paths(report_dir: str | Path, limit: int | None = None) -> list[Path]:
    paths = sorted(Path(report_dir).glob("*.parquet"))
    if limit is not None and limit > 0:
        paths = paths[-limit:]
    return paths


def resolve_phase7_paths(
    *,
    report_dir: str | Path,
    summary_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    if summary_path is None and parquet_path is None:
        summaries = list_phase7_summary_paths(report_dir, limit=1)
        if not summaries:
            raise FileNotFoundError(f"No Phase 7 summary files found under {report_dir}.")
        summary_target = summaries[-1]
        summary_payload = read_json(summary_target)
        parquet_target = Path(summary_payload["parquet_path"])
        return summary_target, parquet_target, summary_payload

    if summary_path is not None:
        summary_target = Path(summary_path)
        summary_payload = read_json(summary_target)
        parquet_target = Path(parquet_path or summary_payload["parquet_path"])
        return summary_target, parquet_target, summary_payload

    parquet_target = Path(parquet_path)  # type: ignore[arg-type]
    summary_name = parquet_target.name.replace(".parquet", "").replace("_summary", "")
    candidate_summaries = sorted(Path(report_dir).glob(f"{summary_name.split('_')[0]}_summary_*.json"))
    if not candidate_summaries:
        raise FileNotFoundError(f"Could not infer Phase 7 summary path for {parquet_target}.")
    summary_target = candidate_summaries[-1]
    return summary_target, parquet_target, read_json(summary_target)


def filter_records(
    rows: Iterable[dict[str, Any]],
    *,
    decision_ids: set[str] | None = None,
    order_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        decision_id = str(row.get("source_decision_id") or row.get("decision_id") or "")
        order_id = str(row.get("order_id") or "")
        if decision_ids and decision_id and decision_id in decision_ids:
            results.append(row)
            continue
        if order_ids and order_id and order_id in order_ids:
            results.append(row)
    return results


def flatten_decision_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    if "decision_metadata" not in flattened.columns:
        return flattened
    metadata = flattened["decision_metadata"].map(lambda value: value if isinstance(value, dict) else {})
    explain = metadata.map(lambda payload: payload.get("explain_features", {}) if isinstance(payload, dict) else {})
    flattened["spread_bps_observed"] = _first_non_null_column(
        flattened,
        "spread_bps_observed",
        explain.map(lambda payload: _coerce_float(payload.get("spread_bps")) if isinstance(payload, dict) else None),
    )
    flattened["estimated_cost_bps_observed"] = _first_non_null_column(
        flattened,
        "estimated_cost_bps_observed",
        explain.map(lambda payload: _coerce_float(payload.get("estimated_cost_bps")) if isinstance(payload, dict) else None),
    )
    flattened["relative_volume_observed"] = _first_non_null_column(
        flattened,
        "relative_volume_observed",
        explain.map(lambda payload: _coerce_float(payload.get("relative_volume")) if isinstance(payload, dict) else None),
    )
    return flattened


def _first_non_null_column(frame: pd.DataFrame, column: str, fallback: pd.Series) -> pd.Series:
    if column not in frame.columns:
        return fallback
    return frame[column].where(frame[column].notna(), fallback)


def _maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
