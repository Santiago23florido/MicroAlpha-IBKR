from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.io import list_phase7_summary_paths, load_phase7_frame, read_json, write_json


def compare_shadow_to_paper_and_market(
    shadow_frame: pd.DataFrame,
    *,
    shadow_report_dir: str | Path,
    phase7_report_dir: str | Path,
    session_id: str,
) -> dict[str, Any]:
    report_dir = Path(shadow_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp_token = session_id.replace("shadow_", "")

    paper_frame = _load_latest_paper_frame(phase7_report_dir)
    shadow_vs_paper = _compare_shadow_to_paper(shadow_frame, paper_frame)
    shadow_vs_market = _compare_shadow_to_market(shadow_frame)

    shadow_vs_paper_path = report_dir / f"shadow_vs_paper_{timestamp_token}.csv"
    shadow_vs_market_path = report_dir / f"shadow_vs_market_{timestamp_token}.csv"
    summary_path = report_dir / f"shadow_alignment_summary_{timestamp_token}.json"
    shadow_vs_paper.to_csv(shadow_vs_paper_path, index=False)
    shadow_vs_market.to_csv(shadow_vs_market_path, index=False)

    summary = {
        "status": "ok",
        "session_id": session_id,
        "shadow_count": int(len(shadow_frame)),
        "paper_count": int(len(paper_frame)),
        "agreement_rate": _safe_ratio(int((shadow_vs_paper["agreement"] == True).sum()), int(len(shadow_vs_paper))),
        "divergence_rate": _safe_ratio(int((shadow_vs_paper["diverged"] == True).sum()), int(len(shadow_vs_paper))),
        "paper_executed_when_shadow_would_not": int((shadow_vs_paper["paper_only"] == True).sum()) if not shadow_vs_paper.empty else 0,
        "shadow_would_act_when_paper_did_not": int((shadow_vs_paper["shadow_only"] == True).sum()) if not shadow_vs_paper.empty else 0,
        "shadow_hit_rate": _safe_ratio(int((shadow_vs_market["shadow_outcome_positive"] == True).sum()), int(len(shadow_vs_market))),
        "shadow_vs_paper_path": str(shadow_vs_paper_path),
        "shadow_vs_market_path": str(shadow_vs_market_path),
        "shadow_alignment_summary_path": str(summary_path),
    }
    write_json(summary_path, summary)
    return summary


def _load_latest_paper_frame(phase7_report_dir: str | Path) -> pd.DataFrame:
    try:
        summary_paths = list_phase7_summary_paths(phase7_report_dir, limit=20)
    except FileNotFoundError:
        return pd.DataFrame()
    candidates = []
    for path in summary_paths:
        payload = read_json(path)
        run_type = str(payload.get("run_type") or "")
        if run_type.startswith("paper_session"):
            candidates.append(payload)
    if not candidates:
        return pd.DataFrame()
    latest = sorted(candidates, key=lambda item: str(item.get("summary_path") or item.get("csv_path") or ""))[-1]
    parquet_path = latest.get("parquet_path")
    if not parquet_path:
        return pd.DataFrame()
    frame = load_phase7_frame(parquet_path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["paper_action"] = frame.get("action", "NO_TRADE")
    frame["paper_execution_status"] = frame.get("execution_status", "UNKNOWN")
    return frame


def _compare_shadow_to_paper(shadow_frame: pd.DataFrame, paper_frame: pd.DataFrame) -> pd.DataFrame:
    if shadow_frame.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "shadow_action", "paper_action", "agreement", "diverged", "paper_only", "shadow_only"])
    shadow = shadow_frame.copy()
    shadow["timestamp"] = pd.to_datetime(shadow["timestamp"], utc=True, errors="coerce")
    shadow["symbol"] = shadow["symbol"].astype(str).str.upper()
    shadow["shadow_action"] = shadow.get("action", "NO_TRADE")
    join_columns = ["timestamp", "symbol"]
    if paper_frame.empty:
        shadow["paper_action"] = None
        shadow["agreement"] = False
        shadow["diverged"] = shadow["shadow_action"].ne("NO_TRADE")
        shadow["paper_only"] = False
        shadow["shadow_only"] = shadow["shadow_action"].ne("NO_TRADE")
        return shadow.loc[:, ["timestamp", "symbol", "shadow_action", "paper_action", "agreement", "diverged", "paper_only", "shadow_only"]]
    merged = shadow.merge(
        paper_frame.loc[:, [*join_columns, "paper_action", "paper_execution_status"]],
        on=join_columns,
        how="outer",
    )
    merged["shadow_action"] = merged.get("shadow_action").fillna("NO_TRADE")
    merged["paper_action"] = merged.get("paper_action").fillna("NO_TRADE")
    merged["agreement"] = merged["shadow_action"] == merged["paper_action"]
    merged["diverged"] = ~merged["agreement"]
    merged["paper_only"] = (merged["paper_action"] != "NO_TRADE") & (merged["shadow_action"] == "NO_TRADE")
    merged["shadow_only"] = (merged["shadow_action"] != "NO_TRADE") & (merged["paper_action"] == "NO_TRADE")
    return merged.loc[:, ["timestamp", "symbol", "shadow_action", "paper_action", "agreement", "diverged", "paper_only", "shadow_only"]]


def _compare_shadow_to_market(shadow_frame: pd.DataFrame) -> pd.DataFrame:
    if shadow_frame.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "action", "confidence_bucket", "future_net_return_bps", "shadow_outcome_positive"])
    frame = shadow_frame.copy()
    frame["confidence_bucket"] = pd.cut(
        pd.to_numeric(frame.get("confidence"), errors="coerce").fillna(0.0),
        bins=[-0.001, 0.5, 0.65, 0.8, 1.0],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)
    future_returns = pd.to_numeric(frame.get("future_net_return_bps"), errors="coerce")
    frame["shadow_outcome_positive"] = ((frame.get("action") == "LONG") & (future_returns > 0)) | ((frame.get("action") == "SHORT") & (future_returns < 0))
    frame["future_net_return_bps"] = future_returns
    return frame.loc[:, ["timestamp", "symbol", "action", "confidence_bucket", "future_net_return_bps", "shadow_outcome_positive"]]


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)
