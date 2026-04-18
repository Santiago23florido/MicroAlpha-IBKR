from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from evaluation.io import read_json, write_json


def compare_runs(
    report_root: str | Path,
    *,
    report_paths: Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    reports = _load_reports(report_root, report_paths=report_paths)
    if not reports:
        raise FileNotFoundError(f"No Phase 8 run_report.json files found under {report_root}.")

    comparison = pd.DataFrame([_comparison_row(report) for report in reports]).sort_values(
        ["session_total_pnl", "sharpe_ratio_simple", "max_drawdown"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    comparison["pnl_rank"] = comparison["session_total_pnl"].rank(method="dense", ascending=False).astype(int)
    comparison["sharpe_rank"] = comparison["sharpe_ratio_simple"].rank(method="dense", ascending=False).astype(int)
    comparison["drawdown_rank"] = comparison["max_drawdown"].rank(method="dense", ascending=True).astype(int)
    comparison["stability_rank"] = comparison["alert_count"].rank(method="dense", ascending=True).astype(int)
    comparison["overall_rank"] = (
        comparison[["pnl_rank", "sharpe_rank", "drawdown_rank", "stability_rank"]].mean(axis=1).rank(method="dense")
    ).astype(int)

    timestamp_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_dir or report_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"compare_runs_{timestamp_token}.csv"
    json_path = out_dir / f"compare_runs_{timestamp_token}.json"
    parquet_path = out_dir / f"compare_runs_{timestamp_token}.parquet"
    comparison.to_csv(csv_path, index=False)
    comparison.to_parquet(parquet_path, index=False)
    write_json(
        json_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_count": int(len(comparison)),
            "best_total_pnl_run": comparison.iloc[0]["run_id"],
            "best_sharpe_run": comparison.sort_values("sharpe_ratio_simple", ascending=False).iloc[0]["run_id"],
            "lowest_drawdown_run": comparison.sort_values("max_drawdown", ascending=True).iloc[0]["run_id"],
            "rows": comparison.to_dict(orient="records"),
        },
    )
    return {
        "status": "ok",
        "run_count": int(len(comparison)),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "parquet_path": str(parquet_path),
        "rows": comparison.to_dict(orient="records"),
    }


def update_economic_leaderboard(
    report_root: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    reports = _load_reports(report_root, report_paths=None)
    if not reports:
        raise FileNotFoundError(f"No Phase 8 run reports found under {report_root}.")
    leaderboard = pd.DataFrame([_comparison_row(report) for report in reports]).sort_values(
        ["session_total_pnl", "sharpe_ratio_simple", "max_drawdown"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    leaderboard["economic_rank"] = range(1, len(leaderboard) + 1)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(target, index=False)
    json_target = target.with_suffix(".json")
    parquet_target = target.with_suffix(".parquet")
    write_json(
        json_target,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(leaderboard)),
            "rows": leaderboard.to_dict(orient="records"),
        },
    )
    leaderboard.to_parquet(parquet_target, index=False)
    return {
        "status": "ok",
        "csv_path": str(target),
        "json_path": str(json_target),
        "parquet_path": str(parquet_target),
        "row_count": int(len(leaderboard)),
    }


def _load_reports(report_root: str | Path, *, report_paths: Sequence[str | Path] | None) -> list[dict[str, Any]]:
    if report_paths:
        return [read_json(path) for path in report_paths]
    return [read_json(path) for path in sorted(Path(report_root).glob("*/run_report.json"))]


def _comparison_row(report: dict[str, Any]) -> dict[str, Any]:
    performance = dict(report.get("performance_summary", {}) or {})
    drift = dict(report.get("drift_summary", {}) or {})
    signal = dict(report.get("signal_summary", {}) or {})
    return {
        "run_id": report.get("run_id"),
        "run_label": report.get("run_label"),
        "generated_at_utc": report.get("generated_at_utc"),
        "phase7_summary_path": report.get("phase7_summary_path"),
        "model_name": report.get("model_name"),
        "feature_set_name": report.get("feature_set_name"),
        "target_mode": report.get("target_mode"),
        "session_total_pnl": performance.get("session_total_pnl"),
        "portfolio_total_pnl": performance.get("portfolio_total_pnl"),
        "win_rate": performance.get("win_rate"),
        "expectancy": performance.get("expectancy"),
        "profit_factor": performance.get("profit_factor"),
        "sharpe_ratio_simple": performance.get("sharpe_ratio_simple"),
        "max_drawdown": performance.get("max_drawdown"),
        "closed_trade_count": performance.get("closed_trade_count"),
        "alert_count": len(report.get("alerts", [])),
        "data_drift_max_psi": drift.get("data_drift_max_psi"),
        "prediction_drift_max_psi": drift.get("prediction_drift_max_psi"),
        "label_drift_max_psi": drift.get("label_drift_max_psi"),
        "score_monotonicity_ratio": signal.get("score_monotonicity_ratio"),
        "probability_monotonicity_ratio": signal.get("probability_monotonicity_ratio"),
    }
