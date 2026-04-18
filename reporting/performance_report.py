from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.performance import evaluate_performance, performance_by_segments


def build_performance_report(
    decision_frame: pd.DataFrame,
    *,
    fills: list[dict[str, Any]] | None = None,
    final_portfolio: dict[str, Any] | None = None,
    thresholds=None,
) -> dict[str, Any]:
    performance = evaluate_performance(
        decision_frame,
        fills=fills,
        final_portfolio=final_portfolio,
        thresholds=thresholds,
    )
    segments = performance_by_segments(decision_frame)
    return {
        "summary": performance["summary"],
        "alerts": performance["alerts"],
        "segment_names": sorted(segments["segment_tables"].keys()),
    }
