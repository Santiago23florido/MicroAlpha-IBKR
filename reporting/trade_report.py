from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.performance import analyze_trade_logs


def build_trade_report(
    decision_frame: pd.DataFrame,
    *,
    orders: list[dict[str, Any]] | None = None,
    fills: list[dict[str, Any]] | None = None,
    reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    analysis = analyze_trade_logs(
        decision_frame,
        orders=orders or [],
        fills=fills or [],
        reports=reports or [],
    )
    return analysis["summary"]
