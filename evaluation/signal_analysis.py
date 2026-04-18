from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.performance import select_outcome_column


def analyze_signal_quality(
    decision_frame: pd.DataFrame,
    *,
    outcome_column: str | None = None,
    degenerate_output_std_floor: float = 1e-6,
) -> dict[str, Any]:
    frame = decision_frame.copy()
    outcome = outcome_column or select_outcome_column(frame)
    if outcome is None:
        return {"summary": {"outcome_column": None}, "alerts": ["no_realized_outcome_available"], "tables": {}}

    frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
    frame = frame.dropna(subset=[outcome]).copy()
    if frame.empty:
        return {"summary": {"outcome_column": outcome}, "alerts": ["realized_outcome_empty_after_filtering"], "tables": {}}

    tables: dict[str, pd.DataFrame] = {}
    alerts: list[str] = []

    score_table = _build_bucket_table(frame, "score", outcome, label="score_deciles")
    if score_table is not None:
        tables["score_deciles"] = score_table

    probability_table = _build_bucket_table(frame, "probability", outcome, label="probability_deciles")
    if probability_table is not None:
        tables["probability_deciles"] = probability_table

    predicted_return_table = _build_bucket_table(frame, "expected_return_bps", outcome, label="predicted_return_deciles")
    if predicted_return_table is not None:
        tables["predicted_return_deciles"] = predicted_return_table

    calibration_table = _build_calibration_table(frame, outcome)
    if calibration_table is not None:
        tables["calibration"] = calibration_table

    score_std = _std(frame.get("score"))
    probability_std = _std(frame.get("probability"))
    if score_std is not None and score_std <= degenerate_output_std_floor:
        alerts.append(f"degenerate_score_distribution:std={score_std:.8f}")
    if probability_std is not None and probability_std <= degenerate_output_std_floor:
        alerts.append(f"degenerate_probability_distribution:std={probability_std:.8f}")
    if "action" in frame.columns:
        action_share = frame["action"].astype(str).value_counts(normalize=True, dropna=False)
        if not action_share.empty and float(action_share.iloc[0]) >= 0.98:
            alerts.append(f"degenerate_action_distribution:dominant_share={float(action_share.iloc[0]):.4f}")

    summary = {
        "outcome_column": outcome,
        "score_outcome_correlation": _corr(frame.get("score"), frame[outcome]),
        "probability_outcome_correlation": _corr(frame.get("probability"), frame[outcome]),
        "score_monotonicity_ratio": _monotonicity_ratio(score_table),
        "probability_monotonicity_ratio": _monotonicity_ratio(probability_table),
        "top_vs_bottom_score_spread": _top_bottom_spread(score_table),
        "top_vs_bottom_probability_spread": _top_bottom_spread(probability_table),
        "score_std": score_std,
        "probability_std": probability_std,
        "sample_count": int(len(frame)),
    }

    return {
        "summary": summary,
        "alerts": alerts,
        "tables": tables,
    }


def _build_bucket_table(frame: pd.DataFrame, column: str, outcome_column: str, *, label: str) -> pd.DataFrame | None:
    if column not in frame.columns:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.notna().sum() < 10:
        return None
    try:
        buckets = pd.qcut(numeric, q=min(10, numeric.nunique()), duplicates="drop")
    except ValueError:
        return None
    working = frame.assign(**{label: buckets.astype(str)})
    table = (
        working.groupby(label, dropna=False)
        .agg(
            count=(outcome_column, "count"),
            mean_outcome=(outcome_column, "mean"),
            median_outcome=(outcome_column, "median"),
            positive_rate=(outcome_column, lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
            avg_signal=(column, "mean"),
        )
        .reset_index()
    )
    return table


def _build_calibration_table(frame: pd.DataFrame, outcome_column: str) -> pd.DataFrame | None:
    if "probability" not in frame.columns:
        return None
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if probability.dropna().empty:
        return None
    bounded = probability.where((probability >= 0.0) & (probability <= 1.0))
    if bounded.dropna().empty:
        return None
    bins = pd.cut(bounded, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    positive_target = (pd.to_numeric(frame[outcome_column], errors="coerce") > 0).astype(float)
    working = pd.DataFrame({"probability_bucket": bins.astype(str), "probability": bounded, "positive_target": positive_target})
    working = working.dropna(subset=["probability"])
    if working.empty:
        return None
    return (
        working.groupby("probability_bucket", dropna=False)
        .agg(
            count=("positive_target", "count"),
            mean_probability=("probability", "mean"),
            observed_positive_rate=("positive_target", "mean"),
        )
        .reset_index()
    )


def _monotonicity_ratio(table: pd.DataFrame | None) -> float | None:
    if table is None or table.empty or "mean_outcome" not in table.columns:
        return None
    diffs = pd.Series(table["mean_outcome"]).diff().dropna()
    if diffs.empty:
        return None
    return float((diffs >= 0).mean())


def _top_bottom_spread(table: pd.DataFrame | None) -> float | None:
    if table is None or table.empty or "mean_outcome" not in table.columns:
        return None
    return float(table["mean_outcome"].iloc[-1] - table["mean_outcome"].iloc[0])


def _std(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.std(ddof=0))


def _corr(left: pd.Series | None, right: pd.Series) -> float | None:
    if left is None:
        return None
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    working = pd.DataFrame({"left": left_numeric, "right": right_numeric}).dropna()
    if len(working) < 2:
        return None
    left_std = float(working["left"].std(ddof=0))
    right_std = float(working["right"].std(ddof=0))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return None
    return float(working["left"].corr(working["right"]))
