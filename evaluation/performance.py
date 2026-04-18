from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping

import numpy as np
import pandas as pd


def build_trade_frame(
    fills: list[dict[str, Any]],
    *,
    final_portfolio: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = sorted(
        fills,
        key=lambda row: (
            str(row.get("filled_at") or ""),
            str(row.get("fill_id") or ""),
        ),
    )
    open_lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    closed_trades: list[dict[str, Any]] = []

    for fill in ordered:
        symbol = str(fill.get("symbol") or "").upper()
        quantity = int(fill.get("quantity", 0) or 0)
        if not symbol or quantity <= 0:
            continue
        fill_price = float(fill.get("fill_price", 0.0) or 0.0)
        fill_time = pd.to_datetime(fill.get("filled_at"), utc=True, errors="coerce")
        fill_commission = float(fill.get("commission", 0.0) or 0.0)
        commission_per_unit = fill_commission / quantity if quantity else 0.0
        delta = _signed_fill_quantity(str(fill.get("action") or ""), quantity)
        remaining = abs(delta)

        if delta > 0:
            while remaining > 0 and open_lots[symbol] and open_lots[symbol][0]["side"] == "SHORT":
                remaining = _close_lot(
                    closed_trades,
                    open_lots[symbol],
                    remaining,
                    exit_price=fill_price,
                    exit_time=fill_time,
                    exit_commission_per_unit=commission_per_unit,
                    exit_fill=fill,
                )
            if remaining > 0:
                open_lots[symbol].append(
                    _build_open_lot(
                        fill=fill,
                        side="LONG",
                        quantity=remaining,
                        entry_price=fill_price,
                        entry_time=fill_time,
                        entry_commission_per_unit=commission_per_unit,
                    )
                )
            continue

        while remaining > 0 and open_lots[symbol] and open_lots[symbol][0]["side"] == "LONG":
            remaining = _close_lot(
                closed_trades,
                open_lots[symbol],
                remaining,
                exit_price=fill_price,
                exit_time=fill_time,
                exit_commission_per_unit=commission_per_unit,
                exit_fill=fill,
            )
        if remaining > 0:
            open_lots[symbol].append(
                _build_open_lot(
                    fill=fill,
                    side="SHORT",
                    quantity=remaining,
                    entry_price=fill_price,
                    entry_time=fill_time,
                    entry_commission_per_unit=commission_per_unit,
                )
            )

    closed_frame = pd.DataFrame(closed_trades)
    if not closed_frame.empty:
        closed_frame["entry_time"] = pd.to_datetime(closed_frame["entry_time"], utc=True, errors="coerce")
        closed_frame["exit_time"] = pd.to_datetime(closed_frame["exit_time"], utc=True, errors="coerce")
        closed_frame["holding_seconds"] = (closed_frame["exit_time"] - closed_frame["entry_time"]).dt.total_seconds()
        closed_frame["win"] = closed_frame["net_pnl"] > 0
        closed_frame["loss"] = closed_frame["net_pnl"] < 0
        closed_frame["cumulative_net_pnl"] = closed_frame["net_pnl"].cumsum()

    open_rows = _build_open_trade_rows(open_lots, final_portfolio)
    open_frame = pd.DataFrame(open_rows)
    return closed_frame, open_frame


def evaluate_performance(
    decision_frame: pd.DataFrame,
    *,
    fills: list[dict[str, Any]] | None = None,
    final_portfolio: Mapping[str, Any] | None = None,
    thresholds=None,
) -> dict[str, Any]:
    closed_trades, open_trades = build_trade_frame(fills or [], final_portfolio=final_portfolio)
    equity_curve = extract_equity_curve(decision_frame)
    first_equity = float(equity_curve["equity_before"].iloc[0]) if not equity_curve.empty else None
    last_equity = float(equity_curve["equity_after"].iloc[-1]) if not equity_curve.empty else None
    session_total_pnl = None if first_equity is None or last_equity is None else round(float(last_equity - first_equity), 9)
    total_commissions = float(sum(float(fill.get("commission", 0.0) or 0.0) for fill in (fills or [])))

    portfolio_realized = _coerce_float(((final_portfolio or {}).get("realized_pnl")))
    portfolio_unrealized = _coerce_float(((final_portfolio or {}).get("unrealized_pnl")))
    portfolio_total = None
    if portfolio_realized is not None or portfolio_unrealized is not None:
        portfolio_total = float((portfolio_realized or 0.0) + (portfolio_unrealized or 0.0))

    gross_total_pnl = float(closed_trades["gross_pnl"].sum()) if not closed_trades.empty else 0.0
    net_total_pnl = float(closed_trades["net_pnl"].sum()) if not closed_trades.empty else 0.0
    trade_count = int(len(closed_trades))
    win_rate = float(closed_trades["win"].mean()) if trade_count else 0.0
    loss_rate = float(closed_trades["loss"].mean()) if trade_count else 0.0
    average_pnl = float(closed_trades["net_pnl"].mean()) if trade_count else 0.0
    average_return_bps = float(closed_trades["return_bps"].mean()) if trade_count else 0.0
    expectancy = average_pnl
    positive_sum = float(closed_trades.loc[closed_trades["net_pnl"] > 0, "net_pnl"].sum()) if trade_count else 0.0
    negative_sum = float(closed_trades.loc[closed_trades["net_pnl"] < 0, "net_pnl"].sum()) if trade_count else 0.0
    profit_factor = None if negative_sum == 0 else float(positive_sum / abs(negative_sum))
    max_drawdown, max_drawdown_pct = _max_drawdown(equity_curve["equity_after"].tolist()) if not equity_curve.empty else (0.0, 0.0)
    sharpe_ratio = _simple_sharpe(equity_curve["period_return"].tolist()) if not equity_curve.empty else 0.0

    summary = {
        "closed_trade_count": trade_count,
        "open_trade_count": int(len(open_trades)),
        "fill_count": int(len(fills or [])),
        "gross_total_pnl": gross_total_pnl,
        "net_total_pnl": net_total_pnl,
        "session_total_pnl": session_total_pnl,
        "portfolio_total_pnl": portfolio_total,
        "portfolio_realized_pnl": portfolio_realized,
        "portfolio_unrealized_pnl": portfolio_unrealized,
        "total_commissions": total_commissions,
        "average_pnl_per_trade": average_pnl,
        "average_return_bps": average_return_bps,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio_simple": sharpe_ratio,
        "cumulative_pnl": float(closed_trades["cumulative_net_pnl"].iloc[-1]) if not closed_trades.empty else 0.0,
        "executed_decision_count": int((decision_frame.get("execution_status") == "FILLED").sum()) if "execution_status" in decision_frame.columns else 0,
    }
    alerts = _performance_alerts(summary, thresholds)
    return {
        "summary": summary,
        "alerts": alerts,
        "closed_trades": closed_trades,
        "open_trades": open_trades,
        "equity_curve": equity_curve,
    }


def performance_by_segments(
    decision_frame: pd.DataFrame,
    *,
    outcome_column: str | None = None,
) -> dict[str, Any]:
    frame = decision_frame.copy()
    outcome = outcome_column or select_outcome_column(frame)
    if outcome is None:
        return {"outcome_column": None, "segment_tables": {}, "alerts": ["no_realized_outcome_available"]}

    frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
    frame = frame.dropna(subset=[outcome]).copy()
    if frame.empty:
        return {"outcome_column": outcome, "segment_tables": {}, "alerts": ["realized_outcome_empty_after_filtering"]}

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["hour_of_day"] = frame["timestamp"].dt.hour
        frame["day_name"] = frame["timestamp"].dt.day_name()

    segment_tables: dict[str, pd.DataFrame] = {}
    for column, name in (
        ("score", "score_deciles"),
        ("probability", "probability_deciles"),
        ("expected_return_bps", "predicted_return_deciles"),
        ("symbol", "symbol"),
        ("hour_of_day", "hour_of_day"),
        ("day_name", "day_of_week"),
    ):
        if column not in frame.columns:
            continue
        prepared = _bucket_if_numeric(frame, column, name)
        if prepared is None:
            continue
        bucket_column, bucket_frame = prepared
        segment_tables[name] = _segment_summary(bucket_frame, bucket_column, outcome)

    for column, name in (
        ("spread_bps_observed", "spread_buckets"),
        ("volume_observed", "volume_buckets"),
        ("relative_volume_observed", "relative_volume_buckets"),
    ):
        if column not in frame.columns:
            continue
        bucket_column = f"{column}_bucket"
        bucket_series = _quantile_bucket(frame[column], bucket_count=5, prefix=column)
        if bucket_series is None:
            continue
        bucket_frame = frame.assign(**{bucket_column: bucket_series})
        segment_tables[name] = _segment_summary(bucket_frame, bucket_column, outcome)

    return {
        "outcome_column": outcome,
        "segment_tables": segment_tables,
        "alerts": [],
    }


def analyze_trade_logs(
    decision_frame: pd.DataFrame,
    *,
    orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    order_frame = pd.DataFrame(orders)
    fill_frame = pd.DataFrame(fills)
    report_frame = pd.DataFrame(reports)

    rejection_rows = []
    if not report_frame.empty and "status" in report_frame.columns:
        rejection_rows.append(report_frame.loc[report_frame["status"].astype(str).isin(["REJECTED", "FAILED"])].copy())
    if "execution_error" in decision_frame.columns:
        decision_rejections = decision_frame.loc[decision_frame["execution_error"].notna(), ["decision_id", "execution_error", "execution_status"]].copy()
        if not decision_rejections.empty:
            rejection_rows.append(decision_rejections.rename(columns={"execution_error": "message", "execution_status": "status"}))
    rejection_frame = pd.concat(rejection_rows, ignore_index=True) if rejection_rows else pd.DataFrame()

    order_status_counts = order_frame["status"].astype(str).value_counts().to_dict() if "status" in order_frame.columns else {}
    report_status_counts = report_frame["status"].astype(str).value_counts().to_dict() if "status" in report_frame.columns else {}
    execution_status_counts = (
        decision_frame["execution_status"].astype(str).value_counts().to_dict()
        if "execution_status" in decision_frame.columns
        else {}
    )
    inconsistencies = _find_trade_log_inconsistencies(order_frame, fill_frame, report_frame)

    return {
        "summary": {
            "order_count": int(len(order_frame)),
            "fill_count": int(len(fill_frame)),
            "report_count": int(len(report_frame)),
            "rejection_count": int(len(rejection_frame)),
            "order_status_counts": {str(key): int(value) for key, value in order_status_counts.items()},
            "report_status_counts": {str(key): int(value) for key, value in report_status_counts.items()},
            "execution_status_counts": {str(key): int(value) for key, value in execution_status_counts.items()},
            "inconsistency_count": int(len(inconsistencies)),
        },
        "rejections": rejection_frame,
        "inconsistencies": pd.DataFrame(inconsistencies),
        "orders": order_frame,
        "fills": fill_frame,
        "reports": report_frame,
    }


def select_outcome_column(frame: pd.DataFrame) -> str | None:
    for column in ("future_net_return_bps", "realized_pnl_delta", "expected_return_bps"):
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    return None


def extract_equity_curve(decision_frame: pd.DataFrame) -> pd.DataFrame:
    if "portfolio_before" not in decision_frame.columns or "portfolio_after" not in decision_frame.columns:
        return pd.DataFrame(columns=["timestamp", "equity_before", "equity_after", "period_return"])

    rows: list[dict[str, Any]] = []
    for _, row in decision_frame.iterrows():
        before = row.get("portfolio_before") if isinstance(row.get("portfolio_before"), dict) else {}
        after = row.get("portfolio_after") if isinstance(row.get("portfolio_after"), dict) else {}
        rows.append(
            {
                "timestamp": pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce"),
                "equity_before": _coerce_float(before.get("equity")),
                "equity_after": _coerce_float(after.get("equity")),
            }
        )
    frame = pd.DataFrame(rows).dropna(subset=["timestamp", "equity_before", "equity_after"])
    if frame.empty:
        return frame
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    denominator = frame["equity_before"].replace(0, np.nan)
    frame["period_return"] = ((frame["equity_after"] - frame["equity_before"]) / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def _segment_summary(frame: pd.DataFrame, bucket_column: str, outcome_column: str) -> pd.DataFrame:
    grouped = frame.groupby(bucket_column, dropna=False)
    summary = grouped.agg(
        count=(outcome_column, "count"),
        mean_outcome=(outcome_column, "mean"),
        median_outcome=(outcome_column, "median"),
        win_rate=(outcome_column, lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
        avg_score=("score", "mean") if "score" in frame.columns else (outcome_column, "count"),
        avg_probability=("probability", "mean") if "probability" in frame.columns else (outcome_column, "count"),
    ).reset_index()
    return summary


def _bucket_if_numeric(frame: pd.DataFrame, column: str, name: str) -> tuple[str, pd.DataFrame] | None:
    if column not in frame.columns:
        return None
    if pd.api.types.is_numeric_dtype(frame[column]):
        bucket_column = f"{column}_bucket"
        buckets = _quantile_bucket(frame[column], bucket_count=10, prefix=name)
        if buckets is None:
            return None
        return bucket_column, frame.assign(**{bucket_column: buckets})
    return column, frame


def _quantile_bucket(series: pd.Series, *, bucket_count: int, prefix: str) -> pd.Series | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < max(bucket_count, 2):
        return None
    try:
        return pd.qcut(numeric, q=min(bucket_count, numeric.nunique()), duplicates="drop").astype(str)
    except ValueError:
        ranked = numeric.rank(method="first")
        try:
            return pd.qcut(ranked, q=min(bucket_count, ranked.nunique()), duplicates="drop").astype(str)
        except ValueError:
            return None


def _build_open_lot(
    *,
    fill: dict[str, Any],
    side: str,
    quantity: int,
    entry_price: float,
    entry_time,
    entry_commission_per_unit: float,
) -> dict[str, Any]:
    return {
        "symbol": str(fill.get("symbol") or "").upper(),
        "side": side,
        "quantity_remaining": int(quantity),
        "entry_price": float(entry_price),
        "entry_time": entry_time,
        "entry_commission_per_unit": float(entry_commission_per_unit),
        "entry_fill_id": fill.get("fill_id"),
        "entry_order_id": fill.get("order_id"),
        "source_decision_id": fill.get("source_decision_id"),
        "source_model_name": fill.get("source_model_name"),
    }


def _close_lot(
    closed_trades: list[dict[str, Any]],
    open_lots: deque[dict[str, Any]],
    remaining_qty: int,
    *,
    exit_price: float,
    exit_time,
    exit_commission_per_unit: float,
    exit_fill: dict[str, Any],
) -> int:
    lot = open_lots[0]
    matched_qty = min(remaining_qty, int(lot["quantity_remaining"]))
    gross_pnl = (
        (exit_price - lot["entry_price"]) * matched_qty
        if lot["side"] == "LONG"
        else (lot["entry_price"] - exit_price) * matched_qty
    )
    commissions = (lot["entry_commission_per_unit"] + exit_commission_per_unit) * matched_qty
    entry_notional = lot["entry_price"] * matched_qty
    net_pnl = gross_pnl - commissions
    return_bps = None if entry_notional == 0 else float((net_pnl / entry_notional) * 10000.0)

    closed_trades.append(
        {
            "symbol": lot["symbol"],
            "side": lot["side"],
            "quantity": matched_qty,
            "entry_time": lot["entry_time"],
            "exit_time": exit_time,
            "entry_price": lot["entry_price"],
            "exit_price": exit_price,
            "gross_pnl": float(gross_pnl),
            "net_pnl": float(net_pnl),
            "return_bps": return_bps,
            "commissions": float(commissions),
            "entry_fill_id": lot["entry_fill_id"],
            "exit_fill_id": exit_fill.get("fill_id"),
            "entry_order_id": lot["entry_order_id"],
            "exit_order_id": exit_fill.get("order_id"),
            "source_decision_id": lot["source_decision_id"],
            "source_model_name": lot["source_model_name"],
        }
    )

    lot["quantity_remaining"] -= matched_qty
    if lot["quantity_remaining"] <= 0:
        open_lots.popleft()
    return remaining_qty - matched_qty


def _build_open_trade_rows(open_lots: dict[str, deque[dict[str, Any]]], final_portfolio: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    positions = dict((final_portfolio or {}).get("positions", {}) or {})
    rows: list[dict[str, Any]] = []
    for symbol, lots in open_lots.items():
        mark_price = _coerce_float(dict(positions.get(symbol, {}) or {}).get("last_price"))
        for lot in lots:
            quantity = int(lot["quantity_remaining"])
            unrealized = None
            if mark_price is not None:
                if lot["side"] == "LONG":
                    unrealized = (mark_price - lot["entry_price"]) * quantity
                else:
                    unrealized = (lot["entry_price"] - mark_price) * quantity
            rows.append(
                {
                    "symbol": symbol,
                    "side": lot["side"],
                    "quantity": quantity,
                    "entry_time": lot["entry_time"],
                    "entry_price": lot["entry_price"],
                    "mark_price": mark_price,
                    "unrealized_pnl": unrealized,
                    "source_decision_id": lot["source_decision_id"],
                    "source_model_name": lot["source_model_name"],
                }
            )
    return rows


def _signed_fill_quantity(action: str, quantity: int) -> int:
    normalized = str(action).upper()
    if normalized in {"BUY", "COVER"}:
        return int(quantity)
    return -int(quantity)


def _simple_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return 0.0
    std = float(series.std(ddof=0))
    if std <= 0:
        return 0.0
    return float((series.mean() / std) * np.sqrt(len(series)))


def _max_drawdown(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    series = pd.Series(values, dtype=float)
    running_max = series.cummax()
    drawdown = running_max - series
    drawdown_pct = drawdown / running_max.replace(0, np.nan)
    return float(drawdown.max()), float(drawdown_pct.max(skipna=True) or 0.0)


def _performance_alerts(summary: Mapping[str, Any], thresholds) -> list[str]:
    alerts: list[str] = []
    if thresholds is None:
        return alerts
    trade_count = int(summary.get("closed_trade_count", 0) or 0)
    if trade_count < int(thresholds.min_trades_required):
        alerts.append(f"insufficient_closed_trades:{trade_count}<{thresholds.min_trades_required}")
    total_pnl = _coerce_float(summary.get("session_total_pnl"))
    if total_pnl is not None and total_pnl < float(thresholds.min_total_pnl):
        alerts.append(f"session_total_pnl_below_threshold:{total_pnl:.6f}<{thresholds.min_total_pnl:.6f}")
    win_rate = float(summary.get("win_rate", 0.0) or 0.0)
    if trade_count and win_rate < float(thresholds.min_win_rate):
        alerts.append(f"win_rate_below_threshold:{win_rate:.4f}<{thresholds.min_win_rate:.4f}")
    expectancy = float(summary.get("expectancy", 0.0) or 0.0)
    if trade_count and expectancy < float(thresholds.min_expectancy):
        alerts.append(f"expectancy_below_threshold:{expectancy:.6f}<{thresholds.min_expectancy:.6f}")
    max_drawdown = float(summary.get("max_drawdown", 0.0) or 0.0)
    if max_drawdown > float(thresholds.max_drawdown):
        alerts.append(f"max_drawdown_exceeds_threshold:{max_drawdown:.6f}>{thresholds.max_drawdown:.6f}")
    return alerts


def _find_trade_log_inconsistencies(
    order_frame: pd.DataFrame,
    fill_frame: pd.DataFrame,
    report_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    inconsistencies: list[dict[str, Any]] = []
    if order_frame.empty:
        return inconsistencies

    fills_by_order = fill_frame.groupby("order_id").size().to_dict() if not fill_frame.empty and "order_id" in fill_frame.columns else {}
    for _, order in order_frame.drop_duplicates(subset=["order_id"], keep="last").iterrows():
        order_id = str(order.get("order_id") or "")
        status = str(order.get("status") or "")
        fill_count = int(fills_by_order.get(order_id, 0))
        if status == "FILLED" and fill_count == 0:
            inconsistencies.append({"order_id": order_id, "issue": "filled_order_without_fill_event"})
        if status in {"REJECTED", "FAILED"} and fill_count > 0:
            inconsistencies.append({"order_id": order_id, "issue": "rejected_or_failed_order_has_fill_event"})
    return inconsistencies


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
