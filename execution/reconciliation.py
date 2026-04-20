from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from config import Settings
from config.phase10_11 import load_phase10_11_config
from config.phase7 import load_phase7_config
from evaluation.io import read_jsonl
from execution import ExecutionJournal, build_execution_backend
from execution.ibkr_state_mapper import map_ibkr_status


TERMINAL_INTERNAL_STATUSES = {"REJECTED", "FILLED", "CANCELLED", "EXPIRED", "FAILED"}


def reconcile_broker_state(
    settings: Settings,
    *,
    backend=None,
    journal: ExecutionJournal | None = None,
    require_ibkr_paper: bool = True,
) -> dict[str, Any]:
    phase7 = load_phase7_config(settings)
    phase10_11 = load_phase10_11_config(settings)
    backend = backend or build_execution_backend(phase7, settings=settings, backend_name="ibkr_paper")
    if require_ibkr_paper and backend.name != "ibkr_paper":
        raise ValueError("Broker reconciliation requires the ibkr_paper backend.")

    if require_ibkr_paper and str(phase7.ibkr_paper.broker_mode).strip().lower() != "paper":
        raise ValueError("Broker reconciliation requires broker_mode=paper.")

    journal = journal or ExecutionJournal(phase7.logging)
    state = journal.load_state()
    internal_orders = list(state.get("orders", []) or [])
    internal_positions = dict((((state.get("portfolio_state") or {}).get("positions")) or {}))
    internal_fills = read_jsonl(journal.fills_path)

    connect_required = backend.name == "ibkr_paper"
    if connect_required:
        backend.connect()
    try:
        broker_open_orders = backend.get_open_orders()
        broker_positions = backend.get_positions()
        broker_executions = backend.get_recent_executions()
    finally:
        if connect_required:
            backend.disconnect()

    order_rows = _reconcile_orders(internal_orders, broker_open_orders, broker_executions)
    fill_rows = _reconcile_fills(internal_fills, broker_executions)
    position_rows = _reconcile_positions(
        internal_positions,
        broker_positions,
        quantity_tolerance=float(phase10_11.reconciliation_tolerances.quantity_tolerance),
        avg_price_tolerance=float(phase10_11.reconciliation_tolerances.average_price_tolerance),
    )

    order_df = pd.DataFrame(order_rows)
    fill_df = pd.DataFrame(fill_rows)
    position_df = pd.DataFrame(position_rows)
    summary = {
        "status": _overall_reconciliation_status(order_df, fill_df, position_df),
        "order_row_count": int(len(order_df)),
        "fill_row_count": int(len(fill_df)),
        "position_row_count": int(len(position_df)),
        "critical_order_mismatches": int(_critical_count(order_df)),
        "critical_fill_mismatches": int(_critical_count(fill_df)),
        "critical_position_mismatches": int(_critical_count(position_df)),
        "percent_reconciled_orders": _percent_reconciled(order_df),
        "percent_reconciled_fills": _percent_reconciled(fill_df),
        "percent_reconciled_positions": _percent_reconciled(position_df),
    }
    return {
        "status": "ok",
        "summary": summary,
        "orders": order_rows,
        "fills": fill_rows,
        "positions": position_rows,
        "broker_snapshot": {
            "open_orders": broker_open_orders,
            "positions": broker_positions,
            "executions": broker_executions,
        },
    }


def _reconcile_orders(
    internal_orders: list[Mapping[str, Any]],
    broker_open_orders: list[Mapping[str, Any]],
    broker_executions: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    broker_status_by_id: dict[int, Mapping[str, Any]] = {}
    for row in broker_open_orders:
        broker_order_id = _to_optional_int(row.get("order_id"))
        if broker_order_id is not None:
            broker_status_by_id[broker_order_id] = row
    broker_exec_order_ids = {
        _to_optional_int(row.get("order_id"))
        for row in broker_executions
        if _to_optional_int(row.get("order_id")) is not None
    }

    rows: list[dict[str, Any]] = []
    seen_broker_order_ids: set[int] = set()
    for order in internal_orders:
        broker_order_id = _to_optional_int(order.get("broker_order_id"))
        internal_status = str(order.get("status") or "")
        order_quantity = _to_optional_float(order.get("quantity"))
        filled_quantity = _to_optional_float(order.get("filled_quantity"))
        broker_row = None if broker_order_id is None else broker_status_by_id.get(broker_order_id)
        broker_seen_in_execs = broker_order_id in broker_exec_order_ids if broker_order_id is not None else False
        if broker_order_id is not None:
            seen_broker_order_ids.add(broker_order_id)

        mismatch = False
        severity = "info"
        reason = "matched"
        broker_status = None
        broker_internal_status = None
        broker_quantity = None
        broker_filled_quantity = None
        if broker_row is None and broker_order_id is None:
            mismatch = True
            severity = "warning"
            reason = "internal_order_missing_broker_order_id"
        elif broker_row is None and internal_status not in TERMINAL_INTERNAL_STATUSES and not broker_seen_in_execs:
            mismatch = True
            severity = "critical"
            reason = "open_internal_order_missing_in_broker"
        elif broker_row is not None:
            broker_status = str(broker_row.get("status") or "")
            broker_internal_status = map_ibkr_status(
                broker_status,
                filled_quantity=broker_row.get("filled_quantity"),
                remaining_quantity=broker_row.get("remaining_quantity"),
                message=broker_row.get("message"),
            ).value
            broker_quantity = _to_optional_float(broker_row.get("quantity"))
            broker_filled_quantity = _to_optional_float(broker_row.get("filled_quantity"))
            if broker_internal_status and internal_status and broker_internal_status != internal_status:
                mismatch = True
                severity = "warning"
                reason = "status_mismatch"
            if broker_quantity is not None and order_quantity is not None and abs(broker_quantity - order_quantity) > 0.0:
                mismatch = True
                severity = "critical"
                reason = "quantity_mismatch"
            if (
                broker_filled_quantity is not None
                and filled_quantity is not None
                and abs(broker_filled_quantity - filled_quantity) > 0.0
            ):
                mismatch = True
                severity = "warning"
                reason = "filled_quantity_mismatch"

        rows.append(
            {
                "order_id": order.get("order_id"),
                "broker_order_id": broker_order_id,
                "source_decision_id": order.get("source_decision_id"),
                "source_model_name": order.get("source_model_name"),
                "symbol": order.get("symbol"),
                "internal_status": internal_status,
                "broker_status": broker_status,
                "broker_internal_status": broker_internal_status,
                "internal_quantity": order_quantity,
                "broker_quantity": broker_quantity,
                "internal_filled_quantity": filled_quantity,
                "broker_filled_quantity": broker_filled_quantity,
                "mismatch": mismatch,
                "severity": severity,
                "reason": reason,
            }
        )

    for row in broker_open_orders:
        broker_order_id = _to_optional_int(row.get("order_id"))
        if broker_order_id is None or broker_order_id in seen_broker_order_ids:
            continue
        rows.append(
            {
                "order_id": None,
                "broker_order_id": broker_order_id,
                "source_decision_id": None,
                "source_model_name": None,
                "symbol": row.get("symbol"),
                "internal_status": None,
                "broker_status": row.get("status"),
                "broker_internal_status": map_ibkr_status(
                    row.get("status"),
                    filled_quantity=row.get("filled_quantity"),
                    remaining_quantity=row.get("remaining_quantity"),
                    message=row.get("message"),
                ).value,
                "internal_quantity": None,
                "broker_quantity": _to_optional_float(row.get("quantity")),
                "internal_filled_quantity": None,
                "broker_filled_quantity": _to_optional_float(row.get("filled_quantity")),
                "mismatch": True,
                "severity": "critical",
                "reason": "broker_order_missing_in_internal_state",
            }
        )
    return rows


def _reconcile_fills(
    internal_fills: list[Mapping[str, Any]],
    broker_executions: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    internal_by_execution_id = {
        str(fill.get("execution_id")): fill
        for fill in internal_fills
        if fill.get("execution_id") not in (None, "")
    }
    seen_exec_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    for execution in broker_executions:
        execution_id = str(execution.get("execution_id") or "")
        internal_fill = internal_by_execution_id.get(execution_id)
        seen_exec_ids.add(execution_id)
        broker_price = _to_optional_float(execution.get("price"))
        broker_qty = _to_optional_float(execution.get("shares"))
        internal_price = _to_optional_float(None if internal_fill is None else internal_fill.get("fill_price"))
        internal_qty = _to_optional_float(None if internal_fill is None else internal_fill.get("quantity"))
        mismatch = internal_fill is None
        severity = "critical" if internal_fill is None else "info"
        reason = "broker_execution_missing_in_internal_journal" if internal_fill is None else "matched"
        if not mismatch and broker_qty is not None and internal_qty is not None and abs(broker_qty - internal_qty) > 0.0:
            mismatch = True
            severity = "critical"
            reason = "fill_quantity_mismatch"
        if not mismatch and broker_price is not None and internal_price is not None and abs(broker_price - internal_price) > 0.05:
            mismatch = True
            severity = "warning"
            reason = "fill_price_mismatch"
        rows.append(
            {
                "execution_id": execution_id or None,
                "broker_order_id": _to_optional_int(execution.get("order_id")),
                "symbol": execution.get("symbol"),
                "internal_fill_id": None if internal_fill is None else internal_fill.get("fill_id"),
                "broker_quantity": broker_qty,
                "internal_quantity": internal_qty,
                "broker_fill_price": broker_price,
                "internal_fill_price": internal_price,
                "mismatch": mismatch,
                "severity": severity,
                "reason": reason,
            }
        )

    for fill in internal_fills:
        execution_id = str(fill.get("execution_id") or "")
        if execution_id and execution_id in seen_exec_ids:
            continue
        rows.append(
            {
                "execution_id": execution_id or None,
                "broker_order_id": _to_optional_int(fill.get("broker_order_id")),
                "symbol": fill.get("symbol"),
                "internal_fill_id": fill.get("fill_id"),
                "broker_quantity": None,
                "internal_quantity": _to_optional_float(fill.get("quantity")),
                "broker_fill_price": None,
                "internal_fill_price": _to_optional_float(fill.get("fill_price")),
                "mismatch": True,
                "severity": "warning",
                "reason": "internal_fill_missing_in_broker_execution_log",
            }
        )
    return rows


def _reconcile_positions(
    internal_positions: Mapping[str, Mapping[str, Any]],
    broker_positions: list[Mapping[str, Any]],
    *,
    quantity_tolerance: float,
    avg_price_tolerance: float,
) -> list[dict[str, Any]]:
    broker_by_symbol = {str(row.get("symbol") or "").upper(): row for row in broker_positions}
    rows: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()

    for symbol, internal in internal_positions.items():
        normalized_symbol = str(symbol).upper()
        broker = broker_by_symbol.get(normalized_symbol, {})
        seen_symbols.add(normalized_symbol)
        internal_qty = _to_optional_float(internal.get("quantity"))
        broker_qty = _to_optional_float(broker.get("position"))
        internal_avg = _to_optional_float(internal.get("average_entry_price"))
        broker_avg = _to_optional_float(broker.get("avgCost"))
        delta_qty = None if internal_qty is None or broker_qty is None else float(internal_qty - broker_qty)
        delta_avg = None if internal_avg is None or broker_avg is None else float(internal_avg - broker_avg)
        mismatch = False
        severity = "info"
        reason = "matched"
        suggested_action = "none"
        if broker == {}:
            mismatch = True
            severity = "critical"
            reason = "internal_position_missing_in_broker"
            suggested_action = "review_open_positions_before_continuing"
        elif delta_qty is not None and abs(delta_qty) > quantity_tolerance:
            mismatch = True
            severity = "critical"
            reason = "position_quantity_mismatch"
            suggested_action = "stop_and_reconcile_positions"
        elif delta_avg is not None and abs(delta_avg) > avg_price_tolerance:
            mismatch = True
            severity = "warning"
            reason = "position_average_price_mismatch"
            suggested_action = "review_average_entry_price"
        rows.append(
            {
                "symbol": normalized_symbol,
                "internal_quantity": internal_qty,
                "broker_quantity": broker_qty,
                "delta_quantity": delta_qty,
                "internal_average_entry_price": internal_avg,
                "broker_average_entry_price": broker_avg,
                "delta_avg_price": delta_avg,
                "match": not mismatch,
                "mismatch": mismatch,
                "severity": severity,
                "reason": reason,
                "suggested_action": suggested_action,
            }
        )

    for symbol, broker in broker_by_symbol.items():
        if symbol in seen_symbols:
            continue
        rows.append(
            {
                "symbol": symbol,
                "internal_quantity": None,
                "broker_quantity": _to_optional_float(broker.get("position")),
                "delta_quantity": None,
                "internal_average_entry_price": None,
                "broker_average_entry_price": _to_optional_float(broker.get("avgCost")),
                "delta_avg_price": None,
                "match": False,
                "mismatch": True,
                "severity": "critical",
                "reason": "broker_position_missing_in_internal_state",
                "suggested_action": "stop_and_reconcile_positions",
            }
        )
    return rows


def _overall_reconciliation_status(
    order_df: pd.DataFrame,
    fill_df: pd.DataFrame,
    position_df: pd.DataFrame,
) -> str:
    critical_total = _critical_count(order_df) + _critical_count(fill_df) + _critical_count(position_df)
    mismatch_total = _mismatch_count(order_df) + _mismatch_count(fill_df) + _mismatch_count(position_df)
    if critical_total > 0:
        return "CRITICAL_MISMATCH"
    if mismatch_total > 0:
        return "MISMATCH"
    return "MATCH"


def _critical_count(frame: pd.DataFrame) -> int:
    if frame.empty or "severity" not in frame.columns:
        return 0
    return int((frame["severity"].astype(str).str.lower() == "critical").sum())


def _mismatch_count(frame: pd.DataFrame) -> int:
    if frame.empty or "mismatch" not in frame.columns:
        return 0
    return int(frame["mismatch"].fillna(False).astype(bool).sum())


def _percent_reconciled(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    if "mismatch" not in frame.columns:
        return None
    matches = (~frame["mismatch"].fillna(False).astype(bool)).sum()
    return float(matches / len(frame))


def _to_optional_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
