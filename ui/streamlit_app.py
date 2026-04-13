from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from broker.orders import calculate_marketable_limit_price
from data.schemas import TradeLifecycleEvent
from engine.runtime import build_runtime
from reporting.performance_report import build_performance_report
from reporting.trade_report import build_trade_report
from risk.risk_manager import ExecutionRequest
from ui.components.common import render_kv_table, render_records_table


st.set_page_config(page_title="MicroAlpha IBKR", layout="wide")
runtime = build_runtime()
settings = runtime.settings


def main() -> None:
    st.title(settings.ui.title)
    st.caption("Local-only SPY ORB + microstructure paper-trading console.")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Market", "Models", "Decisions", "Trades", "Controls"],
    )
    st.sidebar.markdown(
        "\n".join(
            [
                f"`Symbol:` {settings.ib_symbol}",
                f"`DRY_RUN:` {settings.dry_run}",
                f"`SAFE_TO_TRADE:` {settings.safe_to_trade}",
                f"`DATA_MODE:` {settings.trading.data_mode}",
            ]
        )
    )

    if page == "Overview":
        render_overview()
    elif page == "Market":
        render_market()
    elif page == "Models":
        render_models()
    elif page == "Decisions":
        render_decisions()
    elif page == "Trades":
        render_trades()
    else:
        render_controls()


def render_overview() -> None:
    left, right = st.columns(2)
    with left:
        render_kv_table(
            "Mode",
            {
                "symbol": settings.ib_symbol,
                "dry_run": settings.dry_run,
                "safe_to_trade": settings.safe_to_trade,
                "allow_shorts": settings.trading.allow_shorts,
                "data_mode": settings.trading.data_mode,
            },
        )
    with right:
        try:
            payload = runtime.session_engine.test_connection()
            st.success("IB Gateway paper connection is healthy." if payload["connected"] else "IB Gateway is not connected.")
        except Exception as exc:
            st.error(f"Connection check failed: {exc}")

    if st.button("Refresh Overview Data", type="primary"):
        with st.spinner("Fetching live paper account state..."):
            data = fetch_live_overview()
            st.session_state["overview_data"] = data
    overview_data = st.session_state.get("overview_data", {})
    render_records_table("Account Summary", overview_data.get("account_summary", []))
    render_records_table("Open Positions", overview_data.get("positions", []))


def render_market() -> None:
    if st.button("Refresh Market View", type="primary"):
        with st.spinner("Running safe market refresh..."):
            st.session_state["market_data"] = runtime.session_engine.run_cycle(execute_requested=False)
    payload = st.session_state.get("market_data")
    if not payload:
        latest = runtime.session_engine.explain_latest_decision()
        if latest and latest.get("market_status"):
            payload = {
                "market_status": latest.get("market_status"),
                "market_snapshot": latest.get("structured_explanation_data", {}).get("orb", {}),
                "orb_state": latest.get("orb_state"),
                "feature_snapshot": {"feature_values": latest.get("feature_values", {})},
            }
    if not payload:
        st.info("Run a safe market refresh to populate live ORB and feature data.")
        return
    render_kv_table("Market Status", payload.get("market_status", {}))
    render_kv_table("Market Snapshot", payload.get("market_snapshot", {}))
    render_kv_table("ORB State", payload.get("orb_state", {}))
    feature_values = payload.get("feature_snapshot", {}).get("feature_values", {})
    render_kv_table("Latest Feature Values", feature_values)


def render_models() -> None:
    render_kv_table(
        "Active Models",
        {
            "baseline": runtime.model_registry.get_active_model("baseline"),
            "deep": runtime.model_registry.get_active_model("deep"),
            "artifacts_dir": settings.models.artifacts_dir,
            "registry_path": settings.models.registry_path,
        },
    )
    render_records_table("Registered Models", runtime.model_registry.list_models())


def render_decisions() -> None:
    latest = runtime.session_engine.explain_latest_decision()
    if latest and latest.get("status") != "empty":
        st.subheader("Latest Decision")
        st.json(latest, expanded=False)
    else:
        st.info("No decision stored yet. Run a safe session cycle first.")
    render_records_table("Recent Decisions", runtime.session_engine.list_recent_decisions(limit=20))


def render_trades() -> None:
    trades = runtime.session_engine.list_recent_trades(limit=50)
    executions = runtime.session_engine.list_recent_execution_events(limit=50)
    render_kv_table("Trade Summary", build_trade_report(trades))
    render_kv_table("Performance Summary", build_performance_report(trades))
    render_records_table("Recent Trades", trades)
    render_records_table("Execution Events", executions)


def render_controls() -> None:
    st.warning(
        "Paper actions remain blocked unless SAFE_TO_TRADE=true, DRY_RUN=false, and you confirm the action."
    )
    if st.button("Run Safe Session Cycle", type="primary"):
        with st.spinner("Running non-executing session cycle..."):
            st.session_state["controls_session_result"] = runtime.session_engine.run_cycle(execute_requested=False)
    if "controls_session_result" in st.session_state:
        st.json(st.session_state["controls_session_result"], expanded=False)

    st.subheader("Intentional Tiny Paper Test Order")
    order_action = st.selectbox("Action", ["BUY", "SELL"])
    order_quantity = st.number_input("Quantity", min_value=1, value=settings.default_order_quantity, step=1)
    confirm_paper = st.checkbox("I intentionally want a paper order and understand the risks.")
    if st.button("Place Test Order"):
        st.session_state["test_order_result"] = place_test_order(
            action=order_action,
            quantity=int(order_quantity),
            confirm_paper=confirm_paper,
        )
    if "test_order_result" in st.session_state:
        st.json(st.session_state["test_order_result"], expanded=False)

    st.subheader("Manual Cancel / Close")
    cancel_order_id = st.number_input("Order ID to cancel", min_value=0, value=0, step=1)
    confirm_cancel = st.checkbox("Confirm cancel request")
    if st.button("Cancel Order"):
        st.session_state["cancel_result"] = cancel_order(int(cancel_order_id), confirm_cancel)
    if "cancel_result" in st.session_state:
        st.json(st.session_state["cancel_result"], expanded=False)

    confirm_close = st.checkbox("Confirm manual close request")
    if st.button(f"Close {settings.ib_symbol} Position"):
        st.session_state["close_result"] = close_position(confirm_close)
    if "close_result" in st.session_state:
        st.json(st.session_state["close_result"], expanded=False)


def fetch_live_overview() -> dict[str, Any]:
    try:
        runtime.client.connect()
        return {
            "account_summary": runtime.client.get_account_summary(),
            "positions": runtime.client.get_positions(),
        }
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def place_test_order(*, action: str, quantity: int, confirm_paper: bool) -> dict[str, Any]:
    symbol = settings.ib_symbol
    try:
        runtime.client.connect()
        snapshot = runtime.client.get_market_snapshot(
            symbol=symbol,
            exchange=settings.ib_exchange,
            currency=settings.ib_currency,
        )
        limit_price = calculate_marketable_limit_price(
            action=action,
            bid=_coerce_float(snapshot.get("bid")),
            ask=_coerce_float(snapshot.get("ask")),
            last=_coerce_float(snapshot.get("last")),
            buffer_bps=settings.trading.entry_limit_buffer_bps,
        )
        request = ExecutionRequest(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type="limit",
            explicit_command=True,
            limit_price=limit_price,
        )
        risk_decision = runtime.risk_manager.evaluate_execution_request(request)
        preview = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_type": "marketable_limit",
            "limit_price": limit_price,
        }
        if not risk_decision.approved:
            return {"status": "blocked", "reason": risk_decision.reason, "preview": preview}
        if not risk_decision.submit_to_broker:
            return {"status": "dry-run", "reason": risk_decision.reason, "preview": preview}
        if not confirm_paper:
            return {
                "status": "blocked",
                "reason": "UI paper submission requires the confirmation checkbox.",
                "preview": preview,
            }
        payload = runtime.client.submit_marketable_limit_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            bid=_coerce_float(snapshot.get("bid")),
            ask=_coerce_float(snapshot.get("ask")),
            last=_coerce_float(snapshot.get("last")),
            buffer_bps=settings.trading.entry_limit_buffer_bps,
            exchange=settings.ib_exchange,
            currency=settings.ib_currency,
        )
        runtime.trade_store.append_trade(
            TradeLifecycleEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                event_type="manual_test_order_submitted",
                action=action,
                quantity=quantity,
                status=str(payload.get("status", "Submitted")),
                order_id=payload.get("order_id"),
                price=limit_price,
                message="Manual UI paper test order submitted.",
                payload=payload,
            )
        )
        return payload
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def cancel_order(order_id: int, confirmed: bool) -> dict[str, Any]:
    if not confirmed:
        return {"status": "blocked", "reason": "Cancel confirmation checkbox is required."}
    try:
        runtime.client.connect()
        return runtime.client.cancel_order(order_id)
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def close_position(confirmed: bool) -> dict[str, Any]:
    if not confirmed:
        return {"status": "blocked", "reason": "Close confirmation checkbox is required."}
    try:
        runtime.client.connect()
        return runtime.client.close_position(
            symbol=settings.ib_symbol,
            exchange=settings.ib_exchange,
            currency=settings.ib_currency,
        )
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def _coerce_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


main()
