from __future__ import annotations

import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import load_settings
from execution.kraken_paper import run_kraken_paper_sim
from ingestion.lob_capture import lob_capture_status


st.set_page_config(page_title="Kraken Paper Broker", layout="wide")


def main() -> None:
    settings = load_settings(
        os.environ.get("MICROALPHA_ENV_FILE", ".env"),
        config_dir=os.environ.get("MICROALPHA_CONFIG_DIR", "config"),
        environment=os.environ.get("MICROALPHA_ENV"),
    )
    defaults = _ui_defaults(settings)
    st.title("Kraken Paper Broker")
    st.caption("Simulación local sobre LOB capturado. No envía órdenes reales a Kraken.")

    with st.sidebar:
        st.header("Simulación")
        symbol = st.text_input("Symbol", value=defaults["symbol"])
        model_artifact = st.text_input("Model artifact", value=defaults["model_artifact"])
        mode = st.selectbox("Mode", ["live", "replay"], index=0 if defaults["mode"] == "live" else 1)
        duration_minutes = st.number_input(
            "Duration minutes",
            min_value=1.0,
            value=float(defaults["duration_minutes"]),
            step=5.0,
        )
        from_date = st.text_input("From date YYYY-MM-DD", value=defaults["from_date"])
        auto_refresh = st.checkbox("Auto refresh", value=mode == "live")
        refresh_seconds = st.number_input(
            "Refresh seconds",
            min_value=1.0,
            value=float(settings.kraken_lob.paper_ui_refresh_seconds),
            step=1.0,
        )
        run_button = st.button("Refresh Simulation", type="primary")

    run_id = st.session_state.setdefault(
        "kraken_paper_run_id",
        f"ui-{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%S')}",
    )
    if run_button or auto_refresh or "kraken_paper_payload" not in st.session_state:
        with st.spinner("Running local Kraken paper simulation..."):
            try:
                st.session_state["kraken_paper_payload"] = run_kraken_paper_sim(
                    settings,
                    symbol=symbol,
                    model_artifact=model_artifact,
                    duration_minutes=float(duration_minutes),
                    from_date=from_date.strip() or None,
                    run_id=run_id,
                )
                st.session_state.pop("kraken_paper_error", None)
            except Exception as exc:
                st.session_state["kraken_paper_error"] = str(exc)

    _render_capture_status(settings, symbol)
    if "kraken_paper_error" in st.session_state:
        st.error(st.session_state["kraken_paper_error"])
        st.info("Necesitas datos LOB capturados y un modelo deep activo antes de ejecutar la simulación.")
        _maybe_autorefresh(auto_refresh, refresh_seconds)
        return

    payload = dict(st.session_state.get("kraken_paper_payload") or {})
    if not payload:
        st.info("Ejecuta Refresh Simulation para cargar el paper broker.")
        _maybe_autorefresh(auto_refresh, refresh_seconds)
        return

    _render_account(payload)
    _render_market_and_model(payload)
    _render_risk(payload)
    _render_reports(payload)

    if mode == "live":
        st.caption("Live mode: la página relee los chunks locales. Si `start-lob-capture` está corriendo, verá datos nuevos.")
    _maybe_autorefresh(auto_refresh, refresh_seconds)


def _ui_defaults(settings: Any) -> dict[str, Any]:
    return {
        "symbol": os.environ.get("KRAKEN_PAPER_UI_SYMBOL", settings.kraken_lob.symbol),
        "model_artifact": os.environ.get("KRAKEN_PAPER_UI_MODEL_ARTIFACT", "active"),
        "mode": os.environ.get("KRAKEN_PAPER_UI_MODE", "live"),
        "duration_minutes": float(os.environ.get("KRAKEN_PAPER_UI_DURATION_MINUTES", "60")),
        "from_date": os.environ.get("KRAKEN_PAPER_UI_FROM_DATE", ""),
    }


def _render_capture_status(settings: Any, symbol: str) -> None:
    with suppress(Exception):
        status = lob_capture_status(settings, symbol=symbol, provider="kraken")
        state = status.get("state", {}) or {}
        label = f"Capture: {state.get('status', 'unknown')} | rows={state.get('row_count', 0)} | pid_alive={state.get('pid_alive', False)}"
        if state.get("status") == "running":
            st.success(label)
        else:
            st.warning(label)


def _render_account(payload: dict[str, Any]) -> None:
    st.subheader("Cuenta Simulada")
    cols = st.columns(7)
    cols[0].metric("Initial EUR", _money(payload.get("initial_cash_eur")))
    cols[1].metric("Cash EUR", _money(payload.get("final_cash_eur")))
    cols[2].metric("BTC Position", f"{float(payload.get('open_position_qty', 0.0)):.8f}")
    cols[3].metric("Broker Realistic Balance", _money(payload.get("broker_realistic_balance_eur")))
    cols[4].metric("Net PnL EUR", _money(payload.get("net_pnl_eur")), delta=f"{float(payload.get('net_pnl_pct', 0.0)) * 100:.3f}%")
    cols[5].metric("Fees EUR", _money(payload.get("total_fees_eur")))
    cols[6].metric("Trades", str(payload.get("trades", 0)))
    st.caption(f"{payload.get('note', '')} Result is net of configured fees, spread/slippage assumptions, and open-position mark-to-market.")
    warnings = ((payload.get("policy") or {}).get("warnings") or [])
    for warning in warnings:
        st.warning(warning)


def _render_market_and_model(payload: dict[str, Any]) -> None:
    market = payload.get("latest_market") or {}
    decision = payload.get("latest_decision") or {}
    left, right = st.columns([1.1, 1.0])
    with left:
        st.subheader("Order Book")
        cols = st.columns(4)
        cols[0].metric("Bid", _money(market.get("bid")))
        cols[1].metric("Ask", _money(market.get("ask")))
        cols[2].metric("Mid", _money(market.get("mid")))
        cols[3].metric("Spread bps", f"{float(market.get('spread_bps', 0.0)):.2f}")
        book = pd.DataFrame(market.get("levels") or [])
        if not book.empty:
            st.dataframe(book, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Modelo")
        st.metric("Predicción", str(decision.get("predicted_label", "n/a")))
        st.metric("Confianza", f"{float(decision.get('confidence', 0.0)):.4f}")
        st.metric("Acción simulada", str(decision.get("action", "n/a")))
        st.info(f"Motivo: {decision.get('reason', 'n/a')}")
        probs = pd.DataFrame(
            [
                {"class": "down", "probability": decision.get("prob_down", 0.0)},
                {"class": "stationary", "probability": decision.get("prob_stationary", 0.0)},
                {"class": "up", "probability": decision.get("prob_up", 0.0)},
            ]
        )
        st.bar_chart(probs.set_index("class"))


def _render_risk(payload: dict[str, Any]) -> None:
    st.subheader("Políticas")
    policy = payload.get("policy") or {}
    rows = [
        {"policy": "capital_mode", "value": policy.get("initial_cash_mode")},
        {"policy": "minimum_order_base", "value": policy.get("minimum_order_base")},
        {"policy": "minimum_order_notional_eur", "value": policy.get("minimum_order_notional_eur")},
        {"policy": "position_fraction", "value": policy.get("position_fraction")},
        {"policy": "estimated_roundtrip_cost_bps", "value": policy.get("estimated_roundtrip_cost_bps")},
        {"policy": "edge_buffer_bps", "value": policy.get("edge_buffer_bps")},
        {"policy": "model_prob_threshold", "value": policy.get("model_prob_threshold")},
        {"policy": "max_trades_per_day", "value": policy.get("max_trades_per_day")},
        {"policy": "max_daily_loss_pct", "value": policy.get("max_daily_loss_pct")},
        {"policy": "max_open_positions", "value": policy.get("max_open_positions")},
        {"policy": "spot_only", "value": "true"},
        {"policy": "shorts", "value": "disabled"},
        {"policy": "margin", "value": "disabled"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_reports(payload: dict[str, Any]) -> None:
    st.subheader("Reportes")
    paths = {
        "summary": payload.get("report_path"),
        "trades": payload.get("trades_path"),
        "decisions": payload.get("decisions_path"),
        "equity": payload.get("equity_path"),
    }
    st.json(paths, expanded=False)
    equity = _read_csv(paths.get("equity"))
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="coerce")
        chart_columns = [column for column in ["equity_eur", "net_pnl_eur"] if column in equity.columns]
        st.line_chart(equity.dropna(subset=["timestamp"]).set_index("timestamp")[chart_columns])
    trades = _read_csv(paths.get("trades"))
    decisions = _read_csv(paths.get("decisions"))
    left, right = st.columns(2)
    with left:
        st.write("Trades")
        st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
    with right:
        st.write("Últimas decisiones")
        visible_columns = [
            "timestamp",
            "predicted_label",
            "confidence",
            "action",
            "reason",
            "cash_eur",
            "position_qty",
            "equity_eur",
            "net_pnl_eur",
            "predicted_return_bps",
            "required_edge_bps",
        ]
        existing = [column for column in visible_columns if column in decisions.columns]
        st.dataframe(decisions[existing].tail(100), use_container_width=True, hide_index=True)


def _read_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    candidate = Path(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(candidate)


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "0.00"


def _maybe_autorefresh(enabled: bool, refresh_seconds: float) -> None:
    if enabled:
        time.sleep(float(refresh_seconds))
        st.rerun()


if __name__ == "__main__":
    main()
