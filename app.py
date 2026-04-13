from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from broker.ib_client import IBClientError
from broker.orders import (
    calculate_marketable_limit_price,
    create_bracket_order,
    create_limit_order,
    create_market_order,
)
from config import Settings
from engine.runtime import RuntimeServices, build_runtime
from models.train_baseline import train_baseline_model
from models.train_deep import train_deep_model
from risk.risk_manager import ExecutionRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MicroAlpha IBKR ORB + microstructure paper-trading system.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("test-connection", aliases=["check-connection"], help="Verify the IB Gateway paper connection.")
    subparsers.add_parser("server-time", help="Request current IB server time.")
    subparsers.add_parser("account-summary", help="Request the current account summary.")
    subparsers.add_parser("positions", help="Request current positions.")
    subparsers.add_parser("open-orders", help="Request currently open orders.")

    snapshot_parser = subparsers.add_parser(
        "market-snapshot",
        aliases=["snapshot"],
        help="Request the latest market snapshot for the configured symbol.",
    )
    snapshot_parser.add_argument("--symbol", help="Override the configured symbol for data only.")

    session_parser = subparsers.add_parser(
        "run-session",
        aliases=["session-cycle", "session"],
        help="Run one ORB + model + risk session cycle.",
    )
    session_parser.add_argument(
        "--paper",
        action="store_true",
        help="Explicitly request paper execution if all safety gates allow it.",
    )
    session_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Keep the session read-only. This is the default safe behavior.",
    )

    baseline_parser = subparsers.add_parser("train-baseline", help="Train the interpretable baseline model.")
    baseline_parser.add_argument("--data-path", help="CSV or Parquet dataset for research mode.")
    baseline_parser.add_argument("--model-name", default="baseline_logreg")
    baseline_parser.add_argument("--no-set-active", action="store_true")

    deep_parser = subparsers.add_parser("train-deep", help="Train the temporal deep model.")
    deep_parser.add_argument("--data-path", help="CSV or Parquet dataset for research mode.")
    deep_parser.add_argument("--model-name", default="deep_lob_lite")
    deep_parser.add_argument("--epochs", type=int, default=6)
    deep_parser.add_argument("--no-set-active", action="store_true")

    list_parser = subparsers.add_parser(
        "list-models",
        aliases=["models"],
        help="List registered baseline and deep models.",
    )
    list_parser.add_argument("--model-type", choices=["baseline", "deep"])

    set_active_parser = subparsers.add_parser("set-active-model", help="Select the active baseline or deep model.")
    set_active_parser.add_argument("--model-type", required=True, choices=["baseline", "deep"])
    set_active_parser.add_argument("--artifact-id", required=True)

    subparsers.add_parser(
        "explain-latest-decision",
        aliases=["latest-decision"],
        help="Show the most recent stored decision.",
    )

    place_test_parser = subparsers.add_parser(
        "place-test-order",
        aliases=["test-order"],
        help="Intentionally place or preview one tiny marketable-limit paper order.",
    )
    _add_common_order_arguments(place_test_parser)
    place_test_parser.add_argument(
        "--confirm-paper",
        action="store_true",
        help="Required before a real paper order can be sent when all safety flags allow it.",
    )

    cancel_parser = subparsers.add_parser("cancel-order", help="Cancel one tracked order by id.")
    cancel_parser.add_argument("--order-id", required=True, type=int)

    close_parser = subparsers.add_parser("close-position", help="Manually close the configured symbol position.")
    close_parser.add_argument("--symbol", help="Override the configured symbol.")

    launch_parser = subparsers.add_parser(
        "launch-ui",
        aliases=["ui"],
        help="Launch the local Streamlit UI.",
    )
    launch_parser.add_argument("--host", help="Override the UI host.")
    launch_parser.add_argument("--port", type=int, help="Override the UI port.")

    market_order_parser = subparsers.add_parser("market-order", help="Legacy compatibility market order command.")
    _add_common_order_arguments(market_order_parser)

    limit_order_parser = subparsers.add_parser("limit-order", help="Legacy compatibility limit order command.")
    _add_common_order_arguments(limit_order_parser)
    limit_order_parser.add_argument("--limit-price", required=True, type=float)

    bracket_order_parser = subparsers.add_parser("bracket-order", help="Legacy compatibility bracket order command.")
    _add_common_order_arguments(bracket_order_parser)
    bracket_order_parser.add_argument("--entry-limit", required=True, type=float)
    bracket_order_parser.add_argument("--take-profit", required=True, type=float)
    bracket_order_parser.add_argument("--stop-loss", required=True, type=float)

    return parser


def _add_common_order_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", help="Override the configured symbol.")
    parser.add_argument("--action", required=True, choices=["BUY", "SELL"], help="Order side.")
    parser.add_argument("--quantity", type=int, help="Whole-share quantity.")


def print_result(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=False, default=str))


def handle_place_test_order(runtime: RuntimeServices, args: argparse.Namespace) -> int:
    settings = runtime.settings
    symbol = (args.symbol or settings.ib_symbol).upper()
    quantity = args.quantity or settings.default_order_quantity
    try:
        runtime.client.connect()
        snapshot = runtime.client.get_market_snapshot(
            symbol=symbol,
            exchange=settings.ib_exchange,
            currency=settings.ib_currency,
        )
        limit_price = calculate_marketable_limit_price(
            action=args.action,
            bid=_coerce_float(snapshot.get("bid")),
            ask=_coerce_float(snapshot.get("ask")),
            last=_coerce_float(snapshot.get("last")),
            buffer_bps=settings.trading.entry_limit_buffer_bps,
        )
        request = ExecutionRequest(
            symbol=symbol,
            action=args.action.upper(),
            quantity=quantity,
            order_type="limit",
            explicit_command=True,
            limit_price=limit_price,
        )
        decision = runtime.risk_manager.evaluate_execution_request(request)
        preview = {
            "symbol": symbol,
            "action": args.action.upper(),
            "quantity": quantity,
            "order_type": "marketable_limit",
            "limit_price": limit_price,
            "source_bid": snapshot.get("bid"),
            "source_ask": snapshot.get("ask"),
            "source_last": snapshot.get("last"),
        }
        if not decision.approved:
            print_result({"status": "blocked", "reason": decision.reason, "preview": preview})
            return 2
        if not decision.submit_to_broker:
            print_result({"status": "dry-run", "reason": decision.reason, "preview": preview})
            return 0
        if not args.confirm_paper:
            print_result(
                {
                    "status": "blocked",
                    "reason": (
                        "Real paper submission requires --confirm-paper in addition to SAFE_TO_TRADE=true and DRY_RUN=false."
                    ),
                    "preview": preview,
                }
            )
            return 2
        payload = runtime.client.submit_marketable_limit_order(
            symbol=symbol,
            action=args.action.upper(),
            quantity=quantity,
            bid=_coerce_float(snapshot.get("bid")),
            ask=_coerce_float(snapshot.get("ask")),
            last=_coerce_float(snapshot.get("last")),
            buffer_bps=settings.trading.entry_limit_buffer_bps,
            exchange=settings.ib_exchange,
            currency=settings.ib_currency,
        )
        runtime.trade_store.append_trade(
            _trade_event(
                symbol=symbol,
                event_type="manual_test_order_submitted",
                action=args.action.upper(),
                quantity=quantity,
                status=str(payload.get("status", "Submitted")),
                order_id=payload.get("order_id"),
                price=limit_price,
                message="Manual marketable-limit paper order submitted.",
                payload=payload,
            )
        )
        print_result(payload)
        return 0
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def handle_legacy_order(runtime: RuntimeServices, args: argparse.Namespace) -> int:
    symbol = (args.symbol or runtime.settings.ib_symbol).upper()
    quantity = args.quantity or runtime.settings.default_order_quantity
    if args.command == "market-order":
        request = ExecutionRequest(symbol, args.action.upper(), quantity, "market", True)
    elif args.command == "limit-order":
        request = ExecutionRequest(
            symbol,
            args.action.upper(),
            quantity,
            "limit",
            True,
            limit_price=float(args.limit_price),
        )
    else:
        request = ExecutionRequest(
            symbol,
            args.action.upper(),
            quantity,
            "bracket",
            True,
            limit_price=float(args.entry_limit),
            take_profit_price=float(args.take_profit),
            stop_loss_price=float(args.stop_loss),
        )
    decision = runtime.risk_manager.evaluate_execution_request(request)
    if not decision.approved:
        print_result({"status": "blocked", "reason": decision.reason, "request": request.__dict__})
        return 2
    if not decision.submit_to_broker:
        print_result({"status": "dry-run", "reason": decision.reason, "preview": preview_legacy_order(request)})
        return 0

    try:
        runtime.client.connect()
        if request.order_type == "market":
            payload = runtime.client.submit_market_order(
                symbol=symbol,
                action=request.action,
                quantity=quantity,
                exchange=runtime.settings.ib_exchange,
                currency=runtime.settings.ib_currency,
            )
        elif request.order_type == "limit":
            payload = runtime.client.submit_limit_order(
                symbol=symbol,
                action=request.action,
                quantity=quantity,
                limit_price=float(request.limit_price),
                exchange=runtime.settings.ib_exchange,
                currency=runtime.settings.ib_currency,
            )
        else:
            payload = runtime.client.submit_bracket_order(
                symbol=symbol,
                action=request.action,
                quantity=quantity,
                entry_limit_price=float(request.limit_price),
                take_profit_price=float(request.take_profit_price),
                stop_loss_price=float(request.stop_loss_price),
                exchange=runtime.settings.ib_exchange,
                currency=runtime.settings.ib_currency,
            )
        print_result(payload)
        return 0
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def handle_close_position(runtime: RuntimeServices, symbol: str) -> int:
    try:
        runtime.client.connect()
        positions = runtime.client.get_positions()
        matching_position = next(
            (row for row in positions if row["symbol"].upper() == symbol.upper()),
            None,
        )
        if matching_position is None:
            print_result({"status": "error", "message": f"No position found for {symbol.upper()}."})
            return 1
        decision = runtime.risk_manager.evaluate_position_close(
            symbol=symbol,
            position_quantity=float(matching_position["position"]),
            explicit_command=True,
        )
        if not decision.approved:
            print_result({"status": "blocked", "reason": decision.reason, "position": matching_position})
            return 2
        if not decision.submit_to_broker:
            preview = create_market_order(
                "SELL" if float(matching_position["position"]) > 0 else "BUY",
                int(abs(float(matching_position["position"]))),
            )
            print_result(
                {
                    "status": "dry-run",
                    "reason": decision.reason,
                    "position": matching_position,
                    "preview": {
                        "action": preview.action,
                        "quantity": preview.totalQuantity,
                        "order_type": preview.orderType,
                    },
                }
            )
            return 0
        payload = runtime.client.close_position(
            symbol=symbol.upper(),
            exchange=runtime.settings.ib_exchange,
            currency=runtime.settings.ib_currency,
        )
        print_result(payload)
        return 0
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def preview_legacy_order(request: ExecutionRequest) -> dict[str, Any]:
    if request.order_type == "market":
        order = create_market_order(request.action, request.quantity)
        return {"order_type": order.orderType, "action": order.action, "quantity": order.totalQuantity}
    if request.order_type == "limit":
        order = create_limit_order(request.action, request.quantity, float(request.limit_price))
        return {
            "order_type": order.orderType,
            "action": order.action,
            "quantity": order.totalQuantity,
            "limit_price": order.lmtPrice,
        }
    orders = create_bracket_order(
        parent_order_id=1,
        action=request.action,
        quantity=request.quantity,
        entry_limit_price=float(request.limit_price),
        take_profit_price=float(request.take_profit_price),
        stop_loss_price=float(request.stop_loss_price),
    )
    return {
        "order_type": "BRACKET",
        "legs": [
            {
                "order_id": order.orderId,
                "parent_id": getattr(order, "parentId", 0) or None,
                "action": order.action,
                "quantity": order.totalQuantity,
                "type": order.orderType,
                "limit_price": getattr(order, "lmtPrice", None) or None,
                "stop_price": getattr(order, "auxPrice", None) or None,
                "transmit": order.transmit,
            }
            for order in orders
        ],
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    runtime = build_runtime()
    try:
        if args.command in {"test-connection", "check-connection"}:
            print_result(runtime.session_engine.test_connection())
            return 0

        if args.command == "server-time":
            return with_connected_client(runtime, lambda: runtime.client.get_server_time())

        if args.command == "account-summary":
            return with_connected_client(runtime, lambda: runtime.client.get_account_summary())

        if args.command == "positions":
            return with_connected_client(runtime, lambda: runtime.client.get_positions())

        if args.command == "open-orders":
            return with_connected_client(runtime, lambda: runtime.client.get_open_orders())

        if args.command in {"market-snapshot", "snapshot"}:
            symbol = (args.symbol or runtime.settings.ib_symbol).upper()
            return with_connected_client(
                runtime,
                lambda: runtime.client.get_market_snapshot(
                    symbol=symbol,
                    exchange=runtime.settings.ib_exchange,
                    currency=runtime.settings.ib_currency,
                ),
            )

        if args.command in {"run-session", "session-cycle", "session"}:
            payload = runtime.session_engine.run_cycle(execute_requested=bool(args.paper and not args.dry_run))
            print_result(payload)
            return 0

        if args.command == "train-baseline":
            payload = train_baseline_model(
                runtime.settings,
                data_path=args.data_path,
                model_name=args.model_name,
                set_active=not args.no_set_active,
            )
            print_result(payload)
            return 0

        if args.command == "train-deep":
            payload = train_deep_model(
                runtime.settings,
                data_path=args.data_path,
                model_name=args.model_name,
                epochs=args.epochs,
                set_active=not args.no_set_active,
            )
            print_result(payload)
            return 0

        if args.command in {"list-models", "models"}:
            print_result(
                {
                    "active": {
                        "baseline": runtime.model_registry.get_active_model("baseline"),
                        "deep": runtime.model_registry.get_active_model("deep"),
                    },
                    "models": runtime.model_registry.list_models(args.model_type),
                }
            )
            return 0

        if args.command == "set-active-model":
            payload = runtime.model_registry.set_active_model(args.model_type, args.artifact_id)
            print_result(payload)
            return 0

        if args.command in {"explain-latest-decision", "latest-decision"}:
            print_result(runtime.session_engine.explain_latest_decision())
            return 0

        if args.command in {"place-test-order", "test-order"}:
            return handle_place_test_order(runtime, args)

        if args.command in {"market-order", "limit-order", "bracket-order"}:
            return handle_legacy_order(runtime, args)

        if args.command == "cancel-order":
            return with_connected_client(runtime, lambda: runtime.client.cancel_order(args.order_id))

        if args.command == "close-position":
            symbol = (args.symbol or runtime.settings.ib_symbol).upper()
            return handle_close_position(runtime, symbol)

        if args.command in {"launch-ui", "ui"}:
            return launch_ui(runtime.settings, host=args.host, port=args.port)
    except IBClientError as exc:
        runtime.client.logger.error(str(exc))
        print_result({"status": "error", "message": str(exc)})
        return 1
    except Exception as exc:  # pragma: no cover
        runtime.client.logger.exception("Unexpected application failure.")
        print_result(
            {
                "status": "error",
                "message": f"Unexpected failure. Check {runtime.settings.log_file}. Root cause: {exc}",
            }
        )
        return 1
    return 2


def with_connected_client(runtime: RuntimeServices, callback) -> int:
    try:
        runtime.client.connect()
        print_result(callback())
        return 0
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def launch_ui(settings: Settings, *, host: str | None = None, port: int | None = None) -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/streamlit_app.py",
        "--server.address",
        host or settings.ui.host,
        "--server.port",
        str(port or settings.ui.port),
    ]
    return subprocess.run(command, check=False).returncode


def _trade_event(
    *,
    symbol: str,
    event_type: str,
    action: str,
    quantity: float,
    status: str,
    message: str,
    payload: dict[str, Any],
    order_id: int | None = None,
    price: float | None = None,
):
    from data.schemas import TradeLifecycleEvent

    return TradeLifecycleEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol=symbol,
        event_type=event_type,
        action=action,
        quantity=quantity,
        status=status,
        order_id=order_id,
        price=price,
        message=message,
        payload=payload,
    )


def _coerce_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
