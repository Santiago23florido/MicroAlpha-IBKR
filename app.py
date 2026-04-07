from __future__ import annotations

import argparse
import json
from contextlib import suppress
from typing import Any, Callable

from broker.ib_client import IBClient, IBClientError
from broker.orders import create_market_order
from config import Settings, load_settings
from risk.risk_manager import RiskManager
from storage.logger import setup_logger
from strategy.basic_signal import BasicSignalStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safe MVP CLI for Interactive Brokers paper trading via IB Gateway."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-connection", help="Verify the IB Gateway paper connection.")
    subparsers.add_parser("server-time", help="Request current IB server time.")
    subparsers.add_parser("account-summary", help="Request the current account summary.")
    subparsers.add_parser("positions", help="Request current positions.")

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Request a market data snapshot for one symbol.",
    )
    snapshot_parser.add_argument("--symbol", help="Override the configured IB_SYMBOL.")

    order_parser = subparsers.add_parser(
        "paper-test-order",
        help="Explicitly request one manual paper test order.",
    )
    order_parser.add_argument("--symbol", help="Override the configured IB_SYMBOL.")
    order_parser.add_argument(
        "--action",
        required=True,
        choices=["BUY", "SELL"],
        help="Order side.",
    )
    order_parser.add_argument(
        "--quantity",
        required=True,
        type=int,
        help="Whole-share quantity to use for the paper test order.",
    )

    return parser


def print_result(payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=False))


def create_client() -> tuple[Settings, IBClient, RiskManager, BasicSignalStrategy]:
    settings = load_settings()
    logger = setup_logger(settings.log_level, settings.log_file)
    client = IBClient(
        host=settings.ib_host,
        port=settings.ib_port,
        client_id=settings.ib_client_id,
        logger=logger,
        request_timeout=settings.request_timeout_seconds,
        account_summary_group=settings.account_summary_group,
    )
    risk_manager = RiskManager(
        safe_to_trade=settings.safe_to_trade,
        dry_run=settings.dry_run,
    )
    strategy = BasicSignalStrategy()
    return settings, client, risk_manager, strategy


def with_connected_client(
    client: IBClient,
    callback: Callable[[], dict[str, Any] | list[dict[str, Any]]],
) -> int:
    try:
        client.connect()
        payload = callback()
        print_result(payload)
        return 0
    finally:
        with suppress(Exception):
            client.disconnect()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings, client, risk_manager, strategy = create_client()
    logger = client.logger

    strategy_result = strategy.generate_signal()
    logger.info("Strategy placeholder status: %s", strategy_result.reason)

    try:
        if args.command == "check-connection":
            return with_connected_client(
                client,
                lambda: {"connected": client.is_connected()},
            )

        if args.command == "server-time":
            return with_connected_client(client, client.get_server_time)

        if args.command == "account-summary":
            return with_connected_client(client, client.get_account_summary)

        if args.command == "positions":
            return with_connected_client(client, client.get_positions)

        if args.command == "snapshot":
            symbol = (args.symbol or settings.ib_symbol).upper()
            return with_connected_client(
                client,
                lambda: client.get_market_snapshot(symbol),
            )

        if args.command == "paper-test-order":
            symbol = (args.symbol or settings.ib_symbol).upper()
            action = args.action.upper()
            quantity = args.quantity
            decision = risk_manager.evaluate_manual_order(symbol, action, quantity)

            if not decision.approved:
                logger.warning(decision.reason)
                print_result(
                    {
                        "status": "blocked",
                        "reason": decision.reason,
                        "symbol": symbol,
                        "action": action,
                        "quantity": quantity,
                    }
                )
                return 2

            if not decision.submit_to_broker:
                preview_order = create_market_order(action, quantity)
                logger.info("Dry-run paper order preview generated for %s.", symbol)
                print_result(
                    {
                        "status": "dry-run",
                        "reason": decision.reason,
                        "symbol": symbol,
                        "action": preview_order.action,
                        "quantity": preview_order.totalQuantity,
                        "order_type": preview_order.orderType,
                    }
                )
                return 0

            return with_connected_client(
                client,
                lambda: client.submit_paper_test_order(symbol, action, quantity),
            )
    except IBClientError as exc:
        logger.error(str(exc))
        print_result({"status": "error", "message": str(exc)})
        return 1
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected application failure.")
        print_result(
            {
                "status": "error",
                "message": (
                    "Unexpected failure. Check logs/ibkr_mvp.log for details. "
                    f"Root cause: {exc}"
                ),
            }
        )
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
