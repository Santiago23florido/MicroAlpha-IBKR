from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import suppress
from pathlib import Path
from typing import Any, Sequence

from backtest.runner import run_backtest_stub
from broker.ib_client import IBClientError
from config import Settings, load_settings
from engine.runtime import RuntimeServices, build_runtime
from ingestion.collector import collect_market_data
from models.train_baseline import train_baseline_model
from models.train_deep import train_deep_model
from monitoring.healthcheck import build_healthcheck_report
from monitoring.sync import sync_data_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MicroAlpha-IBKR phase 1 foundation. "
            "Use development mode on PC1 for research/backtesting/training and deploy mode on PC2 for collection and operations."
        )
    )
    parser.add_argument("--env-file", default=".env", help="Path to the environment file.")
    parser.add_argument("--config-dir", default="config", help="Path to the YAML configuration directory.")
    parser.add_argument(
        "--environment",
        choices=["development", "deploy"],
        help="Override the target environment declared by config and .env.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Run one safe collection cycle and persist raw artifacts.")
    collect_parser.add_argument("--symbol", help="Override the configured symbol.")
    collect_parser.add_argument("--duration", default="1 D", help="Historical duration passed to IBKR.")
    collect_parser.add_argument("--bar-size", default="1 min", help="Historical bar size passed to IBKR.")
    collect_parser.add_argument("--output-root", help="Optional raw-data destination override.")

    train_parser = subparsers.add_parser("train", help="Train the baseline or deep model from a local dataset.")
    train_parser.add_argument("--model-type", required=True, choices=["baseline", "deep"])
    train_parser.add_argument("--data-path", help="CSV or Parquet dataset for research mode.")
    train_parser.add_argument("--model-name", help="Optional model display name override.")
    train_parser.add_argument("--epochs", type=int, default=6, help="Only used for deep training.")
    train_parser.add_argument("--no-set-active", action="store_true", help="Do not set the new artifact as active.")

    backtest_parser = subparsers.add_parser("backtest", help="Validate backtest inputs and dataset plumbing.")
    backtest_parser.add_argument("--data-path", help="CSV or Parquet dataset for research mode.")
    backtest_parser.add_argument("--symbol", help="Override the configured symbol.")

    session_parser = subparsers.add_parser(
        "run-session",
        aliases=["session-cycle", "session"],
        help="Run one ORB + feature + model + risk session cycle.",
    )
    session_parser.add_argument(
        "--paper",
        action="store_true",
        help="Request paper execution. Safety gates in config and risk still apply.",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        aliases=["launch-ui", "ui"],
        help="Launch the local Streamlit dashboard.",
    )
    dashboard_parser.add_argument("--host", help="Override the Streamlit host.")
    dashboard_parser.add_argument("--port", type=int, help="Override the Streamlit port.")

    healthcheck_parser = subparsers.add_parser(
        "healthcheck",
        aliases=["check-connection"],
        help="Inspect config, paths, and optionally broker connectivity.",
    )
    healthcheck_parser.add_argument("--skip-broker", action="store_true", help="Do not attempt the IBKR connection.")

    snapshot_parser = subparsers.add_parser("snapshot", help="Fetch the current market snapshot for one symbol.")
    snapshot_parser.add_argument("--symbol", help="Override the configured symbol.")

    latest_parser = subparsers.add_parser(
        "latest-decision",
        aliases=["explain-latest-decision"],
        help="Show the most recent stored decision.",
    )
    latest_parser.add_argument("--limit", type=int, default=1, help="Reserved for future expansion.")

    models_parser = subparsers.add_parser(
        "models",
        aliases=["list-models"],
        help="List active and registered model artifacts.",
    )
    models_parser.add_argument("--model-type", choices=["baseline", "deep"])

    sync_parser = subparsers.add_parser("sync-data", help="Prepare or execute a local sync of data artifacts.")
    sync_parser.add_argument("--destination-root", required=True, help="Destination root for the sync plan or copy.")
    sync_parser.add_argument("--execute", action="store_true", help="Perform the file copy instead of a dry-run plan.")

    subparsers.add_parser("show-config", aliases=["config"], help="Print the effective merged configuration.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(args.env_file, config_dir=args.config_dir, environment=args.environment)

    try:
        if args.command in {"show-config", "config"}:
            print_result(settings.as_dict())
            return 0

        if args.command in {"healthcheck", "check-connection"}:
            return handle_healthcheck(args, settings)

        if args.command == "backtest":
            print_result(
                run_backtest_stub(
                    settings,
                    data_path=args.data_path,
                    symbol=args.symbol,
                )
            )
            return 0

        if args.command == "sync-data":
            print_result(
                sync_data_artifacts(
                    settings,
                    destination_root=args.destination_root,
                    execute=args.execute,
                )
            )
            return 0

        runtime = build_runtime(
            env_file=args.env_file,
            config_dir=args.config_dir,
            environment=args.environment,
        )

        if args.command == "collect":
            print_result(
                collect_market_data(
                    runtime.settings,
                    runtime.client,
                    symbol=args.symbol,
                    duration=args.duration,
                    bar_size=args.bar_size,
                    output_root=args.output_root,
                )
            )
            return 0

        if args.command == "train":
            return handle_train(runtime, args)

        if args.command in {"run-session", "session-cycle", "session"}:
            print_result(runtime.session_engine.run_cycle(execute_requested=args.paper))
            return 0

        if args.command in {"dashboard", "launch-ui", "ui"}:
            return launch_dashboard(runtime.settings, env_file=args.env_file, config_dir=args.config_dir, environment=args.environment, host=args.host, port=args.port)

        if args.command == "snapshot":
            return with_connected_client(
                runtime,
                lambda: runtime.client.get_market_snapshot(
                    symbol=(args.symbol or runtime.settings.ib_symbol).upper(),
                    exchange=runtime.settings.ib_exchange,
                    currency=runtime.settings.ib_currency,
                ),
            )

        if args.command in {"latest-decision", "explain-latest-decision"}:
            print_result(runtime.session_engine.explain_latest_decision())
            return 0

        if args.command in {"models", "list-models"}:
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
    except IBClientError as exc:
        print_result({"status": "error", "message": str(exc)})
        return 1
    except NotImplementedError as exc:
        print_result({"status": "placeholder", "message": str(exc)})
        return 2
    except Exception as exc:  # pragma: no cover
        print_result(
            {
                "status": "error",
                "message": f"Unexpected failure. Check {settings.log_file}. Root cause: {exc}",
            }
        )
        return 1

    parser.print_help()
    return 2


def handle_healthcheck(args: argparse.Namespace, settings: Settings) -> int:
    if args.skip_broker:
        print_result(build_healthcheck_report(settings))
        return 0

    runtime = build_runtime(
        env_file=args.env_file,
        config_dir=args.config_dir,
        environment=args.environment,
    )
    print_result(build_healthcheck_report(runtime.settings, broker_client=runtime.client))
    return 0


def handle_train(runtime: RuntimeServices, args: argparse.Namespace) -> int:
    if args.model_type == "baseline":
        payload = train_baseline_model(
            runtime.settings,
            data_path=args.data_path,
            model_name=args.model_name or "baseline_logreg",
            set_active=not args.no_set_active,
        )
    else:
        payload = train_deep_model(
            runtime.settings,
            data_path=args.data_path,
            model_name=args.model_name or "deep_lob_lite",
            epochs=args.epochs,
            set_active=not args.no_set_active,
        )
    print_result(payload)
    return 0


def with_connected_client(runtime: RuntimeServices, callback) -> int:
    try:
        runtime.client.connect()
        print_result(callback())
        return 0
    finally:
        with suppress(Exception):
            runtime.client.disconnect()


def launch_dashboard(
    settings: Settings,
    *,
    env_file: str | None = None,
    config_dir: str | None = None,
    environment: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> int:
    bind_host = host or settings.ui.host
    bind_port = port or settings.ui.port
    browser_host = "127.0.0.1" if bind_host in {"0.0.0.0", "127.0.0.1"} else bind_host
    browser_url = f"http://{browser_host}:{bind_port}"

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/streamlit_app.py",
        "--server.address",
        bind_host,
        "--server.port",
        str(bind_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    env = os.environ.copy()
    if env_file:
        env["MICROALPHA_ENV_FILE"] = str(Path(env_file).resolve())
    if config_dir:
        env["MICROALPHA_CONFIG_DIR"] = str(Path(config_dir).resolve())
    if environment:
        env["MICROALPHA_ENV"] = environment

    print(f"Launching dashboard at {browser_url}", flush=True)
    print(f"If the browser does not open, open {browser_url} manually.", flush=True)
    _open_browser_soon(browser_url)
    return subprocess.run(command, env=env, check=False).returncode


def _open_browser_soon(url: str) -> None:
    def _worker() -> None:
        time.sleep(1.0)
        with suppress(Exception):
            webbrowser.open(url, new=2)

    threading.Thread(target=_worker, name="streamlit-browser-opener", daemon=True).start()


def print_result(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=False, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
