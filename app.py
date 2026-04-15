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
from deployment.lan_sync import pull_from_pc2
from engine.runtime import RuntimeServices, build_runtime
from features.feature_pipeline import (
    inspect_feature_dependencies_for_build,
    list_available_feature_sets,
    run_feature_build_pipeline,
)
from features.validation import validate_feature_store
from ingestion.collector import collect_market_data
from ingestion.ibkr_client import build_collector_ib_client
from models.train_baseline import train_baseline_model
from models.train_deep import train_deep_model
from monitoring.data_quality import validate_imports
from monitoring.healthcheck import build_healthcheck_report
from monitoring.sync import sync_data_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MicroAlpha-IBKR LAN data pipeline with configurable feature sets and dependency-aware feature engineering. "
            "Use deploy mode on PC2 for collection and development mode on PC1 for LAN pull, import validation, and feature generation."
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

    collect_parser = subparsers.add_parser(
        "collect",
        help="Run the PC2 market data collector with polling, buffering and parquet persistence.",
    )
    collect_parser.add_argument("--symbol", help="Override the configured symbol.")
    collect_parser.add_argument("--symbols", nargs="+", help="Override the configured symbol universe.")
    collect_parser.add_argument("--once", action="store_true", help="Run one polling cycle and exit.")
    collect_parser.add_argument("--max-cycles", type=int, help="Stop after this many polling cycles.")
    collect_parser.add_argument("--max-runtime-seconds", type=float, help="Stop after this many seconds.")
    collect_parser.add_argument("--poll-interval", type=float, help="Override the polling interval in seconds.")
    collect_parser.add_argument("--flush-interval", type=float, help="Override the parquet flush interval in seconds.")
    collect_parser.add_argument("--batch-size", type=int, help="Override the in-memory batch size.")
    collect_parser.add_argument("--output-root", help="Optional raw-data destination override.")

    pull_parser = subparsers.add_parser(
        "pull-from-pc2",
        help="Copy new or changed files from the shared PC2 network folder into the local PC1 import area.",
    )
    pull_parser.add_argument("--network-root", help="Override the shared PC2 root path mounted on PC1.")
    pull_parser.add_argument("--destination-root", help="Override the local import root on PC1.")
    pull_parser.add_argument("--categories", nargs="+", choices=["raw", "meta", "logs"])
    pull_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    pull_parser.add_argument("--start-date", help="Only pull market data on or after this session date (YYYY-MM-DD).")
    pull_parser.add_argument("--end-date", help="Only pull market data on or before this session date (YYYY-MM-DD).")
    pull_parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    pull_parser.add_argument("--overwrite-policy", choices=["if_newer", "always", "never"])
    pull_parser.add_argument("--validate-parquet", action=argparse.BooleanOptionalAction, default=None)

    validate_imports_parser = subparsers.add_parser(
        "validate-imports",
        help="Validate imported market parquet files on PC1 before feature generation.",
    )
    validate_imports_parser.add_argument("--input-root", help="Override the imported raw market root.")
    validate_imports_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    validate_imports_parser.add_argument("--start-date", help="Validate files on or after this session date (YYYY-MM-DD).")
    validate_imports_parser.add_argument("--end-date", help="Validate files on or before this session date (YYYY-MM-DD).")

    build_features_parser = subparsers.add_parser(
        "build-features",
        help="Load imported raw market data, validate it, clean it, engineer a selected feature set, and persist feature parquet files.",
    )
    build_features_parser.add_argument("--symbols", nargs="+", help="Override the configured symbol universe.")
    build_features_parser.add_argument("--start-date", help="Filter raw data from this session date (YYYY-MM-DD).")
    build_features_parser.add_argument("--end-date", help="Filter raw data until this session date (YYYY-MM-DD).")
    build_features_parser.add_argument("--input-root", help="Override the imported raw market input root.")
    build_features_parser.add_argument("--output-root", help="Override the processed feature output root.")
    build_features_parser.add_argument("--feature-set", help="Feature set name declared in config/feature_sets.yaml.")

    subparsers.add_parser(
        "list-feature-sets",
        help="List configured feature sets, their families, and the configured default.",
    )

    inspect_dependencies_parser = subparsers.add_parser(
        "inspect-feature-dependencies",
        help="Inspect which indicators are compatible with the current dataset and why incompatible indicators would be skipped.",
    )
    inspect_dependencies_parser.add_argument("--symbols", nargs="+", help="Override the configured symbol universe.")
    inspect_dependencies_parser.add_argument("--start-date", help="Filter raw data from this session date (YYYY-MM-DD).")
    inspect_dependencies_parser.add_argument("--end-date", help="Filter raw data until this session date (YYYY-MM-DD).")
    inspect_dependencies_parser.add_argument("--input-root", help="Override the imported raw market input root.")
    inspect_dependencies_parser.add_argument("--feature-set", help="Feature set name declared in config/feature_sets.yaml.")

    validate_features_parser = subparsers.add_parser(
        "validate-features",
        help="Validate generated feature parquet files for NaNs, constants, infinities, and duplicate columns.",
    )
    validate_features_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    validate_features_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    validate_features_parser.add_argument("--start-date", help="Validate features on or after this session date (YYYY-MM-DD).")
    validate_features_parser.add_argument("--end-date", help="Validate features on or before this session date (YYYY-MM-DD).")

    dev_sync_parser = subparsers.add_parser(
        "dev-sync-and-build",
        help="Pull data from PC2, validate imports, build features, and print a single development summary.",
    )
    dev_sync_parser.add_argument("--network-root", help="Override the shared PC2 root path mounted on PC1.")
    dev_sync_parser.add_argument("--destination-root", help="Override the local import root on PC1.")
    dev_sync_parser.add_argument("--categories", nargs="+", choices=["raw", "meta", "logs"])
    dev_sync_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    dev_sync_parser.add_argument("--start-date", help="Only pull/build data on or after this session date (YYYY-MM-DD).")
    dev_sync_parser.add_argument("--end-date", help="Only pull/build data on or before this session date (YYYY-MM-DD).")
    dev_sync_parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    dev_sync_parser.add_argument("--overwrite-policy", choices=["if_newer", "always", "never"])
    dev_sync_parser.add_argument("--validate-parquet", action=argparse.BooleanOptionalAction, default=None)
    dev_sync_parser.add_argument("--output-root", help="Override the feature output root.")
    dev_sync_parser.add_argument("--feature-set", help="Feature set name declared in config/feature_sets.yaml.")

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

        if args.command == "pull-from-pc2":
            print_result(
                pull_from_pc2(
                    settings,
                    network_root=args.network_root,
                    destination_root=args.destination_root,
                    categories=args.categories,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    dry_run=args.dry_run,
                    overwrite_policy=args.overwrite_policy,
                    validate_parquet=args.validate_parquet,
                )
            )
            return 0

        if args.command == "validate-imports":
            print_result(
                validate_imports(
                    settings,
                    input_root=args.input_root or settings.paths.import_market_dir,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
            )
            return 0

        if args.command == "dev-sync-and-build":
            print_result(
                run_dev_sync_and_build(
                    settings,
                    network_root=args.network_root,
                    destination_root=args.destination_root,
                    categories=args.categories,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    dry_run=args.dry_run,
                    overwrite_policy=args.overwrite_policy,
                    validate_parquet=args.validate_parquet,
                    output_root=args.output_root,
                    feature_set_name=args.feature_set,
                )
            )
            return 0

        if args.command == "list-feature-sets":
            print_result(list_available_feature_sets(settings))
            return 0

        if args.command == "inspect-feature-dependencies":
            print_result(
                inspect_feature_dependencies_for_build(
                    settings,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    input_root=args.input_root or settings.paths.import_market_dir,
                    feature_set_name=args.feature_set,
                )
            )
            return 0

        if args.command == "build-features":
            print_result(
                run_feature_build_pipeline(
                    settings,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    input_root=args.input_root or settings.paths.import_market_dir,
                    output_root=args.output_root,
                    feature_set_name=args.feature_set,
                )
            )
            return 0

        if args.command == "validate-features":
            print_result(
                validate_feature_store(
                    settings,
                    feature_root=args.feature_root or settings.paths.feature_dir,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
            )
            return 0

        if args.command == "collect":
            print_result(
                collect_market_data(
                    settings,
                    symbols=args.symbols,
                    symbol=args.symbol,
                    once=args.once,
                    max_cycles=args.max_cycles,
                    max_runtime_seconds=args.max_runtime_seconds,
                    output_root=args.output_root,
                    poll_interval_seconds=args.poll_interval,
                    flush_interval_seconds=args.flush_interval,
                    batch_size=args.batch_size,
                )
            )
            return 0

        runtime = build_runtime(
            env_file=args.env_file,
            config_dir=args.config_dir,
            environment=args.environment,
        )

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

    from monitoring.logging import setup_logger

    logger = setup_logger(settings.log_level, settings.log_file, logger_name="microalpha.healthcheck")
    collector_client = build_collector_ib_client(settings, logger)
    print_result(build_healthcheck_report(settings, broker_client=collector_client))
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


def run_dev_sync_and_build(
    settings: Settings,
    *,
    network_root: str | None = None,
    destination_root: str | None = None,
    categories: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool | None = None,
    overwrite_policy: str | None = None,
    validate_parquet: bool | None = None,
    output_root: str | None = None,
    feature_set_name: str | None = None,
) -> dict[str, Any]:
    if categories and "raw" not in {category.strip().lower() for category in categories}:
        return {
            "status": "error",
            "message": "dev-sync-and-build requires the 'raw' category because feature generation depends on imported market parquet files.",
        }

    pull_result = pull_from_pc2(
        settings,
        network_root=network_root,
        destination_root=destination_root,
        categories=categories,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
        overwrite_policy=overwrite_policy,
        validate_parquet=validate_parquet,
    )
    if pull_result.get("dry_run"):
        return {
            "status": "planned",
            "pull": pull_result,
            "message": "LAN pull was executed in dry-run mode. Import validation and feature build were skipped.",
        }

    import_root = Path(destination_root) / "raw" / "market" if destination_root else Path(settings.paths.import_market_dir)
    validation_result = validate_imports(
        settings,
        input_root=import_root,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    if pull_result["status"] == "error" or validation_result["status"] == "error":
        return {
            "status": "error",
            "pull": pull_result,
            "validation": validation_result,
            "message": "LAN pull or import validation failed. Feature build was not executed.",
        }

    feature_result = run_feature_build_pipeline(
        settings,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        input_root=import_root,
        output_root=output_root,
        feature_set_name=feature_set_name,
    )
    return {
        "status": "ok",
        "pull": pull_result,
        "validation": validation_result,
        "features": feature_result,
    }


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
