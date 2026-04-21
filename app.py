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
from config.phase6 import set_active_model_selection
from config.phase10_11 import load_phase10_11_config
from governance.releases import (
    governance_status,
    list_model_releases,
    promote_model_release,
    rollback_model_release,
    show_active_release,
)
from deployment.lan_sync import pull_from_pc2
from engine.phase6 import risk_check, run_decisions_offline, run_session, show_active_model
from engine.phase7 import (
    broker_healthcheck,
    execution_status,
    run_paper_session,
    run_paper_session_real,
    run_paper_sim_offline,
    show_execution_backend,
)
from engine.runtime import RuntimeServices, build_runtime
from features.feature_pipeline import (
    inspect_feature_dependencies_for_build,
    list_available_feature_sets,
    run_feature_build_pipeline,
)
from features.validation import validate_feature_store
from ingestion.collector import collect_market_data
from ingestion.ibkr_historical_backfill import (
    export_training_csv_from_backfill,
    ibkr_backfill,
    ibkr_backfill_status,
    ibkr_head_timestamp,
    prepare_ibkr_training_data,
)
from ingestion.ibkr_client import build_collector_ib_client
from ingestion.lob_capture import (
    lob_capture_status,
    run_lob_capture_loop,
    start_lob_capture,
    stop_lob_capture,
)
from labels.labeling import build_labels
from models.experiments import (
    compare_model_variants,
    evaluate_baseline_variant,
    run_phase5_experiments,
    train_baseline_variant,
)
from models.train_baseline import train_baseline_model
from models.train_deep import evaluate_deep_daily, train_deep_model
from monitoring.data_quality import validate_imports
from monitoring.healthcheck import build_healthcheck_report
from monitoring.sync import sync_data_artifacts
from reporting.report_bundle import (
    analyze_signal_report,
    detect_drift_report,
    evaluate_performance_report,
    full_evaluation_run,
    generate_report,
)
from evaluation.compare_runs import compare_runs
from monitoring.alerts import AlertStore
from monitoring.paper_monitor import monitor_paper_session
from ops.incidents import IncidentStore
from ops.orchestrator import (
    full_paper_validation_cycle,
    full_runtime_cycle,
    governance_status_command,
    generate_runbooks,
    postflight_check_command,
    preflight_check_command,
    runtime_status_command_wrapper,
    system_health_report,
)
from ops.runtime_manager import restart_runtime, service_status, start_runtime, stop_runtime
from shadow.session import run_shadow_session
from validation.paper_validation import compare_paper_sessions, reconcile_and_report, run_paper_validation_session
from validation.readiness import generate_readiness_report


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

    build_labels_parser = subparsers.add_parser(
        "build-labels",
        help="Generate flexible labels/targets from a selected feature store and persist them under data/processed/labels.",
    )
    build_labels_parser.add_argument("--feature-set", help="Feature set name declared in config/feature_sets.yaml.")
    build_labels_parser.add_argument("--target-mode", default="classification_binary", help="Target mode declared in config/modeling.yaml.")
    build_labels_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    build_labels_parser.add_argument("--start-date", help="Filter features from this session date (YYYY-MM-DD).")
    build_labels_parser.add_argument("--end-date", help="Filter features until this session date (YYYY-MM-DD).")
    build_labels_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    build_labels_parser.add_argument("--output-root", help="Override the label parquet output root.")

    head_timestamp_parser = subparsers.add_parser(
        "ibkr-head-timestamp",
        help="Query IBKR reqHeadTimestamp to discover the earliest available historical timestamp for a symbol and whatToShow.",
    )
    head_timestamp_parser.add_argument("--symbol", required=True, help="Ticker symbol to inspect.")
    head_timestamp_parser.add_argument("--what-to-show", default="TRADES", help="IBKR whatToShow value such as TRADES.")
    head_timestamp_parser.add_argument("--use-rth", choices=["true", "false"], help="Whether to restrict the request to regular trading hours.")

    backfill_parser = subparsers.add_parser(
        "ibkr-backfill",
        help="Run IBKR historical bar backfill with pacing guards, chunk planning, checkpointing, and deduplicated parquet output.",
    )
    backfill_parser.add_argument("--symbol", required=True, help="Ticker symbol to backfill.")
    backfill_parser.add_argument("--what-to-show", default="TRADES", help="IBKR whatToShow value such as TRADES.")
    backfill_parser.add_argument("--bar-size", default="1 min", help='IBKR bar size such as "1 min".')
    backfill_parser.add_argument("--use-rth", choices=["true", "false"], help="Whether to restrict the request to regular trading hours.")
    backfill_parser.add_argument("--start-date", help="Optional lower bound in YYYY-MM-DD. Example: 2025-01-01")

    backfill_resume_parser = subparsers.add_parser(
        "ibkr-backfill-resume",
        help="Resume an interrupted IBKR historical bar backfill from the saved checkpoint state.",
    )
    backfill_resume_parser.add_argument("--symbol", required=True, help="Ticker symbol to backfill.")
    backfill_resume_parser.add_argument("--what-to-show", default="TRADES", help="IBKR whatToShow value such as TRADES.")
    backfill_resume_parser.add_argument("--bar-size", default="1 min", help='IBKR bar size such as "1 min".')
    backfill_resume_parser.add_argument("--use-rth", choices=["true", "false"], help="Whether to restrict the request to regular trading hours.")
    backfill_resume_parser.add_argument("--start-date", help="Optional lower bound in YYYY-MM-DD. Example: 2025-01-01")

    backfill_status_parser = subparsers.add_parser(
        "ibkr-backfill-status",
        help="Show current IBKR historical backfill state, manifest paths, and checkpoint progress for a symbol.",
    )
    backfill_status_parser.add_argument("--symbol", required=True, help="Ticker symbol to inspect.")
    backfill_status_parser.add_argument("--what-to-show", help="Optional IBKR whatToShow filter.")
    backfill_status_parser.add_argument("--bar-size", help='Optional bar size such as "1 min".')

    export_training_parser = subparsers.add_parser(
        "export-training-csv",
        help="Export a training-ready CSV from an IBKR historical backfill dataset.",
    )
    export_training_parser.add_argument("--symbol", required=True, help="Ticker symbol to export.")
    export_training_parser.add_argument("--what-to-show", default="TRADES", help="IBKR whatToShow value such as TRADES.")
    export_training_parser.add_argument("--bar-size", default="1 min", help='IBKR bar size such as "1 min".')
    export_training_parser.add_argument("--output-path", required=True, help="Training CSV output path.")

    prepare_ibkr_training_parser = subparsers.add_parser(
        "prepare-ibkr-training-data",
        help="Main bootstrap command: discover head timestamp, backfill IBKR historical bars, and export a training-ready CSV.",
    )
    prepare_ibkr_training_parser.add_argument("--symbol", required=True, help="Ticker symbol to prepare.")
    prepare_ibkr_training_parser.add_argument("--what-to-show", default="TRADES", help="IBKR whatToShow value such as TRADES.")
    prepare_ibkr_training_parser.add_argument("--bar-size", default="1 min", help='IBKR bar size such as "1 min".')
    prepare_ibkr_training_parser.add_argument("--use-rth", choices=["true", "false"], help="Whether to restrict the request to regular trading hours.")
    prepare_ibkr_training_parser.add_argument("--start-date", help="Optional lower bound in YYYY-MM-DD. Example: 2025-01-01")
    prepare_ibkr_training_parser.add_argument("--output-path", required=True, help="Training CSV output path.")

    start_lob_capture_parser = subparsers.add_parser(
        "start-lob-capture",
        help="Start a background IBKR Level II capture process that appends multilevel LOB chunks to parquet.",
    )
    start_lob_capture_parser.add_argument("--symbol", required=True, help="Ticker symbol to capture.")
    start_lob_capture_parser.add_argument("--levels", type=int, default=10, help="Requested market depth levels.")
    start_lob_capture_parser.add_argument("--rth", choices=["true", "false"], help="Restrict persistence to RTH only.")

    stop_lob_capture_parser = subparsers.add_parser(
        "stop-lob-capture",
        help="Stop the background IBKR Level II capture process for a symbol.",
    )
    stop_lob_capture_parser.add_argument("--symbol", required=True, help="Ticker symbol to stop.")

    lob_capture_status_parser = subparsers.add_parser(
        "lob-capture-status",
        help="Show current LOB capture state, PID, session, and persisted row counters.",
    )
    lob_capture_status_parser.add_argument("--symbol", required=True, help="Ticker symbol to inspect.")

    build_lob_dataset_parser = subparsers.add_parser(
        "build-lob-dataset",
        help="Build an event-based multilevel LOB dataset from raw capture chunks and emit labels paper-style.",
    )
    build_lob_dataset_parser.add_argument("--symbol", required=True, help="Ticker symbol to build.")
    build_lob_dataset_parser.add_argument("--from-date", required=True, help="Lower session-date bound in YYYY-MM-DD.")
    build_lob_dataset_parser.add_argument("--to-date", help="Upper session-date bound in YYYY-MM-DD.")
    build_lob_dataset_parser.add_argument("--horizon-events", type=int, help="Label horizon in book events.")
    build_lob_dataset_parser.add_argument("--output-path", help="Optional explicit parquet output path.")

    evaluate_deep_daily_parser = subparsers.add_parser(
        "evaluate-deep-daily",
        help="Run walk-forward daily evaluation for the DeepLOB-like model on captured IBKR LOB data.",
    )
    evaluate_deep_daily_parser.add_argument("--symbol", required=True, help="Ticker symbol to evaluate.")
    evaluate_deep_daily_parser.add_argument("--from-date", required=True, help="Lower session-date bound in YYYY-MM-DD.")
    evaluate_deep_daily_parser.add_argument("--epochs", type=int, default=2, help="Epochs per walk-forward retrain.")

    train_baseline_parser = subparsers.add_parser(
        "train-baseline",
        help="Train one configurable phase 5 baseline/model variant from labeled feature data.",
    )
    train_baseline_parser.add_argument("--feature-set", help="Feature set name declared in config/feature_sets.yaml.")
    train_baseline_parser.add_argument("--target-mode", default="classification_binary", help="Target mode declared in config/modeling.yaml.")
    train_baseline_parser.add_argument("--model", default="logistic_regression", help="Model name declared in the phase 5 model factory.")
    train_baseline_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    train_baseline_parser.add_argument("--start-date", help="Filter data from this session date (YYYY-MM-DD).")
    train_baseline_parser.add_argument("--end-date", help="Filter data until this session date (YYYY-MM-DD).")

    evaluate_baseline_parser = subparsers.add_parser(
        "evaluate-baseline",
        help="Reload and evaluate a previously trained phase 5 run using its stored metadata.",
    )
    evaluate_baseline_parser.add_argument("--run-id", help="Registered phase5 run id. Defaults to the latest run.")
    evaluate_baseline_parser.add_argument("--artifact-dir", help="Resolve the run from its artifact directory instead of run id.")

    compare_variants_parser = subparsers.add_parser(
        "compare-model-variants",
        help="Build labels if needed, train multiple model/target/feature-set combinations, and emit a leaderboard.",
    )
    compare_variants_parser.add_argument("--profile", default="default", help="Experiment profile declared in config/modeling.yaml.")
    compare_variants_parser.add_argument("--feature-sets", nargs="+", help="Optional explicit feature sets.")
    compare_variants_parser.add_argument("--target-modes", nargs="+", help="Optional explicit target modes.")
    compare_variants_parser.add_argument("--models", nargs="+", help="Optional explicit model names.")
    compare_variants_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    compare_variants_parser.add_argument("--start-date", help="Filter data from this session date (YYYY-MM-DD).")
    compare_variants_parser.add_argument("--end-date", help="Filter data until this session date (YYYY-MM-DD).")
    compare_variants_parser.add_argument("--max-combinations", type=int, help="Hard cap on attempted model combinations.")

    run_phase5_parser = subparsers.add_parser(
        "run-phase5-experiments",
        help="Main phase 5 command: build features, build labels, train/evaluate variants, and write a leaderboard.",
    )
    run_phase5_parser.add_argument("--profile", default="default", help="Experiment profile declared in config/modeling.yaml.")
    run_phase5_parser.add_argument("--feature-sets", nargs="+", help="Optional explicit feature sets.")
    run_phase5_parser.add_argument("--target-modes", nargs="+", help="Optional explicit target modes.")
    run_phase5_parser.add_argument("--models", nargs="+", help="Optional explicit model names.")
    run_phase5_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    run_phase5_parser.add_argument("--start-date", help="Filter data from this session date (YYYY-MM-DD).")
    run_phase5_parser.add_argument("--end-date", help="Filter data until this session date (YYYY-MM-DD).")
    run_phase5_parser.add_argument("--skip-feature-build", action="store_true", help="Reuse existing feature stores instead of rebuilding them.")

    subparsers.add_parser(
        "show-active-model",
        help="Show the currently configured Phase 6 active model selection and the artifact files it depends on.",
    )

    subparsers.add_parser(
        "show-execution-backend",
        help="Show the configured Phase 7 execution backend and the active paper-execution settings.",
    )

    subparsers.add_parser(
        "broker-healthcheck",
        help="Connect to IBKR Paper, validate paper-mode safety guards, and print broker connectivity status.",
    )

    validation_session_parser = subparsers.add_parser(
        "run-paper-validation-session",
        help="Run one formal paper-validation session with session tracking, monitoring, reconciliation, and readiness outputs.",
    )
    validation_session_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    validation_session_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    validation_session_parser.add_argument("--latest-per-symbol", type=int, help="How many latest rows per symbol to evaluate.")
    validation_session_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")
    validation_session_parser.add_argument("--skip-preflight", action="store_true", help="Skip explicit preflight before the validation session.")

    reconcile_parser = subparsers.add_parser(
        "reconcile-broker-state",
        help="Reconcile internal execution state against IBKR Paper broker state and write reconciliation reports.",
    )
    reconcile_parser.add_argument("--session-id", help="Optional paper-validation session id.")

    monitor_parser = subparsers.add_parser(
        "monitor-paper-session",
        help="Run a conservative monitoring snapshot for the current paper-validation session and emit alerts/incidents.",
    )
    monitor_parser.add_argument("--session-id", help="Optional paper-validation session id.")
    monitor_parser.add_argument("--summary-path", help="Optional explicit Phase 7 summary path.")
    monitor_parser.add_argument("--iterations", type=int, default=1, help="How many monitoring snapshots to collect.")

    subparsers.add_parser(
        "compare-paper-sessions",
        help="Compare tracked paper-validation sessions and generate health and stability leaderboards.",
    )

    readiness_parser = subparsers.add_parser(
        "generate-readiness-report",
        help="Generate the readiness report for the latest or selected paper-validation session.",
    )
    readiness_parser.add_argument("--session-id", help="Optional paper-validation session id.")

    health_report_parser = subparsers.add_parser(
        "system-health-report",
        help="Generate the consolidated system health report across the latest paper-validation artifacts.",
    )
    health_report_parser.add_argument("--session-id", help="Optional paper-validation session id.")

    full_validation_parser = subparsers.add_parser(
        "full-paper-validation-cycle",
        help="Run the full paper validation cycle: session, monitoring, reconciliation, readiness, postflight, and health reporting.",
    )
    full_validation_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    full_validation_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    full_validation_parser.add_argument("--latest-per-symbol", type=int, help="How many latest rows per symbol to evaluate.")
    full_validation_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")

    preflight_parser = subparsers.add_parser(
        "preflight-check",
        help="Run conservative preflight checks before any automated paper-validation session.",
    )
    preflight_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")

    postflight_parser = subparsers.add_parser(
        "postflight-check",
        help="Run postflight validation, archival, and safe-restart assessment for a paper-validation session.",
    )
    postflight_parser.add_argument("--session-id", help="Optional paper-validation session id. Defaults to the latest tracked session.")

    list_alerts_parser = subparsers.add_parser(
        "list-alerts",
        help="List structured alerts generated by validation, monitoring, reconciliation, and ops checks.",
    )
    list_alerts_parser.add_argument("--session-id", help="Optional paper-validation session id.")
    list_alerts_parser.add_argument("--severity", help="Optional severity filter.")
    list_alerts_parser.add_argument("--limit", type=int, default=20, help="Maximum number of alerts to return.")

    list_incidents_parser = subparsers.add_parser(
        "list-incidents",
        help="List structured incidents generated during paper validation and ops hardening.",
    )
    list_incidents_parser.add_argument("--session-id", help="Optional paper-validation session id.")
    list_incidents_parser.add_argument("--severity", help="Optional severity filter.")
    list_incidents_parser.add_argument("--limit", type=int, default=20, help="Maximum number of incidents to return.")

    subparsers.add_parser(
        "generate-runbooks",
        help="Generate the operational runbooks under docs/runbooks/.",
    )

    start_runtime_parser = subparsers.add_parser(
        "start-runtime",
        help="Bootstrap and mark the local PC2 runtime as started for the selected runtime profile.",
    )
    start_runtime_parser.add_argument("--profile", help="Override the runtime profile.")

    stop_runtime_parser = subparsers.add_parser(
        "stop-runtime",
        help="Stop the managed runtime and mark all runtime services as stopped.",
    )

    restart_runtime_parser = subparsers.add_parser(
        "restart-runtime",
        help="Restart the managed runtime for the selected runtime profile.",
    )
    restart_runtime_parser.add_argument("--profile", help="Override the runtime profile.")

    subparsers.add_parser(
        "service-status",
        help="Show the current managed runtime service state.",
    )

    shadow_session_parser = subparsers.add_parser(
        "run-shadow-session",
        help="Run the inference/decision/risk chain in shadow mode without sending paper orders.",
    )
    shadow_session_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    shadow_session_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    shadow_session_parser.add_argument("--latest-per-symbol", type=int, help="How many latest rows per symbol to evaluate.")
    shadow_session_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")

    full_runtime_parser = subparsers.add_parser(
        "full-runtime-cycle",
        help="Run the full runtime cycle using the configured runtime profile, including paper validation or shadow mode.",
    )
    full_runtime_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    full_runtime_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    full_runtime_parser.add_argument("--latest-per-symbol", type=int, help="How many latest rows per symbol to evaluate.")
    full_runtime_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")
    full_runtime_parser.add_argument("--profile", help="Override the runtime profile.")

    subparsers.add_parser(
        "runtime-status",
        help="Show the runtime profile, runtime service state, active release, and latest readiness/governance context.",
    )

    subparsers.add_parser(
        "governance-status",
        help="Show active release governance context, critical incidents/alerts, and promotion audit pointers.",
    )

    subparsers.add_parser(
        "list-model-releases",
        help="List registered model releases and their governance status.",
    )

    show_release_parser = subparsers.add_parser(
        "show-active-release",
        help="Show the active model release and its traceability metadata.",
    )

    promote_parser = subparsers.add_parser(
        "promote-model",
        help="Promote a model release after governance checks and update the active model selection.",
    )
    promote_parser.add_argument("--model-name", help="Model name to promote.")
    promote_parser.add_argument("--run-id", help="Run id to promote.")
    promote_parser.add_argument("--release-id", help="Release id to promote.")
    promote_parser.add_argument("--actor", default="cli", help="Actor or operator initiating the promotion.")
    promote_parser.add_argument("--reason", default="manual promotion", help="Reason for the promotion.")

    rollback_parser = subparsers.add_parser(
        "rollback-model",
        help="Rollback to a previously known release and restore its active-model mapping.",
    )
    rollback_parser.add_argument("--to", required=True, help="Target release id or run id for rollback.")
    rollback_parser.add_argument("--actor", default="cli", help="Actor or operator initiating the rollback.")
    rollback_parser.add_argument("--reason", default="manual rollback", help="Reason for the rollback.")

    execution_status_parser = subparsers.add_parser(
        "execution-status",
        help="Show the current Phase 7 execution state, active backend, positions, and recent order/fill activity.",
    )
    execution_status_parser.add_argument("--limit", type=int, help="How many recent orders/fills/reports to include.")

    set_active_parser = subparsers.add_parser(
        "set-active-model",
        help="Set the Phase 6 active model selection by run id, artifact directory, or model name.",
    )
    set_active_parser.add_argument("--run-id", help="Exact Phase 5 run id to activate.")
    set_active_parser.add_argument("--artifact-dir", help="Artifact directory to activate.")
    set_active_parser.add_argument("--model-name", help="Activate the best available run matching this model name.")

    offline_parser = subparsers.add_parser(
        "run-decisions-offline",
        help="Load historical features, run Phase 6 inference + decision + risk, and persist structured decision outputs.",
    )
    offline_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    offline_parser.add_argument("--start-date", help="Filter data from this session date (YYYY-MM-DD).")
    offline_parser.add_argument("--end-date", help="Filter data until this session date (YYYY-MM-DD).")
    offline_parser.add_argument("--limit", type=int, help="Only evaluate the latest N rows after filtering.")
    offline_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    offline_parser.add_argument("--label-root", help="Override the label parquet root for offline realized-outcome joins.")
    offline_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")

    paper_offline_parser = subparsers.add_parser(
        "run-paper-sim-offline",
        help="Load historical features, run Phase 6 inference + decision + risk, then route approved decisions through the Phase 7 paper/mock execution layer.",
    )
    paper_offline_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    paper_offline_parser.add_argument("--start-date", help="Filter data from this session date (YYYY-MM-DD).")
    paper_offline_parser.add_argument("--end-date", help="Filter data until this session date (YYYY-MM-DD).")
    paper_offline_parser.add_argument("--limit", type=int, help="Only evaluate the latest N rows after filtering.")
    paper_offline_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    paper_offline_parser.add_argument("--label-root", help="Override the label parquet root for offline realized-outcome joins.")
    paper_offline_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")

    evaluate_performance_parser = subparsers.add_parser(
        "evaluate-performance",
        help="Evaluate economic performance, trade metrics, and segment performance for a Phase 7 run.",
    )
    evaluate_performance_parser.add_argument("--summary-path", help="Explicit Phase 7 summary JSON path.")
    evaluate_performance_parser.add_argument("--parquet-path", help="Explicit Phase 7 parquet path.")

    analyze_signals_parser = subparsers.add_parser(
        "analyze-signals",
        help="Analyze signal quality, monotonicity, and calibration for a Phase 7 run.",
    )
    analyze_signals_parser.add_argument("--summary-path", help="Explicit Phase 7 summary JSON path.")
    analyze_signals_parser.add_argument("--parquet-path", help="Explicit Phase 7 parquet path.")

    detect_drift_parser = subparsers.add_parser(
        "detect-drift",
        help="Detect feature, prediction, and label drift for a Phase 7 run.",
    )
    detect_drift_parser.add_argument("--summary-path", help="Explicit Phase 7 summary JSON path.")
    detect_drift_parser.add_argument("--parquet-path", help="Explicit Phase 7 parquet path.")

    compare_runs_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare stored Phase 8 run reports and build an economic ranking table.",
    )
    compare_runs_parser.add_argument("--report-paths", nargs="+", help="Optional explicit Phase 8 run_report.json paths.")
    compare_runs_parser.add_argument("--output-dir", help="Optional output directory for comparison artifacts.")

    generate_report_parser = subparsers.add_parser(
        "generate-report",
        help="Generate the full Phase 8 report bundle for a Phase 7 run.",
    )
    generate_report_parser.add_argument("--summary-path", help="Explicit Phase 7 summary JSON path.")
    generate_report_parser.add_argument("--parquet-path", help="Explicit Phase 7 parquet path.")

    full_eval_parser = subparsers.add_parser(
        "full-evaluation-run",
        help="Alias for generate-report to run the complete Phase 8 evaluation bundle.",
    )
    full_eval_parser.add_argument("--summary-path", help="Explicit Phase 7 summary JSON path.")
    full_eval_parser.add_argument("--parquet-path", help="Explicit Phase 7 parquet path.")

    subparsers.add_parser(
        "risk-check",
        help="Validate the active model, Phase 6 decision/risk configuration, and operational readiness.",
    )

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
    validate_features_parser.add_argument("--feature-set", help="Optional feature set partition to validate.")
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

    lob_runner_parser = subparsers.add_parser(
        "_lob-capture-runner",
        help=argparse.SUPPRESS,
    )
    lob_runner_parser.add_argument("--symbol", required=True)
    lob_runner_parser.add_argument("--levels", type=int, required=True)
    lob_runner_parser.add_argument("--session-id", required=True)
    lob_runner_parser.add_argument("--rth", choices=["true", "false"], required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Validate backtest inputs and dataset plumbing.")
    backtest_parser.add_argument("--data-path", help="CSV or Parquet dataset for research mode.")
    backtest_parser.add_argument("--symbol", help="Override the configured symbol.")

    session_parser = subparsers.add_parser(
        "run-session",
        aliases=["session-cycle", "session"],
        help="Run one Phase 6 operational decision cycle over the latest available feature rows without sending orders.",
    )
    session_parser.add_argument(
        "--paper",
        action="store_true",
        help="Reserved for future use. Phase 6 still does not send orders.",
    )
    session_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    session_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    session_parser.add_argument("--latest-per-symbol", type=int, default=1, help="How many latest rows per symbol to evaluate.")
    session_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")

    paper_session_parser = subparsers.add_parser(
        "run-paper-session",
        help="Run one operational paper/mock session cycle through inference, decision, risk, order management, fills, and journaling.",
    )
    paper_session_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    paper_session_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    paper_session_parser.add_argument("--latest-per-symbol", type=int, help="How many latest rows per symbol to evaluate.")
    paper_session_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")

    real_paper_session_parser = subparsers.add_parser(
        "run-paper-session-real",
        help="Run one operational paper session against the real IBKR Paper backend using the active model and risk checks.",
    )
    real_paper_session_parser.add_argument("--symbols", nargs="+", help="Optional symbol filter.")
    real_paper_session_parser.add_argument("--feature-root", help="Override the feature parquet root.")
    real_paper_session_parser.add_argument("--latest-per-symbol", type=int, help="How many latest rows per symbol to evaluate.")
    real_paper_session_parser.add_argument("--decision-log-path", help="Override the JSONL decision log path.")

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

        if args.command == "build-labels":
            print_result(
                build_labels(
                    settings,
                    feature_set_name=args.feature_set or settings.feature_pipeline.default_feature_set,
                    target_mode=args.target_mode,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    feature_root=args.feature_root,
                    output_root=args.output_root,
                )
            )
            return 0

        if args.command == "ibkr-head-timestamp":
            print_result(
                ibkr_head_timestamp(
                    settings,
                    symbol=args.symbol,
                    what_to_show=args.what_to_show,
                    use_rth=_parse_cli_bool(args.use_rth),
                )
            )
            return 0

        if args.command == "ibkr-backfill":
            print_result(
                ibkr_backfill(
                    settings,
                    symbol=args.symbol,
                    what_to_show=args.what_to_show,
                    bar_size=args.bar_size,
                    use_rth=_parse_cli_bool(args.use_rth),
                    resume=False,
                    start_date=args.start_date,
                )
            )
            return 0

        if args.command == "ibkr-backfill-resume":
            print_result(
                ibkr_backfill(
                    settings,
                    symbol=args.symbol,
                    what_to_show=args.what_to_show,
                    bar_size=args.bar_size,
                    use_rth=_parse_cli_bool(args.use_rth),
                    resume=True,
                    start_date=args.start_date,
                )
            )
            return 0

        if args.command == "ibkr-backfill-status":
            print_result(
                ibkr_backfill_status(
                    settings,
                    symbol=args.symbol,
                    what_to_show=args.what_to_show,
                    bar_size=args.bar_size,
                )
            )
            return 0

        if args.command == "export-training-csv":
            print_result(
                export_training_csv_from_backfill(
                    settings,
                    symbol=args.symbol,
                    what_to_show=args.what_to_show,
                    bar_size=args.bar_size,
                    output_path=args.output_path,
                )
            )
            return 0

        if args.command == "prepare-ibkr-training-data":
            print_result(
                prepare_ibkr_training_data(
                    settings,
                    symbol=args.symbol,
                    what_to_show=args.what_to_show,
                    bar_size=args.bar_size,
                    use_rth=_parse_cli_bool(args.use_rth),
                    output_path=args.output_path,
                    start_date=args.start_date,
                )
            )
            return 0

        if args.command == "start-lob-capture":
            print_result(
                start_lob_capture(
                    settings,
                    symbol=args.symbol,
                    levels=args.levels,
                    rth_only=_parse_cli_bool(args.rth),
                )
            )
            return 0

        if args.command == "stop-lob-capture":
            print_result(stop_lob_capture(settings, symbol=args.symbol))
            return 0

        if args.command == "lob-capture-status":
            print_result(lob_capture_status(settings, symbol=args.symbol))
            return 0

        if args.command == "build-lob-dataset":
            from data.lob_dataset import build_lob_dataset

            print_result(
                build_lob_dataset(
                    settings,
                    symbol=args.symbol,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    horizon_events=args.horizon_events,
                    output_path=args.output_path,
                )
            )
            return 0

        if args.command == "evaluate-deep-daily":
            print_result(
                evaluate_deep_daily(
                    settings,
                    symbol=args.symbol,
                    from_date=args.from_date,
                    epochs=args.epochs,
                )
            )
            return 0

        if args.command == "train-baseline":
            print_result(
                train_baseline_variant(
                    settings,
                    feature_set_name=args.feature_set or settings.feature_pipeline.default_feature_set,
                    target_mode=args.target_mode,
                    model_name=args.model,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
            )
            return 0

        if args.command == "evaluate-baseline":
            print_result(
                evaluate_baseline_variant(
                    settings,
                    run_id=args.run_id,
                    artifact_dir=args.artifact_dir,
                )
            )
            return 0

        if args.command == "compare-model-variants":
            print_result(
                compare_model_variants(
                    settings,
                    feature_sets=args.feature_sets,
                    target_modes=args.target_modes,
                    models=args.models,
                    profile_name=args.profile,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    max_combinations=args.max_combinations,
                )
            )
            return 0

        if args.command == "run-phase5-experiments":
            print_result(
                run_phase5_experiments(
                    settings,
                    feature_sets=args.feature_sets,
                    target_modes=args.target_modes,
                    models=args.models,
                    profile_name=args.profile,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    build_features_first=not args.skip_feature_build,
                )
            )
            return 0

        if args.command == "show-active-model":
            print_result(show_active_model(settings))
            return 0

        if args.command == "show-execution-backend":
            print_result(show_execution_backend(settings))
            return 0

        if args.command == "broker-healthcheck":
            print_result(broker_healthcheck(settings))
            return 0

        if args.command == "run-paper-validation-session":
            print_result(
                run_paper_validation_session(
                    settings,
                    symbols=args.symbols,
                    feature_root=args.feature_root,
                    latest_per_symbol=args.latest_per_symbol,
                    decision_log_path=args.decision_log_path,
                    run_preflight=not args.skip_preflight,
                )
            )
            return 0

        if args.command == "reconcile-broker-state":
            print_result(reconcile_and_report(settings, session_id=args.session_id))
            return 0

        if args.command == "monitor-paper-session":
            print_result(
                monitor_paper_session(
                    settings,
                    session_id=args.session_id,
                    summary_path=args.summary_path,
                    iterations=args.iterations,
                )
            )
            return 0

        if args.command == "compare-paper-sessions":
            print_result(compare_paper_sessions(settings))
            return 0

        if args.command == "generate-readiness-report":
            print_result(generate_readiness_report(settings, session_id=args.session_id))
            return 0

        if args.command == "system-health-report":
            print_result(system_health_report(settings, session_id=args.session_id))
            return 0

        if args.command == "full-paper-validation-cycle":
            print_result(
                full_paper_validation_cycle(
                    settings,
                    symbols=args.symbols,
                    feature_root=args.feature_root,
                    latest_per_symbol=args.latest_per_symbol,
                    decision_log_path=args.decision_log_path,
                )
            )
            return 0

        if args.command == "preflight-check":
            print_result(preflight_check_command(settings, symbols=args.symbols))
            return 0

        if args.command == "postflight-check":
            session_id = args.session_id
            if session_id is None:
                phase10_11 = load_phase10_11_config(settings)
                from validation.session_tracker import SessionTracker

                tracker = SessionTracker(
                    session_root=phase10_11.report_paths.session_root,
                    registry_path=phase10_11.report_paths.registry_path,
                    archive_root=phase10_11.report_paths.archive_root,
                )
                latest = tracker.latest_session()
                if latest is None:
                    print_result({"status": "error", "message": "No tracked paper-validation session exists for postflight-check."})
                    return 1
                session_id = latest["session_id"]
            print_result(postflight_check_command(settings, session_id=session_id))
            return 0

        if args.command == "list-alerts":
            phase10_11 = load_phase10_11_config(settings)
            store = AlertStore(phase10_11.report_paths.alerts_path)
            print_result(
                {
                    "status": "ok",
                    "alerts": store.list_alerts(session_id=args.session_id, severity=args.severity, limit=args.limit),
                    "summary": store.summarize(session_id=args.session_id),
                }
            )
            return 0

        if args.command == "list-incidents":
            phase10_11 = load_phase10_11_config(settings)
            store = IncidentStore(phase10_11.report_paths.incidents_path)
            print_result(
                {
                    "status": "ok",
                    "incidents": store.list_incidents(session_id=args.session_id, severity=args.severity, limit=args.limit),
                    "summary": store.summarize(session_id=args.session_id),
                }
            )
            return 0

        if args.command == "generate-runbooks":
            print_result(generate_runbooks(settings))
            return 0

        if args.command == "start-runtime":
            print_result(start_runtime(settings, profile_name=args.profile))
            return 0

        if args.command == "stop-runtime":
            print_result(stop_runtime(settings))
            return 0

        if args.command == "restart-runtime":
            print_result(restart_runtime(settings, profile_name=args.profile))
            return 0

        if args.command == "service-status":
            print_result(service_status(settings))
            return 0

        if args.command == "run-shadow-session":
            print_result(
                run_shadow_session(
                    settings,
                    symbols=args.symbols,
                    feature_root=args.feature_root,
                    latest_per_symbol=args.latest_per_symbol,
                    decision_log_path=args.decision_log_path,
                )
            )
            return 0

        if args.command == "full-runtime-cycle":
            print_result(
                full_runtime_cycle(
                    settings,
                    symbols=args.symbols,
                    feature_root=args.feature_root,
                    latest_per_symbol=args.latest_per_symbol,
                    decision_log_path=args.decision_log_path,
                    profile_name=args.profile,
                )
            )
            return 0

        if args.command == "runtime-status":
            print_result(runtime_status_command_wrapper(settings))
            return 0

        if args.command == "governance-status":
            print_result(governance_status_command(settings))
            return 0

        if args.command == "list-model-releases":
            print_result(list_model_releases(settings))
            return 0

        if args.command == "show-active-release":
            print_result(show_active_release(settings))
            return 0

        if args.command == "promote-model":
            if not any([args.model_name, args.run_id, args.release_id]):
                print_result({"status": "error", "message": "promote-model requires --model-name, --run-id, or --release-id."})
                return 1
            print_result(
                promote_model_release(
                    settings,
                    model_name=args.model_name,
                    run_id=args.run_id,
                    release_id=args.release_id,
                    actor=args.actor,
                    reason=args.reason,
                )
            )
            return 0

        if args.command == "rollback-model":
            print_result(
                rollback_model_release(
                    settings,
                    to=args.to,
                    actor=args.actor,
                    reason=args.reason,
                )
            )
            return 0

        if args.command == "execution-status":
            print_result(execution_status(settings, limit=args.limit))
            return 0

        if args.command == "evaluate-performance":
            print_result(
                evaluate_performance_report(
                    settings,
                    summary_path=args.summary_path,
                    parquet_path=args.parquet_path,
                )
            )
            return 0

        if args.command == "analyze-signals":
            print_result(
                analyze_signal_report(
                    settings,
                    summary_path=args.summary_path,
                    parquet_path=args.parquet_path,
                )
            )
            return 0

        if args.command == "detect-drift":
            print_result(
                detect_drift_report(
                    settings,
                    summary_path=args.summary_path,
                    parquet_path=args.parquet_path,
                )
            )
            return 0

        if args.command == "compare-runs":
            from config.phase8 import load_phase8_config

            phase8 = load_phase8_config(settings)
            print_result(
                compare_runs(
                    phase8.report_paths.report_dir,
                    report_paths=args.report_paths,
                    output_dir=args.output_dir or phase8.report_paths.compare_runs_dir,
                )
            )
            return 0

        if args.command == "generate-report":
            print_result(
                generate_report(
                    settings,
                    summary_path=args.summary_path,
                    parquet_path=args.parquet_path,
                )
            )
            return 0

        if args.command == "full-evaluation-run":
            print_result(
                full_evaluation_run(
                    settings,
                    summary_path=args.summary_path,
                    parquet_path=args.parquet_path,
                )
            )
            return 0

        if args.command == "set-active-model":
            if not any([args.run_id, args.artifact_dir, args.model_name]):
                print_result(
                    {
                        "status": "error",
                        "message": "set-active-model requires --run-id, --artifact-dir, or --model-name.",
                    }
                )
                return 1
            print_result(
                set_active_model_selection(
                    settings,
                    run_id=args.run_id,
                    artifact_dir=args.artifact_dir,
                    model_name=args.model_name,
                )
            )
            return 0

        if args.command == "run-decisions-offline":
            print_result(
                run_decisions_offline(
                    settings,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    limit=args.limit,
                    feature_root=args.feature_root,
                    label_root=args.label_root,
                    decision_log_path=args.decision_log_path,
                )
            )
            return 0

        if args.command == "run-paper-sim-offline":
            print_result(
                run_paper_sim_offline(
                    settings,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    limit=args.limit,
                    feature_root=args.feature_root,
                    label_root=args.label_root,
                    decision_log_path=args.decision_log_path,
                )
            )
            return 0

        if args.command == "risk-check":
            print_result(risk_check(settings))
            return 0

        if args.command == "validate-features":
            print_result(
                validate_feature_store(
                    settings,
                    feature_root=args.feature_root or settings.paths.feature_dir,
                    feature_set_name=args.feature_set,
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

        if args.command in {"run-session", "session-cycle", "session"}:
            print_result(
                run_session(
                    settings,
                    symbols=args.symbols,
                    feature_root=args.feature_root,
                    latest_per_symbol=args.latest_per_symbol,
                    decision_log_path=args.decision_log_path,
                    execute_requested=args.paper,
                )
            )
            return 0

        if args.command == "run-paper-session":
            print_result(
                run_paper_session(
                    settings,
                    symbols=args.symbols,
                    feature_root=args.feature_root,
                    latest_per_symbol=args.latest_per_symbol,
                    decision_log_path=args.decision_log_path,
                )
            )
            return 0

        if args.command == "run-paper-session-real":
            print_result(
                run_paper_session_real(
                    settings,
                    symbols=args.symbols,
                    feature_root=args.feature_root,
                    latest_per_symbol=args.latest_per_symbol,
                    decision_log_path=args.decision_log_path,
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

        if args.command == "_lob-capture-runner":
            print_result(
                run_lob_capture_loop(
                    settings,
                    symbol=args.symbol,
                    levels=args.levels,
                    session_id=args.session_id,
                    rth_only=_parse_cli_bool(args.rth),
                )
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


def _parse_cli_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean CLI value: {value!r}")


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
