from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from broker.ib_client import IBClient
from config import Settings, load_settings
from data.feature_store import FeatureStore
from engine.session import SessionEngine
from models.registry import ModelRegistry
from monitoring.logging import setup_logger
from risk.risk_manager import RiskManager
from storage.decisions import DecisionStore
from storage.executions import ExecutionAuditStore
from storage.trades import TradeStore


@dataclass(frozen=True)
class RuntimeServices:
    settings: Settings
    client: IBClient
    risk_manager: RiskManager
    feature_store: FeatureStore
    decision_store: DecisionStore
    trade_store: TradeStore
    model_registry: ModelRegistry
    session_engine: SessionEngine


def build_runtime(
    env_file: str | Path | None = None,
    *,
    config_dir: str | Path | None = None,
    environment: str | None = None,
    client_id_override: int | None = None,
) -> RuntimeServices:
    settings = load_settings(env_file, config_dir=config_dir, environment=environment)
    ensure_runtime_directories(settings)
    logger = setup_logger(settings.log_level, settings.log_file)
    audit_store = ExecutionAuditStore(settings.execution_log_file, logger)
    client = IBClient(
        host=settings.ib_host,
        port=settings.ib_port,
        client_id=client_id_override if client_id_override is not None else settings.ib_client_id,
        logger=logger,
        request_timeout=settings.request_timeout_seconds,
        order_follow_up_seconds=settings.order_follow_up_seconds,
        account_summary_group=settings.account_summary_group,
        exchange=settings.ib_exchange,
        currency=settings.ib_currency,
        audit_store=audit_store,
    )
    risk_manager = RiskManager(
        safe_to_trade=settings.safe_to_trade,
        dry_run=settings.dry_run,
        supported_symbols=settings.supported_symbols,
        max_open_positions=settings.risk.max_open_positions,
        max_trades_per_day=settings.risk.max_trades_per_day,
        max_daily_loss_pct=settings.risk.max_daily_loss_pct,
        max_spread_bps=settings.trading.max_spread_bps,
    )
    feature_store = FeatureStore(settings.runtime_db_path, logger, settings.models.sequence_length)
    decision_store = DecisionStore(settings.runtime_db_path, logger)
    trade_store = TradeStore(settings.runtime_db_path, logger)
    model_registry = ModelRegistry(settings.models.registry_path)
    session_engine = SessionEngine(
        settings=settings,
        client=client,
        risk_manager=risk_manager,
        feature_store=feature_store,
        decision_store=decision_store,
        trade_store=trade_store,
        model_registry=model_registry,
    )
    return RuntimeServices(
        settings=settings,
        client=client,
        risk_manager=risk_manager,
        feature_store=feature_store,
        decision_store=decision_store,
        trade_store=trade_store,
        model_registry=model_registry,
        session_engine=session_engine,
    )


def ensure_runtime_directories(settings: Settings) -> None:
    directories = [
        settings.paths.data_root,
        settings.paths.raw_dir,
        settings.paths.market_raw_dir,
        settings.paths.processed_dir,
        settings.paths.feature_dir,
        settings.paths.model_dir,
        settings.paths.model_artifacts_dir,
        settings.paths.log_dir,
        settings.paths.report_dir,
        Path(settings.runtime_db_path).parent,
        Path(settings.execution_log_file).parent,
        Path(settings.log_file).parent,
        Path(settings.models.registry_path).parent,
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
