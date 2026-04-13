from __future__ import annotations

from dataclasses import dataclass

from broker.ib_client import IBClient
from config import Settings, load_settings
from data.feature_store import FeatureStore
from engine.session import SessionEngine
from models.registry import ModelRegistry
from risk.risk_manager import RiskManager
from storage.decisions import DecisionStore
from storage.executions import ExecutionAuditStore
from storage.logger import setup_logger
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


def build_runtime(env_file: str = ".env") -> RuntimeServices:
    settings = load_settings(env_file)
    logger = setup_logger(settings.log_level, settings.log_file)
    audit_store = ExecutionAuditStore(settings.execution_log_file, logger)
    client = IBClient(
        host=settings.ib_host,
        port=settings.ib_port,
        client_id=settings.ib_client_id,
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
