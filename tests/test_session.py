from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import (
    BrokerSettings,
    ModelSettings,
    RiskSettings,
    SessionSettings,
    Settings,
    StorageSettings,
    TradingSettings,
    UISettings,
)
from data.feature_store import FeatureStore
from engine.session import SessionEngine
from models.registry import ModelRegistry
from risk.risk_manager import RiskManager
from storage.decisions import DecisionStore
from storage.trades import TradeStore


class FakeSessionClient:
    def __init__(self) -> None:
        self.connected = False
        self.logger = logging.getLogger("fake-session-client")

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def get_server_time(self) -> dict[str, Any]:
        epoch = int(datetime(2026, 4, 6, 15, 0, tzinfo=timezone.utc).timestamp())
        return {"epoch": epoch, "iso_utc": "2026-04-06T15:00:00+00:00"}

    def get_account_summary(self) -> list[dict[str, str]]:
        return [{"account": "DU123", "tag": "NetLiquidation", "value": "1000", "currency": "USD"}]

    def get_positions(self) -> list[dict[str, Any]]:
        return []

    def get_market_snapshot(self, symbol: str, *, exchange: str, currency: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "snapshot_utc": "2026-04-06T15:00:00+00:00",
            "bid": 500.00,
            "ask": 500.08,
            "last": 500.06,
            "volume": 1000000,
            "bid_size": 200,
            "ask_size": 180,
            "bid_size_00": 200,
            "ask_size_00": 180,
        }

    def get_historical_bars(
        self,
        *,
        symbol: str,
        exchange: str,
        currency: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ) -> list[dict[str, Any]]:
        start = datetime(2026, 4, 6, 13, 30, tzinfo=timezone.utc)
        rows = []
        for minute in range(20):
            timestamp = (start + timedelta(minutes=minute)).isoformat()
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": 499.90 + (minute * 0.01),
                    "high": 500.10 + (minute * 0.02),
                    "low": 499.80 + (minute * 0.01),
                    "close": 500.00 + (minute * 0.01),
                    "volume": 1000 + minute,
                }
            )
        return rows


class AfterHoursSessionClient(FakeSessionClient):
    def get_server_time(self) -> dict[str, Any]:
        epoch = int(datetime(2026, 4, 6, 22, 10, tzinfo=timezone.utc).timestamp())
        return {"epoch": epoch, "iso_utc": "2026-04-06T22:10:00+00:00"}


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        broker=BrokerSettings(
            ib_host="127.0.0.1",
            ib_port=4002,
            ib_client_id=1,
            ib_symbol="SPY",
            ib_exchange="SMART",
            ib_currency="USD",
            supported_symbols=("SPY",),
            account_summary_group="All",
            request_timeout_seconds=5.0,
            order_follow_up_seconds=1.0,
        ),
        session=SessionSettings(
            timezone="America/New_York",
            orb_start=datetime.strptime("09:30", "%H:%M").time(),
            orb_end=datetime.strptime("09:45", "%H:%M").time(),
            primary_session_end=datetime.strptime("11:30", "%H:%M").time(),
            secondary_session_start=datetime.strptime("13:30", "%H:%M").time(),
            secondary_session_end=datetime.strptime("15:00", "%H:%M").time(),
            enable_secondary_session=False,
            flatten_before_close_minutes=5,
        ),
        trading=TradingSettings(
            default_order_quantity=1,
            dry_run=True,
            safe_to_trade=False,
            allow_shorts=False,
            data_mode="paper_or_local",
            allow_session_execution=False,
            entry_limit_buffer_bps=2.0,
            cost_buffer_bps=2.5,
            max_spread_bps=8.0,
            max_hold_minutes=30,
        ),
        risk=RiskSettings(
            max_trades_per_day=2,
            max_daily_loss_pct=1.0,
            max_open_positions=1,
        ),
        models=ModelSettings(
            model_prob_threshold=0.58,
            target_horizon_minutes=3,
            active_baseline_model=None,
            active_deep_model=None,
            baseline_weight=0.4,
            deep_weight=0.6,
            artifacts_dir=str(tmp_path / "models" / "artifacts"),
            registry_path=str(tmp_path / "models" / "artifacts" / "registry.json"),
            sequence_length=16,
        ),
        storage=StorageSettings(
            log_level="INFO",
            log_file=str(tmp_path / "logs" / "test.log"),
            execution_log_file=str(tmp_path / "logs" / "executions.csv"),
            runtime_db_path=str(tmp_path / "runtime" / "microalpha.db"),
        ),
        ui=UISettings(
            host="127.0.0.1",
            port=8501,
            title="MicroAlpha Test UI",
        ),
    )


def build_engine(tmp_path: Path, client: FakeSessionClient) -> tuple[SessionEngine, DecisionStore]:
    settings = build_settings(tmp_path)
    logger = logging.getLogger(f"test-session-{tmp_path.name}")
    feature_store = FeatureStore(settings.runtime_db_path, logger, settings.models.sequence_length)
    decision_store = DecisionStore(settings.runtime_db_path, logger)
    trade_store = TradeStore(settings.runtime_db_path, logger)
    model_registry = ModelRegistry(settings.models.registry_path)
    risk_manager = RiskManager(
        safe_to_trade=settings.safe_to_trade,
        dry_run=settings.dry_run,
        supported_symbols=settings.supported_symbols,
        max_open_positions=settings.risk.max_open_positions,
        max_trades_per_day=settings.risk.max_trades_per_day,
        max_daily_loss_pct=settings.risk.max_daily_loss_pct,
        max_spread_bps=settings.trading.max_spread_bps,
    )
    engine = SessionEngine(
        settings=settings,
        client=client,
        risk_manager=risk_manager,
        feature_store=feature_store,
        decision_store=decision_store,
        trade_store=trade_store,
        model_registry=model_registry,
    )
    return engine, decision_store


def test_session_cycle_runs_without_execution_request(tmp_path: Path) -> None:
    engine, decision_store = build_engine(tmp_path, FakeSessionClient())

    payload = engine.run_cycle()

    assert payload["connection"]["connected"] is True
    assert payload["market_status"]["is_market_open"] is True
    assert payload["execution"]["requested"] is False
    assert payload["execution"]["submitted"] is False
    latest = decision_store.get_latest_decision()
    assert latest is not None
    assert latest["symbol"] == "SPY"


def test_session_cycle_marks_market_closed_after_hours(tmp_path: Path) -> None:
    engine, _ = build_engine(tmp_path, AfterHoursSessionClient())

    payload = engine.run_cycle(execute_requested=True)

    assert payload["market_status"]["is_market_open"] is False
    assert payload["market_status"]["session_window"] == "closed"
    assert payload["execution"]["requested"] is True
    assert payload["execution"]["submitted"] is False
