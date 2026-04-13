from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _parse_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value for {name!r}: {raw_value!r}. "
        "Use true/false, yes/no, on/off, or 1/0."
    )


def _parse_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {name!r}: {raw_value!r}.") from exc


def _parse_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {name!r}: {raw_value!r}.") from exc


def _parse_time(name: str, default: str) -> time:
    raw_value = os.getenv(name, default).strip()
    try:
        hours, minutes = raw_value.split(":", maxsplit=1)
        return time(hour=int(hours), minute=int(minutes))
    except ValueError as exc:
        raise ValueError(
            f"Invalid time value for {name!r}: {raw_value!r}. Use HH:MM in 24-hour time."
        ) from exc


def _parse_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    values = tuple(part.strip().upper() for part in raw_value.split(",") if part.strip())
    if not values:
        raise ValueError(f"Invalid CSV value for {name!r}: {raw_value!r}.")
    return values


@dataclass(frozen=True)
class BrokerSettings:
    ib_host: str
    ib_port: int
    ib_client_id: int
    ib_symbol: str
    ib_exchange: str
    ib_currency: str
    supported_symbols: tuple[str, ...]
    account_summary_group: str
    request_timeout_seconds: float
    order_follow_up_seconds: float


@dataclass(frozen=True)
class SessionSettings:
    timezone: str
    orb_start: time
    orb_end: time
    primary_session_end: time
    secondary_session_start: time
    secondary_session_end: time
    enable_secondary_session: bool
    flatten_before_close_minutes: int
    regular_market_open: time = time(9, 30)
    regular_market_close: time = time(16, 0)


@dataclass(frozen=True)
class TradingSettings:
    default_order_quantity: int
    dry_run: bool
    safe_to_trade: bool
    allow_shorts: bool
    data_mode: str
    allow_session_execution: bool
    entry_limit_buffer_bps: float
    cost_buffer_bps: float
    max_spread_bps: float
    max_hold_minutes: int


@dataclass(frozen=True)
class RiskSettings:
    max_trades_per_day: int
    max_daily_loss_pct: float
    max_open_positions: int


@dataclass(frozen=True)
class ModelSettings:
    model_prob_threshold: float
    target_horizon_minutes: int
    active_baseline_model: str | None
    active_deep_model: str | None
    baseline_weight: float
    deep_weight: float
    artifacts_dir: str
    registry_path: str
    sequence_length: int


@dataclass(frozen=True)
class StorageSettings:
    log_level: str
    log_file: str
    execution_log_file: str
    runtime_db_path: str


@dataclass(frozen=True)
class UISettings:
    host: str
    port: int
    title: str


@dataclass(frozen=True)
class Settings:
    broker: BrokerSettings
    session: SessionSettings
    trading: TradingSettings
    risk: RiskSettings
    models: ModelSettings
    storage: StorageSettings
    ui: UISettings

    @property
    def ib_host(self) -> str:
        return self.broker.ib_host

    @property
    def ib_port(self) -> int:
        return self.broker.ib_port

    @property
    def ib_client_id(self) -> int:
        return self.broker.ib_client_id

    @property
    def ib_symbol(self) -> str:
        return self.broker.ib_symbol

    @property
    def ib_exchange(self) -> str:
        return self.broker.ib_exchange

    @property
    def ib_currency(self) -> str:
        return self.broker.ib_currency

    @property
    def supported_symbols(self) -> tuple[str, ...]:
        return self.broker.supported_symbols

    @property
    def default_order_quantity(self) -> int:
        return self.trading.default_order_quantity

    @property
    def dry_run(self) -> bool:
        return self.trading.dry_run

    @property
    def safe_to_trade(self) -> bool:
        return self.trading.safe_to_trade

    @property
    def log_level(self) -> str:
        return self.storage.log_level

    @property
    def log_file(self) -> str:
        return self.storage.log_file

    @property
    def execution_log_file(self) -> str:
        return self.storage.execution_log_file

    @property
    def request_timeout_seconds(self) -> float:
        return self.broker.request_timeout_seconds

    @property
    def order_follow_up_seconds(self) -> float:
        return self.broker.order_follow_up_seconds

    @property
    def account_summary_group(self) -> str:
        return self.broker.account_summary_group

    @property
    def market_timezone(self) -> str:
        return self.session.timezone

    @property
    def runtime_db_path(self) -> str:
        return self.storage.runtime_db_path

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def load_settings(env_file: str | Path = ".env") -> Settings:
    env_path = Path(env_file)
    load_dotenv(env_path if env_path.exists() else None)

    ib_symbol = os.getenv("IB_SYMBOL", "SPY").upper()
    artifacts_dir = os.getenv("MODEL_ARTIFACTS_DIR", "models/artifacts")
    registry_path = os.getenv("MODEL_REGISTRY_PATH", f"{artifacts_dir}/registry.json")

    return Settings(
        broker=BrokerSettings(
            ib_host=os.getenv("IB_HOST", "127.0.0.1"),
            ib_port=_parse_int("IB_PORT", 4002),
            ib_client_id=_parse_int("IB_CLIENT_ID", 1),
            ib_symbol=ib_symbol,
            ib_exchange=os.getenv("IB_EXCHANGE", "SMART").upper(),
            ib_currency=os.getenv("IB_CURRENCY", "USD").upper(),
            supported_symbols=_parse_csv("SUPPORTED_SYMBOLS", (ib_symbol,)),
            account_summary_group=os.getenv("IB_ACCOUNT_SUMMARY_GROUP", "All"),
            request_timeout_seconds=_parse_float("IB_REQUEST_TIMEOUT_SECONDS", 15.0),
            order_follow_up_seconds=_parse_float("IB_ORDER_FOLLOW_UP_SECONDS", 5.0),
        ),
        session=SessionSettings(
            timezone=os.getenv("TIMEZONE", os.getenv("MARKET_TIMEZONE", "America/New_York")),
            orb_start=_parse_time("ORB_START", "09:30"),
            orb_end=_parse_time("ORB_END", "09:45"),
            primary_session_end=_parse_time("PRIMARY_SESSION_END", "11:30"),
            secondary_session_start=_parse_time("SECONDARY_SESSION_START", "13:30"),
            secondary_session_end=_parse_time("SECONDARY_SESSION_END", "15:00"),
            enable_secondary_session=_parse_bool("ENABLE_SECONDARY_SESSION", False),
            flatten_before_close_minutes=_parse_int("FLATTEN_BEFORE_CLOSE_MINUTES", 5),
        ),
        trading=TradingSettings(
            default_order_quantity=_parse_int("DEFAULT_ORDER_QUANTITY", 1),
            dry_run=_parse_bool("DRY_RUN", True),
            safe_to_trade=_parse_bool("SAFE_TO_TRADE", False),
            allow_shorts=_parse_bool("ALLOW_SHORTS", False),
            data_mode=os.getenv("DATA_MODE", "paper_or_local"),
            allow_session_execution=_parse_bool("ALLOW_SESSION_EXECUTION", False),
            entry_limit_buffer_bps=_parse_float("ENTRY_LIMIT_BUFFER_BPS", 2.0),
            cost_buffer_bps=_parse_float("COST_BUFFER_BPS", 2.5),
            max_spread_bps=_parse_float("MAX_SPREAD_BPS", 8.0),
            max_hold_minutes=_parse_int("MAX_HOLD_MINUTES", 30),
        ),
        risk=RiskSettings(
            max_trades_per_day=_parse_int("MAX_TRADES_PER_DAY", 2),
            max_daily_loss_pct=_parse_float("MAX_DAILY_LOSS_PCT", 1.0),
            max_open_positions=_parse_int("MAX_OPEN_POSITIONS", 1),
        ),
        models=ModelSettings(
            model_prob_threshold=_parse_float("MODEL_PROB_THRESHOLD", 0.58),
            target_horizon_minutes=_parse_int("TARGET_HORIZON_MINUTES", 3),
            active_baseline_model=os.getenv("ACTIVE_BASELINE_MODEL") or None,
            active_deep_model=os.getenv("ACTIVE_DEEP_MODEL") or None,
            baseline_weight=_parse_float("BASELINE_WEIGHT", 0.4),
            deep_weight=_parse_float("DEEP_WEIGHT", 0.6),
            artifacts_dir=artifacts_dir,
            registry_path=registry_path,
            sequence_length=_parse_int("MODEL_SEQUENCE_LENGTH", 16),
        ),
        storage=StorageSettings(
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_file=os.getenv("LOG_FILE", "logs/ibkr_mvp.log"),
            execution_log_file=os.getenv("EXECUTION_LOG_FILE", "logs/executions.csv"),
            runtime_db_path=os.getenv("RUNTIME_DB_PATH", "runtime/microalpha.db"),
        ),
        ui=UISettings(
            host=os.getenv("UI_HOST", "127.0.0.1"),
            port=_parse_int("UI_PORT", 8501),
            title=os.getenv("UI_TITLE", "MicroAlpha IBKR"),
        ),
    )
