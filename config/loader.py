from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

import yaml
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
    return _coerce_time(raw_value, name)


def _coerce_time(value: str, label: str) -> time:
    try:
        hours, minutes = str(value).split(":", maxsplit=1)
        return time(hour=int(hours), minute=int(minutes))
    except ValueError as exc:
        raise ValueError(
            f"Invalid time value for {label!r}: {value!r}. Use HH:MM in 24-hour time."
        ) from exc


def _parse_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return _normalize_symbols(raw_value.split(","))


def _normalize_symbols(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(part.strip().upper() for part in values if str(part).strip())
    if not normalized:
        raise ValueError("Supported symbols must not be empty.")
    return normalized


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(project_root: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


@dataclass(frozen=True)
class BrokerSettings:
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002
    ib_client_id: int = 1
    ib_ui_client_id: int = 101
    ib_symbol: str = "SPY"
    ib_exchange: str = "SMART"
    ib_currency: str = "USD"
    supported_symbols: tuple[str, ...] = ("SPY",)
    account_summary_group: str = "All"
    request_timeout_seconds: float = 15.0
    order_follow_up_seconds: float = 5.0
    mode: str = "paper"


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
class PathSettings:
    project_root: str = "."
    config_dir: str = "config"
    data_root: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    feature_dir: str = "data/features"
    model_dir: str = "data/models"
    model_artifacts_dir: str = "data/models/artifacts"
    log_dir: str = "data/logs"
    report_dir: str = "data/reports"


@dataclass(frozen=True)
class DeploymentSettings:
    environment: str = "development"
    machine_role: str = "pc1"
    role: str = "research"
    broker_mode: str = "paper"
    collector_enabled: bool = False
    training_enabled: bool = True
    backtest_enabled: bool = True
    scheduler_enabled: bool = False
    sync_enabled: bool = False


@dataclass(frozen=True)
class Settings:
    broker: BrokerSettings
    session: SessionSettings
    trading: TradingSettings
    risk: RiskSettings
    models: ModelSettings
    storage: StorageSettings
    ui: UISettings
    paths: PathSettings = field(default_factory=PathSettings)
    deployment: DeploymentSettings = field(default_factory=DeploymentSettings)

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
    def ib_ui_client_id(self) -> int:
        return self.broker.ib_ui_client_id

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

    @property
    def environment(self) -> str:
        return self.deployment.environment

    @property
    def data_root(self) -> str:
        return self.paths.data_root

    @property
    def broker_mode(self) -> str:
        return self.deployment.broker_mode

    @property
    def collector_enabled(self) -> bool:
        return self.deployment.collector_enabled

    @property
    def training_enabled(self) -> bool:
        return self.deployment.training_enabled

    @property
    def backtest_enabled(self) -> bool:
        return self.deployment.backtest_enabled

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(
    env_file: str | Path | None = ".env",
    *,
    config_dir: str | Path | None = None,
    environment: str | None = None,
) -> Settings:
    env_file_override = os.getenv("MICROALPHA_ENV_FILE")
    if env_file_override and env_file in {None, ".env", Path(".env")}:
        resolved_env_file = env_file_override
    else:
        resolved_env_file = env_file or ".env"
    env_path = Path(resolved_env_file)
    load_dotenv(env_path if env_path.exists() else None)

    resolved_config_dir = Path(
        config_dir or os.getenv("MICROALPHA_CONFIG_DIR") or Path(__file__).resolve().parent
    ).resolve()
    project_root = resolved_config_dir.parent

    settings_payload = _read_yaml(resolved_config_dir / "settings.yaml")
    risk_payload = _read_yaml(resolved_config_dir / "risk.yaml")
    symbols_payload = _read_yaml(resolved_config_dir / "symbols.yaml")
    deployment_payload = _read_yaml(resolved_config_dir / "deployment.yaml")

    default_settings = settings_payload.get("defaults", {})
    default_environment = str(default_settings.get("environment", "development"))
    resolved_environment = (
        environment
        or os.getenv("APP_ENV")
        or os.getenv("MICROALPHA_ENV")
        or default_environment
    )

    merged_settings = _deep_merge(
        default_settings,
        settings_payload.get("environments", {}).get(resolved_environment, {}),
    )
    merged_risk = _deep_merge(
        risk_payload.get("defaults", {}),
        risk_payload.get("environments", {}).get(resolved_environment, {}),
    )
    merged_deployment = _deep_merge(
        deployment_payload.get("defaults", {}),
        deployment_payload.get("environments", {}).get(resolved_environment, {}),
    )

    timezone_value = os.getenv(
        "TIMEZONE",
        os.getenv("MARKET_TIMEZONE", str(merged_settings.get("timezone", "America/New_York"))),
    )

    default_symbol = os.getenv(
        "IB_SYMBOL",
        str(symbols_payload.get("default_symbol", merged_settings.get("default_symbol", "SPY"))),
    ).upper()
    supported_symbols = _parse_csv(
        "SUPPORTED_SYMBOLS",
        _normalize_symbols(
            tuple(symbols_payload.get("supported_symbols", (default_symbol,)))
        ),
    )

    data_root_raw = os.getenv("DATA_ROOT", str(merged_settings.get("data_root", "data")))
    path_overrides = merged_settings.get("paths", {})
    raw_dir_raw = str(path_overrides.get("raw_dir", Path(data_root_raw) / "raw"))
    processed_dir_raw = str(path_overrides.get("processed_dir", Path(data_root_raw) / "processed"))
    feature_dir_raw = str(path_overrides.get("feature_dir", Path(data_root_raw) / "features"))
    model_dir_raw = str(path_overrides.get("model_dir", Path(data_root_raw) / "models"))
    model_artifacts_dir_raw = os.getenv(
        "MODEL_ARTIFACTS_DIR",
        str(path_overrides.get("model_artifacts_dir", Path(model_dir_raw) / "artifacts")),
    )
    log_dir_raw = str(path_overrides.get("log_dir", Path(data_root_raw) / "logs"))
    report_dir_raw = str(path_overrides.get("report_dir", Path(data_root_raw) / "reports"))
    log_file_raw = os.getenv("LOG_FILE", str(path_overrides.get("log_file", Path(log_dir_raw) / "microalpha.log")))
    execution_log_raw = os.getenv(
        "EXECUTION_LOG_FILE",
        str(path_overrides.get("execution_log_file", Path(report_dir_raw) / "executions.csv")),
    )
    runtime_db_raw = os.getenv(
        "RUNTIME_DB_PATH",
        str(path_overrides.get("runtime_db_path", Path(processed_dir_raw) / "runtime" / "microalpha.db")),
    )
    registry_path_raw = os.getenv(
        "MODEL_REGISTRY_PATH",
        str(path_overrides.get("model_registry_path", Path(model_artifacts_dir_raw) / "registry.json")),
    )

    paths = PathSettings(
        project_root=str(project_root),
        config_dir=str(resolved_config_dir),
        data_root=_resolve_path(project_root, data_root_raw),
        raw_dir=_resolve_path(project_root, raw_dir_raw),
        processed_dir=_resolve_path(project_root, processed_dir_raw),
        feature_dir=_resolve_path(project_root, feature_dir_raw),
        model_dir=_resolve_path(project_root, model_dir_raw),
        model_artifacts_dir=_resolve_path(project_root, model_artifacts_dir_raw),
        log_dir=_resolve_path(project_root, log_dir_raw),
        report_dir=_resolve_path(project_root, report_dir_raw),
    )

    deployment = DeploymentSettings(
        environment=resolved_environment,
        machine_role=str(merged_deployment.get("machine_role", "pc1" if resolved_environment == "development" else "pc2")),
        role=str(merged_deployment.get("role", "research" if resolved_environment == "development" else "deploy")),
        broker_mode=os.getenv("BROKER_MODE", str(merged_settings.get("broker_mode", "paper"))),
        collector_enabled=_parse_bool(
            "COLLECTOR_ENABLED",
            bool(merged_settings.get("collector_enabled", resolved_environment == "deploy")),
        ),
        training_enabled=_parse_bool(
            "TRAINING_ENABLED",
            bool(merged_settings.get("training_enabled", resolved_environment == "development")),
        ),
        backtest_enabled=_parse_bool(
            "BACKTEST_ENABLED",
            bool(merged_settings.get("backtest_enabled", resolved_environment == "development")),
        ),
        scheduler_enabled=_parse_bool(
            "SCHEDULER_ENABLED",
            bool(merged_deployment.get("scheduler_enabled", resolved_environment == "deploy")),
        ),
        sync_enabled=_parse_bool(
            "SYNC_ENABLED",
            bool(merged_deployment.get("sync_enabled", resolved_environment == "deploy")),
        ),
    )

    session_defaults = merged_settings.get("session", {})
    trading_defaults = merged_settings.get("trading", {})
    model_defaults = merged_settings.get("models", {})
    ui_defaults = merged_settings.get("ui", {})

    return Settings(
        broker=BrokerSettings(
            ib_host=os.getenv("IB_HOST", str(merged_settings.get("ib_host", "127.0.0.1"))),
            ib_port=_parse_int("IB_PORT", int(merged_settings.get("ib_port", 4002))),
            ib_client_id=_parse_int("IB_CLIENT_ID", int(merged_settings.get("ib_client_id", 1))),
            ib_ui_client_id=_parse_int("IB_UI_CLIENT_ID", int(merged_settings.get("ib_ui_client_id", 101))),
            ib_symbol=default_symbol,
            ib_exchange=os.getenv("IB_EXCHANGE", str(merged_settings.get("ib_exchange", "SMART"))).upper(),
            ib_currency=os.getenv("IB_CURRENCY", str(merged_settings.get("ib_currency", "USD"))).upper(),
            supported_symbols=supported_symbols,
            account_summary_group=os.getenv(
                "IB_ACCOUNT_SUMMARY_GROUP",
                str(merged_settings.get("ib_account_summary_group", "All")),
            ),
            request_timeout_seconds=_parse_float(
                "IB_REQUEST_TIMEOUT_SECONDS",
                float(merged_settings.get("ib_request_timeout_seconds", 15.0)),
            ),
            order_follow_up_seconds=_parse_float(
                "IB_ORDER_FOLLOW_UP_SECONDS",
                float(merged_settings.get("ib_order_follow_up_seconds", 5.0)),
            ),
            mode=deployment.broker_mode,
        ),
        session=SessionSettings(
            timezone=timezone_value,
            orb_start=_parse_time("ORB_START", str(session_defaults.get("orb_start", "09:30"))),
            orb_end=_parse_time("ORB_END", str(session_defaults.get("orb_end", "09:45"))),
            primary_session_end=_parse_time(
                "PRIMARY_SESSION_END",
                str(session_defaults.get("primary_session_end", "11:30")),
            ),
            secondary_session_start=_parse_time(
                "SECONDARY_SESSION_START",
                str(session_defaults.get("secondary_session_start", "13:30")),
            ),
            secondary_session_end=_parse_time(
                "SECONDARY_SESSION_END",
                str(session_defaults.get("secondary_session_end", "15:00")),
            ),
            enable_secondary_session=_parse_bool(
                "ENABLE_SECONDARY_SESSION",
                bool(session_defaults.get("enable_secondary_session", False)),
            ),
            flatten_before_close_minutes=_parse_int(
                "FLATTEN_BEFORE_CLOSE_MINUTES",
                int(session_defaults.get("flatten_before_close_minutes", 5)),
            ),
            regular_market_open=_coerce_time(str(session_defaults.get("regular_market_open", "09:30")), "regular_market_open"),
            regular_market_close=_coerce_time(str(session_defaults.get("regular_market_close", "16:00")), "regular_market_close"),
        ),
        trading=TradingSettings(
            default_order_quantity=_parse_int(
                "DEFAULT_ORDER_QUANTITY",
                int(trading_defaults.get("default_order_quantity", 1)),
            ),
            dry_run=_parse_bool("DRY_RUN", bool(trading_defaults.get("dry_run", True))),
            safe_to_trade=_parse_bool(
                "SAFE_TO_TRADE",
                bool(trading_defaults.get("safe_to_trade", merged_settings.get("safe_to_trade", False))),
            ),
            allow_shorts=_parse_bool("ALLOW_SHORTS", bool(trading_defaults.get("allow_shorts", False))),
            data_mode=os.getenv("DATA_MODE", str(trading_defaults.get("data_mode", "paper_or_local"))),
            allow_session_execution=_parse_bool(
                "ALLOW_SESSION_EXECUTION",
                bool(trading_defaults.get("allow_session_execution", False)),
            ),
            entry_limit_buffer_bps=_parse_float(
                "ENTRY_LIMIT_BUFFER_BPS",
                float(trading_defaults.get("entry_limit_buffer_bps", 2.0)),
            ),
            cost_buffer_bps=_parse_float(
                "COST_BUFFER_BPS",
                float(trading_defaults.get("cost_buffer_bps", 2.5)),
            ),
            max_spread_bps=_parse_float(
                "MAX_SPREAD_BPS",
                float(trading_defaults.get("max_spread_bps", 8.0)),
            ),
            max_hold_minutes=_parse_int(
                "MAX_HOLD_MINUTES",
                int(trading_defaults.get("max_hold_minutes", 30)),
            ),
        ),
        risk=RiskSettings(
            max_trades_per_day=_parse_int(
                "MAX_TRADES_PER_DAY",
                int(merged_risk.get("max_trades_per_day", 2)),
            ),
            max_daily_loss_pct=_parse_float(
                "MAX_DAILY_LOSS_PCT",
                float(merged_risk.get("max_daily_loss_pct", 1.0)),
            ),
            max_open_positions=_parse_int(
                "MAX_OPEN_POSITIONS",
                int(merged_risk.get("max_open_positions", 1)),
            ),
        ),
        models=ModelSettings(
            model_prob_threshold=_parse_float(
                "MODEL_PROB_THRESHOLD",
                float(model_defaults.get("model_prob_threshold", 0.58)),
            ),
            target_horizon_minutes=_parse_int(
                "TARGET_HORIZON_MINUTES",
                int(model_defaults.get("target_horizon_minutes", 3)),
            ),
            active_baseline_model=os.getenv("ACTIVE_BASELINE_MODEL") or model_defaults.get("active_baseline_model") or None,
            active_deep_model=os.getenv("ACTIVE_DEEP_MODEL") or model_defaults.get("active_deep_model") or None,
            baseline_weight=_parse_float(
                "BASELINE_WEIGHT",
                float(model_defaults.get("baseline_weight", 0.4)),
            ),
            deep_weight=_parse_float(
                "DEEP_WEIGHT",
                float(model_defaults.get("deep_weight", 0.6)),
            ),
            artifacts_dir=_resolve_path(project_root, model_artifacts_dir_raw),
            registry_path=_resolve_path(project_root, registry_path_raw),
            sequence_length=_parse_int(
                "MODEL_SEQUENCE_LENGTH",
                int(model_defaults.get("sequence_length", 16)),
            ),
        ),
        storage=StorageSettings(
            log_level=os.getenv("LOG_LEVEL", str(merged_settings.get("log_level", "INFO"))).upper(),
            log_file=_resolve_path(project_root, log_file_raw),
            execution_log_file=_resolve_path(project_root, execution_log_raw),
            runtime_db_path=_resolve_path(project_root, runtime_db_raw),
        ),
        ui=UISettings(
            host=os.getenv("UI_HOST", str(ui_defaults.get("host", "127.0.0.1"))),
            port=_parse_int("UI_PORT", int(ui_defaults.get("port", 8501))),
            title=os.getenv("UI_TITLE", str(ui_defaults.get("title", "MicroAlpha IBKR"))),
        ),
        paths=paths,
        deployment=deployment,
    )
