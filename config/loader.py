from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import time
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values


def _parse_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw_value = env.get(name)
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


def _parse_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {name!r}: {raw_value!r}.") from exc


def _parse_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {name!r}: {raw_value!r}.") from exc


def _parse_time(env: Mapping[str, str], name: str, default: str) -> time:
    raw_value = str(env.get(name, default)).strip()
    return _coerce_time(raw_value, name)


def _coerce_time(value: str, label: str) -> time:
    try:
        hours, minutes = str(value).split(":", maxsplit=1)
        return time(hour=int(hours), minute=int(minutes))
    except ValueError as exc:
        raise ValueError(
            f"Invalid time value for {label!r}: {value!r}. Use HH:MM in 24-hour time."
        ) from exc


def _parse_csv(env: Mapping[str, str], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = env.get(name)
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
    ib_collector_client_id: int = 201
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
    dataset_type: str = "bar_bootstrap"
    lob_sequence_length: int = 100
    lob_depth_levels: int = 10
    lob_horizon_events: int = 10
    lob_stationary_threshold_bps: float = 2.0
    lob_train_batch_size: int = 32
    lob_eval_batch_size: int = 64


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
class CollectorSettings:
    mode: str = "snapshot_polling"
    poll_interval_seconds: float = 5.0
    flush_interval_seconds: float = 30.0
    batch_size: int = 50
    reconnect_delay_seconds: float = 10.0
    max_reconnect_attempts: int = 5
    health_log_interval_seconds: float = 60.0


@dataclass(frozen=True)
class LOBCaptureSettings:
    enabled: bool = False
    symbols: tuple[str, ...] = ("SPY",)
    depth_levels: int = 10
    flush_interval_seconds: float = 15.0
    batch_size: int = 250
    output_root: str = "data/raw/ibkr_lob"
    state_root: str = "data/processed/ibkr_lob_state"
    session_root: str = "data/processed/ibkr_lob_sessions"
    dataset_root: str = "data/processed/lob_datasets"
    report_root: str = "data/reports/lob"
    rth_only: bool = True
    reconnect_delay_seconds: float = 10.0
    max_reconnect_attempts: int = 5
    startup_wait_seconds: float = 5.0


@dataclass(frozen=True)
class KrakenLOBSettings:
    enabled: bool = False
    symbol: str = "BTC/EUR"
    depth_levels: int = 10
    flush_interval_seconds: float = 15.0
    batch_size: int = 250
    output_root: str = "data/raw/kraken_lob"
    state_root: str = "data/processed/kraken_lob_state"
    session_root: str = "data/processed/kraken_lob_sessions"
    websocket_url: str = "wss://ws.kraken.com/v2"
    reconnect_delay_seconds: float = 10.0
    max_reconnect_attempts: int = 5
    startup_wait_seconds: float = 5.0
    paper_fee_bps: float = 26.0
    paper_initial_cash_eur: float = 10000.0
    paper_initial_cash_mode: str = "dynamic_minimum"
    paper_min_cash_buffer_bps: float = 1000.0
    paper_position_fraction: float = 0.25
    paper_slippage_bps: float = 2.0
    paper_ui_refresh_seconds: float = 2.0
    paper_ui_port: int = 8502


@dataclass(frozen=True)
class FeaturePipelineSettings:
    gap_threshold_seconds: int = 120
    max_abs_spread_bps: float = 250.0
    forward_fill_limit: int = 2
    drop_outside_regular_hours: bool = True
    rolling_short_window: int = 5
    rolling_medium_window: int = 15
    rolling_long_window: int = 30
    vwap_window: int = 30
    volume_window: int = 30
    label_horizon_rows: int = 1
    train_split_ratio: float = 0.8
    default_feature_set: str = "hybrid_intraday"
    validation_max_nan_ratio: float = 0.35


@dataclass(frozen=True)
class LanSyncSettings:
    pc2_network_root: str | None = None
    source_market_subdir: str = "data/raw/market"
    source_meta_subdir: str = "data/meta"
    source_log_subdir: str = "data/logs"
    include_raw: bool = True
    include_meta: bool = True
    include_logs: bool = False
    dry_run: bool = False
    overwrite_policy: str = "if_newer"
    validate_parquet: bool = True
    allowed_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class PathSettings:
    project_root: str = "."
    config_dir: str = "config"
    data_root: str = "data"
    raw_dir: str = "data/raw"
    market_raw_dir: str = "data/raw/market"
    processed_dir: str = "data/processed"
    feature_dir: str = "data/features"
    meta_dir: str = "data/meta"
    model_dir: str = "data/models"
    model_artifacts_dir: str = "data/models/artifacts"
    log_dir: str = "data/logs"
    report_dir: str = "data/reports"
    sqlite_backup_dir: str = "data/meta/sqlite_backups"
    sync_report_dir: str = "data/reports/sync"
    import_root: str = "imports/from_pc2"
    import_market_dir: str = "imports/from_pc2/raw/market"
    import_meta_dir: str = "imports/from_pc2/meta"
    import_log_dir: str = "imports/from_pc2/logs"
    transfer_log_path: str = "imports/from_pc2/transfer_log.jsonl"
    transfer_report_dir: str = "data/reports/lan_sync"


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
    collector: CollectorSettings = field(default_factory=CollectorSettings)
    lob_capture: LOBCaptureSettings = field(default_factory=LOBCaptureSettings)
    kraken_lob: KrakenLOBSettings = field(default_factory=KrakenLOBSettings)
    feature_pipeline: FeaturePipelineSettings = field(default_factory=FeaturePipelineSettings)
    lan_sync: LanSyncSettings = field(default_factory=LanSyncSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    deployment: DeploymentSettings = field(default_factory=DeploymentSettings)
    env_file: str | None = None

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
    def ib_collector_client_id(self) -> int:
        return self.broker.ib_collector_client_id

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
    file_env = {
        key: str(value)
        for key, value in dotenv_values(env_path if env_path.exists() else None).items()
        if value is not None
    }
    runtime_env = {**file_env, **os.environ}

    resolved_config_dir = Path(
        config_dir or runtime_env.get("MICROALPHA_CONFIG_DIR") or Path(__file__).resolve().parent
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
        or runtime_env.get("APP_ENV")
        or runtime_env.get("MICROALPHA_ENV")
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

    timezone_value = runtime_env.get("TIMEZONE") or runtime_env.get(
        "MARKET_TIMEZONE",
        str(merged_settings.get("timezone", "America/New_York")),
    )

    default_symbol = runtime_env.get(
        "IB_SYMBOL",
        str(symbols_payload.get("default_symbol", merged_settings.get("default_symbol", "SPY"))),
    ).upper()
    supported_symbols = _parse_csv(
        runtime_env,
        "SUPPORTED_SYMBOLS",
        _normalize_symbols(
            tuple(symbols_payload.get("supported_symbols", (default_symbol,)))
        ),
    )

    default_data_root = str(merged_settings.get("data_root", "data"))
    data_root_raw = runtime_env.get("DATA_ROOT", default_data_root)
    path_overrides = merged_settings.get("paths", {})
    data_root_is_overridden = data_root_raw != default_data_root

    def _path_value(
        env_name: str | None,
        key: str,
        relative_default: str,
        *,
        explicit_default: str | None = None,
    ) -> str:
        if env_name:
            env_value = runtime_env.get(env_name)
            if env_value is not None:
                return env_value

        configured = path_overrides.get(key)
        default_value = explicit_default or str(Path(default_data_root) / relative_default)
        if configured is None:
            return str(Path(data_root_raw) / relative_default)
        if data_root_is_overridden and str(configured) == default_value:
            return str(Path(data_root_raw) / relative_default)
        return str(configured)

    raw_dir_raw = _path_value(None, "raw_dir", "raw")
    market_raw_dir_raw = _path_value("MARKET_RAW_DIR", "market_raw_dir", "raw/market")
    processed_dir_raw = _path_value(None, "processed_dir", "processed")
    feature_dir_raw = _path_value(None, "feature_dir", "features")
    meta_dir_raw = _path_value("META_DIR", "meta_dir", "meta")
    model_dir_raw = _path_value(None, "model_dir", "models")
    model_artifacts_dir_raw = _path_value("MODEL_ARTIFACTS_DIR", "model_artifacts_dir", "models/artifacts")
    log_dir_raw = _path_value(None, "log_dir", "logs")
    report_dir_raw = _path_value(None, "report_dir", "reports")
    sqlite_backup_dir_raw = _path_value("SQLITE_BACKUP_DIR", "sqlite_backup_dir", "meta/sqlite_backups")
    sync_report_dir_raw = _path_value("SYNC_REPORT_DIR", "sync_report_dir", "reports/sync")
    import_root_raw = _path_value("IMPORT_ROOT", "import_root", "../imports/from_pc2", explicit_default="imports/from_pc2")
    import_market_dir_raw = _path_value(
        "IMPORT_MARKET_DIR",
        "import_market_dir",
        "../imports/from_pc2/raw/market",
        explicit_default="imports/from_pc2/raw/market",
    )
    import_meta_dir_raw = _path_value(
        "IMPORT_META_DIR",
        "import_meta_dir",
        "../imports/from_pc2/meta",
        explicit_default="imports/from_pc2/meta",
    )
    import_log_dir_raw = _path_value(
        "IMPORT_LOG_DIR",
        "import_log_dir",
        "../imports/from_pc2/logs",
        explicit_default="imports/from_pc2/logs",
    )
    transfer_log_path_raw = _path_value(
        "TRANSFER_LOG_PATH",
        "transfer_log_path",
        "../imports/from_pc2/transfer_log.jsonl",
        explicit_default="imports/from_pc2/transfer_log.jsonl",
    )
    transfer_report_dir_raw = _path_value(
        "TRANSFER_REPORT_DIR",
        "transfer_report_dir",
        "reports/lan_sync",
    )
    log_file_raw = _path_value("LOG_FILE", "log_file", "logs/microalpha.log")
    execution_log_raw = _path_value("EXECUTION_LOG_FILE", "execution_log_file", "reports/executions.csv")
    runtime_db_raw = _path_value("RUNTIME_DB_PATH", "runtime_db_path", "processed/runtime/microalpha.db")
    registry_path_raw = _path_value("MODEL_REGISTRY_PATH", "model_registry_path", "models/artifacts/registry.json")

    paths = PathSettings(
        project_root=str(project_root),
        config_dir=str(resolved_config_dir),
        data_root=_resolve_path(project_root, data_root_raw),
        raw_dir=_resolve_path(project_root, raw_dir_raw),
        market_raw_dir=_resolve_path(project_root, market_raw_dir_raw),
        processed_dir=_resolve_path(project_root, processed_dir_raw),
        feature_dir=_resolve_path(project_root, feature_dir_raw),
        meta_dir=_resolve_path(project_root, meta_dir_raw),
        model_dir=_resolve_path(project_root, model_dir_raw),
        model_artifacts_dir=_resolve_path(project_root, model_artifacts_dir_raw),
        log_dir=_resolve_path(project_root, log_dir_raw),
        report_dir=_resolve_path(project_root, report_dir_raw),
        sqlite_backup_dir=_resolve_path(project_root, sqlite_backup_dir_raw),
        sync_report_dir=_resolve_path(project_root, sync_report_dir_raw),
        import_root=_resolve_path(project_root, import_root_raw),
        import_market_dir=_resolve_path(project_root, import_market_dir_raw),
        import_meta_dir=_resolve_path(project_root, import_meta_dir_raw),
        import_log_dir=_resolve_path(project_root, import_log_dir_raw),
        transfer_log_path=_resolve_path(project_root, transfer_log_path_raw),
        transfer_report_dir=_resolve_path(project_root, transfer_report_dir_raw),
    )

    deployment = DeploymentSettings(
        environment=resolved_environment,
        machine_role=str(merged_deployment.get("machine_role", "pc1" if resolved_environment == "development" else "pc2")),
        role=str(merged_deployment.get("role", "research" if resolved_environment == "development" else "deploy")),
        broker_mode=runtime_env.get("BROKER_MODE", str(merged_settings.get("broker_mode", "paper"))),
        collector_enabled=_parse_bool(
            runtime_env,
            "COLLECTOR_ENABLED",
            bool(merged_settings.get("collector_enabled", resolved_environment == "deploy")),
        ),
        training_enabled=_parse_bool(
            runtime_env,
            "TRAINING_ENABLED",
            bool(merged_settings.get("training_enabled", resolved_environment == "development")),
        ),
        backtest_enabled=_parse_bool(
            runtime_env,
            "BACKTEST_ENABLED",
            bool(merged_settings.get("backtest_enabled", resolved_environment == "development")),
        ),
        scheduler_enabled=_parse_bool(
            runtime_env,
            "SCHEDULER_ENABLED",
            bool(merged_deployment.get("scheduler_enabled", resolved_environment == "deploy")),
        ),
        sync_enabled=_parse_bool(
            runtime_env,
            "SYNC_ENABLED",
            bool(merged_deployment.get("sync_enabled", resolved_environment == "deploy")),
        ),
    )

    session_defaults = merged_settings.get("session", {})
    trading_defaults = merged_settings.get("trading", {})
    model_defaults = merged_settings.get("models", {})
    ui_defaults = merged_settings.get("ui", {})
    collector_defaults = merged_settings.get("collector", {})
    lob_capture_defaults = merged_settings.get("lob_capture", {})
    kraken_lob_defaults = merged_settings.get("kraken_lob", {})
    feature_pipeline_defaults = merged_settings.get("feature_pipeline", {})
    lan_sync_defaults = merged_settings.get("lan_sync", {})

    return Settings(
        broker=BrokerSettings(
            ib_host=runtime_env.get("IB_HOST", str(merged_settings.get("ib_host", "127.0.0.1"))),
            ib_port=_parse_int(runtime_env, "IB_PORT", int(merged_settings.get("ib_port", 4002))),
            ib_client_id=_parse_int(runtime_env, "IB_CLIENT_ID", int(merged_settings.get("ib_client_id", 1))),
            ib_ui_client_id=_parse_int(runtime_env, "IB_UI_CLIENT_ID", int(merged_settings.get("ib_ui_client_id", 101))),
            ib_collector_client_id=_parse_int(
                runtime_env,
                "IB_COLLECTOR_CLIENT_ID",
                int(merged_settings.get("ib_collector_client_id", 201)),
            ),
            ib_symbol=default_symbol,
            ib_exchange=runtime_env.get("IB_EXCHANGE", str(merged_settings.get("ib_exchange", "SMART"))).upper(),
            ib_currency=runtime_env.get("IB_CURRENCY", str(merged_settings.get("ib_currency", "USD"))).upper(),
            supported_symbols=supported_symbols,
            account_summary_group=runtime_env.get(
                "IB_ACCOUNT_SUMMARY_GROUP",
                str(merged_settings.get("ib_account_summary_group", "All")),
            ),
            request_timeout_seconds=_parse_float(
                runtime_env,
                "IB_REQUEST_TIMEOUT_SECONDS",
                float(merged_settings.get("ib_request_timeout_seconds", 15.0)),
            ),
            order_follow_up_seconds=_parse_float(
                runtime_env,
                "IB_ORDER_FOLLOW_UP_SECONDS",
                float(merged_settings.get("ib_order_follow_up_seconds", 5.0)),
            ),
            mode=deployment.broker_mode,
        ),
        session=SessionSettings(
            timezone=timezone_value,
            orb_start=_parse_time(runtime_env, "ORB_START", str(session_defaults.get("orb_start", "09:30"))),
            orb_end=_parse_time(runtime_env, "ORB_END", str(session_defaults.get("orb_end", "09:45"))),
            primary_session_end=_parse_time(
                runtime_env,
                "PRIMARY_SESSION_END",
                str(session_defaults.get("primary_session_end", "11:30")),
            ),
            secondary_session_start=_parse_time(
                runtime_env,
                "SECONDARY_SESSION_START",
                str(session_defaults.get("secondary_session_start", "13:30")),
            ),
            secondary_session_end=_parse_time(
                runtime_env,
                "SECONDARY_SESSION_END",
                str(session_defaults.get("secondary_session_end", "15:00")),
            ),
            enable_secondary_session=_parse_bool(
                runtime_env,
                "ENABLE_SECONDARY_SESSION",
                bool(session_defaults.get("enable_secondary_session", False)),
            ),
            flatten_before_close_minutes=_parse_int(
                runtime_env,
                "FLATTEN_BEFORE_CLOSE_MINUTES",
                int(session_defaults.get("flatten_before_close_minutes", 5)),
            ),
            regular_market_open=_coerce_time(str(session_defaults.get("regular_market_open", "09:30")), "regular_market_open"),
            regular_market_close=_coerce_time(str(session_defaults.get("regular_market_close", "16:00")), "regular_market_close"),
        ),
        trading=TradingSettings(
            default_order_quantity=_parse_int(
                runtime_env,
                "DEFAULT_ORDER_QUANTITY",
                int(trading_defaults.get("default_order_quantity", 1)),
            ),
            dry_run=_parse_bool(runtime_env, "DRY_RUN", bool(trading_defaults.get("dry_run", True))),
            safe_to_trade=_parse_bool(
                runtime_env,
                "SAFE_TO_TRADE",
                bool(trading_defaults.get("safe_to_trade", merged_settings.get("safe_to_trade", False))),
            ),
            allow_shorts=_parse_bool(runtime_env, "ALLOW_SHORTS", bool(trading_defaults.get("allow_shorts", False))),
            data_mode=runtime_env.get("DATA_MODE", str(trading_defaults.get("data_mode", "paper_or_local"))),
            allow_session_execution=_parse_bool(
                runtime_env,
                "ALLOW_SESSION_EXECUTION",
                bool(trading_defaults.get("allow_session_execution", False)),
            ),
            entry_limit_buffer_bps=_parse_float(
                runtime_env,
                "ENTRY_LIMIT_BUFFER_BPS",
                float(trading_defaults.get("entry_limit_buffer_bps", 2.0)),
            ),
            cost_buffer_bps=_parse_float(
                runtime_env,
                "COST_BUFFER_BPS",
                float(trading_defaults.get("cost_buffer_bps", 2.5)),
            ),
            max_spread_bps=_parse_float(
                runtime_env,
                "MAX_SPREAD_BPS",
                float(trading_defaults.get("max_spread_bps", 8.0)),
            ),
            max_hold_minutes=_parse_int(
                runtime_env,
                "MAX_HOLD_MINUTES",
                int(trading_defaults.get("max_hold_minutes", 30)),
            ),
        ),
        risk=RiskSettings(
            max_trades_per_day=_parse_int(
                runtime_env,
                "MAX_TRADES_PER_DAY",
                int(merged_risk.get("max_trades_per_day", 2)),
            ),
            max_daily_loss_pct=_parse_float(
                runtime_env,
                "MAX_DAILY_LOSS_PCT",
                float(merged_risk.get("max_daily_loss_pct", 1.0)),
            ),
            max_open_positions=_parse_int(
                runtime_env,
                "MAX_OPEN_POSITIONS",
                int(merged_risk.get("max_open_positions", 1)),
            ),
        ),
        models=ModelSettings(
            model_prob_threshold=_parse_float(
                runtime_env,
                "MODEL_PROB_THRESHOLD",
                float(model_defaults.get("model_prob_threshold", 0.58)),
            ),
            target_horizon_minutes=_parse_int(
                runtime_env,
                "TARGET_HORIZON_MINUTES",
                int(model_defaults.get("target_horizon_minutes", 3)),
            ),
            active_baseline_model=runtime_env.get("ACTIVE_BASELINE_MODEL") or model_defaults.get("active_baseline_model") or None,
            active_deep_model=runtime_env.get("ACTIVE_DEEP_MODEL") or model_defaults.get("active_deep_model") or None,
            baseline_weight=_parse_float(
                runtime_env,
                "BASELINE_WEIGHT",
                float(model_defaults.get("baseline_weight", 0.4)),
            ),
            deep_weight=_parse_float(
                runtime_env,
                "DEEP_WEIGHT",
                float(model_defaults.get("deep_weight", 0.6)),
            ),
            artifacts_dir=_resolve_path(project_root, model_artifacts_dir_raw),
            registry_path=_resolve_path(project_root, registry_path_raw),
            sequence_length=_parse_int(
                runtime_env,
                "MODEL_SEQUENCE_LENGTH",
                int(model_defaults.get("sequence_length", 16)),
            ),
            dataset_type=runtime_env.get(
                "MODEL_DATASET_TYPE",
                str(model_defaults.get("dataset_type", "bar_bootstrap")),
            ),
            lob_sequence_length=_parse_int(
                runtime_env,
                "LOB_SEQUENCE_LENGTH",
                int(model_defaults.get("lob_sequence_length", 100)),
            ),
            lob_depth_levels=_parse_int(
                runtime_env,
                "LOB_DEPTH_LEVELS",
                int(model_defaults.get("lob_depth_levels", 10)),
            ),
            lob_horizon_events=_parse_int(
                runtime_env,
                "LOB_HORIZON_EVENTS",
                int(model_defaults.get("lob_horizon_events", 10)),
            ),
            lob_stationary_threshold_bps=_parse_float(
                runtime_env,
                "LOB_STATIONARY_THRESHOLD_BPS",
                float(model_defaults.get("lob_stationary_threshold_bps", 2.0)),
            ),
            lob_train_batch_size=_parse_int(
                runtime_env,
                "LOB_TRAIN_BATCH_SIZE",
                int(model_defaults.get("lob_train_batch_size", 32)),
            ),
            lob_eval_batch_size=_parse_int(
                runtime_env,
                "LOB_EVAL_BATCH_SIZE",
                int(model_defaults.get("lob_eval_batch_size", 64)),
            ),
        ),
        storage=StorageSettings(
            log_level=runtime_env.get("LOG_LEVEL", str(merged_settings.get("log_level", "INFO"))).upper(),
            log_file=_resolve_path(project_root, log_file_raw),
            execution_log_file=_resolve_path(project_root, execution_log_raw),
            runtime_db_path=_resolve_path(project_root, runtime_db_raw),
        ),
        ui=UISettings(
            host=runtime_env.get("UI_HOST", str(ui_defaults.get("host", "127.0.0.1"))),
            port=_parse_int(runtime_env, "UI_PORT", int(ui_defaults.get("port", 8501))),
            title=runtime_env.get("UI_TITLE", str(ui_defaults.get("title", "MicroAlpha IBKR"))),
        ),
        collector=CollectorSettings(
            mode=runtime_env.get("COLLECTOR_MODE", str(collector_defaults.get("mode", "snapshot_polling"))),
            poll_interval_seconds=_parse_float(
                runtime_env,
                "COLLECTOR_POLL_INTERVAL_SECONDS",
                float(collector_defaults.get("poll_interval_seconds", 5.0)),
            ),
            flush_interval_seconds=_parse_float(
                runtime_env,
                "COLLECTOR_FLUSH_INTERVAL_SECONDS",
                float(collector_defaults.get("flush_interval_seconds", 30.0)),
            ),
            batch_size=_parse_int(
                runtime_env,
                "COLLECTOR_BATCH_SIZE",
                int(collector_defaults.get("batch_size", 50)),
            ),
            reconnect_delay_seconds=_parse_float(
                runtime_env,
                "COLLECTOR_RECONNECT_DELAY_SECONDS",
                float(collector_defaults.get("reconnect_delay_seconds", 10.0)),
            ),
            max_reconnect_attempts=_parse_int(
                runtime_env,
                "COLLECTOR_MAX_RECONNECT_ATTEMPTS",
                int(collector_defaults.get("max_reconnect_attempts", 5)),
            ),
            health_log_interval_seconds=_parse_float(
                runtime_env,
                "COLLECTOR_HEALTH_LOG_INTERVAL_SECONDS",
                float(collector_defaults.get("health_log_interval_seconds", 60.0)),
            ),
        ),
        lob_capture=LOBCaptureSettings(
            enabled=_parse_bool(
                runtime_env,
                "LOB_CAPTURE_ENABLED",
                bool(lob_capture_defaults.get("enabled", False)),
            ),
            symbols=_parse_csv(
                runtime_env,
                "LOB_CAPTURE_SYMBOLS",
                _normalize_symbols(tuple(lob_capture_defaults.get("symbols", supported_symbols))),
            ),
            depth_levels=_parse_int(
                runtime_env,
                "LOB_CAPTURE_DEPTH_LEVELS",
                int(lob_capture_defaults.get("depth_levels", 10)),
            ),
            flush_interval_seconds=_parse_float(
                runtime_env,
                "LOB_CAPTURE_FLUSH_INTERVAL_SECONDS",
                float(lob_capture_defaults.get("flush_interval_seconds", 15.0)),
            ),
            batch_size=_parse_int(
                runtime_env,
                "LOB_CAPTURE_BATCH_SIZE",
                int(lob_capture_defaults.get("batch_size", 250)),
            ),
            output_root=_resolve_path(
                project_root,
                runtime_env.get(
                    "LOB_CAPTURE_OUTPUT_ROOT",
                    str(lob_capture_defaults.get("output_root", "data/raw/ibkr_lob")),
                ),
            ),
            state_root=_resolve_path(
                project_root,
                runtime_env.get(
                    "LOB_CAPTURE_STATE_ROOT",
                    str(lob_capture_defaults.get("state_root", "data/processed/ibkr_lob_state")),
                ),
            ),
            session_root=_resolve_path(
                project_root,
                runtime_env.get(
                    "LOB_CAPTURE_SESSION_ROOT",
                    str(lob_capture_defaults.get("session_root", "data/processed/ibkr_lob_sessions")),
                ),
            ),
            dataset_root=_resolve_path(
                project_root,
                runtime_env.get(
                    "LOB_CAPTURE_DATASET_ROOT",
                    str(lob_capture_defaults.get("dataset_root", "data/processed/lob_datasets")),
                ),
            ),
            report_root=_resolve_path(
                project_root,
                runtime_env.get(
                    "LOB_CAPTURE_REPORT_ROOT",
                    str(lob_capture_defaults.get("report_root", "data/reports/lob")),
                ),
            ),
            rth_only=_parse_bool(
                runtime_env,
                "LOB_CAPTURE_RTH_ONLY",
                bool(lob_capture_defaults.get("rth_only", True)),
            ),
            reconnect_delay_seconds=_parse_float(
                runtime_env,
                "LOB_CAPTURE_RECONNECT_DELAY_SECONDS",
                float(lob_capture_defaults.get("reconnect_delay_seconds", 10.0)),
            ),
            max_reconnect_attempts=_parse_int(
                runtime_env,
                "LOB_CAPTURE_MAX_RECONNECT_ATTEMPTS",
                int(lob_capture_defaults.get("max_reconnect_attempts", 5)),
            ),
            startup_wait_seconds=_parse_float(
                runtime_env,
                "LOB_CAPTURE_STARTUP_WAIT_SECONDS",
                float(lob_capture_defaults.get("startup_wait_seconds", 5.0)),
            ),
        ),
        kraken_lob=KrakenLOBSettings(
            enabled=_parse_bool(
                runtime_env,
                "KRAKEN_LOB_ENABLED",
                bool(kraken_lob_defaults.get("enabled", False)),
            ),
            symbol=runtime_env.get(
                "KRAKEN_LOB_SYMBOL",
                str(kraken_lob_defaults.get("symbol", "BTC/EUR")),
            ).upper().replace("-", "/"),
            depth_levels=_parse_int(
                runtime_env,
                "KRAKEN_LOB_DEPTH_LEVELS",
                int(kraken_lob_defaults.get("depth_levels", 10)),
            ),
            flush_interval_seconds=_parse_float(
                runtime_env,
                "KRAKEN_LOB_FLUSH_INTERVAL_SECONDS",
                float(kraken_lob_defaults.get("flush_interval_seconds", 15.0)),
            ),
            batch_size=_parse_int(
                runtime_env,
                "KRAKEN_LOB_BATCH_SIZE",
                int(kraken_lob_defaults.get("batch_size", 250)),
            ),
            output_root=_resolve_path(
                project_root,
                runtime_env.get(
                    "KRAKEN_LOB_OUTPUT_ROOT",
                    str(kraken_lob_defaults.get("output_root", "data/raw/kraken_lob")),
                ),
            ),
            state_root=_resolve_path(
                project_root,
                runtime_env.get(
                    "KRAKEN_LOB_STATE_ROOT",
                    str(kraken_lob_defaults.get("state_root", "data/processed/kraken_lob_state")),
                ),
            ),
            session_root=_resolve_path(
                project_root,
                runtime_env.get(
                    "KRAKEN_LOB_SESSION_ROOT",
                    str(kraken_lob_defaults.get("session_root", "data/processed/kraken_lob_sessions")),
                ),
            ),
            websocket_url=runtime_env.get(
                "KRAKEN_LOB_WEBSOCKET_URL",
                str(kraken_lob_defaults.get("websocket_url", "wss://ws.kraken.com/v2")),
            ),
            reconnect_delay_seconds=_parse_float(
                runtime_env,
                "KRAKEN_LOB_RECONNECT_DELAY_SECONDS",
                float(kraken_lob_defaults.get("reconnect_delay_seconds", 10.0)),
            ),
            max_reconnect_attempts=_parse_int(
                runtime_env,
                "KRAKEN_LOB_MAX_RECONNECT_ATTEMPTS",
                int(kraken_lob_defaults.get("max_reconnect_attempts", 5)),
            ),
            startup_wait_seconds=_parse_float(
                runtime_env,
                "KRAKEN_LOB_STARTUP_WAIT_SECONDS",
                float(kraken_lob_defaults.get("startup_wait_seconds", 5.0)),
            ),
            paper_fee_bps=_parse_float(
                runtime_env,
                "KRAKEN_PAPER_FEE_BPS",
                float(kraken_lob_defaults.get("paper_fee_bps", 26.0)),
            ),
            paper_initial_cash_eur=_parse_float(
                runtime_env,
                "KRAKEN_PAPER_INITIAL_CASH_EUR",
                float(kraken_lob_defaults.get("paper_initial_cash_eur", 10000.0)),
            ),
            paper_initial_cash_mode=runtime_env.get(
                "KRAKEN_PAPER_INITIAL_CASH_MODE",
                str(kraken_lob_defaults.get("paper_initial_cash_mode", "dynamic_minimum")),
            ),
            paper_min_cash_buffer_bps=_parse_float(
                runtime_env,
                "KRAKEN_PAPER_MIN_CASH_BUFFER_BPS",
                float(kraken_lob_defaults.get("paper_min_cash_buffer_bps", 1000.0)),
            ),
            paper_position_fraction=_parse_float(
                runtime_env,
                "KRAKEN_PAPER_POSITION_FRACTION",
                float(kraken_lob_defaults.get("paper_position_fraction", 0.25)),
            ),
            paper_slippage_bps=_parse_float(
                runtime_env,
                "KRAKEN_PAPER_SLIPPAGE_BPS",
                float(kraken_lob_defaults.get("paper_slippage_bps", 2.0)),
            ),
            paper_ui_refresh_seconds=_parse_float(
                runtime_env,
                "KRAKEN_PAPER_UI_REFRESH_SECONDS",
                float(kraken_lob_defaults.get("paper_ui_refresh_seconds", 2.0)),
            ),
            paper_ui_port=_parse_int(
                runtime_env,
                "KRAKEN_PAPER_UI_PORT",
                int(kraken_lob_defaults.get("paper_ui_port", 8502)),
            ),
        ),
        feature_pipeline=FeaturePipelineSettings(
            gap_threshold_seconds=_parse_int(
                runtime_env,
                "FEATURE_GAP_THRESHOLD_SECONDS",
                int(feature_pipeline_defaults.get("gap_threshold_seconds", 120)),
            ),
            max_abs_spread_bps=_parse_float(
                runtime_env,
                "FEATURE_MAX_ABS_SPREAD_BPS",
                float(feature_pipeline_defaults.get("max_abs_spread_bps", 250.0)),
            ),
            forward_fill_limit=_parse_int(
                runtime_env,
                "FEATURE_FORWARD_FILL_LIMIT",
                int(feature_pipeline_defaults.get("forward_fill_limit", 2)),
            ),
            drop_outside_regular_hours=_parse_bool(
                runtime_env,
                "FEATURE_DROP_OUTSIDE_REGULAR_HOURS",
                bool(feature_pipeline_defaults.get("drop_outside_regular_hours", True)),
            ),
            rolling_short_window=_parse_int(
                runtime_env,
                "FEATURE_ROLLING_SHORT_WINDOW",
                int(feature_pipeline_defaults.get("rolling_short_window", 5)),
            ),
            rolling_medium_window=_parse_int(
                runtime_env,
                "FEATURE_ROLLING_MEDIUM_WINDOW",
                int(feature_pipeline_defaults.get("rolling_medium_window", 15)),
            ),
            rolling_long_window=_parse_int(
                runtime_env,
                "FEATURE_ROLLING_LONG_WINDOW",
                int(feature_pipeline_defaults.get("rolling_long_window", 30)),
            ),
            vwap_window=_parse_int(
                runtime_env,
                "FEATURE_VWAP_WINDOW",
                int(feature_pipeline_defaults.get("vwap_window", 30)),
            ),
            volume_window=_parse_int(
                runtime_env,
                "FEATURE_VOLUME_WINDOW",
                int(feature_pipeline_defaults.get("volume_window", 30)),
            ),
            label_horizon_rows=_parse_int(
                runtime_env,
                "FEATURE_LABEL_HORIZON_ROWS",
                int(feature_pipeline_defaults.get("label_horizon_rows", 1)),
            ),
            train_split_ratio=_parse_float(
                runtime_env,
                "FEATURE_TRAIN_SPLIT_RATIO",
                float(feature_pipeline_defaults.get("train_split_ratio", 0.8)),
            ),
            default_feature_set=runtime_env.get(
                "FEATURE_SET",
                str(feature_pipeline_defaults.get("default_feature_set", "hybrid_intraday")),
            ),
            validation_max_nan_ratio=_parse_float(
                runtime_env,
                "FEATURE_VALIDATION_MAX_NAN_RATIO",
                float(feature_pipeline_defaults.get("validation_max_nan_ratio", 0.35)),
            ),
        ),
        lan_sync=LanSyncSettings(
            pc2_network_root=runtime_env.get("PC2_NETWORK_ROOT") or lan_sync_defaults.get("pc2_network_root") or None,
            source_market_subdir=runtime_env.get(
                "PC2_SOURCE_MARKET_SUBDIR",
                str(lan_sync_defaults.get("source_market_subdir", "data/raw/market")),
            ),
            source_meta_subdir=runtime_env.get(
                "PC2_SOURCE_META_SUBDIR",
                str(lan_sync_defaults.get("source_meta_subdir", "data/meta")),
            ),
            source_log_subdir=runtime_env.get(
                "PC2_SOURCE_LOG_SUBDIR",
                str(lan_sync_defaults.get("source_log_subdir", "data/logs")),
            ),
            include_raw=_parse_bool(
                runtime_env,
                "LAN_INCLUDE_RAW",
                bool(lan_sync_defaults.get("include_raw", True)),
            ),
            include_meta=_parse_bool(
                runtime_env,
                "LAN_INCLUDE_META",
                bool(lan_sync_defaults.get("include_meta", True)),
            ),
            include_logs=_parse_bool(
                runtime_env,
                "LAN_INCLUDE_LOGS",
                bool(lan_sync_defaults.get("include_logs", False)),
            ),
            dry_run=_parse_bool(
                runtime_env,
                "LAN_DRY_RUN",
                bool(lan_sync_defaults.get("dry_run", False)),
            ),
            overwrite_policy=runtime_env.get(
                "LAN_OVERWRITE_POLICY",
                str(lan_sync_defaults.get("overwrite_policy", "if_newer")),
            ),
            validate_parquet=_parse_bool(
                runtime_env,
                "LAN_VALIDATE_PARQUET",
                bool(lan_sync_defaults.get("validate_parquet", True)),
            ),
            allowed_symbols=_parse_csv(
                runtime_env,
                "LAN_ALLOWED_SYMBOLS",
                _normalize_symbols(tuple(lan_sync_defaults.get("allowed_symbols", supported_symbols))),
            ),
        ),
        paths=paths,
        deployment=deployment,
        env_file=str(env_path),
    )
