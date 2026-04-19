from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from config.loader import Settings
from config.phase6 import load_phase6_config


PHASE7_CONFIG_FILENAME = "phase7.yaml"


@dataclass(frozen=True)
class ExecutionBackendConfig:
    active_execution_backend: str
    paper_mode: bool
    default_order_type: str
    default_position_size: int
    max_position_size: int
    reject_invalid_orders: bool
    allow_partial_fills: bool
    fill_delay_ms: int
    simulate_immediate_fills: bool
    slippage_bps: float
    commission_per_trade: float
    commission_per_share: float
    reject_probability: float
    partial_fill_probability: float
    partial_fill_ratio: float
    spread_aware_fills: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_execution_backend": self.active_execution_backend,
            "paper_mode": self.paper_mode,
            "default_order_type": self.default_order_type,
            "default_position_size": self.default_position_size,
            "max_position_size": self.max_position_size,
            "reject_invalid_orders": self.reject_invalid_orders,
            "allow_partial_fills": self.allow_partial_fills,
            "fill_delay_ms": self.fill_delay_ms,
            "simulate_immediate_fills": self.simulate_immediate_fills,
            "slippage_bps": self.slippage_bps,
            "commission_per_trade": self.commission_per_trade,
            "commission_per_share": self.commission_per_share,
            "reject_probability": self.reject_probability,
            "partial_fill_probability": self.partial_fill_probability,
            "partial_fill_ratio": self.partial_fill_ratio,
            "spread_aware_fills": self.spread_aware_fills,
        }


@dataclass(frozen=True)
class ExecutionLoggingConfig:
    enabled: bool
    journal_dir: str
    state_path: str
    report_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "journal_dir": self.journal_dir,
            "state_path": self.state_path,
            "report_dir": self.report_dir,
        }


@dataclass(frozen=True)
class ExecutionSessionConfig:
    initial_cash: float
    latest_per_symbol: int
    max_orders_per_run: int
    reset_state_on_offline_run: bool
    reset_state_on_session_run: bool
    recent_event_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "latest_per_symbol": self.latest_per_symbol,
            "max_orders_per_run": self.max_orders_per_run,
            "reset_state_on_offline_run": self.reset_state_on_offline_run,
            "reset_state_on_session_run": self.reset_state_on_session_run,
            "recent_event_limit": self.recent_event_limit,
        }


@dataclass(frozen=True)
class IBKRPaperBackendConfig:
    host: str
    port: int
    client_id: int
    exchange: str
    currency: str
    broker_mode: str
    request_timeout_seconds: float
    order_follow_up_seconds: float
    reconnect_attempts: int
    reconnect_delay_seconds: float
    healthcheck_timeout_seconds: float
    max_decision_to_submit_ms: float
    max_submit_to_ack_ms: float
    max_submit_to_fill_ms: float
    safe_to_trade: bool
    allow_session_execution: bool
    supported_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "exchange": self.exchange,
            "currency": self.currency,
            "broker_mode": self.broker_mode,
            "request_timeout_seconds": self.request_timeout_seconds,
            "order_follow_up_seconds": self.order_follow_up_seconds,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_delay_seconds": self.reconnect_delay_seconds,
            "healthcheck_timeout_seconds": self.healthcheck_timeout_seconds,
            "max_decision_to_submit_ms": self.max_decision_to_submit_ms,
            "max_submit_to_ack_ms": self.max_submit_to_ack_ms,
            "max_submit_to_fill_ms": self.max_submit_to_fill_ms,
            "safe_to_trade": self.safe_to_trade,
            "allow_session_execution": self.allow_session_execution,
            "supported_symbols": list(self.supported_symbols),
        }


@dataclass(frozen=True)
class Phase7Config:
    execution: ExecutionBackendConfig
    logging: ExecutionLoggingConfig
    session: ExecutionSessionConfig
    ibkr_paper: IBKRPaperBackendConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution": self.execution.to_dict(),
            "logging": self.logging.to_dict(),
            "session": self.session.to_dict(),
            "ibkr_paper": self.ibkr_paper.to_dict(),
        }


def load_phase7_config(settings: Settings) -> Phase7Config:
    runtime_env = _runtime_env(settings)
    _env_bool_local = lambda name, default: _env_bool(name, default, runtime_env)
    _env_int_local = lambda name, default: _env_int(name, default, runtime_env)
    _env_float_local = lambda name, default: _env_float(name, default, runtime_env)
    payload = _load_yaml(Path(settings.paths.config_dir) / PHASE7_CONFIG_FILENAME)
    defaults = payload.get("defaults", {})
    merged = _deep_merge(defaults, payload.get("environments", {}).get(settings.environment, {}))
    phase6 = load_phase6_config(settings)

    execution_payload = merged.get("execution", {})
    logging_payload = merged.get("logging", {})
    session_payload = merged.get("session", {})
    ibkr_payload = merged.get("ibkr_paper", {})

    return Phase7Config(
        execution=ExecutionBackendConfig(
            active_execution_backend=str(
                runtime_env.get(
                    "ACTIVE_EXECUTION_BACKEND",
                    execution_payload.get("active_execution_backend", "mock"),
                )
            ).strip(),
            paper_mode=_env_bool_local(
                "PAPER_MODE",
                bool(execution_payload.get("paper_mode", True)),
            ),
            default_order_type=str(
                runtime_env.get(
                    "EXECUTION_DEFAULT_ORDER_TYPE",
                    execution_payload.get("default_order_type", "MARKET"),
                )
            ).strip()
            or "MARKET",
            default_position_size=_env_int_local(
                "EXECUTION_DEFAULT_POSITION_SIZE",
                int(execution_payload.get("default_position_size", phase6.sizing.default_position_size)),
            ),
            max_position_size=_env_int_local(
                "EXECUTION_MAX_POSITION_SIZE",
                int(execution_payload.get("max_position_size", phase6.sizing.max_position_size)),
            ),
            reject_invalid_orders=_env_bool_local(
                "EXECUTION_REJECT_INVALID_ORDERS",
                bool(execution_payload.get("reject_invalid_orders", True)),
            ),
            allow_partial_fills=_env_bool_local(
                "EXECUTION_ALLOW_PARTIAL_FILLS",
                bool(execution_payload.get("allow_partial_fills", True)),
            ),
            fill_delay_ms=_env_int_local(
                "EXECUTION_FILL_DELAY_MS",
                int(execution_payload.get("fill_delay_ms", 0)),
            ),
            simulate_immediate_fills=_env_bool_local(
                "EXECUTION_SIMULATE_IMMEDIATE_FILLS",
                bool(execution_payload.get("simulate_immediate_fills", True)),
            ),
            slippage_bps=_env_float_local(
                "EXECUTION_SLIPPAGE_BPS",
                float(execution_payload.get("slippage_bps", 1.0)),
            ),
            commission_per_trade=_env_float_local(
                "EXECUTION_COMMISSION_PER_TRADE",
                float(execution_payload.get("commission_per_trade", 0.25)),
            ),
            commission_per_share=_env_float_local(
                "EXECUTION_COMMISSION_PER_SHARE",
                float(execution_payload.get("commission_per_share", 0.005)),
            ),
            reject_probability=_env_float_local(
                "EXECUTION_REJECT_PROBABILITY",
                float(execution_payload.get("reject_probability", 0.0)),
            ),
            partial_fill_probability=_env_float_local(
                "EXECUTION_PARTIAL_FILL_PROBABILITY",
                float(execution_payload.get("partial_fill_probability", 0.35)),
            ),
            partial_fill_ratio=_env_float_local(
                "EXECUTION_PARTIAL_FILL_RATIO",
                float(execution_payload.get("partial_fill_ratio", 0.5)),
            ),
            spread_aware_fills=_env_bool_local(
                "EXECUTION_SPREAD_AWARE_FILLS",
                bool(execution_payload.get("spread_aware_fills", True)),
            ),
        ),
        logging=ExecutionLoggingConfig(
            enabled=_env_bool_local(
                "EXECUTION_LOGGING_ENABLED",
                bool(logging_payload.get("enabled", True)),
            ),
            journal_dir=_resolve_path(
                settings,
                str(logging_payload.get("journal_dir", runtime_env.get("EXECUTION_JOURNAL_DIR", "data/reports/execution"))),
            ),
            state_path=_resolve_path(
                settings,
                str(
                    logging_payload.get(
                        "state_path",
                        runtime_env.get("EXECUTION_STATE_PATH", "data/processed/runtime/paper_execution_state.json"),
                    )
                ),
            ),
            report_dir=_resolve_path(
                settings,
                str(logging_payload.get("report_dir", runtime_env.get("PHASE7_REPORT_DIR", "data/reports/phase7"))),
            ),
        ),
        session=ExecutionSessionConfig(
            initial_cash=_env_float_local(
                "EXECUTION_INITIAL_CASH",
                float(session_payload.get("initial_cash", 100000.0)),
            ),
            latest_per_symbol=_env_int_local(
                "EXECUTION_LATEST_PER_SYMBOL",
                int(session_payload.get("latest_per_symbol", 1)),
            ),
            max_orders_per_run=_env_int_local(
                "EXECUTION_MAX_ORDERS_PER_RUN",
                int(session_payload.get("max_orders_per_run", 100)),
            ),
            reset_state_on_offline_run=_env_bool_local(
                "EXECUTION_RESET_STATE_ON_OFFLINE_RUN",
                bool(session_payload.get("reset_state_on_offline_run", True)),
            ),
            reset_state_on_session_run=_env_bool_local(
                "EXECUTION_RESET_STATE_ON_SESSION_RUN",
                bool(session_payload.get("reset_state_on_session_run", False)),
            ),
            recent_event_limit=_env_int_local(
                "EXECUTION_RECENT_EVENT_LIMIT",
                int(session_payload.get("recent_event_limit", 5)),
            ),
        ),
        ibkr_paper=IBKRPaperBackendConfig(
            host=str(runtime_env.get("IBKR_PAPER_HOST", ibkr_payload.get("host", settings.ib_host))).strip() or settings.ib_host,
            port=_env_int_local("IBKR_PAPER_PORT", int(ibkr_payload.get("port", settings.ib_port))),
            client_id=_env_int_local(
                "IBKR_PAPER_CLIENT_ID",
                int(ibkr_payload.get("client_id", settings.ib_ui_client_id)),
            ),
            exchange=str(runtime_env.get("IBKR_PAPER_EXCHANGE", ibkr_payload.get("exchange", settings.ib_exchange))).strip().upper()
            or settings.ib_exchange,
            currency=str(runtime_env.get("IBKR_PAPER_CURRENCY", ibkr_payload.get("currency", settings.ib_currency))).strip().upper()
            or settings.ib_currency,
            broker_mode=str(runtime_env.get("IBKR_PAPER_BROKER_MODE", ibkr_payload.get("broker_mode", settings.broker_mode))).strip()
            or settings.broker_mode,
            request_timeout_seconds=_env_float_local(
                "IBKR_PAPER_REQUEST_TIMEOUT_SECONDS",
                float(ibkr_payload.get("request_timeout_seconds", settings.request_timeout_seconds)),
            ),
            order_follow_up_seconds=_env_float_local(
                "IBKR_PAPER_ORDER_FOLLOW_UP_SECONDS",
                float(ibkr_payload.get("order_follow_up_seconds", settings.order_follow_up_seconds)),
            ),
            reconnect_attempts=_env_int_local(
                "IBKR_PAPER_RECONNECT_ATTEMPTS",
                int(ibkr_payload.get("reconnect_attempts", 2)),
            ),
            reconnect_delay_seconds=_env_float_local(
                "IBKR_PAPER_RECONNECT_DELAY_SECONDS",
                float(ibkr_payload.get("reconnect_delay_seconds", 2.0)),
            ),
            healthcheck_timeout_seconds=_env_float_local(
                "IBKR_PAPER_HEALTHCHECK_TIMEOUT_SECONDS",
                float(ibkr_payload.get("healthcheck_timeout_seconds", settings.request_timeout_seconds)),
            ),
            max_decision_to_submit_ms=_env_float_local(
                "IBKR_PAPER_MAX_DECISION_TO_SUBMIT_MS",
                float(ibkr_payload.get("max_decision_to_submit_ms", 2000.0)),
            ),
            max_submit_to_ack_ms=_env_float_local(
                "IBKR_PAPER_MAX_SUBMIT_TO_ACK_MS",
                float(ibkr_payload.get("max_submit_to_ack_ms", 5000.0)),
            ),
            max_submit_to_fill_ms=_env_float_local(
                "IBKR_PAPER_MAX_SUBMIT_TO_FILL_MS",
                float(ibkr_payload.get("max_submit_to_fill_ms", 15000.0)),
            ),
            safe_to_trade=bool(settings.safe_to_trade),
            allow_session_execution=bool(settings.trading.allow_session_execution),
            supported_symbols=tuple(settings.supported_symbols),
        ),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
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


def _resolve_path(settings: Settings, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = Path(settings.paths.project_root) / path
    return str(path.resolve())


def _env_bool(name: str, default: bool, env: dict[str, str] | None = None) -> bool:
    raw_value = (env or os.environ).get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw_value!r}.")


def _env_int(name: str, default: int, env: dict[str, str] | None = None) -> int:
    raw_value = (env or os.environ).get(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _env_float(name: str, default: float, env: dict[str, str] | None = None) -> float:
    raw_value = (env or os.environ).get(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _runtime_env(settings: Settings) -> dict[str, str]:
    file_env = {
        key: str(value)
        for key, value in dotenv_values(settings.env_file if settings.env_file else None).items()
        if value is not None
    }
    return {**file_env, **os.environ}
