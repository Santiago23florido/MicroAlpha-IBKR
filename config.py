from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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
        raise ValueError(
            f"Invalid integer value for {name!r}: {raw_value!r}."
        ) from exc


def _parse_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value for {name!r}: {raw_value!r}."
        ) from exc


@dataclass(frozen=True)
class Settings:
    ib_host: str
    ib_port: int
    ib_client_id: int
    ib_symbol: str
    dry_run: bool
    safe_to_trade: bool
    log_level: str
    log_file: str
    request_timeout_seconds: float
    account_summary_group: str


def load_settings(env_file: str | Path = ".env") -> Settings:
    env_path = Path(env_file)
    load_dotenv(env_path if env_path.exists() else None)

    return Settings(
        ib_host=os.getenv("IB_HOST", "127.0.0.1"),
        ib_port=_parse_int("IB_PORT", 4002),
        ib_client_id=_parse_int("IB_CLIENT_ID", 1),
        ib_symbol=os.getenv("IB_SYMBOL", "SPY").upper(),
        dry_run=_parse_bool("DRY_RUN", True),
        safe_to_trade=_parse_bool("SAFE_TO_TRADE", False),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_file=os.getenv("LOG_FILE", "logs/ibkr_mvp.log"),
        request_timeout_seconds=_parse_float("IB_REQUEST_TIMEOUT_SECONDS", 15.0),
        account_summary_group=os.getenv("IB_ACCOUNT_SUMMARY_GROUP", "All"),
    )
