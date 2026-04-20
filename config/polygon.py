from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from config.loader import Settings


TRAINING_DATA_CONFIG_FILENAME = "training_data.yaml"


@dataclass(frozen=True)
class PolygonBootstrapConfig:
    api_key: str | None
    api_base_url: str
    default_symbol: str
    default_interval: str
    default_start_date: str
    default_end_date: str
    synthetic_spread_bps: float
    default_depth_size: float
    output_root: str
    write_parquet: bool
    write_manifest: bool
    api_rate_limit_sleep_seconds: float
    request_timeout_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_key_configured": bool(self.api_key),
            "api_base_url": self.api_base_url,
            "default_symbol": self.default_symbol,
            "default_interval": self.default_interval,
            "default_start_date": self.default_start_date,
            "default_end_date": self.default_end_date,
            "synthetic_spread_bps": self.synthetic_spread_bps,
            "default_depth_size": self.default_depth_size,
            "output_root": self.output_root,
            "write_parquet": self.write_parquet,
            "write_manifest": self.write_manifest,
            "api_rate_limit_sleep_seconds": self.api_rate_limit_sleep_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
        }


def load_polygon_config(settings: Settings) -> PolygonBootstrapConfig:
    runtime_env = _runtime_env(settings)
    payload = _load_yaml(Path(settings.paths.config_dir) / TRAINING_DATA_CONFIG_FILENAME)
    defaults = payload.get("defaults", {})
    merged = _deep_merge(defaults, payload.get("environments", {}).get(settings.environment, {}))
    polygon_payload = merged.get("polygon", {})

    return PolygonBootstrapConfig(
        api_key=runtime_env.get("POLYGON_API_KEY"),
        api_base_url=str(runtime_env.get("POLYGON_API_BASE_URL", polygon_payload.get("api_base_url", "https://api.polygon.io"))).rstrip("/"),
        default_symbol=str(runtime_env.get("POLYGON_DEFAULT_SYMBOL", polygon_payload.get("default_symbol", "SPY"))).upper(),
        default_interval=str(runtime_env.get("POLYGON_DEFAULT_INTERVAL", polygon_payload.get("default_interval", "1m"))),
        default_start_date=str(runtime_env.get("POLYGON_DEFAULT_START_DATE", polygon_payload.get("default_start_date", "2025-01-01"))),
        default_end_date=str(runtime_env.get("POLYGON_DEFAULT_END_DATE", polygon_payload.get("default_end_date", "2025-03-31"))),
        synthetic_spread_bps=_env_float("POLYGON_SYNTHETIC_SPREAD_BPS", float(polygon_payload.get("synthetic_spread_bps", 2.0)), runtime_env),
        default_depth_size=_env_float("POLYGON_DEFAULT_DEPTH_SIZE", float(polygon_payload.get("default_depth_size", 100.0)), runtime_env),
        output_root=_resolve_path(settings, str(runtime_env.get("TRAINING_DATA_OUTPUT_ROOT", polygon_payload.get("output_root", "data/training/polygon")))),
        write_parquet=_env_bool("TRAINING_DATA_WRITE_PARQUET", bool(polygon_payload.get("write_parquet", True)), runtime_env),
        write_manifest=_env_bool("TRAINING_DATA_WRITE_MANIFEST", bool(polygon_payload.get("write_manifest", True)), runtime_env),
        api_rate_limit_sleep_seconds=_env_float("POLYGON_API_RATE_LIMIT_SLEEP_SECONDS", float(polygon_payload.get("api_rate_limit_sleep_seconds", 12.0)), runtime_env),
        request_timeout_seconds=_env_float("POLYGON_REQUEST_TIMEOUT_SECONDS", float(polygon_payload.get("request_timeout_seconds", 30.0)), runtime_env),
    )


def _runtime_env(settings: Settings) -> dict[str, str]:
    env_path = Path(settings.env_file) if settings.env_file else None
    file_env = {
        key: str(value)
        for key, value in dotenv_values(env_path if env_path and env_path.exists() else None).items()
        if value is not None
    }
    return {**file_env, **os.environ}


def _load_yaml(path: Path) -> dict[str, Any]:
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


def _env_bool(name: str, default: bool, env: dict[str, str]) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


def _env_float(name: str, default: float, env: dict[str, str]) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    return float(raw)
