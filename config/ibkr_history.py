from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from config.loader import Settings, _read_yaml
from dotenv import dotenv_values


@dataclass(frozen=True)
class IBKRHistoricalConfig:
    enabled: bool
    default_symbol: str
    default_what_to_show: str
    default_bar_size: str
    use_rth: bool
    max_concurrent_requests: int
    max_requests_per_10_min: int
    max_same_contract_requests_per_2_sec: int
    dedupe_window_seconds: int
    retry_limit: int
    backoff_seconds: float
    chunk_days_1m: int
    chunk_days_intraday_fallback: int
    output_root: str
    state_root: str
    export_root: str
    write_parquet: bool
    write_manifest: bool
    enable_ticks: bool
    enable_resume: bool
    default_tick_count: int
    synthetic_spread_bps: float
    default_depth_size: float


def load_ibkr_historical_config(settings: Settings) -> IBKRHistoricalConfig:
    config_path = Path(settings.paths.config_dir) / "training_data.yaml"
    payload = _read_yaml(config_path)
    defaults = payload.get("defaults", {}).get("ibkr_historical", {})
    env_payload = payload.get("environments", {}).get(settings.environment, {}).get("ibkr_historical", {})
    merged = {**defaults, **env_payload}
    runtime_env = _runtime_env(settings)

    return IBKRHistoricalConfig(
        enabled=_env_bool(runtime_env, "IBKR_HISTORICAL_ENABLED", bool(merged.get("enabled", True))),
        default_symbol=str(runtime_env.get("IBKR_BACKFILL_DEFAULT_SYMBOL", merged.get("default_symbol", "SPY"))).upper(),
        default_what_to_show=str(runtime_env.get("IBKR_BACKFILL_DEFAULT_WHAT_TO_SHOW", merged.get("default_what_to_show", "TRADES"))).upper(),
        default_bar_size=str(runtime_env.get("IBKR_BACKFILL_DEFAULT_BAR_SIZE", merged.get("default_bar_size", "1 min"))),
        use_rth=_env_bool(runtime_env, "IBKR_BACKFILL_USE_RTH", bool(merged.get("use_rth", True))),
        max_concurrent_requests=_env_int(runtime_env, "IBKR_BACKFILL_MAX_CONCURRENT_REQUESTS", int(merged.get("max_concurrent_requests", 1))),
        max_requests_per_10_min=_env_int(runtime_env, "IBKR_BACKFILL_MAX_REQUESTS_PER_10_MIN", int(merged.get("max_requests_per_10_min", 60))),
        max_same_contract_requests_per_2_sec=_env_int(runtime_env, "IBKR_BACKFILL_MAX_SAME_CONTRACT_REQUESTS_PER_2_SEC", int(merged.get("max_same_contract_requests_per_2_sec", 5))),
        dedupe_window_seconds=_env_int(runtime_env, "IBKR_BACKFILL_DEDUPE_WINDOW_SECONDS", int(merged.get("dedupe_window_seconds", 15))),
        retry_limit=_env_int(runtime_env, "IBKR_BACKFILL_RETRY_LIMIT", int(merged.get("retry_limit", 4))),
        backoff_seconds=_env_float(runtime_env, "IBKR_BACKFILL_BACKOFF_SECONDS", float(merged.get("backoff_seconds", 15.0))),
        chunk_days_1m=_env_int(runtime_env, "IBKR_BACKFILL_CHUNK_DAYS_1M", int(merged.get("chunk_days_1m", 7))),
        chunk_days_intraday_fallback=_env_int(runtime_env, "IBKR_BACKFILL_CHUNK_DAYS_INTRADAY_FALLBACK", int(merged.get("chunk_days_intraday_fallback", 5))),
        output_root=_resolve(settings, str(runtime_env.get("IBKR_BACKFILL_OUTPUT_ROOT", merged.get("output_root", "data/raw/ibkr_backfill")))),
        state_root=_resolve(settings, str(runtime_env.get("IBKR_BACKFILL_STATE_ROOT", merged.get("state_root", "data/processed/ibkr_backfill_state")))),
        export_root=_resolve(settings, str(runtime_env.get("IBKR_BACKFILL_EXPORT_ROOT", merged.get("export_root", "data/training/ibkr")))),
        write_parquet=_env_bool(runtime_env, "IBKR_BACKFILL_WRITE_PARQUET", bool(merged.get("write_parquet", True))),
        write_manifest=_env_bool(runtime_env, "IBKR_BACKFILL_WRITE_MANIFEST", bool(merged.get("write_manifest", True))),
        enable_ticks=_env_bool(runtime_env, "IBKR_BACKFILL_ENABLE_TICKS", bool(merged.get("enable_ticks", False))),
        enable_resume=_env_bool(runtime_env, "IBKR_BACKFILL_ENABLE_RESUME", bool(merged.get("enable_resume", True))),
        default_tick_count=_env_int(runtime_env, "IBKR_BACKFILL_DEFAULT_TICK_COUNT", int(merged.get("default_tick_count", 1000))),
        synthetic_spread_bps=_env_float(runtime_env, "IBKR_BACKFILL_SYNTHETIC_SPREAD_BPS", float(merged.get("synthetic_spread_bps", 2.0))),
        default_depth_size=_env_float(runtime_env, "IBKR_BACKFILL_DEFAULT_DEPTH_SIZE", float(merged.get("default_depth_size", 100.0))),
    )


def _runtime_env(settings: Settings) -> Mapping[str, str]:
    env_path = Path(settings.env_file or ".env")
    file_env = {
        key: str(value)
        for key, value in dotenv_values(env_path if env_path.exists() else None).items()
        if value is not None
    }
    import os
    return {**file_env, **os.environ}


def _resolve(settings: Settings, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = Path(settings.paths.project_root) / path
    return str(path.resolve())


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    return float(raw)
