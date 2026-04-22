from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from config.loader import Settings


ACTIVE_MODEL_FILENAME = "active_model.yaml"
PHASE6_CONFIG_FILENAME = "phase6.yaml"
PHASE6_TARGET_PRIORITY = {
    "classification_binary": 0,
    "regression_point": 1,
    "ordinal_classification": 2,
    "distribution_bins": 3,
    "quantile_regression": 4,
}


@dataclass(frozen=True)
class ActiveModelSelection:
    run_id: str
    artifact_dir: str
    model_name: str
    model_type: str
    feature_set_name: str
    target_mode: str
    selection_reason: str
    source_leaderboard: str | None = None
    updated_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "artifact_dir": self.artifact_dir,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "feature_set_name": self.feature_set_name,
            "target_mode": self.target_mode,
            "selection_reason": self.selection_reason,
            "source_leaderboard": self.source_leaderboard,
            "updated_at_utc": self.updated_at_utc,
        }


@dataclass(frozen=True)
class DecisionConfig:
    score_threshold: float
    probability_threshold: float
    predicted_return_min_bps: float
    net_edge_min_bps: float
    allow_long: bool
    allow_short: bool
    spread_max_bps: float
    cost_max_bps: float
    allowed_trading_start: time
    allowed_trading_end: time
    max_quantile_interval_width_bps: float
    critical_feature_columns: tuple[str, ...]
    explain_feature_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_threshold": self.score_threshold,
            "probability_threshold": self.probability_threshold,
            "predicted_return_min_bps": self.predicted_return_min_bps,
            "net_edge_min_bps": self.net_edge_min_bps,
            "allow_long": self.allow_long,
            "allow_short": self.allow_short,
            "spread_max_bps": self.spread_max_bps,
            "cost_max_bps": self.cost_max_bps,
            "allowed_trading_start": self.allowed_trading_start.isoformat(timespec="minutes"),
            "allowed_trading_end": self.allowed_trading_end.isoformat(timespec="minutes"),
            "max_quantile_interval_width_bps": self.max_quantile_interval_width_bps,
            "critical_feature_columns": list(self.critical_feature_columns),
            "explain_feature_columns": list(self.explain_feature_columns),
        }


@dataclass(frozen=True)
class RiskEngineConfig:
    enabled: bool
    max_trades_per_session: int
    daily_loss_limit_bps: float
    symbol_loss_limit_bps: float
    cooldown_minutes: int
    max_spread_bps: float
    max_estimated_cost_bps: float
    kill_switch_on_invalid_model: bool
    kill_switch_on_anomalous_prediction: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_trades_per_session": self.max_trades_per_session,
            "daily_loss_limit_bps": self.daily_loss_limit_bps,
            "symbol_loss_limit_bps": self.symbol_loss_limit_bps,
            "cooldown_minutes": self.cooldown_minutes,
            "max_spread_bps": self.max_spread_bps,
            "max_estimated_cost_bps": self.max_estimated_cost_bps,
            "kill_switch_on_invalid_model": self.kill_switch_on_invalid_model,
            "kill_switch_on_anomalous_prediction": self.kill_switch_on_anomalous_prediction,
        }


@dataclass(frozen=True)
class SizingConfig:
    default_position_size: int
    max_position_size: int
    min_confidence_for_full_size: float
    min_size_confidence_floor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_position_size": self.default_position_size,
            "max_position_size": self.max_position_size,
            "min_confidence_for_full_size": self.min_confidence_for_full_size,
            "min_size_confidence_floor": self.min_size_confidence_floor,
        }


@dataclass(frozen=True)
class DecisionLoggingConfig:
    enabled: bool
    decision_log_path: str
    report_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "decision_log_path": self.decision_log_path,
            "report_dir": self.report_dir,
        }


@dataclass(frozen=True)
class StrategyConfig:
    enabled_alphas: tuple[str, ...]
    alpha_priority_order: tuple[str, ...]
    alpha_router_mode: str
    regime_detection_enabled: bool
    conservative_decision_mode: bool
    min_net_edge_bps_by_alpha: dict[str, float]
    regime_thresholds: dict[str, float]
    no_trade_filters: tuple[str, ...]
    alpha_specific_thresholds: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled_alphas": list(self.enabled_alphas),
            "alpha_priority_order": list(self.alpha_priority_order),
            "alpha_router_mode": self.alpha_router_mode,
            "regime_detection_enabled": self.regime_detection_enabled,
            "conservative_decision_mode": self.conservative_decision_mode,
            "min_net_edge_bps_by_alpha": self.min_net_edge_bps_by_alpha,
            "regime_thresholds": self.regime_thresholds,
            "no_trade_filters": list(self.no_trade_filters),
            "alpha_specific_thresholds": self.alpha_specific_thresholds,
        }


@dataclass(frozen=True)
class Phase6Config:
    decision: DecisionConfig
    risk: RiskEngineConfig
    sizing: SizingConfig
    logging: DecisionLoggingConfig
    strategy: StrategyConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "risk": self.risk.to_dict(),
            "sizing": self.sizing.to_dict(),
            "logging": self.logging.to_dict(),
            "strategy": self.strategy.to_dict(),
        }


def load_phase6_config(settings: Settings) -> Phase6Config:
    runtime_env = _runtime_env(settings)
    _env_bool_local = lambda name, default: _env_bool(name, default, runtime_env)
    _env_int_local = lambda name, default: _env_int(name, default, runtime_env)
    _env_float_local = lambda name, default: _env_float(name, default, runtime_env)
    _env_time_local = lambda name, default: _env_time(name, default, runtime_env)
    payload = _load_yaml(Path(settings.paths.config_dir) / PHASE6_CONFIG_FILENAME)
    defaults = payload.get("defaults", {})
    merged = _deep_merge(defaults, payload.get("environments", {}).get(settings.environment, {}))
    decision_payload = merged.get("decision", {})
    risk_payload = merged.get("risk", {})
    sizing_payload = merged.get("sizing", {})
    logging_payload = merged.get("logging", {})
    strategy_payload = merged.get("strategy", {})

    return Phase6Config(
        decision=DecisionConfig(
            score_threshold=_env_float_local("DECISION_SCORE_THRESHOLD", float(decision_payload.get("score_threshold", 0.0))),
            probability_threshold=_env_float_local(
                "DECISION_PROBABILITY_THRESHOLD",
                float(decision_payload.get("probability_threshold", 0.58)),
            ),
            predicted_return_min_bps=_env_float_local(
                "DECISION_PREDICTED_RETURN_MIN_BPS",
                float(decision_payload.get("predicted_return_min_bps", 2.5)),
            ),
            net_edge_min_bps=_env_float_local(
                "DECISION_NET_EDGE_MIN_BPS",
                float(decision_payload.get("net_edge_min_bps", 0.5)),
            ),
            allow_long=_env_bool_local("DECISION_ALLOW_LONG", bool(decision_payload.get("allow_long", True))),
            allow_short=_env_bool_local("DECISION_ALLOW_SHORT", bool(decision_payload.get("allow_short", True))),
            spread_max_bps=_env_float_local(
                "DECISION_SPREAD_MAX_BPS",
                float(decision_payload.get("spread_max_bps", 12.0)),
            ),
            cost_max_bps=_env_float_local(
                "DECISION_COST_MAX_BPS",
                float(decision_payload.get("cost_max_bps", 18.0)),
            ),
            allowed_trading_start=_env_time_local(
                "DECISION_ALLOWED_TRADING_START",
                str(decision_payload.get("allowed_trading_start", "09:35")),
            ),
            allowed_trading_end=_env_time_local(
                "DECISION_ALLOWED_TRADING_END",
                str(decision_payload.get("allowed_trading_end", "15:30")),
            ),
            max_quantile_interval_width_bps=_env_float_local(
                "DECISION_MAX_QUANTILE_INTERVAL_WIDTH_BPS",
                float(decision_payload.get("max_quantile_interval_width_bps", 35.0)),
            ),
            critical_feature_columns=tuple(str(value) for value in decision_payload.get("critical_feature_columns", [])),
            explain_feature_columns=tuple(str(value) for value in decision_payload.get("explain_feature_columns", [])),
        ),
        risk=RiskEngineConfig(
            enabled=_env_bool_local("RISK_ENABLED", bool(risk_payload.get("enabled", True))),
            max_trades_per_session=_env_int_local(
                "RISK_MAX_TRADES_PER_SESSION",
                int(risk_payload.get("max_trades_per_session", 4)),
            ),
            daily_loss_limit_bps=_env_float_local(
                "RISK_DAILY_LOSS_LIMIT_BPS",
                float(risk_payload.get("daily_loss_limit_bps", 40.0)),
            ),
            symbol_loss_limit_bps=_env_float_local(
                "RISK_SYMBOL_LOSS_LIMIT_BPS",
                float(risk_payload.get("symbol_loss_limit_bps", 25.0)),
            ),
            cooldown_minutes=_env_int_local(
                "RISK_COOLDOWN_MINUTES",
                int(risk_payload.get("cooldown_minutes", 20)),
            ),
            max_spread_bps=_env_float_local(
                "RISK_MAX_SPREAD_BPS",
                float(risk_payload.get("max_spread_bps", 12.0)),
            ),
            max_estimated_cost_bps=_env_float_local(
                "RISK_MAX_ESTIMATED_COST_BPS",
                float(risk_payload.get("max_estimated_cost_bps", 18.0)),
            ),
            kill_switch_on_invalid_model=_env_bool_local(
                "RISK_KILL_SWITCH_ON_INVALID_MODEL",
                bool(risk_payload.get("kill_switch_on_invalid_model", True)),
            ),
            kill_switch_on_anomalous_prediction=_env_bool_local(
                "RISK_KILL_SWITCH_ON_ANOMALOUS_PREDICTION",
                bool(risk_payload.get("kill_switch_on_anomalous_prediction", True)),
            ),
        ),
        sizing=SizingConfig(
            default_position_size=_env_int_local(
                "POSITION_SIZE_DEFAULT",
                int(sizing_payload.get("default_position_size", 1)),
            ),
            max_position_size=_env_int_local(
                "POSITION_SIZE_MAX",
                int(sizing_payload.get("max_position_size", 3)),
            ),
            min_confidence_for_full_size=_env_float_local(
                "POSITION_MIN_CONFIDENCE_FOR_FULL_SIZE",
                float(sizing_payload.get("min_confidence_for_full_size", 0.72)),
            ),
            min_size_confidence_floor=_env_float_local(
                "POSITION_MIN_CONFIDENCE_FLOOR",
                float(sizing_payload.get("min_size_confidence_floor", 0.55)),
            ),
        ),
        logging=DecisionLoggingConfig(
            enabled=_env_bool_local("DECISION_LOGGING_ENABLED", bool(logging_payload.get("enabled", True))),
            decision_log_path=_resolve_path(
                settings,
                runtime_env.get("DECISION_LOG_PATH", str(logging_payload.get("decision_log_path", "data/reports/decisions/decision_log.jsonl"))),
            ),
            report_dir=_resolve_path(
                settings,
                runtime_env.get("PHASE6_REPORT_DIR", str(logging_payload.get("report_dir", "data/reports/phase6"))),
            ),
        ),
        strategy=StrategyConfig(
            enabled_alphas=_env_csv(
                "ENABLED_ALPHAS",
                tuple(str(value) for value in strategy_payload.get("enabled_alphas", ("low_edge_no_trade_filter", "orb_continuation", "vwap_mean_reversion", "late_session_alpha"))),
                runtime_env,
            ),
            alpha_priority_order=_env_csv(
                "ALPHA_PRIORITY_ORDER",
                tuple(str(value) for value in strategy_payload.get("alpha_priority_order", ("low_edge_no_trade_filter", "orb_continuation", "vwap_mean_reversion", "late_session_alpha"))),
                runtime_env,
            ),
            alpha_router_mode=str(runtime_env.get("ALPHA_ROUTER_MODE", strategy_payload.get("alpha_router_mode", "priority_conservative"))),
            regime_detection_enabled=_env_bool_local("REGIME_DETECTION_ENABLED", bool(strategy_payload.get("regime_detection_enabled", True))),
            conservative_decision_mode=_env_bool_local("CONSERVATIVE_DECISION_MODE", bool(strategy_payload.get("conservative_decision_mode", True))),
            min_net_edge_bps_by_alpha={
                str(key): float(value)
                for key, value in dict(strategy_payload.get("min_net_edge_bps_by_alpha", {})).items()
            },
            regime_thresholds={
                str(key): float(value)
                for key, value in dict(strategy_payload.get("regime_thresholds", {})).items()
            },
            no_trade_filters=_env_csv(
                "NO_TRADE_FILTERS",
                tuple(str(value) for value in strategy_payload.get("no_trade_filters", ("high_cost_regime", "low_liquidity_regime", "noisy_open", "low_edge_midday"))),
                runtime_env,
            ),
            alpha_specific_thresholds=dict(strategy_payload.get("alpha_specific_thresholds", {})),
        ),
    )


def load_active_model_selection(settings: Settings) -> ActiveModelSelection:
    runtime_env = _runtime_env(settings)
    path = Path(settings.paths.config_dir) / ACTIVE_MODEL_FILENAME
    selection = None
    if path.exists():
        payload = _load_yaml(path)
        selection_payload = payload.get("active_model", {})
        if selection_payload:
            selection = _selection_from_payload(selection_payload)

    override_run_id = runtime_env.get("ACTIVE_MODEL_RUN_ID")
    override_artifact_dir = runtime_env.get("ACTIVE_MODEL_ARTIFACT_DIR")
    override_model_name = runtime_env.get("ACTIVE_MODEL_NAME")
    if override_run_id or override_artifact_dir or override_model_name:
        candidate = resolve_phase5_artifact(
            settings,
            run_id=override_run_id,
            artifact_dir=override_artifact_dir,
            model_name=override_model_name,
        )
        return selection_from_candidate(
            settings,
            candidate,
            reason="Resolved from environment override.",
        )

    if selection is not None:
        return selection

    candidate = select_reasonable_default_phase5_artifact(settings)
    selection = selection_from_candidate(
        settings,
        candidate,
        reason="Resolved automatically from the best available Phase 5 artifact.",
    )
    write_active_model_selection(settings, selection)
    return selection


def write_active_model_selection(settings: Settings, selection: ActiveModelSelection) -> str:
    path = Path(settings.paths.config_dir) / ACTIVE_MODEL_FILENAME
    payload = {"active_model": selection.to_dict()}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(path)


def set_active_model_selection(
    settings: Settings,
    *,
    run_id: str | None = None,
    artifact_dir: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    candidate = resolve_phase5_artifact(settings, run_id=run_id, artifact_dir=artifact_dir, model_name=model_name)
    selection = selection_from_candidate(
        settings,
        candidate,
        reason="Explicitly selected via set-active-model.",
    )
    config_path = write_active_model_selection(settings, selection)
    return {
        "status": "ok",
        "config_path": config_path,
        "active_model": selection.to_dict(),
    }


def show_active_model_status(settings: Settings) -> dict[str, Any]:
    selection = load_active_model_selection(settings)
    candidate = resolve_phase5_artifact(settings, run_id=selection.run_id, artifact_dir=selection.artifact_dir)
    required_files = required_phase5_artifact_files(candidate)
    return {
        "status": "ok",
        "active_model": selection.to_dict(),
        "artifact_status": {
            "artifact_dir": candidate["artifact_dir"],
            "required_files": required_files,
            "ready": all(item["exists"] for item in required_files.values()),
        },
    }


def required_phase5_artifact_files(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"path": str(path), "exists": Path(path).exists()}
        for key, path in {
            "artifact_path": candidate["artifact_path"],
            "preprocessing_path": candidate["preprocessing_path"],
            "feature_columns_path": candidate["feature_columns_path"],
            "target_config_path": candidate["target_config_path"],
            "training_metadata_path": candidate["training_metadata_path"],
            "leaderboard_row_path": candidate["leaderboard_row_path"],
        }.items()
    }


def resolve_phase5_artifact(
    settings: Settings,
    *,
    run_id: str | None = None,
    artifact_dir: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    candidates = discover_phase5_artifacts(settings)
    if run_id:
        for candidate in candidates:
            if candidate["run_id"] == run_id:
                return candidate
        raise ValueError(f"No Phase 5 artifact found with run_id={run_id!r}.")

    if artifact_dir:
        resolved_dir = str(_resolve_relative_path(settings, artifact_dir))
        for candidate in candidates:
            if candidate["artifact_dir"] == resolved_dir:
                return candidate
        raise ValueError(f"No Phase 5 artifact found with artifact_dir={artifact_dir!r}.")

    if model_name:
        matching = [candidate for candidate in candidates if candidate["model_name"] == model_name]
        if not matching:
            raise ValueError(f"No Phase 5 artifact found with model_name={model_name!r}.")
        return sorted(
            matching,
            key=lambda item: (
                PHASE6_TARGET_PRIORITY.get(item["target_mode"], 99),
                -float(item.get("ranking_score", float("-inf"))),
                item.get("timestamp_utc", ""),
            ),
        )[0]

    raise ValueError("Provide run_id, artifact_dir, or model_name to resolve a Phase 5 artifact.")


def select_reasonable_default_phase5_artifact(settings: Settings) -> dict[str, Any]:
    candidates = discover_phase5_artifacts(settings)
    if not candidates:
        raise FileNotFoundError(
            f"No Phase 5 artifacts were found under {settings.paths.model_dir}. Run Phase 5 experiments before Phase 6."
        )
    return sorted(
        candidates,
        key=lambda item: (
            PHASE6_TARGET_PRIORITY.get(item["target_mode"], 99),
            -float(item.get("ranking_score", float("-inf"))),
            item.get("timestamp_utc", ""),
        ),
    )[0]


def discover_phase5_artifacts(settings: Settings) -> list[dict[str, Any]]:
    model_root = Path(settings.paths.model_dir)
    candidates: list[dict[str, Any]] = []
    for artifact_dir in sorted(model_root.glob("run_*")):
        if not artifact_dir.is_dir():
            continue
        artifact = _discover_candidate_from_dir(settings, artifact_dir)
        if artifact is not None:
            candidates.append(artifact)
    return candidates


def selection_from_candidate(
    settings: Settings,
    candidate: dict[str, Any],
    *,
    reason: str,
) -> ActiveModelSelection:
    return ActiveModelSelection(
        run_id=str(candidate["run_id"]),
        artifact_dir=_relative_path(settings, candidate["artifact_dir"]),
        model_name=str(candidate["model_name"]),
        model_type=str(candidate["model_type"]),
        feature_set_name=str(candidate["feature_set_name"]),
        target_mode=str(candidate["target_mode"]),
        selection_reason=reason,
        source_leaderboard=_relative_path(settings, candidate.get("source_leaderboard")),
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _discover_candidate_from_dir(settings: Settings, artifact_dir: Path) -> dict[str, Any] | None:
    leaderboard_row_path = artifact_dir / "leaderboard_row.json"
    training_metadata_path = artifact_dir / "training_metadata.json"
    target_config_path = artifact_dir / "target_config.json"
    feature_columns_path = artifact_dir / "feature_columns.json"
    artifact_path = artifact_dir / "model.joblib"
    preprocessing_path = artifact_dir / "preprocessing.joblib"
    required = [
        leaderboard_row_path,
        training_metadata_path,
        target_config_path,
        feature_columns_path,
        artifact_path,
        preprocessing_path,
    ]
    if any(not path.exists() for path in required):
        return None

    leaderboard = _load_json(leaderboard_row_path)
    training_metadata = _load_json(training_metadata_path)
    target_config = _load_json(target_config_path)
    feature_columns = _load_json(feature_columns_path)
    source_leaderboard = leaderboard.get("source_leaderboard") or _infer_source_leaderboard(settings, str(leaderboard.get("run_id", artifact_dir.name)))
    return {
        "run_id": leaderboard.get("run_id", artifact_dir.name),
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_path": str(artifact_path.resolve()),
        "preprocessing_path": str(preprocessing_path.resolve()),
        "feature_columns_path": str(feature_columns_path.resolve()),
        "target_config_path": str(target_config_path.resolve()),
        "training_metadata_path": str(training_metadata_path.resolve()),
        "leaderboard_row_path": str(leaderboard_row_path.resolve()),
        "feature_columns": list(feature_columns),
        "target_config": target_config,
        "training_metadata": training_metadata,
        "model_name": leaderboard.get("model_name", training_metadata.get("model_name")),
        "model_type": leaderboard.get("model_type", leaderboard.get("model_name")),
        "feature_set_name": leaderboard.get("feature_set", training_metadata.get("feature_set_name")),
        "target_mode": leaderboard.get("target_mode", training_metadata.get("target_mode")),
        "ranking_score": float(leaderboard.get("ranking_score", float("-inf"))),
        "validation_metrics": leaderboard.get("validation_metrics", {}),
        "test_metrics": leaderboard.get("test_metrics", {}),
        "hyperparameters": leaderboard.get("hyperparameters", {}),
        "symbols": leaderboard.get("symbols", training_metadata.get("symbols", [])),
        "timestamp_utc": leaderboard.get("timestamp_utc", training_metadata.get("created_at_utc", "")),
        "source_leaderboard": source_leaderboard,
    }


def _selection_from_payload(payload: dict[str, Any]) -> ActiveModelSelection:
    required = [
        "run_id",
        "artifact_dir",
        "model_name",
        "model_type",
        "feature_set_name",
        "target_mode",
        "selection_reason",
    ]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"active_model.yaml is missing required keys: {missing}")
    return ActiveModelSelection(
        run_id=str(payload["run_id"]),
        artifact_dir=str(payload["artifact_dir"]),
        model_name=str(payload["model_name"]),
        model_type=str(payload["model_type"]),
        feature_set_name=str(payload["feature_set_name"]),
        target_mode=str(payload["target_mode"]),
        selection_reason=str(payload["selection_reason"]),
        source_leaderboard=payload.get("source_leaderboard"),
        updated_at_utc=payload.get("updated_at_utc"),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


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


def _relative_path(settings: Settings, value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    with_context = Path(settings.paths.project_root)
    try:
        return str(path.resolve().relative_to(with_context))
    except ValueError:
        return str(path.resolve())


def _resolve_relative_path(settings: Settings, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(settings.paths.project_root) / path
    return path.resolve()


def _runtime_env(settings: Settings) -> dict[str, str]:
    file_env = {
        key: str(value)
        for key, value in dotenv_values(settings.env_file if settings.env_file else None).items()
        if value is not None
    }
    return {**file_env, **os.environ}


def _env_bool(name: str, default: bool, env: dict[str, str] | None = None) -> bool:
    raw = (env or os.environ).get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, env: dict[str, str] | None = None) -> int:
    raw = (env or os.environ).get(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float, env: dict[str, str] | None = None) -> float:
    raw = (env or os.environ).get(name)
    return default if raw is None else float(raw)


def _env_csv(name: str, default: tuple[str, ...], env: dict[str, str] | None = None) -> tuple[str, ...]:
    raw = (env or os.environ).get(name)
    if raw is None:
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or default


def _env_time(name: str, default: str, env: dict[str, str] | None = None) -> time:
    raw = (env or os.environ).get(name, default)
    try:
        return time.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid time for {name}: {raw!r}. Use HH:MM.") from exc


def _infer_source_leaderboard(settings: Settings, run_id: str) -> str | None:
    reports_root = Path(settings.paths.report_dir) / "phase5"
    if not reports_root.exists():
        return None
    for leaderboard_path in sorted(reports_root.glob("leaderboard_*.csv"), reverse=True):
        try:
            if run_id in leaderboard_path.read_text(encoding="utf-8"):
                return str(leaderboard_path.resolve())
        except OSError:
            continue
    return None
