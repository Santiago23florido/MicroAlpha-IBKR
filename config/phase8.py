from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config.loader import Settings


PHASE8_CONFIG_FILENAME = "phase8.yaml"


@dataclass(frozen=True)
class PerformanceThresholdConfig:
    min_trades_required: int
    min_win_rate: float
    max_drawdown: float
    min_expectancy: float
    min_total_pnl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_trades_required": self.min_trades_required,
            "min_win_rate": self.min_win_rate,
            "max_drawdown": self.max_drawdown,
            "min_expectancy": self.min_expectancy,
            "min_total_pnl": self.min_total_pnl,
        }


@dataclass(frozen=True)
class DriftThresholdConfig:
    feature_psi_warning: float
    feature_psi_critical: float
    prediction_psi_warning: float
    label_psi_warning: float
    mean_shift_sigma_warning: float
    degenerate_output_std_floor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_psi_warning": self.feature_psi_warning,
            "feature_psi_critical": self.feature_psi_critical,
            "prediction_psi_warning": self.prediction_psi_warning,
            "label_psi_warning": self.label_psi_warning,
            "mean_shift_sigma_warning": self.mean_shift_sigma_warning,
            "degenerate_output_std_floor": self.degenerate_output_std_floor,
        }


@dataclass(frozen=True)
class EvaluationWindowConfig:
    reference_days: int
    compare_run_limit: int
    min_samples_for_drift: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_days": self.reference_days,
            "compare_run_limit": self.compare_run_limit,
            "min_samples_for_drift": self.min_samples_for_drift,
        }


@dataclass(frozen=True)
class Phase8ReportPathConfig:
    report_dir: str
    economic_leaderboard_path: str
    compare_runs_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_dir": self.report_dir,
            "economic_leaderboard_path": self.economic_leaderboard_path,
            "compare_runs_dir": self.compare_runs_dir,
        }


@dataclass(frozen=True)
class AlertFlagConfig:
    alert_on_negative_pnl: bool
    alert_on_low_win_rate: bool
    alert_on_high_drawdown: bool
    alert_on_feature_drift: bool
    alert_on_prediction_drift: bool
    alert_on_label_drift: bool
    alert_on_degenerate_outputs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_on_negative_pnl": self.alert_on_negative_pnl,
            "alert_on_low_win_rate": self.alert_on_low_win_rate,
            "alert_on_high_drawdown": self.alert_on_high_drawdown,
            "alert_on_feature_drift": self.alert_on_feature_drift,
            "alert_on_prediction_drift": self.alert_on_prediction_drift,
            "alert_on_label_drift": self.alert_on_label_drift,
            "alert_on_degenerate_outputs": self.alert_on_degenerate_outputs,
        }


@dataclass(frozen=True)
class Phase8Config:
    performance_thresholds: PerformanceThresholdConfig
    drift_thresholds: DriftThresholdConfig
    evaluation_window: EvaluationWindowConfig
    report_paths: Phase8ReportPathConfig
    alert_flags: AlertFlagConfig
    auto_generate_after_phase7: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "performance_thresholds": self.performance_thresholds.to_dict(),
            "drift_thresholds": self.drift_thresholds.to_dict(),
            "evaluation_window": self.evaluation_window.to_dict(),
            "report_paths": self.report_paths.to_dict(),
            "alert_flags": self.alert_flags.to_dict(),
            "auto_generate_after_phase7": self.auto_generate_after_phase7,
        }


def load_phase8_config(settings: Settings) -> Phase8Config:
    payload = _load_yaml(Path(settings.paths.config_dir) / PHASE8_CONFIG_FILENAME)
    defaults = payload.get("defaults", {})
    merged = _deep_merge(defaults, payload.get("environments", {}).get(settings.environment, {}))

    performance_payload = merged.get("performance_thresholds", {})
    drift_payload = merged.get("drift_thresholds", {})
    window_payload = merged.get("evaluation_window", {})
    report_payload = merged.get("report_paths", {})
    alert_payload = merged.get("alert_flags", {})

    return Phase8Config(
        performance_thresholds=PerformanceThresholdConfig(
            min_trades_required=_env_int("EVAL_MIN_TRADES_REQUIRED", int(performance_payload.get("min_trades_required", 5))),
            min_win_rate=_env_float("EVAL_MIN_WIN_RATE", float(performance_payload.get("min_win_rate", 0.45))),
            max_drawdown=_env_float("EVAL_MAX_DRAWDOWN", float(performance_payload.get("max_drawdown", 500.0))),
            min_expectancy=_env_float("EVAL_MIN_EXPECTANCY", float(performance_payload.get("min_expectancy", 0.0))),
            min_total_pnl=_env_float("EVAL_MIN_TOTAL_PNL", float(performance_payload.get("min_total_pnl", 0.0))),
        ),
        drift_thresholds=DriftThresholdConfig(
            feature_psi_warning=_env_float("EVAL_FEATURE_PSI_WARNING", float(drift_payload.get("feature_psi_warning", 0.2))),
            feature_psi_critical=_env_float("EVAL_FEATURE_PSI_CRITICAL", float(drift_payload.get("feature_psi_critical", 0.35))),
            prediction_psi_warning=_env_float("EVAL_PREDICTION_PSI_WARNING", float(drift_payload.get("prediction_psi_warning", 0.2))),
            label_psi_warning=_env_float("EVAL_LABEL_PSI_WARNING", float(drift_payload.get("label_psi_warning", 0.2))),
            mean_shift_sigma_warning=_env_float(
                "EVAL_MEAN_SHIFT_SIGMA_WARNING",
                float(drift_payload.get("mean_shift_sigma_warning", 1.5)),
            ),
            degenerate_output_std_floor=_env_float(
                "EVAL_DEGENERATE_OUTPUT_STD_FLOOR",
                float(drift_payload.get("degenerate_output_std_floor", 1e-6)),
            ),
        ),
        evaluation_window=EvaluationWindowConfig(
            reference_days=_env_int("EVAL_REFERENCE_DAYS", int(window_payload.get("reference_days", 5))),
            compare_run_limit=_env_int("EVAL_COMPARE_RUN_LIMIT", int(window_payload.get("compare_run_limit", 10))),
            min_samples_for_drift=_env_int(
                "EVAL_MIN_SAMPLES_FOR_DRIFT",
                int(window_payload.get("min_samples_for_drift", 25)),
            ),
        ),
        report_paths=Phase8ReportPathConfig(
            report_dir=_resolve_path(
                settings,
                str(report_payload.get("report_dir", os.getenv("EVAL_REPORT_DIR", "data/reports/phase8"))),
            ),
            economic_leaderboard_path=_resolve_path(
                settings,
                str(
                    report_payload.get(
                        "economic_leaderboard_path",
                        os.getenv("EVAL_ECONOMIC_LEADERBOARD_PATH", "data/reports/phase5/economic_leaderboard_latest.csv"),
                    )
                ),
            ),
            compare_runs_dir=_resolve_path(
                settings,
                str(report_payload.get("compare_runs_dir", os.getenv("EVAL_COMPARE_RUNS_DIR", "data/reports/phase8/comparisons"))),
            ),
        ),
        alert_flags=AlertFlagConfig(
            alert_on_negative_pnl=_env_bool("EVAL_ALERT_NEGATIVE_PNL", bool(alert_payload.get("alert_on_negative_pnl", True))),
            alert_on_low_win_rate=_env_bool("EVAL_ALERT_LOW_WIN_RATE", bool(alert_payload.get("alert_on_low_win_rate", True))),
            alert_on_high_drawdown=_env_bool("EVAL_ALERT_HIGH_DRAWDOWN", bool(alert_payload.get("alert_on_high_drawdown", True))),
            alert_on_feature_drift=_env_bool("EVAL_ALERT_FEATURE_DRIFT", bool(alert_payload.get("alert_on_feature_drift", True))),
            alert_on_prediction_drift=_env_bool(
                "EVAL_ALERT_PREDICTION_DRIFT",
                bool(alert_payload.get("alert_on_prediction_drift", True)),
            ),
            alert_on_label_drift=_env_bool("EVAL_ALERT_LABEL_DRIFT", bool(alert_payload.get("alert_on_label_drift", False))),
            alert_on_degenerate_outputs=_env_bool(
                "EVAL_ALERT_DEGENERATE_OUTPUTS",
                bool(alert_payload.get("alert_on_degenerate_outputs", True)),
            ),
        ),
        auto_generate_after_phase7=_env_bool(
            "EVAL_AUTO_GENERATE_AFTER_PHASE7",
            bool(merged.get("auto_generate_after_phase7", True)),
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


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw_value!r}.")


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)
