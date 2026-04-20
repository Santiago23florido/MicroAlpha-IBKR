from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import Settings
from config.phase10_11 import load_phase10_11_config


@dataclass(frozen=True)
class ScheduledTask:
    name: str
    stage: str
    interval_seconds: float
    enabled: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "interval_seconds": self.interval_seconds,
            "enabled": self.enabled,
            "description": self.description,
        }


def build_scheduler_plan(settings: Settings) -> dict[str, Any]:
    phase10_11 = load_phase10_11_config(settings)
    tasks = [
        ScheduledTask(
            name="preflight_checks",
            stage="pre_session",
            interval_seconds=float(phase10_11.scheduler_intervals.preflight_delay_seconds),
            enabled=True,
            description="Validate active model, backend, broker mode, paths, and broker reachability before session start.",
        ),
        ScheduledTask(
            name="broker_healthcheck",
            stage="intra_session",
            interval_seconds=float(phase10_11.scheduler_intervals.scheduled_healthcheck_seconds),
            enabled=True,
            description="Run conservative broker connectivity checks during the validation cycle.",
        ),
        ScheduledTask(
            name="post_session_reconciliation",
            stage="post_session",
            interval_seconds=float(phase10_11.scheduler_intervals.scheduled_reconciliation_seconds),
            enabled=True,
            description="Reconcile internal orders, fills, and positions against IBKR Paper state.",
        ),
        ScheduledTask(
            name="report_generation",
            stage="post_session",
            interval_seconds=float(phase10_11.scheduler_intervals.scheduled_reporting_seconds),
            enabled=True,
            description="Generate session summaries, alerts, readiness, and system health artifacts.",
        ),
    ]
    return {
        "status": "ok",
        "scheduler_enabled": bool(settings.deployment.scheduler_enabled),
        "default_monitor_iterations": phase10_11.scheduler_intervals.default_monitor_iterations,
        "tasks": [task.to_dict() for task in tasks],
    }
