from __future__ import annotations

from typing import Any

from config import Settings


def describe_label_pipeline(settings: Settings) -> dict[str, Any]:
    return {
        "status": "placeholder",
        "message": (
            "Label generation is intentionally minimal in phase 1. "
            "The repository now has a dedicated labels module so phase 2 can add "
            "forward-return and execution-aware labels without reshaping the project again."
        ),
        "target_horizon_minutes": settings.models.target_horizon_minutes,
        "environment": settings.environment,
    }
