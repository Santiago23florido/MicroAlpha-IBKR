from __future__ import annotations

from typing import Any

from config import Settings
from data.historical_loader import load_historical_dataset


def run_backtest_stub(
    settings: Settings,
    *,
    data_path: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    frame = load_historical_dataset(settings, data_path)
    ticker = (symbol or settings.ib_symbol).upper()
    return {
        "status": "placeholder",
        "message": (
            "Phase 1 validates backtest inputs and project wiring only. "
            "Strategy simulation and performance outputs are intentionally deferred to phase 2."
        ),
        "symbol": ticker,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "data_source": data_path or "data/sample/spy_microstructure_sample.csv",
        "environment": settings.environment,
        "backtest_enabled": settings.backtest_enabled,
    }
