from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.io import write_json


def write_reconciliation_reports(
    *,
    report_root: str | Path,
    session_dir: str | Path,
    session_id: str,
    reconciliation: dict[str, Any],
) -> dict[str, str]:
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    session_target = root / session_id
    session_target.mkdir(parents=True, exist_ok=True)
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    order_df = pd.DataFrame(list(reconciliation.get("orders", []) or []))
    fill_df = pd.DataFrame(list(reconciliation.get("fills", []) or []))
    position_df = pd.DataFrame(list(reconciliation.get("positions", []) or []))

    paths = {
        "orders_csv": str(session_target / f"orders_{session_id}.csv"),
        "fills_csv": str(session_target / f"fills_{session_id}.csv"),
        "positions_csv": str(session_target / f"positions_{session_id}.csv"),
        "summary_json": str(session_target / f"summary_{session_id}.json"),
    }

    order_df.to_csv(paths["orders_csv"], index=False)
    fill_df.to_csv(paths["fills_csv"], index=False)
    position_df.to_csv(paths["positions_csv"], index=False)
    write_json(paths["summary_json"], reconciliation.get("summary", {}))

    session_paths = {
        "session_orders_csv": str(session_dir / "reconciliation_orders.csv"),
        "session_fills_csv": str(session_dir / "reconciliation_fills.csv"),
        "session_positions_csv": str(session_dir / "reconciliation_positions.csv"),
        "session_summary_json": str(session_dir / "reconciliation_summary.json"),
    }
    order_df.to_csv(session_paths["session_orders_csv"], index=False)
    fill_df.to_csv(session_paths["session_fills_csv"], index=False)
    position_df.to_csv(session_paths["session_positions_csv"], index=False)
    write_json(session_paths["session_summary_json"], reconciliation.get("summary", {}))
    return {**paths, **session_paths}
