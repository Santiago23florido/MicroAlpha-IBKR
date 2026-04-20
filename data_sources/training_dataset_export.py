from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.io import write_json


def export_training_dataset(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    metadata: dict[str, Any],
    write_parquet: bool,
    write_manifest: bool,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot export an empty training dataset.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _validate_training_frame(frame)
    if target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
    else:
        frame.to_csv(target, index=False)

    parquet_path = None
    if write_parquet and target.suffix.lower() != ".parquet":
        parquet_target = target.with_suffix(".parquet")
        frame.to_parquet(parquet_target, index=False)
        parquet_path = str(parquet_target)

    manifest_path = None
    if write_manifest:
        manifest_target = target.with_suffix(".manifest.json")
        manifest_payload = {
            **metadata,
            "output_path": str(target),
            "parquet_path": parquet_path,
            "columns": list(frame.columns),
        }
        manifest_path = write_json(manifest_target, manifest_payload)

    return {
        "status": "ok",
        "output_path": str(target),
        "parquet_path": parquet_path,
        "manifest_path": manifest_path,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
    }


def _validate_training_frame(frame: pd.DataFrame) -> None:
    required = ["timestamp", "symbol", "close", "last", "bid", "ask", "bid_size", "ask_size"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Training dataset is missing required canonical columns: {missing}")
    if frame["timestamp"].isna().all():
        raise ValueError("Training dataset has no usable timestamp values.")
    if pd.to_numeric(frame["close"], errors="coerce").isna().all():
        raise ValueError("Training dataset has no usable close values.")
