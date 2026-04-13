from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    def __init__(self, registry_path: str) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write({"active": {"baseline": None, "deep": None}, "baseline": [], "deep": []})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def list_models(self, model_type: str | None = None) -> list[dict[str, Any]]:
        payload = self._read()
        if model_type is None:
            return payload["baseline"] + payload["deep"]
        return list(payload[model_type])

    def register_model(self, model_type: str, record: dict[str, Any], set_active: bool = False) -> dict[str, Any]:
        payload = self._read()
        rows = [row for row in payload[model_type] if row["artifact_id"] != record["artifact_id"]]
        record = {
            **record,
            "model_type": model_type,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(record)
        payload[model_type] = rows
        if set_active:
            payload["active"][model_type] = record["artifact_id"]
        self._write(payload)
        return record

    def set_active_model(self, model_type: str, artifact_id: str) -> dict[str, Any]:
        payload = self._read()
        matching = next(
            (row for row in payload[model_type] if row["artifact_id"] == artifact_id),
            None,
        )
        if matching is None:
            raise ValueError(f"No {model_type} artifact found with id {artifact_id}.")
        payload["active"][model_type] = artifact_id
        self._write(payload)
        return matching

    def get_active_model(self, model_type: str) -> dict[str, Any] | None:
        payload = self._read()
        artifact_id = payload["active"].get(model_type)
        if artifact_id is None:
            return None
        return next((row for row in payload[model_type] if row["artifact_id"] == artifact_id), None)

    def get_model(self, model_type: str, artifact_id: str) -> dict[str, Any] | None:
        payload = self._read()
        return next((row for row in payload[model_type] if row["artifact_id"] == artifact_id), None)
