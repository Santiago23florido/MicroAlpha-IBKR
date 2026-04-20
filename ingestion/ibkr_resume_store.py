from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResumeHandle:
    state_path: Path
    raw_path: Path
    manifest_path: Path


class IBKRBackfillResumeStore:
    def __init__(self, state_root: str | Path, output_root: str | Path) -> None:
        self.state_root = Path(state_root)
        self.output_root = Path(output_root)

    def resolve(self, *, symbol: str, what_to_show: str, bar_size: str) -> ResumeHandle:
        slug = f"{symbol.upper()}__{what_to_show.upper()}__{bar_size.replace(' ', '_')}"
        state_path = self.state_root / f"{slug}.state.json"
        raw_path = self.output_root / symbol.upper() / what_to_show.upper() / bar_size.replace(" ", "_") / "bars.parquet"
        manifest_path = raw_path.with_name("manifest.json")
        return ResumeHandle(state_path=state_path, raw_path=raw_path, manifest_path=manifest_path)

    def load(self, handle: ResumeHandle) -> dict[str, Any]:
        if not handle.state_path.exists():
            return {}
        try:
            return json.loads(handle.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt backfill resume state: {handle.state_path}") from exc

    def save(self, handle: ResumeHandle, payload: dict[str, Any]) -> str:
        handle.state_path.parent.mkdir(parents=True, exist_ok=True)
        handle.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(handle.state_path)
