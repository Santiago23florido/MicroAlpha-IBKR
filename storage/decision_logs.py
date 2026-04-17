from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class DecisionLogStore:
    def __init__(self, path: str, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True, default=str))
            handle.write("\n")
