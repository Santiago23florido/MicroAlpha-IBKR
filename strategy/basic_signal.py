from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignalResult:
    action: str | None
    reason: str


class BasicSignalStrategy:
    def generate_signal(self, market_context: dict[str, Any] | None = None) -> SignalResult:
        return SignalResult(
            action=None,
            reason="No automated trading logic is enabled in this MVP foundation.",
        )
