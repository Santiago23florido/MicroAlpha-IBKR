from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class IBKRPacingSnapshot:
    total_requests_10m: int
    key_requests_2s: int
    pacing_waits: int
    total_wait_seconds: float


class IBKRHistoricalRateLimiter:
    def __init__(
        self,
        *,
        max_requests_per_10_min: int,
        max_same_contract_requests_per_2_sec: int,
        dedupe_window_seconds: int,
    ) -> None:
        self.max_requests_per_10_min = max_requests_per_10_min
        self.max_same_contract_requests_per_2_sec = max_same_contract_requests_per_2_sec
        self.dedupe_window_seconds = dedupe_window_seconds
        self._global_requests: deque[float] = deque()
        self._per_key_requests: dict[str, deque[float]] = {}
        self._per_signature_requests: dict[str, float] = {}
        self.pacing_waits = 0
        self.total_wait_seconds = 0.0

    def wait_for_slot(self, *, request_key: str, request_signature: str, cost: int = 1) -> float:
        waited = 0.0
        while True:
            now = time.monotonic()
            self._evict_old(now)
            required_wait = max(
                self._dedupe_wait(now, request_signature),
                self._key_window_wait(now, request_key),
                self._global_window_wait(now, cost),
            )
            if required_wait <= 0:
                for _ in range(cost):
                    self._global_requests.append(now)
                per_key = self._per_key_requests.setdefault(request_key, deque())
                for _ in range(cost):
                    per_key.append(now)
                self._per_signature_requests[request_signature] = now
                return waited
            time.sleep(required_wait)
            waited += required_wait
            self.pacing_waits += 1
            self.total_wait_seconds += required_wait

    def snapshot(self, *, request_key: str) -> IBKRPacingSnapshot:
        now = time.monotonic()
        self._evict_old(now)
        return IBKRPacingSnapshot(
            total_requests_10m=len(self._global_requests),
            key_requests_2s=len(self._per_key_requests.get(request_key, deque())),
            pacing_waits=self.pacing_waits,
            total_wait_seconds=round(self.total_wait_seconds, 4),
        )

    def _evict_old(self, now: float) -> None:
        while self._global_requests and (now - self._global_requests[0]) >= 600.0:
            self._global_requests.popleft()
        for key in list(self._per_key_requests):
            queue = self._per_key_requests[key]
            while queue and (now - queue[0]) >= 2.0:
                queue.popleft()
            if not queue:
                self._per_key_requests.pop(key, None)

    def _dedupe_wait(self, now: float, request_signature: str) -> float:
        last = self._per_signature_requests.get(request_signature)
        if last is None:
            return 0.0
        elapsed = now - last
        if elapsed >= self.dedupe_window_seconds:
            return 0.0
        return self.dedupe_window_seconds - elapsed

    def _key_window_wait(self, now: float, request_key: str) -> float:
        queue = self._per_key_requests.get(request_key, deque())
        if len(queue) < self.max_same_contract_requests_per_2_sec:
            return 0.0
        return max(0.0, 2.0 - (now - queue[0]))

    def _global_window_wait(self, now: float, cost: int) -> float:
        if len(self._global_requests) + cost <= self.max_requests_per_10_min:
            return 0.0
        return max(0.0, 600.0 - (now - self._global_requests[0]))
