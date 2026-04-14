"""Phase 4 local-retention cleanup entrypoint for PC2."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("cleanup-local", sys.argv[1:]))
