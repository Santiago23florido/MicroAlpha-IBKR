"""Phase 4 SQLite snapshot backup entrypoint for PC2."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("backup-sqlite", sys.argv[1:]))
