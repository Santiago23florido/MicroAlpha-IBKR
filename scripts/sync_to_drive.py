"""Phase 4 Google Drive sync entrypoint for PC2."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("sync-drive", sys.argv[1:]))
