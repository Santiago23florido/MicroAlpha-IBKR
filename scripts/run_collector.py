#!/usr/bin/env python3
"""Phase 1 collector entrypoint for PC2-style raw data capture."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("collect", sys.argv[1:]))
