#!/usr/bin/env python3
"""Show the current Phase 7 execution status."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("execution-status", sys.argv[1:]))
