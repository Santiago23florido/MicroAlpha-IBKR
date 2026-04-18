#!/usr/bin/env python3
"""Show the configured Phase 7 execution backend."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("show-execution-backend", sys.argv[1:]))
