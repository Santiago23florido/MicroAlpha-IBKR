#!/usr/bin/env python3
"""Run Phase 6 decisions over historical features."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("run-decisions-offline", sys.argv[1:]))
