#!/usr/bin/env python3
"""Run the Phase 6 operational risk/model readiness check."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("risk-check", sys.argv[1:]))
