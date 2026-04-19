#!/usr/bin/env python3
"""Run one Phase 9 operational paper session against IBKR Paper."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("run-paper-session-real", sys.argv[1:]))
