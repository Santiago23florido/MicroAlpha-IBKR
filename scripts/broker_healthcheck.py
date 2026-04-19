#!/usr/bin/env python3
"""Run the Phase 9 IBKR Paper broker healthcheck."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("broker-healthcheck", sys.argv[1:]))
