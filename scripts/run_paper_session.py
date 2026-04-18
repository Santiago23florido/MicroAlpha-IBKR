#!/usr/bin/env python3
"""Run one Phase 7 paper/mock session cycle."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("run-paper-session", sys.argv[1:]))
