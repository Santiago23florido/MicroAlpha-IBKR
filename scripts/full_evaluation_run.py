#!/usr/bin/env python3
"""Run the full Phase 8 evaluation bundle."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("full-evaluation-run", sys.argv[1:]))
