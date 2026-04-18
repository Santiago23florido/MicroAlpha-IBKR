#!/usr/bin/env python3
"""Compare stored Phase 8 run reports."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("compare-runs", sys.argv[1:]))
