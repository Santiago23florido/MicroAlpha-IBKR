#!/usr/bin/env python3
"""Detect drift for a Phase 7 run."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("detect-drift", sys.argv[1:]))
