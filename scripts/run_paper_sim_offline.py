#!/usr/bin/env python3
"""Run the Phase 7 offline paper simulation pipeline."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("run-paper-sim-offline", sys.argv[1:]))
