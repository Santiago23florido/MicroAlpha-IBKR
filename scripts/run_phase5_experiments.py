#!/usr/bin/env python3
"""PC1 main entrypoint for the complete phase 5 experiment workflow."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("run-phase5-experiments", sys.argv[1:]))
