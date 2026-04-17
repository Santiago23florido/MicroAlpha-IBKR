#!/usr/bin/env python3
"""PC1 entrypoint for running the phase 5 comparison search and leaderboard build."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("compare-model-variants", sys.argv[1:]))
