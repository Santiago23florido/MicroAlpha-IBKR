#!/usr/bin/env python3
"""Run the main PC1 development flow: pull from PC2, validate imports, and build features."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("dev-sync-and-build", sys.argv[1:]))
