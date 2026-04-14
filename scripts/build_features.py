#!/usr/bin/env python3
"""PC1 entrypoint for validating imported raw data and generating feature parquet files."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("build-features", sys.argv[1:]))
