#!/usr/bin/env python3
"""Validate generated feature parquet files for missingness and degenerate columns."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("validate-features", sys.argv[1:]))
