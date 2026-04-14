#!/usr/bin/env python3
"""Validate imported PC2 parquet files before feature generation on PC1."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("validate-imports", sys.argv[1:]))
