#!/usr/bin/env python3
"""List the configured feature sets and their default selection."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("list-feature-sets", sys.argv[1:]))
