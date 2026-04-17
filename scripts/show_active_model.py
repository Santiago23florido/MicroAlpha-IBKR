#!/usr/bin/env python3
"""Show the configured Phase 6 active model."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("show-active-model", sys.argv[1:]))
