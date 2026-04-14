#!/usr/bin/env python3
"""Phase 1 healthcheck entrypoint for config, paths, and broker connectivity."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("healthcheck", sys.argv[1:]))
