#!/usr/bin/env python3
"""Generate the full Phase 8 report bundle."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("generate-report", sys.argv[1:]))
