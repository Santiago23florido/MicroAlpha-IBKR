#!/usr/bin/env python3
"""Phase 3 feature pipeline entrypoint for PC1 research data preparation."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("build-features", sys.argv[1:]))
