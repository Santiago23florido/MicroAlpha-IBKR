#!/usr/bin/env python3
"""Phase 1 training entrypoint for baseline or deep models."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("train", sys.argv[1:]))
