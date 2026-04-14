#!/usr/bin/env python3
"""Phase 1 session entrypoint for one ORB decision cycle."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("run-session", sys.argv[1:]))
