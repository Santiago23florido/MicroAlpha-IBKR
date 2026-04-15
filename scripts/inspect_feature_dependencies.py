#!/usr/bin/env python3
"""Inspect dataset compatibility against the configured feature registry."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("inspect-feature-dependencies", sys.argv[1:]))
