#!/usr/bin/env python3
"""PC1 entrypoint for training one phase 5 model variant."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("train-baseline", sys.argv[1:]))
