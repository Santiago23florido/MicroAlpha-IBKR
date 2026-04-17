#!/usr/bin/env python3
"""PC1 entrypoint for generating phase 5 labels from feature parquet stores."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("build-labels", sys.argv[1:]))
