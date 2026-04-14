#!/usr/bin/env python3
"""PC1 entrypoint for pulling new market data from the shared PC2 LAN folder."""

from __future__ import annotations

import sys

from _launcher import run_app_command


if __name__ == "__main__":
    raise SystemExit(run_app_command("pull-from-pc2", sys.argv[1:]))
