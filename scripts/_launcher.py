from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import main


GLOBAL_OPTIONS = {"--env-file", "--config-dir", "--environment"}


def run_app_command(command: str, argv: list[str]) -> int:
    return main(_inject_command(command, argv))


def _inject_command(command: str, argv: list[str]) -> list[str]:
    prefix: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        option_name = argument.split("=", maxsplit=1)[0]
        if option_name not in GLOBAL_OPTIONS:
            break
        prefix.append(argument)
        if "=" not in argument and index + 1 < len(argv):
            prefix.append(argv[index + 1])
            index += 2
            continue
        index += 1
    return [*prefix, command, *argv[index:]]
