#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


EXTRACTORS = ("uni2", "virchow2", "h-optimus-1", "dino-bloom")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("src"),
        help="Root of the source code in this repository.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/private"),
        help="Root of controlled-access inputs. Not included in this public repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=True,
        help="Execute commands. This flag is kept for compatibility and is enabled by default.",
    )


def emit_plan(name: str, commands: list[list[str]], notes: list[str] | None = None) -> None:
    payload = {
        "workflow": name,
        "mode": "controlled-access execution",
        "notes": notes or [],
        "commands": [" ".join(command) for command in commands],
    }
    print(json.dumps(payload, indent=2))


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {description}: {path}\n"
            "This is expected in the public repository. Mount controlled-access data "
            "or upload allowed source code before running with --execute."
        )


def run_or_plan(
    *,
    name: str,
    commands: list[list[str]],
    execute: bool,
    required_paths: list[tuple[Path, str]] | None = None,
    notes: list[str] | None = None,
) -> None:
    emit_plan(name, commands, notes)

    for path, description in required_paths or []:
        require_path(path, description)

    for command in commands:
        subprocess.run(command, check=True)
