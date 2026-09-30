#!/usr/bin/env python3

import argparse
import shlex

from workflow_common import EXTRACTORS, add_common_args, run_or_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STAMP experiments on all tiles.")
    add_common_args(parser)
    args = parser.parse_args()

    config_root = args.data_root / "stamp_configs" / "All_Tiles"
    deploy_root = args.output_dir / "STAMP_Deploy" / "All_Tiles"
    stamp_src = args.project_root / "STAMP" / "src"
    stamp_python = f"PYTHONPATH={shlex.quote(str(stamp_src))}:$PYTHONPATH python -m stamp"
    commands = []
    for task in ("WHO_CLASSE_SIMPLE", "NPM1"):
        for extractor in EXTRACTORS:
            config_dir = config_root / task / extractor
            commands.append(
                [
                    "bash",
                    "-lc",
                    (
                        f"cd {shlex.quote(str(config_dir))} && "
                        f"{stamp_python} train && {stamp_python} deploy && "
                        f"mkdir -p {shlex.quote(str(deploy_root / task / extractor))}"
                    ),
                ]
            )

    run_or_plan(
        name="STAMP all-tiles training and deployment",
        commands=commands,
        execute=args.execute,
        required_paths=[
            (stamp_src / "stamp" / "__main__.py", "included STAMP source code"),
            (config_root, "controlled-access STAMP all-tile config directories"),
            (args.data_root / "features" / "All_Tiles", "controlled-access all-tile features"),
        ],
        notes=[
            "Runs AML-vs-healthy and NPM1 tasks for UNI2, Virchow2, H-Optimus-1, and DinoBloom.",
            "Uses the STAMP source included at src/STAMP/src via PYTHONPATH.",
            "Each private config directory must contain the corresponding STAMP config.yaml with metadata, feature, checkpoint, and output paths.",
            "The public repository intentionally does not include private WSI-derived features.",
        ],
    )


if __name__ == "__main__":
    main()
