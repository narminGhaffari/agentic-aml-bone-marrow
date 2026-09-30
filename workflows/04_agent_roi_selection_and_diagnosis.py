#!/usr/bin/env python3

import argparse
import shlex

from workflow_common import EXTRACTORS, add_common_args, run_or_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agentic ROI selection and downstream diagnosis branches."
    )
    add_common_args(parser)
    args = parser.parse_args()

    staged_script = args.project_root / "Pathology_agent" / "evaluate" / "run_batch_aml_staged.sh"
    agent_src = args.project_root / "Pathology_agent"
    stamp_src = args.project_root / "STAMP" / "src"
    commands = []
    for model in ("gemma-4-31B-it", "qwen"):
        for extractor in EXTRACTORS:
            staged_args = [
                str(staged_script),
                "--csv",
                str(args.data_root / "metadata" / "AML-HEALTHY-SLIDE-DX_test.csv"),
                "--slides-root",
                str(args.data_root / "wsi"),
                "--output-dir",
                str(args.output_dir / "Agent_ROI"),
                "--experiment-root",
                str(args.output_dir / "Agent_ROI"),
                "--model",
                model,
                "--extractor",
                extractor.replace("-", "_") if extractor == "h-optimus-1" else extractor,
                "--stamp-extractor",
                extractor.replace("-", "_") if extractor == "h-optimus-1" else extractor,
                "--tile-size-px",
                "224",
                "--roi-size-px",
                "2048",
                "--max-accepted-rois",
                "5",
                "--target-accepted-rois",
                "5",
                "--default-mpp-um",
                "0.159",
                "--sequential-predictions",
            ]
            commands.append(
                [
                    "bash",
                    "-lc",
                    (
                        f"PYTHONPATH={shlex.quote(str(agent_src))}:"
                        f"{shlex.quote(str(stamp_src))}:$PYTHONPATH "
                        f"bash {shlex.join(staged_args)}"
                    ),
                ]
            )

    run_or_plan(
        name="Agent ROI selection plus downstream VLM or STAMP predictions",
        commands=commands,
        execute=args.execute,
        required_paths=[
            (staged_script, "included agent staged workflow script"),
            (agent_src / "wsi_core_pkg" / "runtime.py", "included agent source code"),
            (stamp_src / "stamp" / "__main__.py", "included STAMP source code"),
            (args.data_root / "metadata" / "AML-HEALTHY-SLIDE-DX_test.csv", "test slide metadata"),
            (args.data_root / "wsi", "controlled-access WSI directory"),
        ],
        notes=[
            "This is the main agentic pipeline entry point: WSI -> candidate fields -> accepted ROIs -> downstream VLM or STAMP outputs.",
            "Uses the included Pathology_agent source and included STAMP source via PYTHONPATH.",
            "Exact manuscript outputs require private WSIs and VLM/model access.",
        ],
    )


if __name__ == "__main__":
    main()
