#!/usr/bin/env python3

import argparse
import shlex

from workflow_common import add_common_args, run_or_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM diagnosis on manual ROIs.")
    add_common_args(parser)
    args = parser.parse_args()

    vlm_script = args.project_root / "VLM_manual_rois" / "Run_VLM.py"
    vlm_src = args.project_root / "VLM_manual_rois"
    model_names = {
        "gemma": "gemma-4-31B-it-h200",
        "qwen": "Qwen3.5-397B-A17B-FP8",
    }

    commands = []
    for model_label, model_name in model_names.items():
        for roi_count in ("5", "10"):
            output_dir = args.output_dir / "VLM_Manual_ROIs" / model_label / f"{roi_count}_rois"
            commands.append(
                [
                    "bash",
                    "-lc",
                    (
                        f"PYTHONPATH={shlex.quote(str(vlm_src))}:$PYTHONPATH "
                        f"VLM_MODEL_NAME={shlex.quote(model_name)} "
                        f"VLM_ROI_COUNT={shlex.quote(roi_count)} "
                        f"VLM_MAIN_DIR={shlex.quote(str(output_dir))} "
                        f"python {shlex.quote(str(vlm_script))}"
                    ),
                ]
            )

    run_or_plan(
        name="VLM diagnosis using manual ROIs",
        commands=commands,
        execute=args.execute,
        required_paths=[
            (vlm_script, "included standalone VLM source code"),
            (args.data_root / "manual_rois", "controlled-access manual ROI image directory"),
        ],
        notes=[
            "Uses the included VLM_manual_rois/Run_VLM.py source via PYTHONPATH.",
            "The wrapper sets VLM_MODEL_NAME, VLM_ROI_COUNT, and VLM_MAIN_DIR for Gemma/Qwen and 5/10 ROI runs.",
            "Private API endpoint/key and ROI images must be mounted under /data/private.",
            "This is included as an experiment entry point, not as a public reproducibility source of manuscript numbers.",
            "No API keys or patient images are included in the public repository.",
        ],
    )


if __name__ == "__main__":
    main()
