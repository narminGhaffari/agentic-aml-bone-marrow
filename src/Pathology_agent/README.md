# Pathology Agent Source

This directory contains the source code for the agentic ROI-selection component used in the manuscript workflow.

The agent reduces a bone marrow whole-slide image into candidate fields, ranks candidate regions using quality filtering and embedding similarity, and then uses a vision-language model to select diagnostically interpretable ROIs.

## Main Modes

| Mode | Purpose |
| --- | --- |
| `aml_roi` | Select ROIs from one WSI and save `roi_collection.json` plus ROI images. |
| `aml_auto` | Select ROIs and run downstream VLM diagnosis. |
| `aml_stamp` | Run downstream STAMP prediction from an existing `roi_collection.json`. |

## Main Entry Points

- `evaluate/run_single_slide.py`: single-slide headless execution.
- `evaluate/run_batch_aml_staged.sh`: cohort-level staged ROI selection and downstream prediction.
- `wsi_core_pkg/runtime.py`: agent runtime orchestration.
- `wsi_core_pkg/tools.py`: slide navigation and ROI-selection tools.
- `wsi_core_pkg/stamp_bridge.py`: bridge from selected ROIs to STAMP feature extraction and deployment.

## Configuration

Runtime settings are read from environment variables and `configs/config.yaml`. Public examples use placeholders such as:

```bash
export OPENAI_API_BASE="https://YOUR_VLM_ENDPOINT/v1"
export OPENAI_API_KEY="YOUR_API_KEY"
```

The public repository does not include private WSIs, API keys, trained checkpoints, patient metadata, extracted features, or manuscript output files.

For runnable commands and expected controlled-access input paths, see:

- ../../README.md
- ../../docs/RUN_EXPERIMENTS.md
- ../../docs/INPUT_SCHEMA.md
