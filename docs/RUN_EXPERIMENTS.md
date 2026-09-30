# Running Experiments

This repository does not include patient data, WSI files, feature files, model checkpoints, or API keys. Replace the example `/data/private` paths below with your own controlled-access local paths.

All commands are intended to be run from the repository root.

## Single-WSI Agent Modes

These commands all use the same script, `src/Pathology_agent/evaluate/run_single_slide.py`. The only difference is the value passed to `--agent`.

| Mode | What it does | When to use it |
| --- | --- | --- |
| `aml_roi` | Selects ROIs only and saves ROI images plus `roi_collection.json`. | You only want the agent-selected regions from one WSI. |
| `aml_auto` | Selects ROIs and then runs VLM diagnosis on those ROIs. | You want an agent plus VLM result for one WSI. |
| `aml_stamp` | Runs STAMP on an existing `roi_collection.json`. | You already selected ROIs with `aml_roi` and now want STAMP prediction on those ROIs. |

Example ROI-selection-only command:

```bash
PYTHONPATH=src/Pathology_agent:src/STAMP/src:$PYTHONPATH \
OPENAI_API_BASE="https://YOUR_VLM_ENDPOINT/v1" \
OPENAI_API_KEY="YOUR_API_KEY" \
python src/Pathology_agent/evaluate/run_single_slide.py \
  --slide /data/private/wsi/one_slide.svs \
  --output-dir results/single_slide_agent_roi \
  --experiment-root results/single_slide_cache \
  --agent aml_roi \
  --model gemma-4-31B-it \
  --extractor uni2 \
  --tile-filter hybrid \
  --tile-size-px 224 \
  --roi-size-px 2048 \
  --max-accepted-rois 5 \
  --target-accepted-rois 5 \
  --default-mpp-um 0.159
```

To run ROI selection plus VLM diagnosis, use the same command but change:

```bash
--agent aml_auto
```

To run ROI selection plus STAMP prediction, use two steps. First run ROI selection with `aml_roi`, then run STAMP on the saved `roi_collection.json`:

```bash
PYTHONPATH=src/Pathology_agent:src/STAMP/src:$PYTHONPATH \
python src/Pathology_agent/evaluate/run_single_slide.py \
  --slide /data/private/wsi/one_slide.svs \
  --output-dir results/single_slide_agent_stamp \
  --experiment-root results/single_slide_cache \
  --agent aml_stamp \
  --roi-input-path results/single_slide_agent_roi/one_slide/roi_collection.json \
  --stamp-train-root /data/private/stamp_train/All_Tiles
```

## Individual Manuscript Experiments

These wrappers are intended to be run individually. They are not single-slide commands, and there is no combined "run everything" launcher because the manuscript workflows involve staged training, deployment, validation, and statistics steps.

## STAMP All Tiles

This runs STAMP using all available tiles for AML-vs-healthy and NPM1 tasks across the configured feature extractors.

```bash
python workflows/01_stamp_all_tiles.py \
  --project-root src \
  --data-root /data/private \
  --output-dir results \
  --execute
```

Expected private inputs include:

```text
/data/private/stamp_configs/All_Tiles/
/data/private/features/All_Tiles/
```

## STAMP Manual ROIs

This runs STAMP using manually annotated ROI tiles.

```bash
python workflows/02_stamp_manual_rois.py \
  --project-root src \
  --data-root /data/private \
  --output-dir results \
  --execute
```

Expected private inputs include:

```text
/data/private/stamp_configs/Manual_ROIs/
/data/private/features/Manual_ROIs/
```

## Standalone VLM On Manual ROIs

This runs VLM diagnosis directly on manual ROIs, without the agent ROI-selection step.

```bash
VLM_BASE_URL="https://YOUR_VLM_ENDPOINT/v1" \
VLM_API_KEY_FILE="/data/private/secrets/vlm_api_key.json" \
python workflows/03_vlm_manual_rois.py \
  --project-root src \
  --data-root /data/private \
  --output-dir results \
  --execute
```

Expected private inputs include:

```text
/data/private/manual_rois/
/data/private/metadata/AML_HEALTHY-CLINI-DX_test.xlsx
/data/private/metadata/AML-HEALTHY-SLIDE-DX.csv
/data/private/secrets/vlm_api_key.json
```

## Agent-Selected ROIs With Downstream VLM Or STAMP

This is the cohort-level agent-selected ROI experiment. The agentic part is ROI selection; the selected ROIs can then be evaluated with downstream VLM diagnosis or downstream STAMP prediction.

```bash
OPENAI_API_BASE="https://YOUR_VLM_ENDPOINT/v1" \
OPENAI_API_KEY="YOUR_API_KEY" \
python workflows/04_agent_roi_selection_and_diagnosis.py \
  --project-root src \
  --data-root /data/private \
  --output-dir results \
  --execute
```

Expected private inputs include:

```text
/data/private/metadata/AML-HEALTHY-SLIDE-DX_test.csv
/data/private/wsi/
/data/private/stamp_train/
```
