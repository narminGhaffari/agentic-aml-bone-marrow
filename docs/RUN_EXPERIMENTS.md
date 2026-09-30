# Running Individual Experiments

This repository does not include patient data, WSI files, feature files, model checkpoints, or API keys. Replace the example paths below with controlled-access local paths.

All commands are intended to be run from the repository root.

## 1. Agentic ROI Selection For One WSI

This runs only the agentic ROI-selection stage and saves the selected ROIs plus `roi_collection.json`.

```bash
PYTHONPATH=src/Pathology_agent:src/STAMP/src:$PYTHONPATH \
OPENAI_API_BASE="https://YOUR_VLM_ENDPOINT/v1" \
OPENAI_API_KEY="YOUR_API_KEY" \
python src/Pathology_agent/evaluate/run_single_slide.py \
  --slide /path/to/one_slide.svs \
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

## 2. Agentic ROI Selection Plus VLM Diagnosis For One WSI

Use the same command, but set:

```bash
--agent aml_auto
```

This performs ROI selection and then runs VLM diagnosis from the selected ROIs.

## 3. Agentic ROI Selection Plus VLM And STAMP For One WSI

Use:

```bash
--agent aml_auto_stamp
```

and provide a private STAMP checkpoint root:

```bash
--stamp-train-root /data/private/stamp_train/All_Tiles
```

This requires trained STAMP checkpoints and the private inputs used for STAMP deployment.

## 4. STAMP On All Tiles

The workflow wrapper runs AML-vs-healthy and NPM1 tasks for UNI2, Virchow2, H-Optimus-1, and DinoBloom.

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

Each STAMP config directory should contain the task-specific `config.yaml` needed by `python -m stamp train` and `python -m stamp deploy`.

## 5. STAMP On Manual ROIs

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

## 6. Standalone VLM Diagnosis On Manual ROIs

The wrapper runs Gemma and Qwen for 5-ROI and 10-ROI settings.

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

## 7. Agent-Selected ROIs With Downstream VLM And STAMP

The wrapper runs the agentic pipeline on the test slide table for Gemma/Qwen and the configured feature extractors.

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

## 8. Run All Four Experiment Families

```bash
OPENAI_API_BASE="https://YOUR_VLM_ENDPOINT/v1" \
OPENAI_API_KEY="YOUR_API_KEY" \
VLM_BASE_URL="https://YOUR_VLM_ENDPOINT/v1" \
VLM_API_KEY_FILE="/data/private/secrets/vlm_api_key.json" \
bash workflows/run_experiments.sh \
  --project-root src \
  --data-root /data/private \
  --output-dir results
```

This command is only meaningful when all controlled-access inputs are available.
