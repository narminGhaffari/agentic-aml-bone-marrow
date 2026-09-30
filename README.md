# Agentic AI Navigates Bone Marrow Aspirate Smears for Autonomous AML Diagnosis

This repository contains the source code and executable workflow structure for the manuscript:

**Agentic AI Navigates Bone Marrow Aspirate Smears for Autonomous AML Diagnosis**

The repository is code-only. Raw bone marrow whole-slide images, manual ROI images, extracted feature files, patient-level metadata, model checkpoints, VLM API keys, patient-level predictions, and manuscript figures/statistics are not included because they contain protected clinical information or require restricted resources.

## Included Experiments

The public workflow covers four computational experiment families:

1. **STAMP on all tiles**
   - AML versus healthy classification.
   - NPM1 mutation prediction among AML patients.
   - UNI2, Virchow2, H-Optimus-1, and DinoBloom feature extractors.

2. **STAMP on manual ROIs**
   - Same tasks and feature extractors as the all-tile STAMP experiment.
   - Uses manually annotated/expert ROI inputs.

3. **Standalone VLM diagnosis on manual ROIs**
   - Gemma and Qwen runs.
   - 5-ROI and 10-ROI settings.

4. **Agent-selected ROIs with downstream diagnosis**
   - Agentic ROI selection from WSI candidate fields.
   - Downstream VLM prediction.
   - Downstream STAMP prediction.

The blinded ROI-quality reader study is intentionally not included in this repository workflow.

## Repository Layout

```text
src/
  Pathology_agent/      Agentic ROI selection, candidate ranking, tools, runtime, and batch workflows
  STAMP/                STAMP source used for training/deployment/heatmap handling
  VLM_manual_rois/      Standalone VLM script for manual ROI diagnosis
  Prompts/              VLM prompt templates

workflows/
  run_experiments.sh    Controlled-access execution entry point for the four experiments
  01_stamp_all_tiles.py
  02_stamp_manual_rois.py
  03_vlm_manual_rois.py
  04_agent_roi_selection_and_diagnosis.py

docs/
  INPUT_SCHEMA.md
  DATA_AVAILABILITY.md
  SOURCE_CODE_SCOPE.md
  RUN_EXPERIMENTS.md
```

## Controlled-Access Execution

Full execution requires private inputs mounted outside the repository, for example:

```text
/data/private/metadata/
/data/private/wsi/
/data/private/features/
/data/private/manual_rois/
/data/private/stamp_configs/
/data/private/secrets/
```

Then run:

```bash
bash workflows/run_experiments.sh --execute \
  --project-root src \
  --data-root /data/private \
  --output-dir results
```

The wrappers explicitly execute source code shipped in this repository:

- STAMP workflows run `python -m stamp` using `PYTHONPATH=src/STAMP/src`.
- Manual-ROI VLM workflows run `src/VLM_manual_rois/Run_VLM.py`.
- Agent workflows run `src/Pathology_agent/evaluate/run_batch_aml_staged.sh` using both `src/Pathology_agent` and `src/STAMP/src`.

For commands to run each individual experiment, including agentic ROI selection on a single WSI, see [Running individual experiments](docs/RUN_EXPERIMENTS.md).

## Data And Model Restrictions

The manuscript results require protected clinical data and trained components that cannot be redistributed here. See:

- [Input schema](docs/INPUT_SCHEMA.md)
- [Data availability](docs/DATA_AVAILABILITY.md)
- [Source code scope](docs/SOURCE_CODE_SCOPE.md)
- [Running individual experiments](docs/RUN_EXPERIMENTS.md)

## Environment

A lightweight environment file is provided:

```bash
conda env create -f environment.yml
```

The full GPU workflow may require additional model-specific dependencies and access permissions for pathology foundation models and VLM endpoints.
