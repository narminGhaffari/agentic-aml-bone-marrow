# Source Code Scope

The repository includes a clean source-code snapshot under:

```text
src/
```

This snapshot is intended to show how the experiments were implemented, not to expose
protected clinical data.
Local machine-specific absolute paths have been replaced with portable placeholders
such as `/data/private`, `results`, and `src`.

## Included

- `Pathology_agent/`: agentic ROI selection, candidate ranking, slide tools, runtime,
  STAMP bridge, VLM diagnosis flow, batch experiment entry points, and tests.
- `VLM_manual_rois/`: standalone VLM script used for manual-ROI diagnosis.
- `Prompts/`: prompt templates used by the VLM workflows.
- `STAMP/src/`: STAMP package source used for feature extraction, training,
  deployment, and heatmap/tile handling.
- `workflows/*.py` and `workflows/*.sh`: wrapper scripts that connect the source
  tree to the expected controlled-access input layout.

## Included Experiment Families

1. STAMP on all tiles.
2. STAMP on manual ROIs.
3. Standalone VLM diagnosis on manual ROIs.
4. Agent-selected ROIs with downstream VLM or STAMP prediction.

## Excluded

- whole-slide images,
- ROI images,
- extracted feature files,
- patient-level metadata,
- trained model checkpoints,
- patient-level predictions,
- manuscript figures,
- manuscript statistical output tables,
- blinded ROI-quality reader-study workflows and outputs,
- local virtual environments, git internals, caches, and generated outputs.

## What Requires Controlled-Access Data

A full execution requires mounting the private data under `/data/private`, including
metadata, WSIs, STAMP configs, extracted features/checkpoints, and VLM/API credentials.
Those inputs cannot be distributed in the public repository.
