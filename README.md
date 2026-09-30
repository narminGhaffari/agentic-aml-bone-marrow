# Agentic AI for Bone Marrow AML Diagnosis

Source code for the manuscript:

**Agentic AI Navigates Bone Marrow Aspirate Smears for Autonomous AML Diagnosis**

This repository contains the computational workflow used to study agentic region-of-interest (ROI) selection in bone marrow aspirate whole-slide images, with downstream AML and NPM1 prediction using VLM- and STAMP-based models.

The repository is code-only. Raw whole-slide images, manual ROI images, extracted features, patient-level metadata, trained checkpoints, VLM API keys, patient-level predictions, manuscript figures, and statistical output tables are not included because they contain protected clinical information or require controlled-access resources.

## ✨ What This Repository Contains

- 🧭 **Agentic ROI selection** from bone marrow whole-slide images.
- 🔎 **Deterministic candidate ranking** using tissue filtering, quality filtering, and pathology foundation-model embeddings.
- 🧠 **Downstream VLM diagnosis** from selected or manually annotated ROIs.
- 🧬 **STAMP training/deployment workflows** for all-tile, manual-ROI, and agent-selected ROI settings.
- 📦 **Executable workflow wrappers** for the four main computational experiment families.
- 📚 **Documentation** for expected private inputs, source-code scope, and individual run commands.

## 🧪 Experiment Families

The public workflow structure covers four experiment families:

1. **STAMP on all tiles**
   - AML versus healthy classification.
   - NPM1 mutation prediction among AML patients.
   - UNI2, Virchow2, H-Optimus-1, and DinoBloom feature extractors.

2. **STAMP on manual ROIs**
   - Same prediction tasks and feature extractors as the all-tile STAMP experiment.
   - Uses manually annotated ROI inputs.

3. **Standalone VLM diagnosis on manual ROIs**
   - Gemma and Qwen runs.
   - 5-ROI and 10-ROI settings.

4. **Agent-selected ROIs with downstream prediction**
   - The agentic part is ROI selection.
   - Selected ROIs can then be evaluated with downstream VLM diagnosis or downstream STAMP prediction.

## 📁 Repository Layout

```text
src/
  Pathology_agent/      Agentic ROI selection, candidate ranking, tools, runtime, and batch workflows
  STAMP/                STAMP source used for training, deployment, heatmaps, and tile handling
  VLM_manual_rois/      Standalone VLM script for manual ROI diagnosis
  Prompts/              Prompt templates used by the VLM workflows

workflows/
  run_experiments.sh    Entry point for the four cohort-level experiment families
  01_stamp_all_tiles.py
  02_stamp_manual_rois.py
  03_vlm_manual_rois.py
  04_agent_roi_selection_and_diagnosis.py

docs/
  INPUT_SCHEMA.md       Expected controlled-access input layout
  RUN_EXPERIMENTS.md    Commands for each experiment and single-WSI agent runs
  DATA_AVAILABILITY.md  Data-access restrictions
  SOURCE_CODE_SCOPE.md  What is included and excluded
```

## 🚀 Quick Start

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate agentic-aml-bone-marrow
```

Full execution requires private inputs mounted outside the repository, for example:

```text
/data/private/metadata/
/data/private/wsi/
/data/private/features/
/data/private/manual_rois/
/data/private/stamp_configs/
/data/private/stamp_train/
/data/private/secrets/
```

Run all four cohort-level experiment families:

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

For individual commands, including agentic ROI selection on a single WSI, see [Running Experiments](docs/RUN_EXPERIMENTS.md).

## 🧭 Single-WSI Agent Modes

The single-slide entry point is:

```bash
python src/Pathology_agent/evaluate/run_single_slide.py
```

It supports three main agentic-use modes:

| Mode | Meaning |
| --- | --- |
| `aml_roi` | ROI selection only. |
| `aml_auto` | ROI selection followed by VLM diagnosis. |
| `aml_stamp` | STAMP prediction from an existing agent-selected `roi_collection.json`. |

More complete examples are provided in [docs/RUN_EXPERIMENTS.md](docs/RUN_EXPERIMENTS.md).

## 🔗 How The Workflow Connects

The wrapper scripts execute the source code shipped in this repository:

- STAMP workflows run `python -m stamp` using `PYTHONPATH=src/STAMP/src`.
- Manual-ROI VLM workflows run `src/VLM_manual_rois/Run_VLM.py`.
- Agent workflows run `src/Pathology_agent/evaluate/run_batch_aml_staged.sh` using both `src/Pathology_agent` and `src/STAMP/src`.

## 🔒 Data And Model Access

The manuscript results require protected clinical data and trained components that cannot be redistributed publicly. The code is provided so that the workflow structure, implementation details, and execution entry points are transparent.

See:

- [Input schema](docs/INPUT_SCHEMA.md)
- [Data availability](docs/DATA_AVAILABILITY.md)
- [Source code scope](docs/SOURCE_CODE_SCOPE.md)
- [Running experiments](docs/RUN_EXPERIMENTS.md)

## 🙏 Acknowledgements

This repository includes and builds on the STAMP framework for weakly supervised pathology modeling. Please see the STAMP project for details about the underlying STAMP package and citation information:

https://github.com/KatherLab/STAMP
