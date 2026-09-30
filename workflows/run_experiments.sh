#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="src"
DATA_ROOT="/data/private"
OUTPUT_DIR="results"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      shift
      ;;
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --project-root "${PROJECT_ROOT}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --execute
)

echo "1/4 STAMP all tiles"
"${PYTHON_BIN}" workflows/01_stamp_all_tiles.py "${COMMON_ARGS[@]}"

echo "2/4 STAMP manual ROIs"
"${PYTHON_BIN}" workflows/02_stamp_manual_rois.py "${COMMON_ARGS[@]}"

echo "3/4 VLM manual ROIs"
"${PYTHON_BIN}" workflows/03_vlm_manual_rois.py "${COMMON_ARGS[@]}"

echo "4/4 Agent ROI selection and downstream diagnosis"
"${PYTHON_BIN}" workflows/04_agent_roi_selection_and_diagnosis.py "${COMMON_ARGS[@]}"

echo "Experiment workflow completed."
